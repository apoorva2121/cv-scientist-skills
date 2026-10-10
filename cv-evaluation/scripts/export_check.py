"""Stage 10-11: export a trained classifier to ONNX, prove parity with PyTorch on real images, and measure latency.

Usage:
    python export_check.py --model runs/final/model.pt --image-dir IMAGE_DIR --files files.txt [--n 200]
        [--int8] [--threads 4] [--out export/]
    python export_check.py --module my_model.py:build --image-dir IMAGE_DIR --files files.txt [...]
    python export_check.py --module ../../cv-training/scripts/train_probe.py:load_probe --module-arg runs/probe/model.pt ...

--model is the model.pt that cv-training/scripts/train_classifier.py writes (timm architecture, classes and config).
--module FILE:FUNC is for any other classifier (e.g. a frozen backbone plus a linear head fitted in scikit-learn):
FUNC() returns (model, preprocess, classes), where model is a torch.nn.Module mapping a (N,3,H,W) float
batch to logits, preprocess(path) returns the (3,H,W) tensor exactly as evaluation produced it, and classes
is the class list. Put the head inside the module (copy the sklearn coef_/intercept_ into an nn.Linear).
Detectors are not covered: the legacy exporter fails on torchvision Faster R-CNN (the proposal count gets
fixed from the dummy input); keep detectors in PyTorch or use the dynamo exporter once onnxscript is installed.
The same evaluation transform (decode with EXIF, resize/crop policy, normalisation) feeds both runtimes, so
the test isolates the export itself. Serving code must reproduce that transform: check it with
cv-preprocessing/scripts/serve_parity.py.

What it does (facts checked 2026-09-28 in the cv-skills environment):
  1. Exports with the legacy TorchScript-based exporter (dynamo=False). In torch 2.14 the default
     dynamo exporter needs the `onnxscript` package; install it to use the new exporter. The legacy path
     still works but warns that it is deprecated.
  2. Parity on N real images: max |p_torch - p_onnx|, argmax agreement. Fails (exit 1) if any argmax
     differs or the probability difference exceeds --atol.
  3. Latency, batch 1, after warm-up: p50 / p90 / p99 in ms for PyTorch and ONNX Runtime (CPU EP), with
     the thread count stated. Latency depends on hardware, threads and other load: report all three.
  4. --int8: static QDQ INT8 quantisation (onnxruntime.quantization.quantize_static) calibrated on up to
     100 of the images, then parity against fp32 and latency again. INT8 accuracy can drop (and some
     CPUs without VNNI saturate activations): keep INT8 only if argmax agreement and the task metric hold.
Outputs: model.onnx (+ model_int8.onnx), export_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "cv-data-audit" / "scripts"))
sys.path.insert(0, str(HERE.parents[1] / "cv-training" / "scripts"))


def pct(ts: list[float]) -> dict:
    a = np.array(ts) * 1000
    return {"p50_ms": round(float(np.percentile(a, 50)), 2), "p90_ms": round(float(np.percentile(a, 90)), 2),
            "p99_ms": round(float(np.percentile(a, 99)), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--module", default="", help="FILE:FUNC returning (model, preprocess, classes)")
    ap.add_argument("--module-arg", action="append", default=[], help="argument passed to FUNC (repeatable), e.g. a model path")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--files", required=True, help="text file with one image file name per line")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="export")
    a = ap.parse_args()

    import onnxruntime as ort
    import timm
    import torch
    import torchvision.transforms.v2 as T

    from embed_images import load_rgb
    from train_classifier import eval_resize

    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [f.strip() for f in Path(a.files).read_text().splitlines() if f.strip()][: a.n]
    if a.module:
        import importlib.util
        mpath, fname = a.module.rsplit(":", 1)
        spec = importlib.util.spec_from_file_location("export_module", mpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model, preprocess, classes = getattr(mod, fname)(*a.module_arg)
        model.eval()
        X = torch.stack([preprocess(str(Path(a.image_dir) / f)) for f in files])
        cfg_run = {"model": a.module, "img_size": int(X.shape[-1])}
    else:
        if not a.model:
            raise SystemExit("give --model (train_classifier model.pt) or --module FILE:FUNC")
        ck = torch.load(a.model, map_location="cpu", weights_only=True)  # tensors + plain config only
        cfg_run = ck["config"]
        model = timm.create_model(cfg_run["model"], pretrained=False, num_classes=len(ck["classes"]))
        model.load_state_dict(ck["state_dict"])
        model.eval()
        dcfg = timm.data.resolve_data_config({}, model=model)
        tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(dcfg["mean"], dcfg["std"])])
        fn = eval_resize(cfg_run.get("eval_transform", "crop"), cfg_run["img_size"])
        X = torch.stack([tf(fn(load_rgb(Path(a.image_dir) / f))) for f in files])
    size = cfg_run["img_size"]

    onnx_path = out / "model.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, (X[:1],), str(onnx_path), dynamo=False, input_names=["image"], output_names=["logits"],
                          dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads

    def ort_run(path):
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        logits = np.concatenate([sess.run(None, {"image": X[i:i + 32].numpy()})[0] for i in range(0, len(X), 32)])
        return sess, logits

    def softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    with torch.inference_mode():
        p_t = torch.softmax(torch.cat([model(X[i:i + 32]) for i in range(0, len(X), 32)]), 1).numpy()
    sess, z_o = ort_run(onnx_path)
    p_o = softmax(z_o)
    parity = {"n_images": len(files), "max_abs_prob_diff": float(np.abs(p_t - p_o).max()),
              "argmax_agreement": float((p_t.argmax(1) == p_o.argmax(1)).mean())}

    def latency(call, reps=100, warm=10):
        x1 = X[:1]
        for _ in range(warm):
            call(x1)
        ts = []
        for i in range(reps):
            x1 = X[i % len(X)][None]
            t0 = time.perf_counter()
            call(x1)
            ts.append(time.perf_counter() - t0)
        return pct(ts)

    with torch.inference_mode():
        lat_t = latency(lambda x: model(x))
    lat_o = latency(lambda x: sess.run(None, {"image": x.numpy()}))
    report = {"model": cfg_run["model"], "img_size": size, "threads": a.threads, "exporter": "torch.onnx.export dynamo=False, opset 17",
              "onnxruntime": ort.__version__, "torch": torch.__version__, "parity_fp32": parity,
              "latency_batch1": {"pytorch": lat_t, "onnxruntime_cpu": lat_o},
              "onnx_size_mb": round(onnx_path.stat().st_size / 1e6, 2)}
    ok = parity["argmax_agreement"] == 1.0 and parity["max_abs_prob_diff"] <= a.atol

    if a.int8:
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter([{"image": X[i:i + 1].numpy()} for i in range(min(100, len(X)))])

            def get_next(self):
                return next(self.it, None)

        q_path = out / "model_int8.onnx"
        quantize_static(str(onnx_path), str(q_path), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
        sess_q, z_q = ort_run(q_path)
        p_q = softmax(z_q)
        report["int8"] = {"calibration_images": min(100, len(X)),
                          "argmax_agreement_vs_fp32": float((p_q.argmax(1) == p_o.argmax(1)).mean()),
                          "max_abs_prob_diff_vs_fp32": float(np.abs(p_q - p_o).max()),
                          "latency_batch1": latency(lambda x: sess_q.run(None, {"image": x.numpy()})),
                          "size_mb": round(q_path.stat().st_size / 1e6, 2),
                          "note": "calibration images are also in the agreement set here; confirm the task metric on held-out data"}
    report["verdict"] = "PASS: export matches PyTorch" if ok else "FAIL: export differs from PyTorch; do not ship it"
    (out / "export_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
