"""Stage 3 shortcut screen: does anything other than the object itself predict the label?

Usage:
  Part 1 - non-content attributes (cheap, always run):
    python shortcut_screen.py attributes --audit image_audit.csv --meta metadata.csv --label label
        [--key file] [--extra camera_id,site] [--group capture_session]
  Part 2 - region masking with a probe (image content outside the object):
    python shortcut_screen.py masking --images IMAGE_DIR --meta metadata.csv --label label [--key file]
        [--group capture_session] [--per-class 60] [--model vit_small_patch14_dinov2.lvd142m] [--img-size 224]

Part 1 scores each attribute ALONE as a label predictor (cross-validated, grouped if --group):
file format, colour mode, width, height, file size, EXIF orientation, near-constant border, brightness,
contrast, blur, plus any --extra metadata columns (camera, site, scanner). An attribute that predicts the
label well above chance is either a legitimate cue or a shortcut; decide which, in writing.

Part 2 runs two complementary tests on held-out groups:
  "cue used"    fit a linear probe on unmasked images, re-embed a per-class sample with one region
                blanked (4 corners at 15% of the side, an 8% border, the centre as a control) and
                report per-class recall changes. Shows whether THIS probe relies on a region.
  "cue present" fit a probe on each peripheral region ALONE (corner crops; the border with the
                interior blanked) and report per-class recall and precision. Shows whether the DATA
                contains a cue there, whatever model is trained later. A cue can be invisible to the
                masking test (when the object alone is enough for the probe) yet obvious to this one.
Blanked regions are filled with the image mean, not black, to avoid creating a new cue.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = {"top_left": (0, 0, .15, .15), "top_right": (.85, 0, 1, .15), "bottom_left": (0, .85, .15, 1),
           "bottom_right": (.85, .85, 1, 1), "border_8pct": None, "centre_control": (.35, .35, .65, .65)}


def attributes(a) -> None:
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    au = pd.read_csv(a.audit)
    meta = pd.read_csv(a.meta)
    df = meta.merge(au[au["readable"]], left_on=a.key, right_on="file", how="inner", suffixes=("", "_audit"))
    y = df[a.label].astype(str)
    base = y.value_counts(normalize=True).max()
    cols = ["format", "mode", "width", "height", "bytes", "exif_orientation", "border_uniform", "mean", "std", "blur_var"]
    cols += [c for c in a.extra.split(",") if c]
    cv = (StratifiedGroupKFold(5, shuffle=True, random_state=0) if a.group else StratifiedKFold(5, shuffle=True, random_state=0))
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        x = df[[c]].copy()
        if not pd.api.types.is_numeric_dtype(x[c]):
            x[c] = pd.factorize(x[c].astype(str))[0]
        acc = cross_val_score(DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=0), x.to_numpy(float), y,
                              cv=cv, groups=df[a.group] if a.group else None, scoring="balanced_accuracy").mean()
        top = (df.groupby(y)[c].agg(lambda s: s.astype(str).value_counts(normalize=True).index[0]
                                    if not pd.api.types.is_numeric_dtype(s) else round(float(s.median()), 2)))
        rows.append({"attribute": c, "balanced_acc_alone": round(float(acc), 3), "per_class_typical": top.to_dict()})
    res = pd.DataFrame(rows).sort_values("balanced_acc_alone", ascending=False)
    chance = 1 / y.nunique()
    print(f"classes={y.nunique()} chance balanced accuracy={chance:.3f} (majority share {base:.3f})")
    print(res.to_string(index=False))
    flagged = res[res.balanced_acc_alone > chance + 0.1]
    print(f"\n{len(flagged)} attribute(s) predict the label well above chance on their own: "
          f"{list(flagged.attribute)}. For each, write whether it is a legitimate cue or a shortcut.")


def masking(a) -> None:
    import timm
    import torch
    from PIL import Image
    from sklearn.linear_model import LogisticRegression

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from embed_images import load_rgb

    meta = pd.read_csv(a.meta)
    root = Path(a.images)
    ok = [(f, l) for f, l in zip(meta[a.key], meta[a.label].astype(str)) if (root / f).exists()]
    rng = np.random.default_rng(0)
    df = pd.DataFrame(ok, columns=["file", "label"])
    if a.group:
        df["group"] = meta.set_index(a.key).loc[df.file, a.group].to_numpy()
    model = timm.create_model(a.model, pretrained=True, num_classes=0, img_size=a.img_size).eval()
    cfg = timm.data.resolve_data_config({"input_size": (3, a.img_size, a.img_size)}, model=model)
    tf = timm.data.create_transform(**cfg)

    def embed(imgs):
        with torch.inference_mode():
            return torch.nn.functional.normalize(model(torch.stack([tf(i) for i in imgs])), dim=1).numpy()

    # region crops are small, so the cue-present test embeds them at a lower size (default 112 px: ~4x cheaper)
    rsize = a.region_size
    rmodel = model if rsize == a.img_size else timm.create_model(a.model, pretrained=True, num_classes=0, img_size=rsize).eval()
    rtf = timm.data.create_transform(**timm.data.resolve_data_config({"input_size": (3, rsize, rsize)}, model=rmodel))

    def rembed(imgs):
        with torch.inference_mode():
            return torch.nn.functional.normalize(rmodel(torch.stack([rtf(i) for i in imgs])), dim=1).numpy()

    def load(f):
        try:
            return load_rgb(root / f)
        except Exception:  # noqa: BLE001
            return None

    # hold out whole groups for the masked evaluation sample
    groups = df["group"] if a.group else df["file"]
    uniq = groups.unique()
    rng.shuffle(uniq)
    test_groups = set(uniq[: max(1, len(uniq) // 4)])
    is_test = groups.isin(test_groups).to_numpy()
    def per_class(frame, n):  # explicit loop: pandas 3 groupby.apply drops the grouping column
        return pd.concat([g.sample(min(len(g), n), random_state=0) for _, g in frame.groupby("label")])

    train = per_class(df[~is_test], 4 * a.per_class)
    test = per_class(df[is_test], a.per_class)
    tr_imgs = [(load(f), l) for f, l in zip(train.file, train.label)]
    tr_imgs = [(i, l) for i, l in tr_imgs if i is not None]
    if not a.cue_only:
        Xtr = np.concatenate([embed([i for i, _ in tr_imgs[s:s + 64]]) for s in range(0, len(tr_imgs), 64)])
        probe = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr * 10, [l for _, l in tr_imgs])
    te = [(load(f), l) for f, l in zip(test.file, test.label)]
    te = [(i, l) for i, l in te if i is not None]
    labels = np.array([l for _, l in te])

    def masked(img, region):
        arr = np.asarray(img).copy()
        h, w = arr.shape[:2]
        fill = arr.reshape(-1, 3).mean(0).astype(arr.dtype)
        if region is None:
            b = max(1, int(0.08 * min(h, w)))
            arr[:b], arr[-b:], arr[:, :b], arr[:, -b:] = fill, fill, fill, fill
        else:
            x0, y0, x1, y1 = region
            arr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = fill
        return Image.fromarray(arr)

    def recall(imgs):
        pred = np.concatenate([probe.predict(embed(imgs[s:s + 64]) * 10) for s in range(0, len(imgs), 64)])
        return pd.Series(pred == labels).groupby(labels).mean()

    if not a.cue_only:
        base = recall([i for i, _ in te])
        table = {"none": base}
        for name, reg in REGIONS.items():
            table[name] = recall([masked(i, reg) for i, _ in te])
        res = pd.DataFrame(table).round(3)
        drop = res.drop(columns="none").rsub(res["none"], axis=0)
        print(f"probe trained on {len(tr_imgs)} images; masked evaluation on {len(te)} held-out images (whole groups held out)")
        print("recall per class with each region masked:\n" + res.to_string())
        worst = drop.drop(columns="centre_control").stack().sort_values(ascending=False).head(5)
        print("\nlargest recall drops from masking a peripheral region (class, region, drop):")
        for (cls, reg), d in worst.items():
            print(f"  {cls:15s} {reg:14s} {d:+.3f}")
        print("A drop of more than about 0.10 from a small corner or border is a shortcut candidate: inspect those images.")

    # "Cue present" test: can a probe predict the class from a peripheral region ALONE?
    # Masking only shows whether THIS probe uses a region; a model trained later may still latch onto a
    # cue the data contains. Region-only prediction finds the cue regardless of the model.
    def region_only(img, region):
        if region is None:  # keep the border, blank the interior
            arr = np.asarray(img).copy()
            h, w = arr.shape[:2]
            b = max(1, int(0.08 * min(h, w)))
            arr[b:-b, b:-b] = arr.reshape(-1, 3).mean(0).astype(arr.dtype)
            return Image.fromarray(arr)
        w, h = img.size
        x0, y0, x1, y1 = region
        return img.crop((int(x0 * w), int(y0 * h), max(int(x1 * w), int(x0 * w) + 1), max(int(y1 * h), int(y0 * h) + 1)))

    tr_small = [(i, l) for i, l in tr_imgs]
    tr_small = [x for c in sorted({l for _, l in tr_small}) for x in [t for t in tr_small if t[1] == c][: 2 * a.per_class]]
    y_tr = [l for _, l in tr_small]
    cue = {}
    for name, reg in REGIONS.items():
        if name == "centre_control":
            continue
        Xr = np.concatenate([rembed([region_only(i, reg) for i, _ in tr_small[s:s + 64]]) for s in range(0, len(tr_small), 64)])
        pr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(Xr * 10, y_tr)
        Xt = np.concatenate([rembed([region_only(i, reg) for i, _ in te[s:s + 64]]) for s in range(0, len(te), 64)])
        pred = pr.predict(Xt * 10)
        rec = pd.Series(pred == labels).groupby(labels).mean()
        prec = pd.Series(pred == labels).groupby(pred).mean().reindex(rec.index)
        cue[name] = rec.round(3)
        cue[name + "_precision"] = prec.round(3)
    cue_df = pd.DataFrame(cue)
    chance = 1 / len(set(labels))
    print(f"\n'cue present' test: per-class recall (and precision) from a peripheral region ALONE; chance = {chance:.2f}")
    print(cue_df.to_string())
    hits = [(c, r, cue_df.at[c, r], cue_df.at[c, r + "_precision"]) for r in REGIONS if r != "centre_control"
            for c in cue_df.index if cue_df.at[c, r] > chance + 0.3 and (cue_df.at[c, r + "_precision"] or 0) > 0.5]
    corners = ["top_left", "top_right", "bottom_left", "bottom_right"]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c, r, rc, pc in sorted(hits, key=lambda x: -x[2]):
        print(f"  CUE: class '{c}' recognisable from {r} alone (recall {rc:.2f}, precision {pc:.2f}): inspect that region")
        if r in corners:
            flagged = [k for k in corners if (c, k) in {(h[0], h[1]) for h in hits}]
            if len(flagged) == 1:
                print(f"     only ONE corner flags '{c}': object geometry is symmetric, so this points to an artefact "
                      "(watermark, stamp, burnt-in text, ruler, frame) rather than the object itself")
        # contact sheet: the region cropped from this class (top row) and from other classes (bottom row)
        reg = REGIONS.get(r)
        mine = [i for i, l in te if l == c][:12]
        other = [i for i, l in te if l != c][:12]
        tile = 96
        sheet = Image.new("RGB", (tile * 12, tile * 2), "white")
        for row, imgs in enumerate((mine, other)):
            for k, im in enumerate(imgs):
                sheet.paste(region_only(im, reg).convert("RGB").resize((tile, tile)), (k * tile, row * tile))
        path = out_dir / f"cue_{c}_{r}.png"
        sheet.save(path)
        print(f"     LOOK at {path} before judging (top row: '{c}', bottom row: other classes). "
              "Decide from what the crops show, not from a plausible story.")
    if not hits:
        print("  no peripheral region predicts any class well on its own")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("attributes")
    p1.add_argument("--audit", required=True)
    p1.add_argument("--meta", required=True)
    p1.add_argument("--label", required=True)
    p1.add_argument("--key", default="file")
    p1.add_argument("--extra", default="")
    p1.add_argument("--group", default="")
    p2 = sub.add_parser("masking")
    p2.add_argument("--images", required=True)
    p2.add_argument("--meta", required=True)
    p2.add_argument("--label", required=True)
    p2.add_argument("--key", default="file")
    p2.add_argument("--group", default="")
    p2.add_argument("--per-class", type=int, default=60)
    p2.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    p2.add_argument("--img-size", type=int, default=224)
    p2.add_argument("--out-dir", default=".", help="where cue contact sheets are saved")
    p2.add_argument("--cue-only", action="store_true", help="run only the 'cue present' test (the part that finds "
                    "watermarks and stamps); skips the masking half")
    p2.add_argument("--region-size", type=int, default=112, help="embedding size for region crops (patch-14 models: 112, 168, 224)")
    a = ap.parse_args()
    attributes(a) if a.cmd == "attributes" else masking(a)


if __name__ == "__main__":
    main()
