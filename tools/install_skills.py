"""Install (or refresh) the ten CV skills into the Claude Code user skills folder.

    python tools/install_skills.py            # copy cv-*/ into ~/.claude/skills/, replacing older copies
    python tools/install_skills.py --dry-run  # show what would change
    python tools/install_skills.py --target D:/other/skills

Copies SKILL.md, references/ and scripts/ of every cv-* folder, skipping caches. The repository stays the
source of truth: rerun after improving a skill. Skill scripts find each other through the sibling folders,
so all ten must be installed side by side.
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=str(Path.home() / ".claude" / "skills"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    target = Path(a.target)
    skills = sorted(p for p in ROOT.glob("cv-*") if (p / "SKILL.md").exists())
    if len(skills) != 10:
        print(f"WARNING: expected 10 skills, found {len(skills)}")
    for s in skills:
        dst = target / s.name
        state = "new" if not dst.exists() else ("unchanged" if not filecmp.dircmp(s, dst, ignore=["__pycache__"]).diff_files
                                                  and (dst / "SKILL.md").read_bytes() == (s / "SKILL.md").read_bytes() else "update")
        print(f"{s.name:22s} {state}")
        if a.dry_run:
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(s, dst, ignore=IGNORE)
    if not a.dry_run:
        print(f"installed {len(skills)} skills into {target}")


if __name__ == "__main__":
    main()
