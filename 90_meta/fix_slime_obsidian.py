#!/usr/bin/env python3
"""Fix slime_reading Obsidian vault integration: renames, wikilinks, H1 titles."""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLIME = ROOT / "slime_reading"

# Vault-level collisions: rename slime copies with Slime- prefix
RENAMES = [
    ("00-方法论/00-方法论-00-MOC.md", "00-方法论/Slime-00-方法论-00-MOC.md"),
    ("00-方法论/00-方法论-01-核心概念.md", "00-方法论/Slime-00-方法论-01-核心概念.md"),
    ("00-方法论/00-方法论-02-源码走读.md", "00-方法论/Slime-00-方法论-02-源码走读.md"),
    ("00-方法论/00-方法论-03-数据流与交互.md", "00-方法论/Slime-00-方法论-03-数据流与交互.md"),
    ("00-方法论/00-方法论-04-关键问题.md", "00-方法论/Slime-00-方法论-04-关键问题.md"),
    ("00-方法论/00-方法论-05-checkpoint.md", "00-方法论/Slime-00-方法论-05-checkpoint.md"),
    ("01-启动与入口/01-启动与入口-00-MOC.md", "01-启动与入口/Slime-01-启动与入口-00-MOC.md"),
    ("PLAN.md", "Slime-PLAN.md"),
    ("progress.md", "Slime-progress.md"),
]

# Old wikilink stem -> new (within slime_reading and vault index for slime rows)
WIKILINK_MAP = {
    "00-方法论-00-MOC": "Slime-00-方法论-00-MOC",
    "00-方法论-01-核心概念": "Slime-00-方法论-01-核心概念",
    "00-方法论-02-源码走读": "Slime-00-方法论-02-源码走读",
    "00-方法论-03-数据流与交互": "Slime-00-方法论-03-数据流与交互",
    "00-方法论-04-关键问题": "Slime-00-方法论-04-关键问题",
    "00-方法论-05-checkpoint": "Slime-00-方法论-05-checkpoint",
    "01-启动与入口-00-MOC": "Slime-01-启动与入口-00-MOC",
    "PLAN": "Slime-PLAN",
    "progress": "Slime-progress",
}

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:\|([^\]]+))?\]\]")


def move_file(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"SKIP missing: {src.relative_to(ROOT)}")
        return
    if dst.exists():
        print(f"SKIP exists: {dst.relative_to(ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"RENAMED {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def rename_files() -> None:
    for old_rel, new_rel in RENAMES:
        move_file(SLIME / old_rel, SLIME / new_rel)


def replace_wikilinks_in_text(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    changes = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal changes
        target, alias = m.group(1), m.group(2)
        base = target.split("/")[-1]
        if base in mapping:
            changes += 1
            new_target = mapping[base]
            if alias:
                return f"[[{new_target}|{alias}]]"
            return f"[[{new_target}]]"
        return m.group(0)

    return WIKILINK_RE.sub(repl, text), changes


def fix_wikilinks_in_dir(directory: Path, mapping: dict[str, str]) -> int:
    total = 0
    for md in sorted(directory.rglob("*.md")):
        if "_TEMPLATE" in md.parts:
            continue
        text = md.read_text(encoding="utf-8")
        new_text, n = replace_wikilinks_in_text(text, mapping)
        if n:
            md.write_text(new_text, encoding="utf-8")
            total += n
            print(f"  wikilinks {n:3d}  {md.relative_to(ROOT)}")
    return total


def parse_frontmatter_title(text: str) -> str | None:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None
    for line in m.group(1).splitlines():
        if line.startswith("title:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    return None


def fix_h1_from_title(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    title = parse_frontmatter_title(text)
    if not title:
        return False
    m = re.search(r"^# .+$", text, re.MULTILINE)
    if not m:
        return False
    old_h1 = m.group(0)
    new_h1 = f"# {title}"
    if old_h1 == new_h1:
        return False
    if not (old_h1.startswith("# 批次") or old_h1.startswith("# 阶段")):
        return False
    new_text = text[: m.start()] + new_h1 + text[m.end() :]
    path.write_text(new_text, encoding="utf-8")
    print(f"  H1  {path.relative_to(ROOT)}: {old_h1!r} -> {new_h1!r}")
    return True


def fix_h1_titles() -> int:
    count = 0
    for md in sorted(SLIME.rglob("*.md")):
        if "_TEMPLATE" in md.parts:
            continue
        if fix_h1_from_title(md):
            count += 1
    return count


def main() -> int:
    print("=== 1. Rename colliding slime files ===")
    rename_files()

    print("\n=== 2. Update wikilinks in slime_reading ===")
    n_slime = fix_wikilinks_in_dir(SLIME, WIKILINK_MAP)

    print("\n=== 3. Update wikilinks in vault index (slime-specific only) ===")
    index = ROOT / "index.md"
    idx_text = index.read_text(encoding="utf-8")
    # Only replace in Slime section: methodology + stage I MOC rows
    slime_section = idx_text.split("## 阶段 MOC · Slime", 1)
    if len(slime_section) == 2:
        head, tail = slime_section
        tail_parts = tail.split("---", 1)
        slime_table = tail_parts[0]
        rest = "---" + tail_parts[1] if len(tail_parts) > 1 else ""
        new_table, n_idx = replace_wikilinks_in_text(slime_table, WIKILINK_MAP)
        idx_text = head + "## 阶段 MOC · Slime" + new_table + rest
    # Fix maintenance + quick entry ambiguous links
    idx_text = idx_text.replace(
        "| 阅读进度 | [[progress]] |",
        "| SGLang 阅读进度 | [[progress]] |\n| Slime 阅读进度 | [[Slime-progress]] |",
    )
    idx_text = idx_text.replace(
        "| [[PLAN]] | 写作规范与批次计划 |",
        "| [[PLAN]] | SGLang 写作规范与批次计划 |\n| [[Slime-PLAN]] | Slime 写作规范与批次计划 |",
    )
    index.write_text(idx_text, encoding="utf-8")
    print(f"  index.md updated")

    print("\n=== 4. Fix H1 titles (remove 批次 NN from headings) ===")
    h1_count = fix_h1_titles()
    print(f"  {h1_count} files updated")

    print("\n=== SUMMARY ===")
    print(f"wikilink replacements in slime: {n_slime}")
    print(f"H1 fixes: {h1_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
