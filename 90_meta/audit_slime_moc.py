#!/usr/bin/env python3
"""Audit slime_reading batch docs: six-piece set, code blocks, tags, Mermaid."""
import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent / "slime_reading"
doc_suffixes = [
    ("00-MOC", "moc"),
    ("01-核心概念", "concept"),
    ("02-源码走读", "walkthrough"),
    ("03-数据流与交互", "dataflow"),
    ("04-关键问题", "faq"),
    ("05-checkpoint", "checkpoint"),
]

# Batch 30 uses onboard naming + index docs (AGENT-DISPATCH §批次30).
BATCH30_REQUIRED = [
    "08-总结与索引-00-MOC.md",
    "08-总结与索引-01-项目总览.md",
    "08-总结与索引-02-架构分层.md",
    "08-总结与索引-03-关键概念.md",
    "08-总结与索引-04-导读路径.md",
    "08-总结与索引-05-文件地图.md",
    "08-总结与索引-06-复杂度热点.md",
    "08-总结与索引-07-可观测与CI.md",
    "全链路RL训练追踪.md",
    "Slime-业务域流程.md",
    "Slime-模块依赖图.md",
    "Slime-术语表.md",
    "与SGLang阅读对照.md",
    "08-总结与索引-05-checkpoint.md",
]
BATCH30_CODE_MIN = 80
BATCH30_DOC_TYPES = {
    "moc",
    "concept",
    "walkthrough",
    "dataflow",
    "faq",
    "checkpoint",
    "index",
}

STUB_LENGTH_EXEMPT_SUFFIXES = {"00-MOC", "05-checkpoint"}
PLACEHOLDER_MARKERS = ("（模块职责）", "待补充")

CODE_BLOCK_RE = re.compile(r"^```", re.MULTILINE)
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
MERMAID_BLOCK_RE = re.compile(r"```mermaid\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
FENCED_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)


def parse_frontmatter(text: str) -> dict:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}
    data = {}
    for line in m.group(1).splitlines():
        if line.strip().startswith("- "):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            data[key.strip()] = val.strip().strip('"').strip("'")
    return data


def parse_tags(text: str) -> list[str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return []
    tags = []
    in_tags = False
    for line in m.group(1).splitlines():
        if line.strip().startswith("tags:"):
            in_tags = True
            continue
        if in_tags:
            if line.strip().startswith("- "):
                tags.append(line.strip()[2:].strip())
            elif line and not line.startswith(" "):
                in_tags = False
    return tags


def is_batch_dir(moc: Path) -> bool:
    prefix = moc.stem.replace("-00-MOC", "")
    if prefix != moc.parent.name:
        return False
    if "_TEMPLATE" in str(moc):
        return False
    text = moc.read_text(encoding="utf-8", errors="ignore")
    fm = parse_frontmatter(text)
    batch = fm.get("batch", "")
    return bool(re.fullmatch(r"\d{2}", batch))


def count_code_blocks(text: str) -> int:
    return len(CODE_BLOCK_RE.findall(text)) // 2 if "```" in text else text.count("```") // 2


def prose_without_code(text: str) -> str:
    return FENCED_BLOCK_RE.sub("", text)


def has_placeholder(text: str) -> bool:
    prose = prose_without_code(text)
    if any(marker in prose for marker in PLACEHOLDER_MARKERS):
        return True
    return bool(re.search(r"(?<![\w/])TODO:", prose))


def is_stub(text: str, suffix: str) -> bool:
    if suffix in STUB_LENGTH_EXEMPT_SUFFIXES:
        return has_placeholder(text)
    if len(text.strip()) < 500:
        return True
    return has_placeholder(text)


def find_mermaid_backslash_n(text: str) -> list[str]:
    issues = []
    for i, block in enumerate(MERMAID_BLOCK_RE.findall(text), 1):
        if "\\n" in block:
            issues.append(f"mermaid#{i}")
    return issues


def has_batch30_doc_tag(tags: list[str]) -> bool:
    for tag in tags:
        if tag.startswith("slime/doc/"):
            doc_type = tag.split("/", 2)[-1]
            if doc_type in BATCH30_DOC_TYPES:
                return True
    return False


def audit_batch30(batch_dir: Path, prefix: str, batch_num: str) -> tuple:
    actual = {f.name for f in batch_dir.glob("*.md")}
    missing = [fname for fname in BATCH30_REQUIRED if fname not in actual]

    stubs = []
    tag_issues = []
    mermaid_issues = []
    total_code_blocks = 0

    for md in sorted(batch_dir.glob("*.md")):
        text = md.read_text(encoding="utf-8", errors="ignore")
        total_code_blocks += count_code_blocks(text)

    for fname in BATCH30_REQUIRED:
        fpath = batch_dir / fname
        if not fpath.exists():
            continue
        text = fpath.read_text(encoding="utf-8", errors="ignore")
        suffix = "05-checkpoint" if fname.endswith("-05-checkpoint.md") else (
            "00-MOC" if fname.endswith("-00-MOC.md") else fname
        )
        if is_stub(text, suffix):
            stubs.append((fname, "placeholder" if has_placeholder(text) else len(text)))

        tags = parse_tags(text)
        if f"slime/batch/{batch_num}" not in tags:
            tag_issues.append(f"{fname}: missing slime/batch/{batch_num}")
        if not has_batch30_doc_tag(tags):
            tag_issues.append(f"{fname}: missing slime/doc/* (onboard types)")

        m_issues = find_mermaid_backslash_n(text)
        if m_issues:
            mermaid_issues.append((fname, m_issues))

    code_issue = total_code_blocks < BATCH30_CODE_MIN
    return missing, stubs, total_code_blocks, code_issue, tag_issues, mermaid_issues


def audit_standard_batch(batch_dir: Path, prefix: str, batch_num: str) -> tuple:
    actual = {f.name for f in batch_dir.glob("*.md")}
    missing = []
    for suffix, _ in doc_suffixes:
        fname = f"{prefix}-{suffix}.md"
        if fname not in actual:
            missing.append(fname)

    stubs = []
    tag_issues = []
    mermaid_issues = []
    total_code_blocks = 0

    for suffix, doc_type in doc_suffixes:
        fname = f"{prefix}-{suffix}.md"
        fpath = batch_dir / fname
        if not fpath.exists():
            continue
        text = fpath.read_text(encoding="utf-8", errors="ignore")
        total_code_blocks += count_code_blocks(text)

        if is_stub(text, suffix):
            stubs.append((fname, "placeholder" if has_placeholder(text) else len(text)))

        tags = parse_tags(text)
        if f"slime/batch/{batch_num}" not in tags:
            tag_issues.append(f"{fname}: missing slime/batch/{batch_num}")
        if f"slime/doc/{doc_type}" not in tags:
            tag_issues.append(f"{fname}: missing slime/doc/{doc_type}")

        m_issues = find_mermaid_backslash_n(text)
        if m_issues:
            mermaid_issues.append((fname, m_issues))

    code_issue = total_code_blocks < 15
    return missing, stubs, total_code_blocks, code_issue, tag_issues, mermaid_issues


batch_dirs = []
for moc in root.rglob("*-00-MOC.md"):
    if is_batch_dir(moc):
        batch_dirs.append(moc.parent)

results_missing = []
results_stubs = []
results_code = []
results_tags = []
results_mermaid = []

for batch_dir in sorted(set(batch_dirs), key=lambda p: p.as_posix()):
    mocs = list(batch_dir.glob("*-00-MOC.md"))
    if not mocs:
        continue
    moc = mocs[0]
    prefix = moc.stem.replace("-00-MOC", "")
    fm = parse_frontmatter(moc.read_text(encoding="utf-8", errors="ignore"))
    batch_num = fm.get("batch", "??")

    if batch_num == "30":
        missing, stubs, total_code, code_issue, tag_issues, mermaid_issues = audit_batch30(
            batch_dir, prefix, batch_num
        )
        dirpath = str(batch_dir.relative_to(root))
    else:
        missing, stubs, total_code, code_issue, tag_issues, mermaid_issues = audit_standard_batch(
            batch_dir, prefix, batch_num
        )
        dirpath = str(batch_dir.relative_to(root))

    if missing:
        results_missing.append((prefix, batch_num, dirpath, missing))
    if stubs:
        results_stubs.append((prefix, batch_num, stubs))
    if code_issue:
        results_code.append((prefix, batch_num, total_code))
    if tag_issues:
        results_tags.append((prefix, batch_num, tag_issues))
    if mermaid_issues:
        results_mermaid.append((prefix, batch_num, mermaid_issues))

print("=== SLIME READING AUDIT ===")
print(f"Root: {root}")
print(f"Batch dirs scanned: {len(set(batch_dirs))}")

print("\n=== MISSING DOCS (six-piece / batch30 required) ===")
if not results_missing:
    print("(none)")
else:
    for prefix, batch_num, dirpath, missing in results_missing:
        print(f"\n批 {batch_num} {prefix} ({dirpath})")
        print(f"  missing: {missing}")

print("\n=== STUBS/THIN (<500 chars or placeholder) ===")
if not results_stubs:
    print("(none)")
else:
    for prefix, batch_num, stubs in results_stubs:
        print(f"批 {batch_num} {prefix}: {stubs}")

print("\n=== CODE BLOCKS (<15 per batch, batch30 <80) ===")
if not results_code:
    print("(none — all batches meet threshold)")
else:
    for prefix, batch_num, count in results_code:
        print(f"批 {batch_num} {prefix}: {count} blocks")

print("\n=== FRONTMATTER TAG ISSUES ===")
if not results_tags:
    print("(none)")
else:
    for prefix, batch_num, issues in results_tags:
        print(f"批 {batch_num} {prefix}:")
        for issue in issues:
            print(f"  - {issue}")

print("\n=== MERMAID \\n ISSUES (should use <br/>) ===")
if not results_mermaid:
    print("(none)")
else:
    for prefix, batch_num, issues in results_mermaid:
        print(f"批 {batch_num} {prefix}: {issues}")

print("\n=== SUMMARY ===")
print(f"Batches with missing docs: {len(results_missing)}")
print(f"Batches with stubs: {len(results_stubs)}")
print(f"Batches with low code blocks: {len(results_code)}")
print(f"Batches with tag issues: {len(results_tags)}")
print(f"Batches with Mermaid \\n: {len(results_mermaid)}")
ok = not any([results_missing, results_stubs, results_code, results_tags, results_mermaid])
print(f"Overall: {'PASS' if ok else 'FAIL'}")
