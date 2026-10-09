#!/usr/bin/env python3
"""Лінт позначень методички (блок перевірки, п. 2).

Використання:  python3 tools/lint_notation.py [файли.tex ...]
Без аргументів перевіряє chapters/*.tex і appendices/*.tex.
Код виходу 1, якщо знайдено порушення.
"""
import re, sys, glob, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

RULES = [
    (r"\\zeta\b", "ζ заборонено як тороїдальний кут — використовуйте \\varphi (\\tor)"),
    (r"\\vartheta\b", "ϑ заборонено — полоїдальний кут лише \\theta (\\pol)"),
    (r"\\chi\b(?!\s*_|\s*\{)", "χ лише з індексом (температуропровідність χ_s); як кут заборонено"),
    (r"(?<![A-Za-z\\])\\phi\b", "\\phi заборонено: тороїдальний кут — \\varphi, потенціал — \\Phi"),
    (r"\\cite\{[^}]*(Aksenov2015|Zimnikov2016|Tolochkevych2015)", "виключене джерело (рішення Q7)"),
    (r"\bq_\{?\\parallel", "q — лише запас стійкості; тепловий потік позначайте Q"),
    (r"\\iota\b", "ι не використовуємо для токамака — лише q (ι=1/q допускається тільки в тексті про стеларатори; позначте # lint-ok)"),
]

MATH_DECIMAL = re.compile(r"(?<![\w.{])\d+\.\d+(?![\w.])")
MATH_SPANS = re.compile(r"\$[^$]+\$|\\\[.*?\\\]|\\begin\{(equation|align|gather|multline)\*?\}.*?\\end\{\1\*?\}", re.S)


def lint(path):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    problems = []
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        code = line.split("%")[0] if not line.lstrip().startswith("%") else ""
        if "lint-ok" in line:
            continue
        for pat, msg in RULES:
            if re.search(pat, code):
                problems.append((i, msg, line.strip()))
    # десяткова кома в математиці: 0{,}5, а не 0.5
    for m in MATH_SPANS.finditer(text):
        span = m.group(0)
        if "lint-ok" in span:
            continue
        for d in MATH_DECIMAL.finditer(span):
            ln = text.count("\n", 0, m.start() + d.start()) + 1
            problems.append((ln, "десяткова крапка в математиці — пишіть {,} або \\num{}", d.group(0)))
    return problems


def main():
    files = sys.argv[1:] or sorted(glob.glob(str(ROOT / "chapters/*.tex")) + glob.glob(str(ROOT / "appendices/*.tex")))
    total = 0
    for f in files:
        for ln, msg, ctx in lint(f):
            total += 1
            print(f"{pathlib.Path(f).name}:{ln}: {msg}\n    {ctx[:120]}")
    print(f"--- порушень: {total}")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
