#!/usr/bin/env python3
"""Перевіряє всі \\code{path}{first}{last}{caption} у файлах путівника.

Для кожного виклику: файл існує (через guide/src/ посилання), діапазон у межах файлу,
друкує перший рядок діапазону (щоб очима переконатися, що він починається з потрібної
функції/класу). Код виходу 1 при помилці.
Використання: python3 tools/check_code_refs.py [chapters/*.tex ...]
"""
import re, sys, glob, pathlib

GUIDE = pathlib.Path(__file__).resolve().parent.parent
PAT = re.compile(r"\\code\{([^}]*)\}\{(\d+)\}\{(\d+)\}")


def main():
    files = sys.argv[1:] or sorted(glob.glob(str(GUIDE / "chapters/*.tex")) + glob.glob(str(GUIDE / "appendix/*.tex")))
    bad = 0
    for f in files:
        text = pathlib.Path(f).read_text(encoding="utf-8")
        for m in PAT.finditer(text):
            rel, a, b = m.group(1), int(m.group(2)), int(m.group(3))
            src = GUIDE / "src" / rel
            ln = text.count("\n", 0, m.start()) + 1
            tag = f"{pathlib.Path(f).name}:{ln}  {rel} [{a}-{b}]"
            if not src.exists():
                print(f"MISSING  {tag}"); bad += 1; continue
            lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
            if not (1 <= a <= b <= len(lines)):
                print(f"RANGE    {tag}  (файл має {len(lines)} рядків)"); bad += 1; continue
            first = lines[a - 1].strip()
            print(f"ok       {tag}  → {first[:80]}")
    print(f"--- помилок: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
