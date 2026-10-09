#!/usr/bin/env python3
"""Build c4_report.html — the C4 (GS_score) test as a one-page report in the lab style of the
C1/C2 reports (C1's <style> block reused). Numbers from eval.json and gref.json.

    python3 make_report.py <fmf-shield-outline.svg> <fmf-shield-solid.svg>
"""
import base64
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[1] / "03-deep-dives" / "D1-psi-to-scalars" / "c1" / "report_template.html"


def b64(p):
    return "data:image/svg+xml;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def main():
    c1 = C1.read_text(encoding="utf-8")
    head = c1[:c1.index('<div class="wrap">')].replace("<title>Перевірка гіпотези C1</title>",
                                                       "<title>Перевірка гіпотези C4</title>")
    data = {"ev": json.loads((HERE / "eval.json").read_text()), "gref": json.loads((HERE / "gref.json").read_text())}
    body = (HERE / "report_body.html").read_text(encoding="utf-8").replace("/*DATA*/null", json.dumps(data))
    html = (head + body).replace("CREST_LIGHT", b64(sys.argv[1])).replace("CREST_DARK", b64(sys.argv[2]))
    assert not re.search(r"CREST_|/\*DATA\*/", html)
    (HERE / "c4_report.html").write_text(html, encoding="utf-8")
    print("c4_report.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
