#!/usr/bin/env python3
"""Build c2_report.html — the C2 test as a one-page report, in the same lab style as
../c1/c1_report.html (its <style> block is reused verbatim). Numbers come from eval.json,
weights.json and sk_remeasure.json; the FMF crest SVGs are passed on the command line.

    python3 make_report.py <fmf-shield-outline.svg> <fmf-shield-solid.svg>
"""
import base64
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def b64(p):
    return "data:image/svg+xml;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def main():
    c1 = (HERE.parent / "c1" / "report_template.html").read_text(encoding="utf-8")
    head = c1[:c1.index('<div class="wrap">')].replace("<title>Перевірка гіпотези C1</title>",
                                                       "<title>Перевірка гіпотези C2</title>")
    head = head.replace("</style>", ".kv{display:flex; justify-content:space-between; gap:12px; "
                        "border-bottom:1px solid var(--line); padding-block:6px}\n"
                        ".kv:last-child{border-bottom:none}\n.kv b{font-weight:500}\n</style>")
    ev = json.loads((HERE / "eval.json").read_text())
    w = json.loads((HERE / "weights.json").read_text())
    sk = json.loads((HERE / "sk_remeasure.json").read_text())
    d1 = [sk["d1_demo3_e0.1"][f"band-{a}-{b}"] for a, b in sk["bands"]]
    data = {"ev": ev, "w": w, "bands": sk["bands"], "names": sk["scalars"],
            "sk": {"d1": d1, "e01": sk["median_sigma"]["0.1"], "e03": sk["median_sigma"]["0.3"],
                   "e10": sk["median_sigma"]["1.0"]}}
    body = (HERE / "report_body.html").read_text(encoding="utf-8")
    body = body.replace("/*DATA*/null", json.dumps(data))
    html = (head + body).replace("CREST_LIGHT", b64(sys.argv[1])).replace("CREST_DARK", b64(sys.argv[2]))
    assert not re.search(r"CREST_|/\*DATA\*/", html)
    (HERE / "c2_report.html").write_text(html, encoding="utf-8")
    print("c2_report.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
