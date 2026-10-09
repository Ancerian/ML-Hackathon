#!/usr/bin/env python3
"""Build c6_report.html — the C6 + C4-Calibration test as a one-page report in the lab style of the
C1/C2/C4 reports (C1's <style> block reused, crests taken from c4_report.html). Numbers from
c6_eval.json, c4cal_eval.json, diag_pool.json and diag_shot61.json.

    python3 make_report.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[1] / "03-deep-dives" / "D1-psi-to-scalars" / "c1" / "report_template.html"
C4 = HERE.parent / "c4" / "c4_report.html"


def main():
    c1 = C1.read_text(encoding="utf-8")
    head = c1[:c1.index('<div class="wrap">')].replace("<title>Перевірка гіпотези C1</title>",
                                                       "<title>Перевірка гіпотез C6 і C4</title>")
    assert "<title>Перевірка гіпотез C6 і C4</title>" in head
    c4 = C4.read_text(encoding="utf-8")
    light = re.search(r'class="crest crest-light" src="([^"]+)"', c4).group(1)
    dark = re.search(r'class="crest crest-dark" src="([^"]+)"', c4).group(1)
    data = {"c6": json.loads((HERE / "c6_eval.json").read_text()), "c4": json.loads((HERE / "c4cal_eval.json").read_text()),
            "pool": json.loads((HERE / "diag_pool.json").read_text()), "s61": json.loads((HERE / "diag_shot61.json").read_text())}
    body = (HERE / "report_body.html").read_text(encoding="utf-8").replace("/*DATA*/null", json.dumps(data))
    html = (head + body).replace("CREST_LIGHT", light).replace("CREST_DARK", dark)
    assert not re.search(r"CREST_|/\*DATA\*/", html)
    (HERE / "c6_report.html").write_text(html, encoding="utf-8")
    print("c6_report.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
