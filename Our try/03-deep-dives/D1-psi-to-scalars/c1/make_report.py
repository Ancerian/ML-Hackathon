#!/usr/bin/env python3
"""Build c1_report.html — a one-page report of the C1 test, styled after the lab's brand book
(AI-лабораторія імені професора В. М. Горшкова: Inter / IBM Plex Mono / STIX Two, paper-ink-blue).

All numbers come from results_c1.json and local_models.json in this folder. The FMF crest SVGs
(official files from the lab's design system) are passed on the command line and embedded unchanged.

    python3 make_report.py <fmf-shield-outline.svg> <fmf-shield-solid.svg>
"""
import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def b64(p):
    return "data:image/svg+xml;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def main():
    res = json.loads((HERE / "results_c1.json").read_text())
    loc = json.loads((HERE / "local_models.json").read_text())
    rows = res["rows"]
    samples = [("DIII-D, 3 demo shots", "DIII-D · 3 демо-розряди", "demo"),
               ("DIII-D, 28 shots", "DIII-D · 28 розрядів", "d28"),
               ("MAST, 3 shots", "MAST · 3 розряди", "mast"),
               ("Cerfon-Freidberg, 12 shapes", "Серфон–Фрайдберг · 12 форм", "cf")]
    data = {"names": ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"],
            "samples": [{"key": k, "label": lab,
                         "alpha": rows[f"{src} | white"]["alpha"],
                         "ci": rows[f"{src} | white"]["ci95"],
                         "smooth": rows[f"{src} | smooth"]["alpha"],
                         "c1": rows[f"{src} | white"]["C1_freq_k3"]}
                        for src, lab, k in samples]}
    m1 = loc["M1"]
    hR = m1["h"][0]
    data["m1"] = {"etas": m1["etas"], "hR": hR, "lambdas": m1["lambdas"],
                  "white": [[s[0] for s in m1["white"][str(l)]["slopes"]] for l in m1["lambdas"]],
                  "smooth": [[s[0] for s in m1["smooth"][str(l)]["slopes"]] for l in m1["lambdas"]],
                  "e14": m1["E14"]["dr_sqrt_lambda"], "r6": [m1["R6_alpha_R"][str(l)] for l in m1["lambdas"]]}
    html = (HERE / "report_template.html").read_text(encoding="utf-8")
    html = html.replace("/*DATA*/null", json.dumps(data))
    html = html.replace("CREST_LIGHT", b64(sys.argv[1])).replace("CREST_DARK", b64(sys.argv[2]))
    (HERE / "c1_report.html").write_text(html, encoding="utf-8")
    print("c1_report.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
