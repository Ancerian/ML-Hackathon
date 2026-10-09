#!/usr/bin/env python3
"""INDEX.md для репозиторію: усі пункти реєстрів (ID, суть, статус, де в реєстрі, код) у Markdown.

ID і статуси ПАРСЯТЬСЯ з реєстрів функціями build_index.py; короткий зміст береться з його SUM.
    python3 guide/tools/make_index_md.py      (з кореня репозиторію) -> INDEX.md
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_index as bi                                                     # noqa: E402

ROOT = HERE.parents[1]
REGFILE = {"ESTABLISHED.md": "Our try/00-decisions/ESTABLISHED.md", "CONJECTURES.md": "Our try/00-decisions/CONJECTURES.md",
           "DECISIONS.md": "Our try/00-decisions/DECISIONS.md", "open-questions.md": "Our try/00-decisions/open-questions.md",
           "verification-log.md": "Our try/99-bibliography/verification-log.md", "findings.md": "tokamak-3d-viz/findings.md",
           "COIL-BACKEND.md": "tokamak-3d-viz/docs/COIL-BACKEND.md"}


def md(s):
    """LaTeX-фрагмент SUM -> Markdown (GitHub рендерить $...$)."""
    s = s.replace("~", " ").replace(r"\,", " ").replace("---", "—").replace("--", "–")
    s = re.sub(r"\\texttt\{([^}]*)\}", r"`\1`", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", s)
    s = s.replace(r"\_", "_").replace(r"\%", "%").replace(r"\&", "&")
    s = re.sub(r"\$\\to\$", "→", s)
    s = s.replace("$\\leftarrow$", "←").replace("\\dd ", "d").replace("\\dd", "d")
    s = s.replace("{,}", ",")
    return s.replace("|", "\\|")


def code_link(ref):
    if not ref:
        return "—"
    path, _, func = ref.partition(":")
    for k, v in bi.PREFIX.items():
        if path.startswith(k):
            path = v + path[len(k):]
            break
    if path.startswith("Tokamak-GS-solver/"):
        return f"`{path}`" + (f" · `{func}`" if func else "") + " (оригінал, див. `Tokamak-GS-solver/README.md`)"
    url = path.replace(" ", "%20")
    return f"[`{path.rsplit('/', 1)[-1]}`]({url})" + (f" · `{func}`" if func else "")


def main():
    parsed = {}
    for fn in (bi.parse_est, bi.parse_conj, bi.parse_dec, bi.parse_oq, bi.parse_ver, bi.parse_viz, bi.parse_coil):
        parsed.update(fn())
    out = ["# Покажчик усіх пунктів реєстрів", "",
           "Згенеровано `guide/tools/make_index_md.py` з реєстрів проєкту: ID і статуси парсяться з самих файлів, "
           "короткий зміст — із покажчика путівника (`guide/tools/build_index.py`). Номер рядка веде у файл реєстру.", ""]
    total = 0
    for title, reg, pre in bi.GROUPS:
        ids = sorted((i for i in parsed if bi.REG_OF(i) == (reg, pre)), key=bi.sort_key)
        total += len(ids)
        out += [f"## {title} — {len(ids)}", "", "| ID | Суть | Статус | Реєстр | Код |", "|---|---|---|---|---|"]
        for iid in ids:
            fname, line, st = parsed[iid]
            summ, code, _ = bi.SUM[iid]
            reg_path = REGFILE.get(fname, fname)
            out.append(f"| **{iid}** | {md(summ)} | {md(st)} | [{fname}:{line}]({reg_path.replace(' ', '%20')}#L{line}) | {code_link(code)} |")
        out.append("")
    out.insert(4, f"Усього **{total}** пунктів.\n")
    (ROOT / "INDEX.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("INDEX.md:", total, "items")


if __name__ == "__main__":
    main()
