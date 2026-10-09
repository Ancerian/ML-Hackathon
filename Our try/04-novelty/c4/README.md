# C4 (половина: GS_score) — перевірка «GS_score змінює впорядкування бейзлайнів» (2026-10-01)

**Вердикт: C4-GS спростовано → R12.** Критерії записано до навчання й оцінки в [THEORY.md](THEORY.md).
Встановлено E38 (карти потоку нейромереж не є рівновагами). C4 лишається активною лише в частині
Calibration (разом із C6). Розбір — `../metric-design.md`, розділ «Перевірка C4».

| Файл | Що робить |
|---|---|
| `THEORY.md` | означення GS_score (відносно істини) і S′_GS, ваги, скан, критерій стійкої перестановки. **Попередня реєстрація** |
| `gref.py` | калібрування g_ref: медіана g(істина + 1 % шуму) на 28 розрядах → `gref.json` (g_ref = 0,634; g(істина) = 0,011) |
| `train.py` | 7 бейзлайнів `experiments.py` на розбитті C2 (UNet_Lite і PCA+Ridge — прогнози C2) → `preds/` |
| `evaluate.py` | терміни офіційного скорера, g для кожного кадру, S′_GS при w_r = 0…0,30, бутстреп 1000× по тестових розрядах, пари, вердикт → `eval.json` |
| `summarise.py` | рисунок `c4_gs.png` |

## Відтворення (~10 хв; потрібні кеш і прогнози C2)

```bash
cd "fusion equilibrium challenge/starter"
P="../../Our try/04-novelty/c4"
.venv/bin/python "$P/gref.py"        # ~3 с
.venv/bin/python "$P/train.py"       # ~2,5 хв
.venv/bin/python "$P/evaluate.py"    # ~5 хв
.venv/bin/python "$P/summarise.py"
```
