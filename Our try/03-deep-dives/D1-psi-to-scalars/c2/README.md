# C2 — перевірка «спектрально зважена втрата покращує реальні моделі» (2026-09-30)

**Вердикт: C2 спростовано → R11.** Критерії записано до навчання в [THEORY.md](THEORY.md).
Побічно зафіксовано E37 (розчеплення R²ψ і Consistency на реальній моделі).
Повний розбір — `../findings.md`, розділ «Перевірка C2».

| Файл | Що робить |
|---|---|
| `THEORY.md` | правило ваги з S(k), 4 руки (flat / S(k) / H¹ / shuffled), протокол, критерії. **Попередня реєстрація** |
| `fetch_split.py` | кешує розбиття `SCORE.md`: потокові розряди 0–39 (навчання 0–35, валідація 36–39) і 60–67 (тест) у `fusion equilibrium challenge/downloaded_huggingface/c2_cache/` |
| `sk_remeasure.py` | S(k) на 28 розрядах при e = 0,1 / 0,3 / 1,0 (функції D1 і `../c1/sweep_core.py`) |
| `train.py` | робоча точка e_op, ваги (`weights.json`), PCA+Ridge (зважена PCA) і UNet_Lite (зважена втрата у просторі Фур'є), 3 зерна; прогнози в `preds/` |
| `evaluate.py` | функції офіційного скорера (`local_score.py`, v3.1), парні різниці, бутстреп по тестових розрядах, вердикт → `eval.json` |
| `summarise.py` | рисунок `c2_arms.png` |

## Відтворення (~80 хв, з них ~74 хв — 12 прогонів UNet на MPS)

```bash
cd "fusion equilibrium challenge/starter"
P="../../Our try/03-deep-dives/D1-psi-to-scalars/c2"
.venv/bin/python "$P/fetch_split.py"      # один раз, 48 розрядів
.venv/bin/python "$P/sk_remeasure.py"     # ~40 с
.venv/bin/python "$P/train.py"            # ~75 хв
.venv/bin/python "$P/evaluate.py"         # ~4 хв
.venv/bin/python "$P/summarise.py"
```
`preds/` (~340 МБ прогнозів) можна видалити після `evaluate.py` — вони відтворюються `train.py`.
