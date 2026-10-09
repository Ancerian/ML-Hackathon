# C5 — нелінійна перевірка: результат

**Вердикт: СПРОСТОВАНО → R13** (`00-decisions/ESTABLISHED.md`). Критерії — `THEORY.md`, зафіксовані
2026-10-01 до запуску (SHA-256 `1ea23e22…81e1a4`, не редагувався). Побічно — **E40**.

## Головний результат (шум 3 %)

| | Стенд A — Tokamap (E35) | Стенд B — `tokamak-3d-viz`, DIII-D 203702 |
|---|---|---|
| успіх (s) | **0 / 40** | **0 / 3** |
| хибні мінімуми (f) | 0 % | 0 % (і в скані пар фаз — 0) |
| технічні збої | 0 % | 1 / 18 (5,6 %) |
| найкращий χ² / χ²_true | 2,6 … 193, медіана 110 | 303 / 484 / 503 |
| медіанна макс. похибка амплітуди / фази | 32 % / 77° | 54 % / 144° |
| CPU на екземпляр (усі старти) | 16 с | 5,3–7,3 год |

Правило `THEORY.md` §5: «на будь-якому стенді s < 50 % ⇒ спростовано». Спрацювало на обох.
При 10 % шуму (стенд A, лише для звіту): 2/40 успіхів, 1/40 із хибним мінімумом.

**Як читати.** Хибних мінімумів немає: жоден старт не знайшов **іншого** розв'язку, що пояснює
дані на рівні шуму. Тобто інформація в спостережуваному є. Але **жоден** старт не дійшов до рівня
шуму взагалі: розв'язувач зупиняється (status 3, `xtol`) у χ², у десятки–сотні разів більшому за
істинний. Перешкода — оптимізація, а не ідентифіковність.

## Post-hoc діагностика (не впливає на вердикт) — E40

`diag_tokamap.py` → `diag_tokamap.json`, стенд A, 8 екземплярів:

| старт | χ² / χ²_true | у допуску |
|---|---|---|
| в істині | 0,99–1,00 | 8/8 — код коректний |
| істина + 5 % / 10° | **41–100** | 7/8 (параметри майже не зрушили) |
| істина + 20 % / 45° | 111–779 | 0/8 |

Безшумовий 1-D розріз χ² через істину (екземпляр 0): по φ₁ у ±20° — **8** локальних мінімумів,
по колу — **24**; по ln a₁ у ±0,2 — 3. Зсув φ₁ на 0,5° дає χ² 245 в один бік і 2,9 в інший
(рівень шуму χ² ≈ 160). Скан пар фаз на стенді B (сітка 30°): 8 і 14 локальних мінімумів, усі
віддалені — з χ²_0 ≥ 2 184 ≫ N_obs = 132, тож хибними за §4 не зараховані.

Механізм: екскурсія ptp за скінченне число обертів — максимум по дискретних точках орбіти;
малий зсув параметра змінює, яка точка дає екстремум, і нев'язка стрибає.

## Що це означає (висновки поза реєстром позначено)

- Для концепту: задача Рівня 3 **у цій постановці** (екскурсія + градієнтний фіт) не годиться як
  завдання базового рівня — команда без глобальної оптимізації чи сурогата не отримає нічого.
- *Гіпотеза, не перевірено:* обумовленість VE22–VE25, VE31 рахувалась скінченними різницями з
  кроком 15 % амплітуди й 0,3 рад — це січна, що усереднює нерівності; локальний ландшафт вона не
  описує. Висновок «складність задає спостережуване» може стосуватися лише згладженої задачі.
- *Гіпотеза, не перевірено:* гладше спостережуване (середнє або квантиль замість ptp, довші
  траси, усереднення по засівах у межах комірки) розгладить ландшафт і поверне задачу в межі
  базового фіту. Це кандидат у нову здогадку.

## Відтворення

```bash
cd "fusion equilibrium challenge/starter"
.venv/bin/python "../../Our try/03-deep-dives/D3-poincare-inverse/c5/run_tokamap.py"   # ≈2 хв, 10 ядер
.venv/bin/python "../../Our try/03-deep-dives/D3-poincare-inverse/c5/run_viz.py"       # ≈3,2 год, 9–10 ядер
.venv/bin/python "../../Our try/03-deep-dives/D3-poincare-inverse/c5/diag_tokamap.py"  # ≈1 хв
```

Стенд B під час прогону ділив процесор з іншою сесією (`run_bake`, `run_inverse`, `run_analysis`
`tokamak-3d-viz`), тож час — верхня оцінка. Файли: `results_tokamap.json`, `results_viz.json`,
`results_viz_fits.jsonl`, `log_*.txt`.

---

## English summary

**C5 refuted (→ R13)** against criteria frozen before the run. A fixed baseline solver (multi-start
bounded least squares, forward-difference Jacobian) recovering 3 amplitudes + 3 phases from the
phase-resolved excursion at 3 % noise succeeded on **0/40** instances on the corrected multi-mode
Tokamap and **0/3** on the `tokamak-3d-viz` stand; no spurious minimum was found on either, so the
information is there — but no start ever reached the noise level (best χ²/χ²_true median 110 on A, range 2.6–193;
303–503 on B). Post-hoc (E40): starting at the truth the fit stays put (code is correct); starting
5 %/10° away it stalls at 41–100 × χ²_true; a 1-D cut shows 8 local minima within ±20° of one phase
and 24 around the circle, with a 0.5° shift moving χ² by 245 one way and 2.9 the other. The
excursion observable (a ptp over discrete orbit points) is non-smooth, so the misfit landscape is
glassy. Unverified implications: the linearised condition numbers (VE22–VE25, VE31; 15 % finite-
difference steps) describe a smoothed secant, not the local landscape; a smoother observable might
bring the task back within reach of a baseline fit.
