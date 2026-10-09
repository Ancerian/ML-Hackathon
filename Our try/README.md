# Our try — аналіз і концепт хакатону ФМФ

Робоча папка: літературний аналіз, два власні розбори з кодом, дизайн метрики, концепт.
Артефакти **двомовні**: основний текст українською, у кінці кожного файлу — English summary.

Останнє оновлення: **2026-09-29** (узгодження реєстрів; журнал правок —
`00-decisions/CHANGELOG.md`; було 2026-09-22).

---

## Стан

| Блок | Файли | Стан |
|---|---|---|
| **00 Рішення** | `DECISIONS.md`, `open-questions.md`, **`ESTABLISHED.md`**, **`CONJECTURES.md`** | ✅ |
| **01 Ландшафт** | `competitions-and-benchmarks.md`, `datasets.md`, `saturated-vs-headroom.md` | ✅ |
| **02 Карта поля** | `A-inverse-equilibrium.md`, `B-pde-neural-solvers.md`, `C-uq-and-bayesian.md`, `D-superconductivity.md`, `E-disruption-why-not.md`, `F-poincare-and-topology.md` | ✅ |
| **03 Глибокі розбори** | `D1-psi-to-scalars/`, `D2-gs-nonuniqueness/`, `D3-poincare-inverse/` | ✅ з кодом, числами і графіками (D3 — пілот C5; `multimode.py` зламаний навмисно, R8) |
| **04 Новизна** | `our-contribution.md`, `metric-design.md`, `gs_residual_probe.py`, `topology_probe.py`, `topology_ec_curve.py` | ✅ три прототипи перевірено на реальних даних |
| **05 Концепт** | `hackathon-concept-v1.md`, `hackathon-concept-v1.en.md` | ✅ v1 |
| **99 Бібліографія** | `verification-log.md`, `refs.bib` | ✅ |

---

## Два реєстри — головне правило папки

**`00-decisions/ESTABLISHED.md`** — лише те, що **виміряно власним кодом** або **прочитано
у відкритому джерелі**. Плюс окремий розділ **спростованого** — вісім власних тверджень (R1–R8; до 2026-09-29
тут стояло «сім»), які не видаляються.

**`00-decisions/CONJECTURES.md`** — усе неперевірене, з зазначенням **що його спростує**
і **скільки коштує перевірка**. Запис звідси **не можна** цитувати в концепті
без позначки «гіпотеза».

Перевірено → переїжджає з другого в перший. **Нічого не зникає мовчки.**

## Якщо читати лише три файли

1. **`00-decisions/DECISIONS.md`** — що вирішено і за яких умов рішення переглядається.
2. **`01-landscape/saturated-vs-headroom.md`** — де вже стеля, а де є за що змагатися.
3. **`05-concept/hackathon-concept-v1.md`** — сам концепт.

## Якщо цікавить, що ми **виміряли**, а не переказали

- **`03-deep-dives/D1-psi-to-scalars/findings.md`** — відображення ψ → похідні скаляри
  **не ліпшицеве в L²**. Показники підсилення α ∈ [0,35; 0,70], усі < 1.
  `R_axis` досягає похибки 1σ уже при **R²ψ = 0,99993**.
- **`03-deep-dives/D2-gs-nonuniqueness/findings.md`** — L²-середнє двох гілок має резидуал
  **1,22** проти **6,4·10⁻⁸** у справжньої гілки. Опуклий контроль (Бін) — **2,5·10⁻¹³**.
- **`04-novelty/metric-design.md`** — GS-резидуал обчислюваний **лише з ψ**:
  істина 0,009 → +1% шуму 0,650, **стрибок у 70 разів**, інваріантність до масштабу пройдена.
- **`02-field-map/F-poincare-and-topology.md`** — критичний скелет ψ: істина дає **рівно одну
  еліптичну точку і суму індексів +1** (Пуанкаре–Хопф виконується точно); при R²ψ = 0,993 —
  **20 магнітних островів, яких не існує**. І побічний результат: **сума індексів зберігається,
  тож вона як метрика марна — працює лише кількість**.

## Якщо цікавить, де ми **помилялися**

Вісім власних тверджень перевірено й **спростовано** — повний перелік R1–R8 у
`00-decisions/ESTABLISHED.md`, розділ C. (До 2026-09-29 тут стояло «два», хоча нижче вже було
п'ять пунктів.) Усі задокументовані, а не тихо прибрані; ті, що мають окремий розбір:

- **`99-bibliography/verification-log.md`, V-A2** — «ніхто не нав'язує інтегральну крайову умову
  жорстко в нейроархітектурі» → **СПРОСТОВАНО** (McClenaghan et al., 2024).
- **`02-field-map/D-superconductivity.md`** — «критичний стан Біна і вільна межа плазми —
  одна математика» → **СПРОСТОВАНО**. Опукла проти неопуклої; активна множина проти лінії рівня;
  нульове перехресне цитування за 50 років. **Переформульована версія виявилась кращою за
  початкову ідею.**
- **`99-bibliography/verification-log.md`, V-D1** — «симплектична нейромережа для відображень
  силових ліній — наша ідея» → **СПРОСТОВАНО** (Burby, Tang, Maulik, 2021).
- **`99-bibliography/verification-log.md`, V-D3** — канонічні змінні гамільтоніана силової лінії
  названо хибно («ψ як імпульс»). Правильно: **імпульс = тороїдальний потік, гамільтоніан =
  полоїдальний**. Виправлено до того, як потрапило в матеріали.
  ⚠️ *2026-09-29:* та сама помилка **повторилася в коді** `tokamak-3d-viz` (ширина острова,
  завищення в √q) і виправлена — `ESTABLISHED.md` R4, `tokamak-3d-viz/findings.md` VR15.
- **`verification-log.md`, V-D4** — рекомендація замінити лічильник критичних точок на χ-криву
  була **хибна**: χ-крива менш чутлива, бо за теорією Морса вона є його інтегральною версією.
- Решта — R6 (α = ½ не універсальний), R7 (H5: топологія переживає перенесення),
  R8 (багатомодовий Tokamap не симплектичний) — описані в `ESTABLISHED.md` §C і
  `03-deep-dives/D3-poincare-inverse/findings.md`.

---

## Як відтворити обчислення

Усе працює у наявному оточенні starter kit'а, без нових залежностей.

```bash
cd "../fusion equilibrium challenge/starter"

# D1 — обумовленість (≈30 с)
.venv/bin/python "../../Our try/03-deep-dives/D1-psi-to-scalars/conditioning.py" --frames 60 --seeds 3
.venv/bin/python "../../Our try/03-deep-dives/D1-psi-to-scalars/plots.py"

# D2 — неєдиність (≈10 с)
.venv/bin/python "../../Our try/03-deep-dives/D2-gs-nonuniqueness/branch_mean.py"
.venv/bin/python "../../Our try/03-deep-dives/D2-gs-nonuniqueness/plots.py"

# прототипи членів метрики (≈20 с / 40 с / 60 с)
.venv/bin/python "../../Our try/04-novelty/gs_residual_probe.py"
.venv/bin/python "../../Our try/04-novelty/topology_probe.py"
.venv/bin/python "../../Our try/04-novelty/topology_ec_curve.py"

# контроль цілісності скорера — має друкувати рівно 1.0 і 0.0
.venv/bin/python local_score.py --mode perfect --n-shots 2
.venv/bin/python local_score.py --mode zeros   --n-shots 2
```

## Правила ведення цієї папки

1. **Маркери достовірності обов'язкові:** `[V]` джерело відкрито, `[S]` зі зведення пошуку,
   `[?]` непідтверджено.
2. **Негативні твердження** («ніхто не робив X») **перевіряються окремо** і йдуть у
   `verification-log.md` з вердиктом. Формулювання — «нам не відомо про», не «не існує».
3. **Спростовані твердження не видаляються**, а позначаються як спростовані. Помилка,
   яку видалили, повертається через місяць.
4. **Числа перераховуються.** Кожне числове твердження з чужого джерела або перевірене,
   або позначене `[S]`.

---

## English

Working folder for the ФМФ hackathon: literature analysis, two original measured deep dives,
metric design, and the concept itself. All documents are bilingual — Ukrainian body, English
summary at the end of each file.

**Three files if you read nothing else:** `00-decisions/DECISIONS.md` (what was decided and what
would reverse it), `01-landscape/saturated-vs-headroom.md` (where the ceiling is and where the
headroom is), `05-concept/hackathon-concept-v1.en.md` (the concept).

**What we measured rather than summarised:** the map ψ ↦ derived scalars is **not Lipschitz in L²**
(amplification exponents all below 1; `R_axis` hits a 1σ error at R²ψ = 0.99993); an L2-trained
regressor on a multivalued target returns a **branch mean with residual 1.22 against 6.4 × 10⁻⁸**
for a true branch, while a convex control stays at 2.5 × 10⁻¹³; and the Grad–Shafranov residual is
computable **from ψ alone**, separating ground truth (0.009) from 1%-noise (0.650) by 70×.

**Where we were wrong**, documented rather than quietly deleted — eight items so far (R1–R8 in
`00-decisions/ESTABLISHED.md` §C; this line said "five" until 2026-09-29). The five with their own
write-ups: the claim that
no neural architecture hard-constrains the Green's-function boundary condition is **refuted**
(McClenaghan et al. 2024); the proposal that Bean's critical state and the plasma free-boundary
problem share a mathematical structure is **refuted** (convex vs non-convex), though the
reformulation that survived is better than the original idea; symplectic neural maps for field
lines **already exist** (Burby, Tang & Maulik 2021); the canonical variables of the field-line
Hamiltonian were **stated wrongly** in draft (momentum is the *toroidal* flux, the Hamiltonian is
the *poloidal* flux) and corrected before it reached any hackathon material; and the recommendation
to replace the critical-point counter with an Euler-characteristic curve was **wrong on sensitivity**
— by Morse theory the curve is the counter's integral, not an independent measurement. The
canonical-variable error (R4) later **recurred in code** — the visualisation's island-width formula
used ψ as the momentum and overestimated widths by √q — and was fixed on 2026-09-29.

House rules: confidence tags on every claim; negative claims verified separately and phrased as
"we are not aware of"; refuted claims marked, never deleted; numbers recomputed.
