# D2 — Неєдиність і «середнє гілок»
# / Non-uniqueness and the mean-of-branches pathology

**Статус: виконано, з кодом і числами.** Запущено 2026-09-22.
**Обсяг чесно обмежений:** це **сурогат механізму**, а не Град–Шафранов. Див. «Межі» нижче.

---

## Твердження, яке перевіряємо

Кожна ML-робота з реконструкції рівноваги припускає, що відображення
(діагностики) → ψ є **функцією**. Ham & Farrell (*Nucl. Fusion* 64, 034001, 2024) і
Pentland et al. (arXiv:2503.05674, MAST-U) показують, що це може бути **багатозначне
відношення**: різні рівноваги з **однаковими вимірами**. Дослівно з роботи по MAST-U:
*«дві дуже різні форми профілю можуть давати однаковий потік на осі»*, коли підгонку
обмежують лише магнітні виміри.

Якщо так, то регресор, що мінімізує `E‖f(x) − y‖²`, збігається до **умовного середнього гілок**.
А **середнє двох розв'язків нелінійного PDE не є розв'язком**.

> **І ось ключове:** ця шкода **невидима для L²-метрики** — адже середнє **за означенням**
> є L²-оптимальним передбаченням. Вона видима **лише в резидуалі PDE**.
> Це і є повний аргумент на користь оцінювання резидуала.

**Перевірка V-A3** (`99-bibliography/verification-log.md`) підтвердила, що цього ніхто не міряв:
`all:"Grad-Shafranov" AND all:"multiple solutions" AND all:"neural"` → **0 результатів**.
Робота по MAST-U **не містить машинного навчання взагалі**. А в повному тексті статті
флагманського челенджу — **0 збігів** для «multivalued», «non-unique», «ill-posed», «ambiguous».

## Дизайн: опуклий контроль проти неопуклого тесту

Цю структуру нам **нав'язав** результат огляду надпровідності
(`02-field-map/D-superconductivity.md`): чесне питання — не «чи це одна задача», а
**«чи відрізняється режим відмови між ОПУКЛОЮ задачею з вільною межею і НЕОПУКЛОЮ»**.

| | Контроль | Тест |
|---|---|---|
| Задача | **критичний стан Біна**, 1D плита, стале `J_c` | **1D Братý / Гельфанд**: `−u″ = λeᵘ`, `u(0)=u(1)=0` |
| Структура | **опукла**, максимально монотонний оператор | **неопукла**, знаконевизначений функціонал |
| Розв'язки | **ЄДИНИЙ** (Prigozhin 1996, Теорема 2) | **РІВНО ДВА** при λ < λ*, один при λ*, жодного далі |
| Чому саме ця | строго доведена єдиність | Bartolucci et al. (arXiv:2106.04331) відносять задачу плазми з вільною межею **саме до сімейства Гельфанда / середньопольових вихорів / напівлінійних біфуркацій** |

Обидві мають **замкнені розв'язки**, тож результат точний, а не залежний від розв'язувача.

**Валідація реалізації:** обчислена точка складки **λ\* = 3,513830719** збігається з
літературним значенням 3,513830719 **до останньої цифри**.

## Результат

```
=== ТЕСТ: 1D Братý (неопукла, ДВІ гілки на кожну λ) ===
  lambda    th_lo    th_hi    res(lo)    res(hi)   res(MEAN)   L2 sep
   0.500   1.0336  13.0382   3.40e-07   1.10e-08      4.2609   3.3624
   1.000   1.5172  10.9387   1.64e-07   1.26e-08      2.1018   2.6545
   2.000   2.3576   8.5072   7.22e-08   1.41e-08      0.7249   1.7514
   3.000   3.3735   6.5766   3.95e-08   1.63e-08      0.1745   0.9207
   3.400   4.1084   5.5623   2.94e-08   1.88e-08      0.0347   0.4193
   3.500   4.5519   5.0543   2.52e-08   2.37e-08      0.0041   0.1450

=== КОНТРОЛЬ: 1D плита Біна (опукла, ОДИН розв'язок на кожне H_a) ===
  усі H_a:  res(розв'язку) = res(середнього) ≈ 2e-13   (множина розв'язків — одноелементна)
```

| Величина | Значення |
|---|---|
| Неопукла: резидуал **справжньої гілки** | **6,4·10⁻⁸** (числовий шум) |
| Неопукла: резидуал **L²-середнього** | **1,22** |
| **Відношення** | **1,9·10⁷ разів гірше** |
| Опукла: резидуал L²-середнього | **2,5·10⁻¹³** — без змін |

**Відносний резидуал 1,22 означає, що нев'язка того самого порядку, що й саме джерело.**
При λ = 0,5 він **4,26** — учетверо більший за норму джерела. Тобто «розв'язок» не просто
трохи хибний: він **узагалі не задовольняє рівняння**.

**І структура залежності правильна:** при λ → λ* гілки зливаються, і резидуал середнього
спадає до 0,0041. Біля складки середнє майже є розв'язком, далеко від неї — ні.
Це рівно те, чого вимагає теорія біфуркацій, і воно слугує додатковою перевіркою коду.

## Що з цього випливає

1. **L²-метрика структурно сліпа до цієї шкоди.** Середнє **за означенням** мінімізує
   очікувану L²-похибку. Жодне збільшення ваги L²-члена цього не виявить.
   **Тільки член із резидуалом бачить, що це не розв'язок.**
2. **Інформація не втрачена — її знищує цільова функція.** Кожна окрема гілка має резидуал
   на рівні числового шуму. Проблема не в даних і не в потужності мережі, а в тому, що
   **L²-регресія — неправильна постановка для багатозначної цілі**. Правильні інструменти
   відомі: мережі зі змішаним розподілом (MDN), умовні генеративні моделі, оборотні мережі
   (Ardizzone et al., arXiv:1808.04730: *«традиційні мережі повертають лише один найімовірніший
   розв'язок або середнє»*).
3. **Опуклість — це вододіл, а не деталь.** Контроль Біна проходить процедуру без подряпин.
   Різниця між 2,5·10⁻¹³ і 1,22 — це і є вся різниця між опуклою і неопуклою задачею
   з вільною межею, виміряна однією величиною.

## Межі цього результату — читати обов'язково

- **Це НЕ Град–Шафранов.** Братý — сурогат механізму, обраний тому, що його гілки відомі
  в замкненій формі, а Bartolucci et al. поміщають задачу плазми в те саме сімейство.
  **Перенесення висновку на GS спирається на цитати, а не на цей розрахунок.**
- **Наступний крок названо явно:** дефляційна континуація поверх `FreeGSNKE` на геометрії
  MAST-U за рецептом Pentland et al., щоб отримати справжні пари рівноваг з однаковими
  синтетичними діагностиками, і повторити вимір на них. Це вимагає встановлення FreeGSNKE
  і дефляційної обв'язки — тобто окремої сесії.
- Плита Біна — 1D і зі сталим `J_c`. Випадок Біна–Кіма (`J_c` залежить від поля) **лишається
  квазіваріаційним**, і навіть його існування встановлено лише нещодавно (Yousept, *M2AN* 2021).
- Резидуал Біна обчислюється лише в проникненій області, з відступом від злому;
  сама вільна межа виключена.

## Відтворення

```bash
cd "fusion equilibrium challenge/starter"
.venv/bin/python "../../Our try/03-deep-dives/D2-gs-nonuniqueness/branch_mean.py"
.venv/bin/python "../../Our try/03-deep-dives/D2-gs-nonuniqueness/plots.py"
```

Артефакти: `results.json`, `branch_mean.png`. Час виконання ≈ 10 с.

---

## English summary

**The claim under test.** Every ML paper on equilibrium reconstruction assumes the map
diagnostics → ψ is a *function*. Ham & Farrell (2024) and Pentland et al. (2025, MAST-U) show it
can be a multivalued *relation* — distinct equilibria with identical measurements ("two very
different profile shapes can produce the same flux on axis"). If so, an L2-trained regressor
converges to the **conditional mean of the branches**, and the mean of two solutions of a
nonlinear PDE is not a solution. **That damage is invisible to an L2 metric — the mean is by
definition the L2-optimal predictor — and visible only in the PDE residual.** Verification V-A3
confirmed nobody has measured this: zero arXiv hits, and the flagship benchmark's paper contains
zero occurrences of "multivalued", "non-unique" or "ill-posed".

**Design — convex control vs non-convex test**, a structure forced on us by the superconductivity
review. Control: the **Bean critical state** (1D slab, constant `J_c`), provably **unique**
(Prigozhin 1996, Thm 2, maximal monotone). Test: **1D Bratu/Gelfand** `−u″ = λeᵘ`, which has
**exactly two** solutions below the fold — and Bartolucci et al. place the plasma free-boundary
problem in precisely this Gelfand/mean-field-vortex family. Both have closed-form solutions, so
the result is exact rather than solver-dependent. Implementation validated: the computed fold
point **λ\* = 3.513830719** matches the literature to the last digit.

**Result.** Non-convex: an actual branch has residual **6.4 × 10⁻⁸** (numerical noise); the L2
mean has **1.22** — a relative residual *of the same order as the source term itself*, and **4.26**
at λ = 0.5. That is **1.9 × 10⁷ times worse**. Convex control: **2.5 × 10⁻¹³**, unchanged, because
the solution set is a singleton. The λ-dependence is also correct — as branches merge at the fold
the mean's residual falls to 0.004, exactly as bifurcation theory requires, which doubles as a code
check.

**Implications.** (1) The L2 objective is *structurally* blind to this; no reweighting of an L2
term will expose it, only a residual term will. (2) The information is not lost — each individual
branch is exact — the *objective* destroys it; the right tools (mixture density networks,
conditional generative models, invertible networks) are known. (3) Convexity is the watershed, and
the gap between 2.5 × 10⁻¹³ and 1.22 measures it in a single number.

**Limits, to be read.** This is **not Grad–Shafranov**: Bratu is a mechanism surrogate, and the
transfer to GS rests on citations, not on this computation. The named next step is **deflated
continuation on FreeGSNKE** over MAST-U geometry, following Pentland et al., to obtain genuine
equilibrium pairs with identical synthetic diagnostics and repeat the measurement — a separate
session's work. The Bean slab is 1D with constant `J_c`; the Bean–Kim case remains quasi-variational.
