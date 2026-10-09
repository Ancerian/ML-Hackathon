# D3 — Пілот оберненої задачі на перетині Пуанкаре
# / Poincaré inverse problem: feasibility pilot

**Статус: пілот виконано. Гіпотезу C5 НЕ ЗАКРИТО** — прояснено, що саме її блокує.
*Оновлення 2026-10-01:* **C5 закрито — спростовано → R13** (нелінійна перевірка, розділ «Перевірка C5» нижче; доказ — `c5/`).
Дата: **2026-09-23**.

---

## Навіщо пілот

Рівень 3 обрано основним напрямком інтеграції. Перш ніж закладати на нього тижні,
треба знати, чи обернена задача взагалі **коректно поставлена**: чи можна відновити
прихований спектр збурень (m, n) зі спостережуваної структури.

Пілот мав відповісти **до** побудови інфраструктури, а не після.

## Що працює

**Tokamap реалізовано і перевірено** (`tokamap.py`, ~40 рядків):

```
det J:  x_L=0,00 → max|det−1| = 3,4·10⁻⁹
        x_L=1,00 → max|det−1| = 5,6·10⁻⁸
Ψ ≥ 0 зберігається при x_L = 0,5 / 1,0 / 2,0
перехід до хаосу монотонний: частка орбіт із розкидом > 0,3
        0,0 при x_L ≤ 0,5  →  0,083 (0,8)  →  0,250 (1,2)  →  0,417 (2,0)
```

**Точно симплектичне, Ψ ≥ 0 за побудовою, хаос з'являється як належить.**
Для навчального використання це готовий інструмент: три рівняння, жодних залежностей.

## Дві власні помилки, знайдені пілотом

### Помилка 1: багатомодове узагальнення **не симплектичне**

Я припустив, що структуру Tokamap можна перенести на суму мод, замінивши
`V(T) = (x_L/2π)sin(2πT)` на `Σ_k (a_k/2π)sin(2π m_k T + φ_k)`.

```
a = 0,00  →  max|det J − 1| = 3,4·10⁻⁹     ✓
a = 0,20  →  max|det J − 1| = 5,0·10⁻¹     ✗
a = 0,50  →  max|det J − 1| = 1,7          ✗
```

**Не переноситься.** Член корекції T′ прив'язаний до конкретної твірної функції Tokamap,
і наївна підстановка V′(T) її руйнує. **Така мапа не є відображенням силових ліній:
вона не зберігає потік, тож будь-який результат на ній фізично нічого не означає.**

### Помилка 2: спостережувана була **непридатна**

Перший тест ідентифіковності брав **одну** випадкову початкову фазу на радіус.
Шумова підлога **лише від зміни seed'а** склала **0,3244**, а всі «сигнали» — 0,38–0,43.
Тобто тест не вимірював нічого: різниці лежали всередині розкиду шуму.

**Виправлення:** усереднення по **24 початкових фазах** на радіус.

```
шумова підлога:  0,3244  →  0,0006     (зниження у 540 разів)
```

## Що це дало

На **перевірено симплектичній** одномодовій мапі, з виправленою спостережуваною:

| зміна x_L | відносна зміна спостережуваної | SNR до підлоги |
|---|---|---|
| +3% | 0,0341 | **57** |
| +10% | 0,1028 | **171** |
| +20% | 0,2063 | 342 |
| +50% | 0,5233 | 869 |

**Одна амплітуда відновлюється тривіально.** SNR 57 при зміні на 3% — це не задача
для хакатону, це підстановка.

## Вердикт по C5

**Гіпотеза не закрита, але її блокер локалізовано точно.**

| Передумова | Стан |
|---|---|
| Валідна симплектична мапа для **однієї** моди | ✅ є, перевірено до 10⁻⁸ |
| Спостережувана з прийнятною шумовою підлогою | ✅ розв'язано, 540× покращення |
| Валідна симплектична мапа для **кількох** мод | ❌ **не зроблено — це й є блокер** |
| Ідентифіковність спектра (m, n) | ❓ **непротестовано**, бо потребує попереднього рядка |

**Одномодова версія задачі занадто легка.** Уся складність — і вся наукова цінність —
у багатомодовому випадку, а він упирається в коректне симплектичне узагальнення.

## Що робити далі

Три шляхи, за зростанням вартості:

1. **Вивести правильну твірну функцію** для багатомодового випадку. Найдешевше, якщо
   виведення тримається; але це справжня робота, а не 20 рядків.
2. **Взяти `pyoculus`** (MIT, Python) — у ньому вже є нерухомі точки, лишки Гріна,
   інваріантні многовиди, площі turnstile і QFM. Найкоротший шлях до робочого стенду.
3. **Взяти готову багатогармонічну мапу з літератури** (мапи ергодичного магнітного
   лімітера, сімейство дивертрних мап Пунджабі) замість власного узагальнення.

**Рекомендація: (2), і перевірити (1) як вправу на валідацію.** Причина проста —
сьогодні власне узагальнення зламалось мовчки, і зламалось би так само вдруге.

## Відтворення

```bash
cd "fusion equilibrium challenge/starter"
.venv/bin/python "../../Our try/03-deep-dives/D3-poincare-inverse/tokamap.py"          # ~20 c
.venv/bin/python "../../Our try/03-deep-dives/D3-poincare-inverse/multimode.py"        # ~40 c  (показує ПОЛАМАНУ мапу — лишено навмисно)
.venv/bin/python "../../Our try/03-deep-dives/D3-poincare-inverse/identifiability.py"  # ~90 c
```

`multimode.py` **навмисно лишено в репозиторії разом із його провалом** — це найдешевший
спосіб не повторити ту саму помилку.

---

## Перевірка C5 (2026-10-01): нелінійна задача — спростовано → R13

Критерії записано до запуску (`c5/THEORY.md`, план затверджено користувачем). Базовий
розв'язувач — багатостартовий МНК (`least_squares`, прямі різниці з фіксованими великими
кроками), 3 моди, амплітуди + фази, фазово-роздільна екскурсія, 3 % шуму.

| | Tokamap з h(T) (E35) | `tokamak-3d-viz` |
|---|---|---|
| успіх (амплітуди ≤ 10 %, фази ≤ 20°) | **0 / 40** | **0 / 3** |
| хибні мінімуми | 0 | 0 |
| найкращий χ² / χ²_true | медіана 110 | 303–503 |

**Інформація є, дістатися до неї базовий фіт не може.** Зі старту в істині фіт стоїть на місці,
зі старту на 5 % / 10° поруч — застрягає при χ² у 41–100 разів більшому. Ландшафт скляний: 8
локальних мінімумів у ±20° по одній фазі (E40). Причина — спостережуване: ptp за скінченне число
обертів є максимумом по дискретних точках і стрибає з параметром. Те, що пілот виправив для
шумової підлоги (усереднення по фазах), не робить його **гладким**.

*Не перевірено:* локальна обумовленість `tokamak-3d-viz` (VE22–VE25, VE31) рахувалась кроком 15 %
і описує згладжену січну; гладше спостережуване може повернути задачу базовому фіту.
Деталі й відтворення — `c5/README.md`.

---

## English summary

**Purpose.** Level 3 was chosen as the primary integration direction, so before committing weeks
we asked whether the inverse problem is even well-posed: can a hidden (m, n) perturbation spectrum
be recovered from observable structure?

**What works.** The **Tokamap is implemented and verified** — ~40 lines, three equations, no
dependencies. It is **exactly symplectic** (max |det J − 1| = 5.6 × 10⁻⁸ at x_L = 1.0),
preserves Ψ ≥ 0 by construction, and shows the expected monotone transition to chaos (fraction of
orbits with excursion > 0.3 rising 0 → 0.083 → 0.250 → 0.417 as x_L goes 0.5 → 0.8 → 1.2 → 2.0).

**Two of our own errors, both found by the pilot.** First, the **multi-mode generalisation broke
symplecticity** (max |det J − 1| = 1.7 at a = 0.5): the Tokamap's T′ correction is tied to a
specific generating function, and substituting a sum of modes destroys it. That map is therefore
**not a field-line map** and nothing measured on it means anything physically. Second, the first
identifiability test used **one** random initial phase per radius, giving a reseed-only noise floor
of **0.3244** while every tested signal sat at 0.38–0.43 — the test measured nothing. Averaging over
**24 initial phases** dropped the floor to **0.0006**, a **540×** improvement.

**What that bought.** On the verified single-mode map with the corrected observable, a **3% change
in x_L sits at SNR 57**, and 10% at SNR 171. So a single amplitude is **trivially** recoverable —
which means the single-parameter version is not a task, it is a substitution.

**Verdict on conjecture C5: not closed, but the blocker is now located precisely.** A valid
symplectic map for one mode exists; a usable observable exists; a valid symplectic map for
**several** modes does **not** — and spectrum identifiability cannot be tested until it does. All
the difficulty, and all the scientific value, lives in the multi-mode case.

**Recommendation:** adopt **`pyoculus`** (MIT, Python — fixed points, Greene residues, invariant
manifolds, turnstile areas, QFM surfaces already implemented) rather than generalising the map
ourselves, and treat deriving the correct generating function as a validation exercise. The reason
is simply that our own generalisation **broke silently today**, and would break the same way twice.
`multimode.py` is kept in the repository **together with its failure**, as the cheapest way not to
repeat it.

**C5 test, 2026-10-01 — refuted (R13).** Against pre-registered criteria (`c5/THEORY.md`), a multi-start least-squares baseline recovered 3 amplitudes + 3 phases from the phase-resolved excursion at 3 % noise on 0/40 instances of the corrected multi-mode Tokamap and 0/3 on the `tokamak-3d-viz` stand. No spurious minima — the information is there — but no start reached the noise level: the excursion is a maximum over discrete orbit points, so the misfit landscape is glassy (E40: 8 local minima within ±20° of one phase). Phase averaging fixed the noise floor, not the smoothness.
