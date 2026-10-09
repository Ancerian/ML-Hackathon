# C. Невизначеність і баєсів підхід / UQ and Bayesian inference

Маркери: `[V]` джерело відкрито, `[S]` зі зведення пошуку, `[?]` непідтверджено.

> **Стисло:** GP-томографія дає **UQ у замкненій формі, але без силового балансу**.
> Нелінійний баєсів GS дає **силовий баланс, але без замкненої форми**.
> **Дешевої середини не існує** — і це наша вісь новизни №2.

---

## C.1. Мінерва: баєсова графова модель

**J. Svensson & A. Werner**, *Large Scale Bayesian Data Analysis for Nuclear Fusion
Experiments*, IEEE WISP 2007, DOI 10.1109/WISP.2007.4447579 `[V]`
<https://scipub.euro-fusion.org/wp-content/uploads/2014/11/EFDC070504.pdf>

**Структурна ідея.** Кожна діагностика отримує **власний прямий модельний вузол**, пріори
задані явно, і спільний апостеріор будується композицією. Ключове: різнорідні діагностики
об'єднуються **на рівні правдоподібностей**, а не на рівні постоброблених похідних величин.
Модульність означає, що модель силового балансу можна підміняти
(томографія ↔ ітеративний GS ↔ повний нелінійний GS).

Продовження: **J. Svensson**, *Modelling of JET diagnostics using Bayesian graphical models*,
**Contrib. Plasma Phys. 51 (2011)** `[V]`
<https://onlinelibrary.wiley.com/doi/10.1002/ctpp.201000058>

## C.2. Струмова томографія — найелегантніший результат у всьому треді

**J. Svensson & A. Werner**, *Current tomography for axisymmetric plasmas*,
**Plasma Phys. Control. Fusion 50 (2008) 085002** `[S]`

**Вирішальний математичний хід:** покласти **умовний авторегресійний (CAR) / гаусів-процесний
пріор на тороїдальну густину струму J_φ(R,Z)**. Оскільки прямa модель магнітної діагностики
**лінійна за J_φ** (функції Гріна), маємо

```
правдоподібність (гаусова, лінійний оператор)  ×  пріор (гаусів)
        ⇒  апостеріор гаусів У ЗАМКНЕНІЙ ФОРМІ
```

Тобто **середнє і повна коваріація аналітично**. Одна лінійна система. Без MCMC,
без циклу навчання. **Реалізується за півдня людиною, що знає гаусове обумовлення.**

> **Чесне обмеження, яке треба вимовити:** **струмова томографія — це не рівновага.**
> Вона повністю відкидає силовий баланс; GP-пріор замінює обмеження GS. Ви отримуєте
> каліброване UQ на J_φ, але результат **не зобов'язаний** задовольняти
> Δ*ψ = −μ₀RJ_φ із узгодженими p(ψ), F(ψ).
> Щойно ви накладаєте силовий баланс — **спряженість гине**, і потрібні MCMC/HMC
> або варіаційний вивід.

## C.3. Силовий баланс як гіпотеза, що перевіряється

**G. T. von Nessi & M. J. Hole**, *A unified method for inference of tokamak equilibria and
validation of force-balance models based on Bayesian analysis*, arXiv:1209.3068
(*J. Phys. A*, 2013) `[V]` <https://arxiv.org/html/1209.3068>

**Математично найцікавіше обрамлення у треді:** трактувати силовий баланс **як гіпотезу,
яку можна перевірити**, а не як жорстке обмеження. Тобто порівняння моделей / фактори Баєса
по фізичних моделях.

Супутнє: **M. J. Hole et al.**, arXiv:1403.1321 `[V]` — вузьке місце
високовимірного інтегрування/квадратур, зроблене явним.

## C.4. Сучасна баєсова інтеграція діагностик

- **S. Kwak, J. Svensson et al.** — баєсів вивід рівноваги, що поєднує магнітні виміри,
  інтерферометрію, поляриметрію, Томсон і Li-beam на JET; апостеріори ℓi та β_p для L- і H-моди.
  EPS 2019 P4.1019 `[V]` — **журнальну версію знайти й дозвірити** `[?]`.
- *Tomography for plasma imaging: a unifying framework for Bayesian inference*,
  arXiv:2506.20232 `[V]` — сучасне уніфікувальне викладення, добрий педагогічний вхід.
- *Simultaneous kinetic profile and magnetic equilibrium inference with Bayesian integrated
  data analysis in preparation for ITER*, arXiv:2502.07805 `[S]`.
- Огляди: *Machine learning and Bayesian inference in nuclear fusion research*,
  **PPCF (2023)**, DOI 10.1088/1361-6587/acc60f `[V]`; *A review of the Bayesian method in
  nuclear fusion diagnostic research*, **J. Fusion Energy (2024)** `[V]`.

## C.5. UQ у машинному навчанні для рівноваги

**EFIT-Prime** (Madireddy et al., **Phys. Plasmas 31, 092505 (2024)**) `[S]` — **глибокий
ансамбль на основі пошуку архітектури**, що розділяє **алеаторну** (невизначеність даних)
та **епістемічну** (невизначеність моделі).

*Обмеження, які треба назвати:*
- «Фізично обмежений» означає **м'які штрафи**, а не задоволений резидуал GS.
- Глибокі ансамблі дорогі.
- **Їхнє UQ не калібровано в жодному частотному сенсі.** Тобто ніхто не показав, що
  90%-й інтервал накриває істину у 90% випадків.

**Bayesian neural network for plasma equilibria in KSTAR** — S. Joung, дисертація KAIST,
arXiv:2301.11555 (2023) `[V]` — корисне як довідка «UQ для нейромережевої рівноваги».

## C.6. Конформні передбачення — чого бракує полю

Конформні передбачення дають **гарантії покриття без припущень про розподіл**: за обмінюваності
даних побудований інтервал накриває істину з імовірністю ≥ 1−α, **незалежно** від того,
наскільки погана базова модель.

Це рівно та властивість, якої немає ні в глибоких ансамблів, ні в MC-dropout, і якої
**фузійна ML-література практично не використовує** `[?]` (перевірити цілеспрямовано).

**Чому це наша тема.** Це чиста математика (обмінюваність, порядкові статистики,
скоригована на скінченну вибірку квантиль), вона **дешева обчислювально**, і вона дає
твердження, яке можна **перевірити емпірично за хвилини**: побудувати криву калібрування
(номінальне покриття проти емпіричного) і подивитися, чи лежить вона на діагоналі.

**Ускладнення, яке робить задачу нетривіальною і цікавою:** ψ — це **поле 65×65**,
а не скаляр. Поточкові конформні інтервали дають 4225 маргінальних гарантій, а не одну
спільну. Потрібні **одночасні / функціональні** конформні смуги, і тут є справжній
математичний вибір (max-статистика, проєкція на PCA-базис, конформізація похідних скалярів).

## C.7. Порожня ніша — сформульовано прямо

| Підхід | Силовий баланс | UQ | Вартість |
|---|---|---|---|
| GP-струмова томографія | ✗ відкинуто | ✓ замкнена форма | одна лінійна система |
| Нелінійний баєсів GS | ✓ | ✗ потрібен MCMC | дорого |
| Глибокі ансамблі (EFIT-Prime) | ~ м'який штраф | ~ некалібровано | дуже дорого |
| **Порожнє місце** | ✓ | ✓ дешево | ? |

**Кандидати на заповнення, які ніхто не пробував** `[?]`:
- **лапласова апроксимація навколо GS-многовиду** (гаусів апостеріор із гесіана в точці
  розв'язку класичного солвера);
- **лінеаризований GS** із гаусовим апостеріором плюс калібрувальна поправка;
- **конформізація виходу класичного солвера**, а не нейромережі — гарантії покриття
  без відмови від фізики.

Усі три **тракта́бельні за 48 годин на синтетичних даних**, і всі три дають публікабельний
результат незалежно від знака відповіді.

---

## English summary

**The gap, stated plainly:** GP current tomography gives **closed-form UQ without force balance**;
nonlinear Bayesian GS gives **force balance without closed form**; deep ensembles give neither
cheaply nor with calibration. **No cheap middle exists** — this is our novelty axis #2.

**Minerva** (Svensson & Werner 2007) is a Bayesian graphical framework where every diagnostic gets
a forward-model node and heterogeneous data combine **at the likelihood level**, not as
post-processed derived quantities — so the force-balance model is swappable.

**Current tomography** (Svensson & Werner 2008) is the most elegant result in the thread: a
CAR/Gaussian-process prior on J_φ(R,Z) combined with a **linear** Green's-function forward model
yields a **Gaussian posterior in closed form** — mean and full covariance analytically, one linear
solve, no MCMC, implementable in an afternoon. The honest caveat: **it is not an equilibrium**. It
discards force balance entirely; imposing force balance destroys conjugacy.

**von Nessi & Hole** offer the most interesting framing mathematically — force balance as a
*testable hypothesis* via Bayesian model comparison rather than a hard constraint.

**EFIT-Prime** is the most serious ML-UQ work (NAS-based deep ensemble separating aleatoric from
epistemic uncertainty), but its "physics constraints" are soft penalties and **its UQ is not
calibrated in any frequentist sense** — nobody has shown a 90% interval covers truth 90% of the time.

**Conformal prediction** supplies exactly the missing property: distribution-free coverage
guarantees under exchangeability, cheap to compute, and empirically checkable in minutes via a
calibration curve. The non-trivial twist that makes it a real research question: ψ is a **65×65
field**, so pointwise conformal intervals give 4,225 marginal guarantees rather than one joint one.
Simultaneous/functional conformal bands are needed, and the choice among them (max-statistic,
PCA-basis projection, conformalising the derived scalars) is a genuine mathematical decision.

**Untried candidates to fill the gap**, all tractable in 48 hours on synthetic data: a **Laplace
approximation around the GS solution manifold**; a **linearised-GS Gaussian posterior** with a
calibrated correction; and **conformalising a classical solver's output** rather than a network's —
coverage guarantees without abandoning physics.
