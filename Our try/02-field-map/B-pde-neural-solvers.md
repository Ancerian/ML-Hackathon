# B. Нейронні розв'язувачі PDE: PINN, Deep Ritz, оператори
# / Neural PDE solvers, with the negative literature given equal weight

Маркери: `[V]` джерело відкрито, `[S]` зі зведення пошуку, `[?]` непідтверджено.

> **Принцип цього файлу.** Негативна література тут **не додаток, а рівноправна частина**.
> Хакатон, який роздає студентам PINN і не каже, де вони провально не працюють, —
> це не фізмат, це маркетинг.

---

## B.1. PINN для рівняння Града–Шафранова

| Робота | Внесок | Код | Обмеження |
|---|---|---|---|
| **Jang, Kaptanoglu, Gaur, Pan, Landreman, Dorland**, *Grad–Shafranov equilibria via data-free PINNs*, **Phys. Plasmas 31, 032510 (2024)**, arXiv:2311.13491 `[V]` | PINN без даних (чиста мінімізація резидуала) для кількох типів крайових умов; систематичний компроміс точність↔швидкість; **параметризований PINN** із (p, A, κ, δ) як додатковими входами ⇒ одна мережа на ціле сімейство рівноваг | не вказано | фіксована межа; **немає X-точки**; точність значно нижча за МСЕ |
| **Rizqan, Hole, Gretton**, arXiv:2504.21155 (2025) `[V]` | PINN проти FNO із фокусом на узагальненні **по змінних крайових умовах**; використано **Marabou** — інструмент формальної верифікації нейромереж — щоб *довести* властивості навчених мереж | не вказано | розбіжності між обчисленнями в PyTorch і Marabou (плаваюча кома) — чесне й рідкісне визнання |
| **Zhou & Zhu**, arXiv:2507.16636 (2025) `[V]` | **багатостадійний PINN**: мережа 2-ї стадії вчить резидуал 1-ї. Похибка **O(10⁻⁸)** проти аналітичних розв'язків | не вказано | лише розв'язки класу Solov'ev; немає вільної межі, немає експериментальних даних; **автори обмежень не обговорюють** |
| **Rossi, Gelfusa, Murari & JET contributors**, **Nucl. Fusion 63, 126059 (2023)** `[V]` | три архітектури — S3MHDnet, S2MHDnet, GradShafrNet — для реконструкції рівноваги, інверсії інтерферометра, томографії болометра; валідовано на аналітиці й даних JET | не вказано | **напрочуд відвертий розділ обмежень**: точність падає з тороїдальною асиметрією; 3D гірше за 2D через розрідженість сенсорів; **без внутрішніх обмежень ядро погане**; гіперпараметри підбираються щоразу |
| **Rutigliano, Murari, Gaudio, Gelfusa, Rossi**, **PPCF 68(4), 045002 (2026)** `[V]` | **систематичне параметричне дослідження.** Три архітектури: **MHD-Net** (м'які обмеження), **GS-Net** (жорстко кодує p і F як функції потоку), **H-Net** (гібрид). Висновки: **адаптивне зважування втрат б'є статичне**; **GS-Net збігається найшвидше (~2·10⁻⁸ фізичних втрат) на симетричних випадках, але КАТАСТРОФІЧНО ПРОВАЛЮЄТЬСЯ на асиметричних n/T** | **ТАК** — <https://github.com/QEP-Repository/PINN_MHD_Equilibria_NetOptimisation> | лише синтетичні сценарії TokaLab; одна геометрія; SOL спрощено; ≤5000 епох може бути недозбіжністю |
| **Rutigliano et al.**, *Multi-diagnostics reconstruction… applications to JET*, **Nucl. Fusion (2026)** `[S]` | монотонне покращення при додаванні діагностик. **На відміну від класичних кодів реконструкції, PINN відновлює nₑ і Tₑ ОДНОЧАСНО з ψ** | — | специфічно для JET |
| **Kaltsas & Throumoulopoulos**, arXiv:2109.12850 `[S]` | PINN для **узагальненого GS із течією** (GS–Бернуллі) — складніший, менш вивчений оператор | — | — |

> **Урок із GS-Net.** Жорстко закодовані структурні припущення купують швидкість ціною
> **звуженого класу гіпотез**. Це підручникова ілюстрація для лекції на відкритті.

## B.2. Deep Ritz і варіаційні нейронні методи

Рівняння GS **має варіаційну структуру** — формулювання Temam саме таке: розв'язки як
критичні точки функціонала за наявності обмеження. Тож Deep Ritz — математично **природна**
і **недодосліджена** альтернатива PINN саме тут.

- **E, Weinan & Yu, Bing**, *The Deep Ritz method*, **Commun. Math. Stat. 6 (2018) 1–12** `[S]`
  (точна пагінація не звірена).
- **Чому це може мати значення для GS.** DRM мінімізує ∫(½|∇u|² − fu) замість ‖Δu − f‖².
  Це **знижує диференціальний порядок у функції втрат із 2 до 1** — а саме високий порядок
  Krishnapriyan et al. звинувачують у поганій обумовленості м'якого штрафу.
  Ціна: замість похибки autodiff з'являється **похибка квадратури**, і природним простором
  стає **H¹ замість C²**.
- Порівняння лоб у лоб: *Neural PDE solvers with physics constraints: PINNs, DRM, and WANs*,
  arXiv:2510.09693 `[S]`.
- Пом'якшення спектрального зміщення: *Deep Ritz with Fourier feature mapping*,
  arXiv:2502.06865 `[S]`.

> ⚠️ **Перевірка V-A1 виконана: опублікованого розв'язувача Deep Ritz для Града–Шафранова
> немає** (0 збігів в arXiv API і UI; корпус «Grad-Shafranov» на arXiv — 278 робіт, тобто
> пошук іде по малій добре окресленій множині). Уся нейронна література по GS —
> це **мінімізація резидуала (PINN/колокація) або кероване регресування**, ніколи не
> мінімізація енергетичного функціонала. Формулювати слід як **«нам не відомо про»**,
> бо пошук arXiv покриває назву й анотацію, але не повний текст, і не покриває CNKI
> та репозиторії дисертацій.
>
> **І є важливий «майже-збіг», який треба цитувати, бо він ПОСИЛЮЄ нашу позицію:**
> arXiv:**2609.13101**, *Quantum Variational Approaches to Plasma Equilibrium: Cost Function
> Design for the Grad–Shafranov Equation* `[V]`. Вона будує **рівно той об'єкт**, який потрібен
> для Deep Ritz — зважений енергетичний функціонал
> `J(ψ) = ∬ {−½ψΔ*ψ − μ₀R²P(ψ) − ½F²(ψ)} (1/R) dΩ`
> (вага `1/R` робить Δ* самоспряженим) — і стверджує, що **розв'язання GS еквівалентне
> мінімізації J(ψ)**. Але мінімізує його **варіаційною квантовою схемою**, а не нейромережею,
> і Deep Ritz не цитує.
> **Отже відмовка «у GS немає енергетичного функціонала» знята явно й у літературі.**
>
> **У нас уже є реалізація:** `Tokamak-GS-solver/src/examples/deep_ritznet.py`.

## B.3. Нейронні оператори

- **P. G. Krastev** (Harvard), *Millisecond-scale neural operator surrogates for double-null
  free-boundary Grad–Shafranov equilibria*, arXiv:2608.05555 (2026) `[V]` —
  **геометрично зумовлений FNO**, навчений на рівновагах із FreeGS. Середня відносна
  L²-похибка **0,05%**; X-точка з точністю 0,2 см; O-точка 0,03 см; прискорення 640× на GPU.
  *Обмеження (визнане автором):* одна фіксована геометрія машини й задана топологія.
  > **Математична примітка.** Заявлено «середній нормований резидуал 2,29, що **збігається
  > з самим бейзлайном FreeGS**». Це і є цікаве число: сурогат **успадковує**, а не покращує
  > похибку дискретизації свого вчителя. **Сурогат не може бути точнішим за свої мітки** —
  > це стеля апроксимаційної теорії, яку прикладна література чітко не проговорює.
- **Yoo, Howes, Ghai, Kobayashi, Chakraborty, Alam**, arXiv:2606.15512 (2026) `[V]` —
  вісім геометрично різних конфігурацій із аналітичного сімейства **Solov'ev**;
  чотири стратегії перенесення. **Висновок: передтренування на одній геометрії переноситься
  погано; передтренування на багатьох геометріях дає ефективну за даними адаптацію** —
  <4% відносної L² лише зі 100 міченими цільовими рівновагами.
  *Обмеження:* Solov'ev аналітичний і низькоскладний — далеко від реальних дивертованих рівноваг.
- **Ding, Zhang, Shi et al.**, *Physics-informed neural operator learning for nonlinear
  Grad–Shafranov equation*, arXiv:2511.19114 (2025) `[V]` — найкращий варіант
  **Transformer–KAN Neural Operator (TKNO)**. Кероване: 0,25% середньої відносної L²;
  напівкероване: **0,48% інтерполяція проти 4,76% екстраполяція**.

> **Найцитованіший математичний факт у цьому треді, і основа нашої осі новизни №1:**
> суто керовані моделі досягали **чудової L²-похибки за великих фізичних резидуалів** —
> тобто **мала похибка розв'язку не тягне малий резидуал**, бо навчене відображення
> **не є проєкцією Гальоркіна** і немає переносу стійкості на кшталт леми Сеа.
> А **десятикратна деградація інтерполяція→екстраполяція** — це стіна узагальнення в числах.

### Апроксимаційна теорія, яку варто дати студентам

- **Lanthaler, Mishra, Karniadakis**, *Error estimates for DeepONets*, **Trans. Math. Appl.
  6(1), tnac001 (2022)**, arXiv:2102.09618 `[V]` — **розклад: повна похибка = кодування +
  апроксимація + реконструкція**, із **верхніми і нижніми** оцінками, прив'язаними до
  **спектрального спадання коваріаційних операторів**. Для загальних ліпшицевих операторів —
  прокляття розмірності; для конкретних класів PDE воно доказово долається.

> **Чому це конкретно корисно.** Член похибки реконструкції — це **рівно твердження про
> PCA / ширину Колмогорова многовиду розв'язків**. Тож питання «скільки PCA-мод насправді
> має простір рівноваг DIII-D?» є водночас (а) обчислюваним за півдня на наявних даних
> і (б) **строгою нижньою межею** для будь-якого операторного методу з лінійним декодером.
> У нас PCA вже порахована: `my_experiments/pca_variance.png`, перша компонента — 94,2%.

## B.4. Чесна математична критика — обов'язкове читання

1. **Krishnapriyan, Gholami, Zhe, Kirby, Mahoney**, *Characterizing possible failure modes in
   PINNs*, **NeurIPS 34 (2021)**, arXiv:2109.01050 `[V]`
   **Суть:** PINN успішні на тривіальних задачах і **провалюються на лише трохи складніших** —
   не через виразність, а тому що **ландшафт оптимізації з м'якою регуляризацією стає
   погано обумовленим зі зростанням коефіцієнта PDE**. **Відмова в оптимізаторі, не в апроксиматорі.**
   *Пряма релевантність для GS:* GS нелінійне і, у випадку вільної межі, негладке.
   Аналог «режиму великого коефіцієнта» — високе β / сильне формування / околиця X-точки.
2. **Wang, Teng, Perdikaris**, *Understanding and mitigating gradient flow pathologies in PINNs*,
   **SIAM J. Sci. Comput. 43(5), A3055 (2021)** `[V]` — **чисельна жорсткість ⇒ незбалансовані
   зворотні градієнти** між резидуальним і крайовим членами.
3. **Wang, Yu, Perdikaris**, *When and why PINNs fail to train: an NTK perspective*,
   **J. Comput. Phys. 449, 110768 (2022)** `[V]` — у границі нескінченної ширини динаміка
   навчання керується NTK, **власні значення якого для резидуальної і крайової компонент
   різняться на порядки**, тож компоненти втрат збігаються з шалено різними швидкостями.
   > Це **теоретична підкладка** емпіричного результату «адаптивне зважування б'є статичне»
   > у Rutigliano et al. (PPCF 2026) — красиве замкнене коло для доповіді.
4. **Grossmann, Komorowska, Latz, Schönlieb**, *Can physics-informed neural networks beat the
   finite element method?*, **IMA J. Appl. Math. 89(1) (2024) 143–174** `[V]`
   **Вердикт без прикрас:** *«За часом розв'язання і точністю PINN не змогли перевершити
   метод скінченних елементів.»* Важливе застереження: PINN були **швидшими в обчисленні
   вже розв'язаної PDE** — тобто виграш у **амортизованому інференсі, не в розв'язуванні**.
5. **Mishra & Molinaro**, *Estimates on the generalization error of PINNs*,
   **IMA J. Numer. Anal. 43(1) (2023) 1–43** `[V]` — строга рамка: **похибка узагальнення
   ≤ C(похибка навчання, число точок квадратури)**, де C керується **сталою стійкості
   базової PDE**.
   > **Вирішальне застереження для GS:** ці оцінки вимагають оцінки стійкості PDE.
   > **Для задачі з вільною межею в режимі неєдиності такої сталої не існує** —
   > отже теорія формально **не застосовна**. Сказати це вголос на відкритті хакатону —
   > це рівно той «погляд математика», якого ми хочемо.

### Де PINN справді виграють — чесний позитивний бік

- **Обернені задачі та асиміляція даних** із розрідженими, різнорідними, шумними даними.
  Класичному солверу потрібна коректна пряма постановка; PINN поглинає часткові виміри як
  додаткові члени втрат без переcітковування. **Сильна версія: Rutigliano 2026 —
  спільна оцінка ψ + nₑ + Tₑ, чого не робить жоден стандартний код реконструкції.**
- **Параметричні сімейства й оптимізація форми** (параметризований PINN у Jang et al.).
- **Безсітковий вихід** довільної роздільності, декомпозиція області.
- **Амортизований інференс:** 640× у Krastev, 0,11 мс у EFIT-mini, <100 мкс на TCV.
- **Як передобумовлювач/ініціалізатор класичного солвера** (EFIT-mini) — найкраще з обох світів
  і **єдина архітектура, що витримує критику Grossmann et al.**

## B.5. Чому вільна межа особливо важка для нейронних методів

1. **Область є невідомою самої задачі.** Ω_p = {(R,z) : ψ < ψ_b}. Методи на точках колокації
   мусять семплювати область, якої ще не знають. Література про задачу Стефана впирається
   в ту саму стіну: **Wang & Perdikaris**, *Deep learning of free boundary and Stefan problems*,
   **J. Comput. Phys. 428, 109914 (2021)** `[V]` — підхід із двох мереж: одна на розв'язок,
   друга на рухому межу.
2. **ψ_b — негладкий функціонал від ψ.** Це або max по точках дотику лімітера, або значення
   в **сідловій точці (X-точка)**, де ∇ψ = 0. Обидві операції — max/критична точка:
   **відображення ψ ↦ ψ_b щонайбільше ліпшицеве, не диференційовне**, і його autodiff-градієнт
   розривний щоразу, коли перемикається активна точка лімітера або гілка X-точки.
   Autodiff крізь argmax по сітці дає градієнт, майже всюди нульовий і подекуди нескінченний.
3. **Зміни топології дискретні.** Лімітована → дивертована, single-null → double-null.
   Неперервна мережа не може подати розривний топологічний індекс. Чесний обхід у Krastev:
   **задати положення X-точок як входи** і обмежитися фіксованою топологією.
4. **Структура нелінійної задачі на власні значення.** Обмеження на повний струм плазми діє
   як умова на власне значення (формулювання Temam). У стандартних втратах PINN **немає
   механізму нав'язати нелінійне обмеження на власне значення**.
5. **Множинність.** Навіть за ідеальних даних ціль може бути **гілкою, а не точкою**
   (Ham–Farrell 2024; Pentland et al. 2025). Градієнтний спуск по резидуалу **не має механізму
   вибору гілки**; дефляція має.
6. **Різкі градієнти на сепаратрисі** — відома слабкість PINN.
7. **Тип крайової умови.** Вільна межа в GS використовує **інтегральну (нелокальну) крайову
   умову** через вільнопросторову функцію Гріна на обчислювальній межі. Це **не поточкова
   умова** і її не можна нав'язати жорстко через функцію відстані.
   > ⚠️ **ВИПРАВЛЕНО після перевірки V-A2.** Первісне твердження «ніхто не нав'язує інтегральну
   > умову жорстко» **спростовано**. Прямий контрприклад: **McClenaghan et al.,
   > *Phys. Plasmas* 31, 082507 (2024)**, DOI 10.1063/5.0213625 `[V]` — вихідний простір мережі
   > є лінійною оболонкою відгуків функцій Гріна, тож ψ = Σ cᵢ (G ∗ Jᵢ) задовольняє
   > співвідношення **точно за побудовою**: без члена втрат на межі й без λ для підбору.
   > Поза фузійною областю те саме роблять BI-GreenNet (arXiv:2204.13247) і
   > Green-Integral-Constrained Neural Solver (arXiv:2604.21411) `[S]`.
   >
   > **Що з твердження вціліло (вужча, перевірена версія):** немає опублікованої
   > нейроархітектури, яка жорстко нав'язує інтегральну умову **всередині прямого
   > розв'язувача GS із вільною межею**. McClenaghan — це обернена задача/реконструкція,
   > а не прямий free-boundary солвер. Підтверджені «обхідники» на прямому боці:
   > Krastev (arXiv:2608.05555) явно «обходить інтегральні обмеження вільної межі,
   > задаючи положення X-точок як вхід»; Rizqan (arXiv:2504.21155) використовує м'яку
   > умову з λ = 100 `[V]`.

## B.6. Відкритий код, придатний для starter kit

| Репозиторій | Що це |
|---|---|
| <https://github.com/QEP-Repository/PINN_MHD_Equilibria_NetOptimisation> `[V]` | **найякісніший відтворюваний артефакт PINN-для-рівноваги**: код + дані до PPCF 2026 |
| <https://github.com/ZINZINBIN/Tokamak-GS-solver> `[V]` | **у нас уже локально.** Числовий (FreeGS-подібний) і PINN бекенди; PINN приймає 0D-параметри (β_p, q₉₅, Iₚ, ℓi); **є вільна межа і є `deep_ritznet.py`** (⚠️ 2026-09-29: розв'язувач із вільною межею до цієї дати не запускався з README — див. E34; виправлено частково, `Tokamak-GS-solver/CHANGES-2026-09-29.md`) |
| <https://tokalab.github.io/> `[V]` | віртуальний токамак: VirtualLab, SimPla, **SynDiag**. **Істина відома** — знімає ризик «день 1 на дані» |
| <https://github.com/a1k12/characterizing-pinns-failure-modes> `[V]` | код до Krishnapriyan et al. |
| <https://github.com/TamaraGrossmann/FEM-vs-PINNs> `[V]` | **готовий протокол порівняння** МСЕ проти PINN |
| <https://github.com/erikjonperez/PINN-for-magnetic-confinement-in-a-Tokamak> `[V]` | маленький PINN для ψ(R,Z), <0,04% проти FEMM. Добрий «hello world» |

---

## English summary

**This file treats the negative literature as a first-class citizen.** A hackathon that hands
students PINNs without telling them where PINNs provably fail is marketing, not a physics and
mathematics faculty.

**PINNs for Grad–Shafranov**: Jang et al. (data-free, parameterised over shape); Rossi et al.
(JET, with an unusually candid limitations section); Rutigliano et al. (PPCF 2026, **open code**,
systematic architecture study — adaptive loss weighting beats static, and the hard-constrained
GS-Net converges fastest on symmetric cases but **fails catastrophically on asymmetric profiles**,
a textbook illustration that structural assumptions buy speed at the cost of hypothesis-class width).

**Deep Ritz**: GS has a variational structure (Temam 1975), so DRM is the mathematically natural
and under-explored option — it lowers the differential order in the loss from 2 to 1, trading
autodiff error for quadrature error and C² for H¹. **No published Deep Ritz solver for GS was
found** (verification V-A1 pending). We already hold an implementation in
`Tokamak-GS-solver/src/examples/deep_ritznet.py`.

**Neural operators**: Krastev's FNO reaches 0.05% relative L² but its residual merely *matches*
FreeGS — a surrogate cannot be more accurate than its labels, an approximation-theory ceiling the
applied literature rarely states. Ding et al. supply our headline fact: **excellent L² error
alongside large physics residuals**, because a learned map is not a Galerkin projection and there
is no Céa-lemma stability transfer. Lanthaler–Mishra–Karniadakis decompose operator-learning error
into encoding + approximation + reconstruction, where the reconstruction term **is** a Kolmogorov
n-width statement — making "how many PCA modes does the DIII-D equilibrium manifold have?" both a
half-day computation and a rigorous lower bound for any linear decoder.

**The critique**: Krishnapriyan (failure is in the optimisation landscape, not expressivity);
Wang et al. (gradient pathologies; NTK eigenvalues of residual and boundary terms differ by orders
of magnitude — the theory behind adaptive weighting); Grossmann et al. (**PINNs do not beat the
FEM**; the win is amortised inference); Mishra & Molinaro (generalisation bounds require a PDE
stability constant — **which does not exist for free-boundary GS in the non-unique regime**, so
the theory formally does not apply).

**Where PINNs genuinely win**: inverse problems with sparse heterogeneous data (Rutigliano's joint
ψ + nₑ + Tₑ inference, which no standard reconstruction code does), parametric families, meshless
output, amortised inference, and — most defensibly — as a **neural initialiser for a classical
certifier** (EFIT-mini), the one architecture that survives the Grossmann critique.

**Why free boundaries are hard for neural methods**: the domain is itself unknown; ψ_b is a
max/saddle functional, Lipschitz but not differentiable, with discontinuous autodiff gradients at
branch switches; topology changes are discrete; there is a nonlinear eigenvalue constraint with no
standard loss mechanism; solutions may be branches rather than points; and the boundary condition
is **integral and nonlocal**, which no published neural architecture appears to hard-constrain.
