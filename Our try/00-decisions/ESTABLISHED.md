# Реєстр встановленого / Established register

**Правило входу:** сюди потрапляє лише те, що **виміряно власним кодом** або **прочитано
у відкритому джерелі**. Не здогадка, не правдоподібність, не «так роблять усі».
Здогадки живуть у [CONJECTURES.md](CONJECTURES.md).

Останнє оновлення: **2026-09-29** (узгодження реєстрів, крок «Р»; було 2026-09-23).
Кожну правку цього оновлення записано в [CHANGELOG.md](CHANGELOG.md).
Доповнено **2026-09-29** (за згодою користувача): реєстрація результатів перевірок путівника —
E8 (число PCA-5), E16 (примітка k = 50), E31 (застереження), нова **E35**, примітка до R8, нова
**R9** (колишня C8); CHANGELOG №64–91.
Доповнено **2026-09-30** (перевірка C1, план затверджено користувачем): нова **E36**, нова **R10** (колишня C1),
обмеження до E1 і E2, E14 і R6 відтворено; CHANGELOG №95–100. Того ж дня (перевірка C2): нова **R11** (колишня C2), нова **E37**; CHANGELOG №103–107.
Доповнено **2026-10-01** (перевірка C4, половина GS_score): нова **R12**, нова **E38**; CHANGELOG №110–114.
Того ж дня (за згодою користувача): нова **E39** (розд. B, Landreman 2026, з власною чисельною перевіркою); CHANGELOG №115–120. Того ж дня (перевірка C5, план затверджено користувачем): нова **R13** (колишня C5), нова **E40**; CHANGELOG №124–130.

---

## A. Виміряно власним кодом

| # | Твердження | Число | Де |
|---|---|---|---|
| **E1** | Відображення ψ → похідні скаляри **не ліпшицеве в L²** | усі α ∈ [0,35; 0,70]; `R_axis` досягає 1σ при **R²ψ = 0,99993**; при R²ψ = 0,99 похибка осі 3,4 см. *Уточнення 2026-09-30 (перевірка C1, E36):* α < 1 виміряно для **білого (повузлового)** шуму; це режим стрибків дискретного максимуму на сітці скорера. Для гладкого збурення (\|k\| < 4) α ≈ 0,69–1,08 на всіх вибірках. На 28 розрядах DIII-D і на MAST для білого шуму всі α теж < 1 (0,34–0,83) | `03-deep-dives/D1-psi-to-scalars/` |
| **E2** | Показники α **розпадаються на три кластери** | 0,53–0,55 (положення екстремуму) / 0,35–0,42 (контур крізь сідло) / 0,70 (інтеграли). ⚠️ *Обмеження 2026-09-30 (перевірка C1 → R10):* розбиття стабільне **лише на 3 демо-розрядах** (96,7 % бутстрепу по розрядах); на 28 розрядах DIII-D — **0 %** (`li` 0,70 → 0,47), MAST — 3,3 %, синтетика Серфона–Фрайдберга — 0,3 %. Числа α для демо-розрядів відтворено до 4 знаків; кластери — властивість вибірки, а не механізмів (`03-deep-dives/D1-psi-to-scalars/c1/`) | те саме |
| **E3** | Хвіст PCA шкідливіший за білий шум тієї ж енергії | `Z_axis`: **64,4 σ** проти 6,3 σ при e = 0,30 | те саме |
| **E4** | L²-середнє двох гілок **не є розв'язком**, і L² цього не бачить | резидуал **1,22** проти **6,4·10⁻⁸** у гілки; опуклий контроль **2,5·10⁻¹³** | `03-deep-dives/D2-gs-nonuniqueness/` |
| **E5** | Реалізація D2 валідна | точка складки Братý **λ\* = 3,513830719** збігається з літературою до останньої цифри | те саме |
| **E6** | GS-резидуал **обчислюваний лише з поданого ψ** | істина **0,009** → +1% шуму **0,650** (×70); ×3% → 0,921 | `04-novelty/gs_residual_probe.py` |
| **E7** | GS-член **інваріантний до масштабу** | ψ×1,05 не змінює значення | те саме |
| **E8** | GS-член **сліпий** до гладкої похибки й низькорангового усічення | гаусове згладжування 0,010; PCA-5 **0,0093** — проти істини 0,009. *Було 0,016 (без скрипта; не відтворюється).* Уточнення 2026-09-29: число 0,016 не мало скрипта-джерела; відтворення тими самими функціями (PCA-базис скінченних кадрів розряду 203702, усічення до 5 компонент, медіана по 12 кадрах) дає **0,0093** (k = 10: 0,0096; k = 50: 0,0092). Висновок «сліпий» не змінюється — лише посилюється | те саме; `04-novelty/pca5_check.py` (копія `guide/figures/g04_pca5_check.py`) |
| **E9** | Топологічний скелет істини **точно канонічний** | рівно **1 еліптична точка, 0 сідел, сума індексів +1** | `04-novelty/topology_probe.py` |
| **E10** | Скелет стійкий до 3% шуму, руйнується на 10% | при R²ψ = 0,993 — **20,5 зайвих нерухомих точок**. ⚠️ *Уточнення 2026-09-29 (N4):* поріг 3 % — **пограничний**. `topology_probe.py` на 3 % дає медіану 1 O / 0 X (0 зайвих), а `topology_ec_curve.py` на **іншій реалізації шуму** — медіану лічильника **1,5** (спрацьовує). Обидва перезапущено 2026-09-29, числа відтворились. «Стійкий до 3 %» читати як «на межі стійкості» | те саме; `topology_ec_curve.py` |
| **E11** | **Сума індексів зберігається навіть у хаосі** — Пуанкаре–Хопф змушує пари O–X | сума ≈ +1 при 20 зайвих ⇒ **як метрика марна, працює лише кількість**. *Уточнення 2026-09-29 (N4):* медіана суми при 10 % = **1,5** (виміряно `topology_probe.py`), тобто «близька до +1», а не рівна +1; висновок «сума марна, працює кількість» не змінюється | те саме |
| **E12** | На істині **χ(ψ ≤ t) ≡ 1** на всіх рівнях — еталон без налаштування | кожна підрівнева множина є диском | `04-novelty/topology_ec_curve.py` |
| **E13** | χ-крива **менш чутлива** за лічильник | мовчить на 3%, де лічильник спрацьовує. За Морсом вона — його інтеграл. *Уточнення 2026-09-29 (N4):* порівняння зроблено в одному скрипті на одній реалізації шуму (лічильник — медіана 1,5). З E10 («на 3 % лічильник мовчить») це не суперечить по суті: 3 % — пограничний рівень, і спрацювання лічильника там залежить від реалізації шуму | те саме |
| **E14** | Зміщення осі під шумом **∝ 1/√λ** від кривизни | \|Δx\|·√λ стала в межах **±7%** на 16-кратному діапазоні λ. ~~⚠️ **Без відтворюваного скрипта (N5, 2026-09-29):** синтетичний тест 23.09 не збережено. Числа лишаються як записані, але не відтворювані, доки скрипт не відновлено~~ **Відтворено 2026-09-30:** \|ΔR\|·√λ при η = 10⁻² для λ = 1…16 — 0,0424…0,0474, тобто **±5,6 %**. Умова: η ≫ η\* = λh²/2 (режим стрибків, E36). Для Z за тих самих η закон не виконується (±19 %), бо ΔZ = 5 см і для великих λ режим стрибків ще не настав | `03-deep-dives/D1-psi-to-scalars/c1/e14_curvature.py` (було: синтетичний тест 23.09 — скрипта немає) |
| **E15** | **Топологічний критерій машинно-незалежний** | DIII-D і MAST: **обидві 1O/0X, сума +1**, попри протилежні конвенції знаку, різні сітки й різну геометрію | `04-novelty/h5_transfer_topology.py` |
| **E16** | **Топологія переживає перенесення там, де значення — ні** | MAST→базис DIII-D: R²ψ падає до 0,66, **скелет лишається 1O/0X**. ⚠️ *Примітка 2026-09-29 (k = 50; твердження для k = 5 не зачеплене):* у таблиці `h5_transfer_topology.py` рядок k = 50 (R²ψ 0,963) теж друкує «1O/0X», але це медіана **N_X = 0,5**, округлена форматом `.0f` до 0 (Python округлює 0,5 до парного). Покадрово: (1,0,+1) ×4, (1,1,0) ×3, (2,1,+1) ×1 — **4/8 кадрів мають X-точку всередині LCFS**. При k = 20 (рядок таблиці зі знаком −1, 6/8 вилучено) — 1 кадр (1,1,0), медіана 0. При k = 5 усі вилучені кадри 1O/0X (5/8 вилучено; зі знаком, поверненим у конвенцію MAST, — 8/8, див. R9). Отже, топологічні дефекти при перенесенні **можливі** на великих k; H5 варто повторити при порівнянному R²ψ | те саме; `04-novelty/h5_frames_check.py` (копія `guide/figures/g06_h5_frames.py` + покадровий MAST→DIII-D) |
| **E17** | Харнес скорера цілий | `perfect` → **1,000000**; `zeros` → **0,000000** | `local_score.py` |
| **E30** | **Tokamap реалізовано і він точно симплектичний** | max\|det J − 1\| = **5,6·10⁻⁸** при x_L = 1,0; Ψ ≥ 0 зберігається; перехід до хаосу монотонний | `03-deep-dives/D3-poincare-inverse/tokamap.py` |
| **E31** | **Спостережувана на одній початковій фазі непридатна** | шумова підлога **0,3244**; при усередненні по 24 фазах — **0,0006**, тобто **540× краще**. ⚠️ *Застереження 2026-09-29:* дві підлоги виміряно на **різних стендах** — 0,3244 на тримодовій мапі `multimode.py` (a = 0,3, одна фаза; мапа ще й не симплектична, R8), 0,0006 на одномодовій Tokamap (x_L = 0,6, 24 фази; перезапущено 2026-09-29: медіана 0,0006, макс. 0,0012). Висновок «одна фаза непридатна» стоїть (у `multimode.py` усі сигнали 0,38–0,43 лежать у шумі), але **відношення «540×» змішує ефект усереднення зі зміною стенда** і не є чистою мірою усереднення. Чиста міра: `N_PHASE = 1` в `identifiability.py` (не виміряно) | `.../identifiability.py`; `.../multimode.py` |
| **E32** | **Відновлення однієї амплітуди тривіальне** | зміна x_L на 3% → **SNR 57**; на 10% → SNR 171. Не задача для хакатону | те саме |
| **E33** | **Функція Гріна в `Tokamak-GS-solver` отримувала k замість параметра m = k²** (`ellipk/ellipe` у `src/numerical/compute.py` і `src/GSsolver/GradShafranov.py`). Знахідка N2 координатора. **Статус: виправлено 2026-09-29** | похибка старої G: **+9 %** біля нитки, **+42 %** при k² = 0,95 (3,355·10⁻⁷ проти 2,355·10⁻⁷ Вб/рад/А, звірено квадратурою), ×8 при k² = 0,5. Після виправлення відхилення від квадратури Біо–Савара ~10⁻¹⁵, симетрія G до 10⁻¹². Усі `result/PINN_*.png`, зроблені до 29.09, використовували хибну G | `Tokamak-GS-solver/CHANGES-2026-09-29.md` §2; `tests/test_green.py`; `manual/tools/COORD_NOTES.md` (розд. 3) |
| **E34** | **`Tokamak-GS-solver` не запускався з README**: `ModuleNotFoundError src.grad_shafranov`, `SyntaxError` у `src/numerical/grad_shafranov.py:98`, `GSMatrix` не визначено, `profiles.py` імпортував відсутній `_computes`, не було `environment.yaml`; PINN без даних і з примусовою CUDA. Знахідка N7 координатора. **Статус: виправлено 2026-09-29 (частково)** | приклад вільної межі збігається за **116** ітерацій Пікара (~0,55 с); `pytest` — 22 тести (за CHANGES; під час узгодження реєстрів не перезапускались). **Лишилось:** фіксована межа (`test_boundary.py`) падає; LCFS обмежена прямокутником сітки; PINN не навчено й не перевірено | `Tokamak-GS-solver/CHANGES-2026-09-29.md` §1, §3, §5; `tests/test_free_boundary_smoke.py`; `manual/tools/COORD_NOTES.md` (розд. 3) |
| **E35** | **Корінь R8 знайдено: багатомодове узагальнення Tokamap має брати первісну h(T), а не V′(T).** У твірній функції Tokamap рівняння для Ψ′ містить V(T) (через P = Ψ − 1 − V(T)), а поправка до T′ — її **первісну** h(T)/(1+Ψ′)² з h′ = V, тобто h(T) = −Σ a_k/(4π² m_k)·cos(2π m_k T + φ_k) (коефіцієнт **a_k/m_k**). `multimode.py` бере −V′(T)/(4π²) (коефіцієнт **a_k·m_k**) — різниця в **m_k²** для кожної моди; для одномодової m = 1 обидва збігаються, тому E30 не зачеплена. Додано 2026-09-29 (перевірка путівника, зареєстровано за згодою користувача). `multimode.py` **не змінено** — лишається доказом R8 | max\|det J − 1\| на 400 точках, три моди m = 2, 3, 4: a = 0 — 3,4·10⁻⁹ (обидві); a = 0,2 — `multimode.py` **0,50**, виправлена **9,5·10⁻⁸**; a = 0,5 — **1,69** проти **3,6·10⁻⁷** (перезапущено 2026-09-29) | `03-deep-dives/D3-poincare-inverse/multimode_fixed_check.py` (копія `guide/figures/g07_multimode_check.py`) |
| **E36** | **Механізм підсилення для положення осі: перехід 1 → ½ за η/η\*.** Скорер шукає вісь як локальний максимум на вузлах плюс параболічне уточнення. Для повузлового шуму амплітуди η нахил похибки дорівнює **1**, поки η < η\* = λh²/2 (уточнення лінійне), і **≈ 0,4–0,5**, коли η ≳ η\* (дискретний максимум стрибає на Δr ~ √(2η/λ)). Криві для різних λ лягають на одну в координатах η/η\*. Для гладкого збурення нахил 1 до геометричного насичення. Звідси: α ≈ ½ для осі в D1 — це режим стрибків, а не властивість екстремуму; R6 (α, що росте з λ) — наслідок фіксованого вікна η. Додано 2026-09-30 (перевірка C1) | параболоїд на сітці DIII-D, λ = 1…16, 300 спроб на точку: початок переходу в межах множника 2–3 від η\*; α у вікні η ∈ [10⁻⁵; 10⁻²] = 0,746 / 0,819 / 0,884 / 0,938 / 0,976 для λ = 1, 2, 4, 8, 16 | `03-deep-dives/D1-psi-to-scalars/c1/local_models.py` (частина A), рис. `c1/c1_m1_crossover.png` |
| **E37** | **Розчеплення R²ψ і Consistency на навченій моделі.** На тих самих 8 відкладених розрядах DIII-D (потокові 60–67) UNet_Lite відновлює ψ майже ідеально, але похідні скаляри в неї гірші, ніж у PCA+Ridge з майже марною картою потоку. Теза E1, тепер не на штучних збуреннях, а на реальній моделі. Додано 2026-09-30 (побічно з перевірки C2) | UNet_Lite (3 зерна): R²ψ = **0,976** (0,972–0,981), Consistency **0,179**, R²(`Z_axis`) = −1,92, R²(`kappa`) = −1,39; PCA+Ridge: R²ψ = **0,090**, Consistency **0,270**. Композитний S: 0,663 проти 0,194 | `03-deep-dives/D1-psi-to-scalars/c2/evaluate.py`, `c2/eval.json` |
| **E38** | **Карти потоку нейромереж не є рівновагами ГШ, хоча R²ψ у них 0,96–0,98.** Нев'язка найкращого підбору (p′, FF′) у моделей із виходом-картою (Simple MLP, Conv Decoder, UNet_Lite) більша, ніж в істини з 1 % білого шуму. Моделі, що прогнозують коефіцієнти PCA (лінійні, Random Forest), мають g, близьке до істини, — через гладкість низькорангового базису, а не фізику (R²ψ ≈ 0,09). Отже, GS_score антикорелює з R²ψ по моделях; як адитивний член він нагороджує гладкість (ризик E8 справдився). Додано 2026-10-01 (побічно з перевірки C4) | 8 тестових розрядів (потокові 60–67), 1 521 кадр; медіана g: істина **0,0097**, мережі **0,81–0,96**, MLP (sklearn) 0,54, лінійні/RF **0,017–0,034**; g(істина + 1 % шуму) = **0,634** (28 розрядів); Спірмен(GS_score, R²ψ) = **−0,60**, (GS_score, Consistency) = +0,43 | `04-novelty/c4/evaluate.py`, `c4/eval.json`, `c4/gref.json` |
| **E40** | **Ландшафт нев'язки екскурсійного спостережуваного «скляний»: базовий фіт застрягає навіть поруч з істиною.** Екскурсія ptp за скінченне число обертів — максимум по дискретних точках орбіти, тож малий зсув параметра змінює, яка точка дає екстремум, і нев'язка стрибає. Інформація при цьому є: хибних мінімумів на рівні шуму не знайдено (R13). Додано 2026-10-01 (post-hoc діагностика перевірки C5; на вердикт не впливала) | Стенд A (Tokamap, E35), 8 екземплярів: старт в істині — χ²/χ²_true = 0,99–1,00 (код коректний); старт з істини + 5 % / 10° — **41–100**; + 20 % / 45° — 111–779. Безшумовий 1-D розріз (екземпляр 0): по φ₁ у ±20° — **8** локальних мінімумів, по колу — **24**; по ln a₁ у ±0,2 — 3; зсув φ₁ на 0,5° — χ² 245 в один бік і 2,9 в інший (рівень шуму ≈ 160). Стенд B (`tokamak-3d-viz`), скан пар фаз 12 × 12 (крок 30°): 8 і 14 локальних мінімумів, усі віддалені з χ²_0 ≥ 2 184 ≫ N_obs = 132 | `03-deep-dives/D3-poincare-inverse/c5/diag_tokamap.py`, `c5/diag_tokamap.json`, `c5/results_viz.json` (scans) |

## B. Прочитано у відкритих джерелах

| # | Твердження | Джерело |
|---|---|---|
| **E18** | В осесиметрії система **інтегровна**, тож перетин Пуанкаре = **рівно контури ψ** | двоrядковий доказ `B·∇ψ_p = 0`; Escande & Momo 2024 |
| **E19** | Канонічний словник: час = φ, координата = θ, **імпульс = ТОРОЇДАЛЬНИЙ потік**, **гамільтоніан = ПОЛОЇДАЛЬНИЙ**. ⚠️ *2026-09-29:* помилка R4 **повторилася** в коді `tokamak-3d-viz` (формула ширини острова з ψ як імпульсом, завищення в √q) і виправлена — див. R4 нижче і viz VR15 | Escande & Momo 2024; Kallinikos та ін. 2023 |
| **E20** | **Одна резонансна мода не дає хаосу** — інтегровна в гелікоїдальній системі | Escande & Momo, Eq. 6 |
| **E21** | Розв'язок GS **може бути неєдиним** на реальній геометрії | Ham & Farrell 2024; Pentland та ін. 2025 (MAST-U, дефляційна континуація) |
| **E22** | Мала L²-похибка **не тягне** малий фізичний резидуал | Ding та ін., arXiv:2511.19114: 0,25% L² при великих резидуалах |
| **E23** | **Жоден фузійний бенчмарк не оцінює резидуал PDE** | FEC, TokaMark, PDEBench, The Well — перевірено поіменно |
| **E24** | Критичний стан Біна **опуклий і однозначно розв'язний**; задача плазми **неопукла й багатозначна** | Prigozhin 1996 Тм.2; Bartolucci та ін. 2021; Schaeffer 1977 |
| **E25** | Прогноз зриву насичений **і** має закриті дані | AUC 0,974; DisruptionBench архівовано без даних |
| **E26** | Обсяг корпусу Sophelio | **103,86 ГБ, 9 121 розряд** (виміряно через HF API) |
| **E27** | Consistency усереднює **сім** скалярів, не вісім | Codabench Evaluation + картка HF + `common.py` |
| **E28** | Ліцензії | HDB5 **CC BY 4.0**; ConStellaration **MIT**; Sophelio **CC BY 4.0**; FAIR-MAST **CC BY-SA 4.0** з поширенням через §4(b) |
| **E29** | Код starter kit — справді **MIT** | `NOASSERTION` на GitHub спричинений дописаним «NOTE ON SCOPE» |
| **E39** | **Інтегровність (точні вкладені поверхні) можлива і без осесиметрії.** Існують явні гладкі тороїдальні 3D-рівноваги МГД з ∇p ≠ 0 скрізь, крім осі, і з відхиленням від осесиметрії порядку одиниці: сімейство з ι ≡ 2 (усі лінії замкнені) і сімейство з широм ι. Обернене до E18 («немає симетрії ⇒ немає поверхонь») хибне. Перевірено **власним кодом**: усі рівняння статті виконуються до 10⁻¹⁰–10⁻¹², числа (β_V = 2/57, ι(0) = 2,28690, ι(δ) ≈ 2,2878) відтворено; помилок не знайдено. *Застереження (у статті не сказано):* у сімейства 1 саме поле B має точну неперервну **неізометричну** симетрію (еліптичне обертання, B = A·B₀(A⁻¹x) від Солов'ева), яку порушують |B|, густина струму й тиск (*до 2026-10-02 тут стояло «яку порушує лише тиск» — неточно: перевірка `99-bibliography/landreman2026/symmetry_recheck.py` показала, що |B|² і J теж не інваріантні*), — контрприклад до Греда слабкий; у сімейства 2 афінної симетрії немає, але шир у прикладі (3.27) лише +0,04 % (на рис. 2 — +9,3 / −1,4 / −2,6 %). Узгоджено з E19: ψ_t = ψ/2 при ϵ → 0 ⇒ ι = dψ_p/dψ_t = 2. Додано 2026-10-01 (за згодою користувача) | Landreman, arXiv:2609.26742 v2 (2026) `[V]`, `library/ARXIV_Landreman_2026_…pdf`; перевірка — `99-bibliography/landreman2026/` |

---

## C. Спростоване — тринадцять власних тверджень

*(До 2026-09-29 заголовок казав «сім», хоча в таблиці R1–R8 — вісім; «вісім» → «дев'ять» 2026-09-29 з додаванням R9; «дев'ять» → «десять» 2026-09-30 з додаванням R10; «десять» → «одинадцять» того ж дня з додаванням R11; «одинадцять» → «дванадцять» 2026-10-01 з додаванням R12; «дванадцять» → «тринадцять» того ж дня з додаванням R13.)*

**Не видаляються.** Помилка, яку прибрали, повертається через місяць.

| # | Твердження | Чим спростовано |
|---|---|---|
| **R1** | «Жодна нейроархітектура не нав'язує інтегральну крайову умову жорстко» | McClenaghan та ін., *Phys. Plasmas* 31, 082507 (2024) — робить це архітектурно |
| **R2** | «Критичний стан Біна і вільна межа плазми — одна математика» | Опукла проти неопуклої; активна множина проти лінії рівня; **нульове перехресне цитування за 50 років** |
| **R3** | «Симплектична нейромережа для відображень силових ліній — наша ідея» | Burby, Tang, Maulik, PPCF 63, 024001 (2021) — HénonNet |
| **R4** | «ψ як канонічний імпульс» | Імпульс — **тороїдальний** потік; ψ_p — **гамільтоніан**. ⚠️ **Повторилося 2026-09 (N1):** у `tokamak-3d-viz` `perturbation.island_width_psin` рахувала W = 4√(εq²/\|dq/dψ\|), тобто брала ψ за імпульс, і завищувала ширину острова в √q. **Виправлено 2026-09-29** на 4√(εq/\|dq/dψ_N\|): трасування дає 0,983 (2/1) і 0,996 (3/1) від нової формули; S за замовчуванням 0,714 → 0,458. Це повторення R4, не нове спростування батьківського реєстру. Деталі: `tokamak-3d-viz/findings.md` VR15; `tokamak-3d-viz/docs/CHANGES-2026-09-29-island-width.md` |
| **R5** | «χ-крива краща за лічильник критичних точок» | Менш чутлива; за Морсом це його інтегральна версія |
| **R6** | «α = ½ — універсальний закон для положення критичної точки» | Синтетика дає **0,687** (λ=1) і **0,841** (λ=4) — **залежить від λ**, тобто не степеневий закон. ~~⚠️ **Без відтворюваного скрипта (N5, 2026-09-29):** ця синтетика — той самий тест 23.09, що й E14, і його не збережено~~ **Механізм відтворено 2026-09-30** (`03-deep-dives/D1-psi-to-scalars/c1/e14_curvature.py`): α у фіксованому вікні η росте з λ — 0,746 / 0,884 для λ = 1 / 4 (вікно 10⁻⁵…10⁻²). Точні 0,687 / 0,841 не відтворюються, бо вікно тесту 23.09 невідоме. Причину пояснює E36 |
| **R7** | «Провал перенесення має топологічну складову» (H5) | **Топологія вціліває**: R²ψ падає до 0,66, скелет лишається 1O/0X. Інверсія корисніша за гіпотезу — див. E15, E16. *Примітка 2026-09-29:* при k = 50 на 4/8 кадрів MAST→DIII-D є X-точка (медіана N_X = 0,5 друкувалась як «0X») — див. примітку до E16; спростування для k = 5 (R²ψ 0,66) це не скасовує, але H5 на великих k варто перевірити при порівнянному R²ψ |
| **R8** | «Структуру Tokamap можна перенести на суму мод підстановкою V′(T)» | **Симплектичність ламається**: max\|det J − 1\| = **1,7** при a = 0,5. Твірна функція цього не витримує. *Корінь знайдено 2026-09-29 → **E35**:* поправка до T′ мусить брати первісну h(T) (коеф. a_k/m_k), а не V′(T) (a_k·m_k) — різниця m_k². З виправленням max\|det J − 1\| = 9,5·10⁻⁸ (a = 0,2) і 3,6·10⁻⁷ (a = 0,5). Спростоване твердження («підстановкою V′(T)») лишається спростованим; `multimode.py` не змінено |
| **R9** | «Провали вилучення LCFS MAST при перенесенні (3/8 при k = 5, 2/8 при k = 20) — систематичний сигнал» (**колишня C8**, перенесена 2026-09-29) | **Артефакт обробки знака.** Провали трапляються **лише** в рядках, де `project` обрав знак s = −1 (k = 5, 20); у рядках із s = +1 (k = 10, 50) — 0. `skeleton()` викликає `extract_lcfs` **без** `axis_sign`, тож для інвертованої реконструкції спрацьовує «first-success» fallback скорера (`lcfs.py`, docstring `extract_lcfs_with_sign`: fallback не best-fit і на неправильно орієнтованій мапі може не знайти контур). Якщо реконструкцію повернути в конвенцію MAST (помножити на s), провалів **0 з 8 при всіх k = 5, 10, 20, 50** для обох знаків. Перезапущено 2026-09-29. Нових даних (FAIR-MAST) для цього не треба. Доказ: `04-novelty/c8_sign_check.py` (копія `guide/figures/g06_c8_sign_check.py`); `04-novelty/h5_transfer_topology.py` (рядки k = 5 і 20: «3/8», «2/8 LCFS extraction failed», знак −1) |
| **R10** | «Три кластери α (E2) відповідають трьом механізмам: положення екстремуму / контур крізь сідло / інтеграл» (**колишня C1**, перенесена 2026-09-30) | **Кластери — властивість вибірки.** Критерії записано до запуску (`03-deep-dives/D1-psi-to-scalars/c1/THEORY.md`). Розбиття C1 у бутстрепі по розрядах (білий шум, α як у D1): 3 демо-розряди DIII-D — 96,7 %; **28 розрядів DIII-D — 0 %** (`li` 0,70 → 0,47, у ДІ осі); MAST — 3,3 % (`Z_axis` 0,71); 12 однонульових рівноваг Серфона–Фрайдберга через скорер — 0,3 %. Для гладкого збурення кластерів немає (α ≈ 0,69–1,08). Конкурентна гіпотеза «положення ≈ ½ проти значення/інтеграла» теж не пройшла (0–1,2 %). Що лишилось: механізм осі (E36). Доказ: `c1/real_sweep.py`, `c1/cf_xpoint.py`, `c1/results_c1.json` |
| **R11** | «Спектрально зважена втрата, виведена з S(k), покращує реальні моделі» (**колишня C2**, перенесена 2026-09-30) | Критерії записано до навчання (`03-deep-dives/D1-psi-to-scalars/c2/THEORY.md`). S(k) переміряно на 28 розрядах; у робочій точці (e_op = 0,245 → e = 0,3) ваги кілець спадають з k від 0,91 до 0,49. Розбиття `SCORE.md` (навчання 0–35, тест 60–67). PCA+Ridge: \|Δ\| < 10⁻⁴ (вага діє лише через базис PCA). UNet_Lite, 3 зерна: ΔConsistency(S(k) − flat) = −0,011 [−0,044; +0,012]; H¹ −0,017, перемішана −0,002, усі ДІ містять 0; розкид між зернами (0,114–0,231) більший за будь-який ефект ваги. Доказ: `c2/train.py`, `c2/evaluate.py`, `c2/eval.json` |
| **R12** | «GS_score змінить упорядкування бейзлайнів» (**половина C4**, перенесена 2026-10-01; частина про Calibration лишається в C4) | Критерії записано до запуску (`04-novelty/c4/THEORY.md`): стійка перестановка = ≥ 95 % бутстреп-повторень. При головній вазі w_r = 0,167 точковий порядок змінюється (τ Кендалла = 0,857; MLP (sklearn) над UNet_Lite і Conv Decoder), але лише в 75 % і 74 % повторень; при w_r = 0,30 — 94 % і 84 %. Групи «лінійні» (S ≈ 0,19) і «мережі» (S ≈ 0,65) розділені надто сильно, щоб GS_score з вагою ≤ 0,30 їх переставив. Доказ: `c4/evaluate.py`, `c4/eval.json` |
| **R13** | «Обернена задача Рівня 3 (спектр (m,n)) трактабельна за 48 год зі starter kit» (**колишня C5**, перенесена 2026-10-01) | Критерії записано до запуску (`03-deep-dives/D3-poincare-inverse/c5/THEORY.md`): базовий багатостартовий нелінійний МНК, 3 моди, амплітуди + фази, фазово-роздільна екскурсія, 3 % шуму; успіх = усі амплітуди в межах 10 %, усі фази в межах 20°. Стенд A (виправлений Tokamap, E35): **0/40** успіхів (16 стартів); стенд B (`tokamak-3d-viz`, DIII-D 203702, 2/1–5/2–3/1): **0/3** (6 стартів). Хибних мінімумів 0 на обох; технічних збоїв 0 % і 5,6 %. Жоден старт не дійшов до рівня шуму: найкращий χ²/χ²_true — 2,6…193 (медіана 110) на A, 303–503 на B; медіанна макс. похибка 32 % / 77° (A), 54 % / 144° (B). При 10 % шуму (A, для звіту): 2/40. CPU на екземпляр B — 5,3–7,3 год. Перешкода — оптимізація на негладкому ландшафті (E40), а не ідентифіковність. Спростовано для базового розв'язувача, а не «нерозв'язно» (`THEORY.md` §7). Що лишилось: VE21–VE25, VE30, VE31 як вимірювання **локальної січної** обумовленості — не зачеплені, але й не описують ландшафт. Доказ: `c5/run_tokamap.py`, `c5/run_viz.py`, `c5/results_*.json` |

**Співвідношення на 23.09.2026: вісім спростовано з приблизно чотирнадцяти висунутих.**
**На 29.09.2026 (після додавання R9): дев'ять спростовано.** R9 — колишня активна гіпотеза C8, тож
знаменник «~14» не перераховувався (він і раніше був наближеним).
**На 30.09.2026 (після додавання R10): десять спростовано.** R10 — колишня активна гіпотеза C1.
**Того ж дня (після додавання R11): одинадцять.** R11 — колишня активна гіпотеза C2.
**На 01.10.2026 (після додавання R12): дванадцять.** R12 — половина C4 (GS_score); частина C4 про Calibration лишається активною.
**Того ж дня (після додавання R13): тринадцять.** R13 — колишня активна гіпотеза C5.
Повторення R4 у коді візуалізації (N1) рахується як рецидив R4, а не як окреме спростування
(до 2026-09-29 тут стояло «не як дев'яте»; дев'ятим тепер є R9 — колишня C8, інше твердження); власні спростування візуалізації ведуться окремо (VR1–VR16 у
`tokamak-3d-viz/findings.md`).

---

## English summary

**Entry rule:** only what was **measured by our own code** or **read in an open source** enters here.
Guesses live in `CONJECTURES.md`.

**Measured (27 items: E1–E17, E30–E38, E40; "26" until E40 was added later on 2026-10-01; "25" until E38 was added on 2026-10-01; "24" until E37 was added later on 2026-09-30; this line said "17" until 2026-09-29, then "22" until E35 was added the same day, then "23" until E36 was added on 2026-09-30):** the ψ → derived-scalar map is **not Lipschitz in L²** (`R_axis` hits a 1σ
error at R²ψ = 0.99993); the amplification exponents fall into **three clusters**; the PCA tail is
**10× more damaging** than white noise of equal energy; an L2 branch-mean carries residual **1.22**
against **6.4 × 10⁻⁸** for a true branch while a convex control stays at 2.5 × 10⁻¹³; the
Grad–Shafranov residual is **computable from ψ alone** (0.009 truth vs 0.650 at 1% noise,
scale-invariant, but blind to smoothing and low-rank truncation); the topological skeleton of truth
is **exactly one elliptic point, zero saddles, index sum +1**, robust to 3% noise and producing
**20 spurious fixed points** at 10%; the **index sum is conserved** so only the count is informative;
χ(ψ ≤ t) ≡ 1 on truth; axis displacement scales as **1/√λ** to ±7%; and — newly — the **topological
criterion is machine-independent** (DIII-D and MAST both give 1O/0X/+1 despite opposite sign
conventions and different geometry) and **topology survives cross-machine projection where values
do not**. The Tokamap is exactly symplectic, a single-phase observable is useless (540× noise-floor
gain from phase averaging — the two floors come from different stands, see the caveats below), and single-amplitude recovery is trivial (E30–E32).

**Added 2026-09-29:** the `Tokamak-GS-solver` Green's function passed k instead of m = k² to the
elliptic integrals (+42 % at k² = 0.95, ×8 at k² = 0.5) — **fixed 2026-09-29**, checked against
quadrature (E33); the solver did not run from its README — **fixed 2026-09-29, partially**
(free-boundary example converges; fixed-boundary path and PINN still not working) (E34).
**Caveats added 2026-09-29:** the 3 % noise threshold of E10/E13 is borderline (median count 1 in
one script, 1.5 in the other, different noise realisations); the median index sum at 10 % noise
is 1.5, "close to +1" (E11); E14 (and R6) have **no reproducible script** — the 23.09 synthetic
test was not saved (**restored 2026-09-30**: ±5.6 % for R over a 16× range of λ; the R6 mechanism reproduced).
**Guide checks registered 2026-09-29 (with user approval):** E8's PCA-5 figure 0.016 had no generating
script; a reconstruction with the same functions gives **0.0093** (was 0.016, no script) — the
"blind" verdict only strengthens. E16's note: the k = 50 row of the H5 table prints "1O/0X" because
the median N_X = 0.5 is rounded by `.0f`; 4 of 8 frames carry an interior X-point (the k = 5 claim is
unaffected). E31's "540×" compares two different stands (0.3244 from the broken three-mode
`multimode.py`, 0.0006 from the single-mode Tokamap with 24 phases); the averaging conclusion stands,
the ratio is not a clean measure of it. **New E35:** the root cause of R8 — the multi-mode Tokamap
must use the antiderivative h(T) (coefficient a_k/m_k), not V′(T) (a_k·m_k), a factor m_k² apart;
with the fix max|det J − 1| = 9.5 × 10⁻⁸ (a = 0.2) and 3.6 × 10⁻⁷ (a = 0.5).

**Read (13 items; "12" until E39 was added on 2026-10-01):** axisymmetry makes the Poincaré section exactly the ψ contours; the canonical
dictionary has **toroidal** flux as momentum and **poloidal** flux as Hamiltonian; a single resonant
mode gives islands but no chaos; GS solutions can be non-unique on real geometry; small L² error
does not imply small residual; **no fusion benchmark scores a PDE residual**; Bean is convex and
unique while the plasma problem is non-convex and multivalued; plus dataset sizes and licences.
**E39 (added 2026-10-01):** exact nested flux surfaces do **not** require axisymmetry — Landreman
(arXiv:2609.26742) gives explicit smooth toroidal 3D MHD equilibria (ι ≡ 2, and a sheared-ι family);
our own code reproduces every equation to 10⁻¹⁰–10⁻¹² and finds no errors. Caveats: the ι = 2 family
is a linear image of a Solov'ev field, so B keeps an exact non-isometric symmetry, broken by |B|, J and p ("only by p" until 2026-10-02 — inaccurate);
the sheared example (3.27) has only +0.04 % shear. Consistent with E19 (ι = dψ_p/dψ_t = 2).

**Refuted — thirteen of our own claims** (R1–R13; "twelve" until R13 was added later on 2026-10-01; "eleven" until R12 was added on 2026-10-01; "ten" until R11 was added later on 2026-09-30; this line said "seven" until 2026-09-29, then "eight" until R9 was added the same day, then "nine" until R10 was added on 2026-09-30), kept rather
than deleted: the Green's-function hard-constraint claim; "Bean and plasma are the same mathematics";
symplectic neural maps as novel; ψ as canonical momentum; the χ-curve as superior to the counter;
α = ½ as a universal law; H5, the topological component of cross-machine collapse — where **the
inversion proved more useful than the hypothesis**; and the claim that the Tokamap structure carries
over to a sum of modes by substituting V′(T) (symplecticity breaks, max|det J − 1| = 1.7; root cause
found 2026-09-29, see E35); and — **R9, formerly conjecture C8** — that the MAST LCFS-extraction
failures under cross-machine projection are a systematic signal: they occur only in rows where the
sign −1 was chosen, because `skeleton()` calls `extract_lcfs` without `axis_sign` and falls into the
scorer's first-success fallback; with the sign restored there are 0 failures at every k.
**R4 recurred** in the visualisation code (island width computed with ψ as the momentum, √q too
large) and was fixed on 2026-09-29 (viz register VR15).

**Added 2026-09-30 (C1 test, plan approved by the user).** **R10, formerly conjecture C1:** the three α clusters of E2 correspond to three mechanisms — refuted against criteria fixed before the run: the partition holds on the 3 demo shots (96.7 % of shot-bootstrap resamples) but on 0 % over 28 DIII-D shots, 3.3 % on MAST and 0.3 % on 12 single-null Cerfon–Freidberg equilibria; smooth perturbations show no clusters (α ≈ 0.69–1.08). E2 now carries that restriction, and E1 a note that α < 1 is measured for per-node noise. **E36:** the axis error has slope 1 below η* = λh²/2 and ≈ 0.4–0.5 above it (the discrete maximum jumps); curves collapse on η/η*; this also explains R6. E14 is reproducible again (±5.6 % for R).

**Added later on 2026-09-30 (C2 test).** **R11, formerly conjecture C2:** a spectrally weighted loss derived from S(k) improves real models — refuted against criteria fixed before training: PCA+Ridge is insensitive (|Δ| < 10⁻⁴), UNet_Lite gets slightly worse under every weighting (ΔConsistency −0.002…−0.017, all 95 % CIs contain 0), S(k) beats neither control. **E37:** on the same 8 held-out shots UNet_Lite has R²ψ = 0.976 but Consistency 0.179, below PCA+Ridge (R²ψ = 0.090, Consistency 0.270).

**Added 2026-10-01 (C4 test, GS_score half).** **R12:** GS_score changes the ranking of the baselines — refuted against criteria fixed in advance: at the design weight the point ranking moves (τ = 0.857) but no swap holds in ≥ 95 % of shot-bootstrap replicates (74–75 %; 94 % at w_r = 0.30). **E38:** map-output networks (Simple MLP, Conv Decoder, UNet_Lite) reach R²ψ = 0.96–0.98 yet GS-inconsistency 0.81–0.96, worse than truth + 1 % noise (0.634); PCA-coefficient models reach 0.017–0.034 thanks to smoothness (R²ψ ≈ 0.09); GS_score anti-correlates with R²ψ (Spearman −0.60).

**Added later on 2026-10-01 (C5 test, plan approved by the user).** **R13, formerly conjecture C5:** the Level-3 spectrum inverse problem is tractable for a baseline solver — refuted against criteria fixed in advance: multi-start bounded least squares recovering 3 amplitudes + 3 phases from the phase-resolved excursion at 3 % noise succeeded on 0/40 Tokamap instances and 0/3 `tokamak-3d-viz` instances; no spurious minima, but no start reached the noise level (best χ²/χ²_true median 110 on A, 303–503 on B). **E40:** the excursion misfit landscape is glassy — from the truth the fit stays put, from 5 %/10° away it stalls at 41–100 × χ²_true; 8 local minima within ±20° of one phase, 24 around the circle.
