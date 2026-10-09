# Покажчик усіх пунктів реєстрів

Згенеровано `guide/tools/make_index_md.py` з реєстрів проєкту: ID і статуси парсяться з самих файлів, короткий зміст — із покажчика путівника (`guide/tools/build_index.py`). Номер рядка веде у файл реєстру.

Усього **167** пунктів.

## Батьківський реєстр встановленого (E) — 40

| ID | Суть | Статус | Реєстр | Код |
|---|---|---|---|---|
| **E1** | Відображення $\psi\to$ скаляри не ліпшицеве в $L^2$ | встановлено; уточнено (→E36) | [ESTABLISHED.md:23](Our%20try/00-decisions/ESTABLISHED.md#L23) | [`conditioning.py`](Our%20try/03-deep-dives/D1-psi-to-scalars/conditioning.py) · `scalars_of` |
| **E2** | Показники $\alpha$ розпадаються на три кластери | встановлено; лише 3 розряди (→R10) | [ESTABLISHED.md:24](Our%20try/00-decisions/ESTABLISHED.md#L24) | [`conditioning.py`](Our%20try/03-deep-dives/D1-psi-to-scalars/conditioning.py) · `main` |
| **E3** | Хвіст PCA шкідливіший за білий шум тієї ж енергії | встановлено (виміряно) | [ESTABLISHED.md:25](Our%20try/00-decisions/ESTABLISHED.md#L25) | [`conditioning.py`](Our%20try/03-deep-dives/D1-psi-to-scalars/conditioning.py) · `pert_pca_tail` |
| **E4** | $L^2$-середнє двох гілок не є розв'язком | встановлено (виміряно) | [ESTABLISHED.md:26](Our%20try/00-decisions/ESTABLISHED.md#L26) | [`branch_mean.py`](Our%20try/03-deep-dives/D2-gs-nonuniqueness/branch_mean.py) · `rel_residual_bratu` |
| **E5** | Точка складки Братý $\lambda^*$ відтворена до останньої цифри | встановлено (виміряно) | [ESTABLISHED.md:27](Our%20try/00-decisions/ESTABLISHED.md#L27) | [`branch_mean.py`](Our%20try/03-deep-dives/D2-gs-nonuniqueness/branch_mean.py) · `lam_star` |
| **E6** | GS-нев'язка обчислювана лише з поданого $\psi$ | встановлено (виміряно) | [ESTABLISHED.md:28](Our%20try/00-decisions/ESTABLISHED.md#L28) | [`gs_residual_probe.py`](Our%20try/04-novelty/gs_residual_probe.py) · `gs_inconsistency` |
| **E7** | GS-член інваріантний до масштабу $\psi$ | встановлено (виміряно) | [ESTABLISHED.md:29](Our%20try/00-decisions/ESTABLISHED.md#L29) | [`gs_residual_probe.py`](Our%20try/04-novelty/gs_residual_probe.py) · `gs_inconsistency` |
| **E8** | GS-член сліпий до гладкої похибки й PCA-усічення (PCA-5: 0,0093) | встановлено; число уточнено | [ESTABLISHED.md:30](Our%20try/00-decisions/ESTABLISHED.md#L30) | [`pca5_check.py`](Our%20try/04-novelty/pca5_check.py) |
| **E9** | Скелет істини канонічний: 1 O-точка, 0 сідел | встановлено (виміряно) | [ESTABLISHED.md:31](Our%20try/00-decisions/ESTABLISHED.md#L31) | [`topology_probe.py`](Our%20try/04-novelty/topology_probe.py) · `critical_points` |
| **E10** | Скелет стійкий до 3 % шуму, руйнується на 10 % | встановлено; уточнено (N4) | [ESTABLISHED.md:32](Our%20try/00-decisions/ESTABLISHED.md#L32) | [`topology_probe.py`](Our%20try/04-novelty/topology_probe.py) · `critical_points` |
| **E11** | Сума індексів зберігається й у хаосі; інформативна лише кількість | встановлено; уточнено (N4) | [ESTABLISHED.md:33](Our%20try/00-decisions/ESTABLISHED.md#L33) | [`topology_probe.py`](Our%20try/04-novelty/topology_probe.py) · `_winding` |
| **E12** | На істині $\chi_{\mathrm{E}}(\psi\le t)\equiv1$ на всіх рівнях | встановлено (виміряно) | [ESTABLISHED.md:34](Our%20try/00-decisions/ESTABLISHED.md#L34) | [`topology_ec_curve.py`](Our%20try/04-novelty/topology_ec_curve.py) · `ec_curve` |
| **E13** | Крива Ейлера менш чутлива за лічильник критичних точок | встановлено; уточнено (N4) | [ESTABLISHED.md:35](Our%20try/00-decisions/ESTABLISHED.md#L35) | [`topology_ec_curve.py`](Our%20try/04-novelty/topology_ec_curve.py) · `ec_curve` |
| **E14** | Зміщення осі під шумом $\propto1/\sqrt{\lambda}$ (відтворено для $R$) | встановлено; відтворено 2026-09-30 | [ESTABLISHED.md:36](Our%20try/00-decisions/ESTABLISHED.md#L36) | [`local_models.py`](Our%20try/03-deep-dives/D1-psi-to-scalars/c1/local_models.py) · `part_a` |
| **E15** | Топологічний критерій машинно-незалежний (DIII-D і MAST) | встановлено (виміряно) | [ESTABLISHED.md:37](Our%20try/00-decisions/ESTABLISHED.md#L37) | [`h5_transfer_topology.py`](Our%20try/04-novelty/h5_transfer_topology.py) · `skeleton` |
| **E16** | Топологія переживає перенесення, значення $\psi$ — ні | встановлено; примітка ($k=50$) | [ESTABLISHED.md:38](Our%20try/00-decisions/ESTABLISHED.md#L38) | [`h5_transfer_topology.py`](Our%20try/04-novelty/h5_transfer_topology.py) · `skeleton` |
| **E17** | Харнес скорера цілий: perfect $\to1$, zeros $\to0$ | встановлено (виміряно) | [ESTABLISHED.md:39](Our%20try/00-decisions/ESTABLISHED.md#L39) | [`local_score.py`](fusion%20equilibrium%20challenge/starter/local_score.py) · `score_shot` |
| **E18** | В осесиметрії перетин Пуанкаре — рівно контури $\psi$ | встановлено (прочитано) | [ESTABLISHED.md:55](Our%20try/00-decisions/ESTABLISHED.md#L55) | [`fieldline.py`](tokamak-3d-viz/src/tokviz/fieldline.py) · `invariant_drift` |
| **E19** | Імпульс — тороїдальний потік, гамільтоніан — полоїдальний | встановлено (прочитано) | [ESTABLISHED.md:56](Our%20try/00-decisions/ESTABLISHED.md#L56) | [`perturbation.py`](tokamak-3d-viz/src/tokviz/perturbation.py) · `island_width_psin` |
| **E20** | Одна резонансна мода не дає хаосу | встановлено (прочитано) | [ESTABLISHED.md:57](Our%20try/00-decisions/ESTABLISHED.md#L57) | — |
| **E21** | Розв'язок GS може бути неєдиним на реальній геометрії | встановлено (прочитано) | [ESTABLISHED.md:58](Our%20try/00-decisions/ESTABLISHED.md#L58) | — |
| **E22** | Мала $L^2$-похибка не тягне малого фізичного резидуалу | встановлено (прочитано) | [ESTABLISHED.md:59](Our%20try/00-decisions/ESTABLISHED.md#L59) | — |
| **E23** | Жоден фузійний бенчмарк не оцінює резидуал PDE | встановлено (прочитано) | [ESTABLISHED.md:60](Our%20try/00-decisions/ESTABLISHED.md#L60) | — |
| **E24** | Стан Біна опуклий і однозначний; задача плазми — ні | встановлено (прочитано) | [ESTABLISHED.md:61](Our%20try/00-decisions/ESTABLISHED.md#L61) | — |
| **E25** | Прогноз зриву насичений і має закриті дані | встановлено (прочитано) | [ESTABLISHED.md:62](Our%20try/00-decisions/ESTABLISHED.md#L62) | — |
| **E26** | Корпус Sophelio: 103,86 ГБ, 9 121 розряд | встановлено (прочитано) | [ESTABLISHED.md:63](Our%20try/00-decisions/ESTABLISHED.md#L63) | — |
| **E27** | Consistency усереднює сім скалярів, не вісім | встановлено (прочитано) | [ESTABLISHED.md:64](Our%20try/00-decisions/ESTABLISHED.md#L64) | — |
| **E28** | Ліцензії HDB5, ConStellaration, Sophelio, FAIR-MAST | встановлено (прочитано) | [ESTABLISHED.md:65](Our%20try/00-decisions/ESTABLISHED.md#L65) | — |
| **E29** | Код starter kit справді MIT | встановлено (прочитано) | [ESTABLISHED.md:66](Our%20try/00-decisions/ESTABLISHED.md#L66) | — |
| **E30** | Tokamap реалізовано; він точно симплектичний | встановлено (виміряно) | [ESTABLISHED.md:40](Our%20try/00-decisions/ESTABLISHED.md#L40) | [`tokamap.py`](Our%20try/03-deep-dives/D3-poincare-inverse/tokamap.py) · `jacobian_det` |
| **E31** | Спостережуване на одній фазі непридатне; «540$\times$» порівнює різні стенди | встановлено; із застереженням | [ESTABLISHED.md:41](Our%20try/00-decisions/ESTABLISHED.md#L41) | [`identifiability.py`](Our%20try/03-deep-dives/D3-poincare-inverse/identifiability.py) · `profile` |
| **E32** | Відновлення однієї амплітуди тривіальне (SNR 57) | встановлено (виміряно) | [ESTABLISHED.md:42](Our%20try/00-decisions/ESTABLISHED.md#L42) | [`identifiability.py`](Our%20try/03-deep-dives/D3-poincare-inverse/identifiability.py) · `main` |
| **E33** | Функція Гріна GS-solver брала $k$ замість $m=k^2$ | виправлено | [ESTABLISHED.md:43](Our%20try/00-decisions/ESTABLISHED.md#L43) | `Tokamak-GS-solver/src/numerical/compute.py` · `GreenFunction` (оригінал, див. `Tokamak-GS-solver/README.md`) |
| **E34** | GS-solver не запускався з README; полагоджено частково | виправлено частково | [ESTABLISHED.md:44](Our%20try/00-decisions/ESTABLISHED.md#L44) | `Tokamak-GS-solver/src/numerical/grad_shafranov.py` · `GSsolverFreeBoundary` (оригінал, див. `Tokamak-GS-solver/README.md`) |
| **E35** | Корінь R8: поправка $T'$ бере первісну $h(T)$, а не $V'(T)$ | встановлено (виміряно) | [ESTABLISHED.md:45](Our%20try/00-decisions/ESTABLISHED.md#L45) | [`multimode_fixed_check.py`](Our%20try/03-deep-dives/D3-poincare-inverse/multimode_fixed_check.py) · `step_fixed` |
| **E36** | Похибка осі: нахил 1 при $\eta<\eta^*=\lambda h^2/2$, $\approx\tfrac12$ вище | встановлено (виміряно) | [ESTABLISHED.md:46](Our%20try/00-decisions/ESTABLISHED.md#L46) | [`local_models.py`](Our%20try/03-deep-dives/D1-psi-to-scalars/c1/local_models.py) · `part_a` |
| **E37** | UNet: $R^2_\psi=0,976$, але Consistency 0,179 < PCA+Ridge 0,270 | встановлено (виміряно) | [ESTABLISHED.md:47](Our%20try/00-decisions/ESTABLISHED.md#L47) | [`evaluate.py`](Our%20try/03-deep-dives/D1-psi-to-scalars/c2/evaluate.py) · `main` |
| **E38** | Мережі: $R^2_\psi\approx0,97$, але $g=0,81$–$0,96$ — не рівноваги | встановлено (виміряно) | [ESTABLISHED.md:48](Our%20try/00-decisions/ESTABLISHED.md#L48) | [`evaluate.py`](Our%20try/04-novelty/c4/evaluate.py) · `main` |
| **E39** | Вкладені поверхні без осесиметрії можливі (Landreman 2026; перевірено власним кодом) | встановлено (прочитано) | [ESTABLISHED.md:67](Our%20try/00-decisions/ESTABLISHED.md#L67) | [`verify_landreman.py`](Our%20try/99-bibliography/landreman2026/verify_landreman.py) · `check` |
| **E40** | Ландшафт нев'язки екскурсії «скляний»: базовий фіт застрягає біля істини | встановлено (виміряно) | [ESTABLISHED.md:49](Our%20try/00-decisions/ESTABLISHED.md#L49) | [`stand_tokamap.py`](Our%20try/03-deep-dives/D3-poincare-inverse/c5/stand_tokamap.py) · `observable` |

## Спростоване батьківського реєстру (R) — 13

| ID | Суть | Статус | Реєстр | Код |
|---|---|---|---|---|
| **R1** | «Ніхто не нав'язує інтегральну крайову умову жорстко» | спростовано | [ESTABLISHED.md:79](Our%20try/00-decisions/ESTABLISHED.md#L79) | — |
| **R2** | «Бін і вільна межа плазми — одна математика» | спростовано | [ESTABLISHED.md:80](Our%20try/00-decisions/ESTABLISHED.md#L80) | — |
| **R3** | «Симплектична нейромережа для силових ліній — наша ідея» | спростовано | [ESTABLISHED.md:81](Our%20try/00-decisions/ESTABLISHED.md#L81) | — |
| **R4** | «$\psi$ як канонічний імпульс»; повторилося в коді viz | спростовано; рецидив виправлено | [ESTABLISHED.md:82](Our%20try/00-decisions/ESTABLISHED.md#L82) | [`perturbation.py`](tokamak-3d-viz/src/tokviz/perturbation.py) · `island_width_psin` |
| **R5** | «Крива Ейлера краща за лічильник критичних точок» | спростовано | [ESTABLISHED.md:83](Our%20try/00-decisions/ESTABLISHED.md#L83) | [`topology_ec_curve.py`](Our%20try/04-novelty/topology_ec_curve.py) · `ec_curve` |
| **R6** | «$\alpha=\tfrac12$ — універсальний закон» (механізм відтворено) | спростовано; механізм відтворено (→E36) | [ESTABLISHED.md:84](Our%20try/00-decisions/ESTABLISHED.md#L84) | [`local_models.py`](Our%20try/03-deep-dives/D1-psi-to-scalars/c1/local_models.py) · `part_a` |
| **R7** | «Провал перенесення має топологічну складову» (H5) | спростовано | [ESTABLISHED.md:85](Our%20try/00-decisions/ESTABLISHED.md#L85) | [`h5_transfer_topology.py`](Our%20try/04-novelty/h5_transfer_topology.py) · `skeleton` |
| **R8** | «Tokamap переноситься на суму мод підстановкою» | спростовано; корінь знайдено (→E35) | [ESTABLISHED.md:86](Our%20try/00-decisions/ESTABLISHED.md#L86) | [`multimode.py`](Our%20try/03-deep-dives/D3-poincare-inverse/multimode.py) · `det_J` |
| **R9** | «Провали LCFS MAST — систематичний сигнал» (колишня C8) | спростовано (артефакт; ←C8) | [ESTABLISHED.md:87](Our%20try/00-decisions/ESTABLISHED.md#L87) | [`c8_sign_check.py`](Our%20try/04-novelty/c8_sign_check.py) |
| **R10** | «Три кластери $\alpha$ — три механізми» (колишня C1) | спростовано (←C1) | [ESTABLISHED.md:88](Our%20try/00-decisions/ESTABLISHED.md#L88) | [`real_sweep.py`](Our%20try/03-deep-dives/D1-psi-to-scalars/c1/real_sweep.py) · `run_machine` |
| **R11** | «Спектральна вага з $S(k)$ покращує моделі» (колишня C2) | спростовано (←C2) | [ESTABLISHED.md:89](Our%20try/00-decisions/ESTABLISHED.md#L89) | [`train.py`](Our%20try/03-deep-dives/D1-psi-to-scalars/c2/train.py) · `unet` |
| **R12** | «GS_score змінить упорядкування бейзлайнів» (половина C4) | спростовано (←половина C4) | [ESTABLISHED.md:90](Our%20try/00-decisions/ESTABLISHED.md#L90) | [`evaluate.py`](Our%20try/04-novelty/c4/evaluate.py) · `main` |
| **R13** | «Обернена задача Рівня 3 трактабельна за 48 год» (колишня C5) | спростовано | [ESTABLISHED.md:91](Our%20try/00-decisions/ESTABLISHED.md#L91) | [`fit_common.py`](Our%20try/03-deep-dives/D3-poincare-inverse/c5/fit_common.py) · `run_one` |

## Гіпотези (C) — 10

| ID | Суть | Статус | Реєстр | Код |
|---|---|---|---|---|
| **C1** | Три кластери $\alpha$ відповідають трьом механізмам (→R10) | закрито: спростовано (→R10) | [CONJECTURES.md:21](Our%20try/00-decisions/CONJECTURES.md#L21) | [`real_sweep.py`](Our%20try/03-deep-dives/D1-psi-to-scalars/c1/real_sweep.py) · `run_machine` |
| **C2** | Спектрально зважена втрата покращує реальні моделі (→R11) | закрито: спростовано (→R11) | [CONJECTURES.md:22](Our%20try/00-decisions/CONJECTURES.md#L22) | [`train.py`](Our%20try/03-deep-dives/D1-psi-to-scalars/c2/train.py) · `unet` |
| **C3** | Число обумовленості $1/\sqrt{\lambda}$ придатне для стратифікації тесту | активна; E14 відновлено, $\lambda$ широкий | [CONJECTURES.md:23](Our%20try/00-decisions/CONJECTURES.md#L23) | — |
| **C4** | Калібрування змінить упорядкування бейзлайнів (GS-половина →R12) | активна лише Calibration (GS →R12) | [CONJECTURES.md:24](Our%20try/00-decisions/CONJECTURES.md#L24) | [`evaluate.py`](Our%20try/04-novelty/c4/evaluate.py) · `main` |
| **C5** | Обернена задача Рівня 3 трактабельна за 48 год (→R13) | закрито: спростовано (→R13) | [CONJECTURES.md:25](Our%20try/00-decisions/CONJECTURES.md#L25) | [`run_inverse.py`](tokamak-3d-viz/src/run_inverse.py) · `observable` |
| **C6** | Одночасні конформні смуги — правильна форма UQ-члена | активна | [CONJECTURES.md:26](Our%20try/00-decisions/CONJECTURES.md#L26) | — |
| **C7** | Deep Ritz нестабільний на плазмі, стабільний на Біні | активна | [CONJECTURES.md:27](Our%20try/00-decisions/CONJECTURES.md#L27) | — |
| **C8** | Провали вилучення LCFS на MAST — систематичний сигнал (→R9) | закрито: спростовано (→R9) | [CONJECTURES.md:28](Our%20try/00-decisions/CONJECTURES.md#L28) | [`h5_transfer_topology.py`](Our%20try/04-novelty/h5_transfer_topology.py) · `skeleton` |
| **C9** | Коваріантне подання покращить перенесення між машинами | відкладено | [CONJECTURES.md:58](Our%20try/00-decisions/CONJECTURES.md#L58) | — |
| **C10** | Частина похибки моделей — успадковане зміщення EFIT | відкладено | [CONJECTURES.md:59](Our%20try/00-decisions/CONJECTURES.md#L59) | — |

## Відкриті питання (Q) — 13

| ID | Суть | Статус | Реєстр | Код |
|---|---|---|---|---|
| **Q1** | Чи реалістична реконструкція $\psi$ без магнітної діагностики | відкрите | [open-questions.md:12](Our%20try/00-decisions/open-questions.md#L12) | — |
| **Q2** | Наскільки виродження $p'$/$FF'$ псує реконструкцію | відкрите | [open-questions.md:13](Our%20try/00-decisions/open-questions.md#L13) | — |
| **Q3** | Чи трапляється неєдиність GS у робочих режимах | відкрите | [open-questions.md:14](Our%20try/00-decisions/open-questions.md#L14) | — |
| **Q4** | Реалістичні $\tau$ і $\Delta H$ для NbTi/Nb$_3$Sn | відкрите | [open-questions.md:15](Our%20try/00-decisions/open-questions.md#L15) | — |
| **Q5** | Наскільки камера екранує швидке $dB/dt$ | відкрите | [open-questions.md:16](Our%20try/00-decisions/open-questions.md#L16) | — |
| **Q6** | Доступ до кластера AI-лабораторії | відкрите | [open-questions.md:22](Our%20try/00-decisions/open-questions.md#L22) | — |
| **Q7** | Дати, місце, розмір команди, журі | відкрите | [open-questions.md:23](Our%20try/00-decisions/open-questions.md#L23) | — |
| **Q8** | Чи потрібна платформа зі скорингом | відкрите | [open-questions.md:24](Our%20try/00-decisions/open-questions.md#L24) | — |
| **Q9** | Хто реально будує starter kit і скільки годин | відкрите | [open-questions.md:25](Our%20try/00-decisions/open-questions.md#L25) | — |
| **Q10** | Умови поширення даних `fusionsimulator.io` | відкрите; не публікувати похідне | [open-questions.md:31](Our%20try/00-decisions/open-questions.md#L31) | — |
| **Q11** | Ліцензія ITPA HDB5 (CC BY 4.0) | закрите | [open-questions.md:32](Our%20try/00-decisions/open-questions.md#L32) | — |
| **Q12** | Атрибуція скорера й даних Sophelio | закрите | [open-questions.md:33](Our%20try/00-decisions/open-questions.md#L33) | — |
| **Q13** | Перевипуск MAST із CC BY-SA у CC BY — підстава? | відкрите; блокує перевипуск | [open-questions.md:34](Our%20try/00-decisions/open-questions.md#L34) | — |

## Рішення (D) — 9

| ID | Суть | Статус | Реєстр | Код |
|---|---|---|---|---|
| **D1** | Ядро — реконструкція $\psi$, а не прогноз зриву | прийнято | [DECISIONS.md:14](Our%20try/00-decisions/DECISIONS.md#L14) | — |
| **D2** | Наскрізний конвеєр — лише в наративі | прийнято | [DECISIONS.md:15](Our%20try/00-decisions/DECISIONS.md#L15) | — |
| **D3** | Два треки: базовий і дослідницький | прийнято | [DECISIONS.md:16](Our%20try/00-decisions/DECISIONS.md#L16) | — |
| **D4** | Дані — гібрид sim-to-real | прийнято | [DECISIONS.md:17](Our%20try/00-decisions/DECISIONS.md#L17) | — |
| **D5** | Синтетика з FreeGS/FreeGSNKE/TokaLab, без власного генератора | прийнято | [DECISIONS.md:18](Our%20try/00-decisions/DECISIONS.md#L18) | — |
| **D6** | Метрика $S'$: Sophelio + GS-резидуал + калібрування UQ | прийнято | [DECISIONS.md:19](Our%20try/00-decisions/DECISIONS.md#L19) | — |
| **D7** | Артефакти двомовні | прийнято | [DECISIONS.md:20](Our%20try/00-decisions/DECISIONS.md#L20) | — |
| **D8** | Надпровідність — лише як опуклий контроль (Бін) у D2 | прийнято; закрито | [DECISIONS.md:21](Our%20try/00-decisions/DECISIONS.md#L21) | — |
| **D9** | Квенч магніта — не оцінювана задача | прийнято | [DECISIONS.md:22](Our%20try/00-decisions/DECISIONS.md#L22) | — |

## Журнал верифікації (V-) — 17

| ID | Суть | Статус | Реєстр | Код |
|---|---|---|---|---|
| **V-A1** | Deep Ritz до рівняння GS не застосовували (пошук arXiv) | підтверджено (попередньо) | [verification-log.md:20](Our%20try/99-bibliography/verification-log.md#L20) | — |
| **V-A2** | Жорстка інтегральна умова вже є (McClenaghan 2024; →R1) | спростовано | [verification-log.md:48](Our%20try/99-bibliography/verification-log.md#L48) | — |
| **V-A3** | Патологію «середнього гілок» для GS не квантифіковано | підтверджено (попередньо) | [verification-log.md:68](Our%20try/99-bibliography/verification-log.md#L68) | — |
| **V-A4** | Жоден фузійний ML-бенчмарк не оцінює резидуал GS (→E23) | підтверджено із застереженнями | [verification-log.md:91](Our%20try/99-bibliography/verification-log.md#L91) | — |
| **V-D1** | Симплектична мережа для силових ліній — HénonNet (→R3) | спростовано | [verification-log.md:121](Our%20try/99-bibliography/verification-log.md#L121) | — |
| **V-D2** | Топологічна вісь: новизна застосування, не методу | частково існує | [verification-log.md:140](Our%20try/99-bibliography/verification-log.md#L140) | — |
| **V-D3** | Канонічні змінні: «$\psi$ як імпульс» хибне (→R4) | чернетку спростовано | [verification-log.md:163](Our%20try/99-bibliography/verification-log.md#L163) | — |
| **V-D4** | Власні виміри топології (→E9–E13, R5) | перевірено (власні виміри) | [verification-log.md:176](Our%20try/99-bibliography/verification-log.md#L176) | — |
| **V-B1** | Обсяг корпусу: 103,86 ГБ (→E26) | перевірено | [verification-log.md:194](Our%20try/99-bibliography/verification-log.md#L194) | — |
| **V-B2** | Кількість розрядів: 9 121 (→E26) | перевірено | [verification-log.md:195](Our%20try/99-bibliography/verification-log.md#L195) | — |
| **V-B3** | Consistency: сім скалярів (→E27) | перевірено | [verification-log.md:196](Our%20try/99-bibliography/verification-log.md#L196) | — |
| **V-B4** | Канонічний репозиторій starter kit — Sophelio | перевірено | [verification-log.md:197](Our%20try/99-bibliography/verification-log.md#L197) | — |
| **V-B5** | Топові бали недоступні без облікового запису | перевірено | [verification-log.md:198](Our%20try/99-bibliography/verification-log.md#L198) | — |
| **V-C1** | ITPA HDB5 — CC BY 4.0 (→Q11) | перевірено [V] | [verification-log.md:206](Our%20try/99-bibliography/verification-log.md#L206) | — |
| **V-C2** | ConStellaration — MIT | перевірено [V] | [verification-log.md:207](Our%20try/99-bibliography/verification-log.md#L207) | — |
| **V-C3** | Sophelio FEC — CC BY 4.0 | перевірено [V] | [verification-log.md:208](Our%20try/99-bibliography/verification-log.md#L208) | — |
| **V-C4** | FAIR-MAST — CC BY-SA 4.0 (→Q13) | перевірено [V] | [verification-log.md:209](Our%20try/99-bibliography/verification-log.md#L209) | — |

## Візуалізація: встановлене (VE) — 34

| ID | Суть | Статус | Реєстр | Код |
|---|---|---|---|---|
| **VE1** | Пошук осі збігається з EFIT до 0,0002 мм | встановлено (виміряно) | [findings.md:26](tokamak-3d-viz/findings.md#L26) | [`equilibrium.py`](tokamak-3d-viz/src/tokviz/equilibrium.py) · `find_axis` |
| **VE2** | LCFS набору даних — справді контур $\psi$ | встановлено (виміряно) | [findings.md:31](tokamak-3d-viz/findings.md#L31) | [`equilibrium.py`](tokamak-3d-viz/src/tokviz/equilibrium.py) · `set_boundary_from_contour` |
| **VE3** | $F$ з $q_{95}$ дає $B_\varphi=1,927$ Тл без знання $B_0$ | встановлено (виміряно) | [findings.md:35](tokamak-3d-viz/findings.md#L35) | [`equilibrium.py`](tokamak-3d-viz/src/tokviz/equilibrium.py) · `calibrate_F` |
| **VE4** | $q$ контурним інтегралом і трасуванням збігаються до 0,16 % | встановлено (виміряно) | [findings.md:41](tokamak-3d-viz/findings.md#L41) | [`fieldline.py`](tokamak-3d-viz/src/tokviz/fieldline.py) · `q_from_tracing` |
| **VE5** | Інтегратор зберігає $\psi$ до $10^{-7}$ за 200 обертів | встановлено (виміряно) | [findings.md:46](tokamak-3d-viz/findings.md#L46) | [`fieldline.py`](tokamak-3d-viz/src/tokviz/fieldline.py) · `invariant_drift` |
| **VE6** | Растеризація $\cos m\theta^*$ на 65$\times$65 дає фальшиві острови | встановлено (виміряно) | [findings.md:51](tokamak-3d-viz/findings.md#L51) | [`perturbation.py`](tokamak-3d-viz/src/tokviz/perturbation.py) · `Perturbation` |
| **VE7** | Аналітичні похідні збурення збігаються зі скінченними різницями | встановлено (виміряно) | [findings.md:60](tokamak-3d-viz/findings.md#L60) | [`test_physics.py`](tokamak-3d-viz/tests/test_physics.py) · `test_perturbation_derivatives_match_finite_difference` |
| **VE8** | Ланцюг $m/n$ дає рівно $m$ островів у розрізі | встановлено (виміряно) | [findings.md:63](tokamak-3d-viz/findings.md#L63) | [`islands.py`](tokamak-3d-viz/src/tokviz/islands.py) · `classify_fixed_points` |
| **VE9** | Чириков $S=0,458<1$ (було 0,714); поверхні виживають | встановлено (виміряно) | [findings.md:69](tokamak-3d-viz/findings.md#L69) | [`perturbation.py`](tokamak-3d-viz/src/tokviz/perturbation.py) · `chirikov` |
| **VE10** | Перший рендер Cycles на Metal: $\sim$100 с компіляції | встановлено (виміряно) | [findings.md:75](tokamak-3d-viz/findings.md#L75) | — |
| **VE11** | Продуктивність Cycles на M4: GPU лише $\sim$1,2$\times$ | встановлено (виміряно) | [findings.md:86](tokamak-3d-viz/findings.md#L86) | — |
| **VE11a** | Час рендера з тестової сцени занижено вдвічі | встановлено (виміряно) | [findings.md:79](tokamak-3d-viz/findings.md#L79) | — |
| **VE12** | Blender не має імпортера STEP/IGES | встановлено (прочитано) | [findings.md:250](tokamak-3d-viz/findings.md#L250) | — |
| **VE13** | Зміни API Blender 4.2→5.x, що ламають скрипти | встановлено (прочитано) | [findings.md:252](tokamak-3d-viz/findings.md#L252) | — |
| **VE14** | EEVEE: об'ємна емісія не освітлює поверхні | встановлено (прочитано) | [findings.md:257](tokamak-3d-viz/findings.md#L257) | — |
| **VE15** | ITER-2024: перша стінка — вольфрам замість берилію | встановлено (прочитано) | [findings.md:260](tokamak-3d-viz/findings.md#L260) | — |
| **VE16** | Число обертання трасуванням відтворює $1/q$ до 0,03 % | встановлено (виміряно) | [findings.md:90](tokamak-3d-viz/findings.md#L90) | [`analysis.py`](tokamak-3d-viz/src/tokviz/analysis.py) · `rotation_number` |
| **VE17** | Хаотичні орбіти групуються рівно на резонансах | встановлено (виміряно) | [findings.md:95](tokamak-3d-viz/findings.md#L95) | [`analysis.py`](tokamak-3d-viz/src/tokviz/analysis.py) · `classify` |
| **VE18** | Виправлена маятникова ширина збігається з виміряною до 0–5 % | встановлено (виміряно) | [findings.md:100](tokamak-3d-viz/findings.md#L100) | [`analysis.py`](tokamak-3d-viz/src/tokviz/analysis.py) · `measured_island_width` |
| **VE19** | Хаос існує задовго до $S=1$; перехід через 1 не перевірено | встановлено (виміряно) | [findings.md:136](tokamak-3d-viz/findings.md#L136) | [`analysis.py`](tokamak-3d-viz/src/tokviz/analysis.py) · `classify` |
| **VE20** | Згущення засіву завищує частку хаосу в 1,6–1,9 раза | встановлено (виміряно) | [findings.md:130](tokamak-3d-viz/findings.md#L130) | [`run_analysis.py`](tokamak-3d-viz/src/run_analysis.py) · `main` |
| **VE21** | $\Bvec=\nabla\times(\psi\nabla\varphi)$ бездивергентне для будь-якого числа мод | встановлено (виміряно) | [findings.md:151](tokamak-3d-viz/findings.md#L151) | [`fieldline.py`](tokamak-3d-viz/src/tokviz/fieldline.py) · `FieldLines` |
| **VE22** | Відновлення самих амплітуд тривіальне (обумовленість 1,2) | встановлено (виміряно) | [findings.md:159](tokamak-3d-viz/findings.md#L159) | [`run_inverse.py`](tokamak-3d-viz/src/run_inverse.py) · `observable` |
| **VE23** | Складність оберненої задачі задає спостережуване, а не фізика | встановлено (виміряно) | [findings.md:164](tokamak-3d-viz/findings.md#L164) | [`run_inverse.py`](tokamak-3d-viz/src/run_inverse.py) · `observable` |
| **VE24** | Перевага роздільності = фазова інформація + обсяг даних | встановлено (виміряно) | [findings.md:174](tokamak-3d-viz/findings.md#L174) | [`run_inverse.py`](tokamak-3d-viz/src/run_inverse.py) · `main` |
| **VE25** | Затиснута мода 5/2 гірша лише без фазової інформації | встановлено (виміряно) | [findings.md:181](tokamak-3d-viz/findings.md#L181) | [`run_inverse.py`](tokamak-3d-viz/src/run_inverse.py) · `main` |
| **VE26** | Паразитні гармоніки стенда не дають видимих островів | встановлено (виміряно) | [findings.md:187](tokamak-3d-viz/findings.md#L187) | [`run_parasitic.py`](tokamak-3d-viz/src/run_parasitic.py) · `main` |
| **VE27** | Біо–Савар перевірено проти аналітичної петлі | встановлено (виміряно) | [findings.md:197](tokamak-3d-viz/findings.md#L197) | [`rmp_coils.py`](tokamak-3d-viz/src/tokviz/rmp_coils.py) · `biot_savart` |
| **VE28** | Поле масиву $4\cdot10^{-23}$ Тл — через точку симетрії | встановлено (виміряно) | [findings.md:202](tokamak-3d-viz/findings.md#L202) | [`rmp_coils.py`](tokamak-3d-viz/src/tokviz/rmp_coils.py) · `icoil_array` |
| **VE29** | Інстанси Geometry Nodes не успадковують матеріали | встановлено (виміряно) | [findings.md:207](tokamak-3d-viz/findings.md#L207) | — |
| **VE30** | Слід на стінці зважений до краю (VC6) | встановлено (виміряно) | [findings.md:212](tokamak-3d-viz/findings.md#L212) | [`run_footprint.py`](tokamak-3d-viz/src/run_footprint.py) · `observable` |
| **VE31** | Обумовленість гіршає з числом мод плавно (VC7) | встановлено (виміряно) | [findings.md:220](tokamak-3d-viz/findings.md#L220) | [`run_mode_scan.py`](tokamak-3d-viz/src/run_mode_scan.py) · `main` |
| **VE32** | Гістограма ударів непридатна для скінченних різниць | встановлено (виміряно) | [findings.md:231](tokamak-3d-viz/findings.md#L231) | [`footprint.py`](tokamak-3d-viz/src/tokviz/footprint.py) · `connection_length_profile` |
| **VE33** | Немонотонність частки хаосу при 40 зернах — шум (VC5) | встановлено (виміряно) | [findings.md:236](tokamak-3d-viz/findings.md#L236) | [`run_analysis.py`](tokamak-3d-viz/src/run_analysis.py) · `main` |

## Візуалізація: спростоване (VR) — 16

| ID | Суть | Статус | Реєстр | Код |
|---|---|---|---|---|
| **VR1** | «Геометричного кута досить для розміщення островів» | спростовано | [findings.md:267](tokamak-3d-viz/findings.md#L267) | [`equilibrium.py`](tokamak-3d-viz/src/tokviz/equilibrium.py) · `theta_star_field` |
| **VR2** | «Паразитні гармоніки інтерполяції можна ігнорувати» | спростовано | [findings.md:271](tokamak-3d-viz/findings.md#L271) | [`perturbation.py`](tokamak-3d-viz/src/tokviz/perturbation.py) · `Perturbation` |
| **VR3** | «Об'ємну сітку плазми можна різати разом з машиною» | спростовано | [findings.md:275](tokamak-3d-viz/findings.md#L275) | — |
| **VR4** | «Камери можна кадрувати за $R_0$» | спростовано | [findings.md:279](tokamak-3d-viz/findings.md#L279) | — |
| **VR5** | «Спільна ціль TRACK_TO для всіх камер» | спростовано | [findings.md:283](tokamak-3d-viz/findings.md#L283) | — |
| **VR6** | «Можу навести замкнену форму кривої Princeton D» | спростовано | [findings.md:286](tokamak-3d-viz/findings.md#L286) | [`surfaces.py`](tokamak-3d-viz/src/tokviz/surfaces.py) · `d_shape` |
| **VR7** | «`angle2 == 0` означає зсув $0^\circ$» | спростовано | [findings.md:291](tokamak-3d-viz/findings.md#L291) | — |
| **VR8** | «Число обертання — просте середнє за вікном» | спростовано | [findings.md:295](tokamak-3d-viz/findings.md#L295) | [`analysis.py`](tokamak-3d-viz/src/tokviz/analysis.py) · `rotation_number` |
| **VR9** | «Острів розпізнається за екскурсією $\psi_N$» | спростовано | [findings.md:301](tokamak-3d-viz/findings.md#L301) | [`analysis.py`](tokamak-3d-viz/src/tokviz/analysis.py) · `classify` |
| **VR10** | «Захоплення $\nu$ — близькість до раціонального» | спростовано | [findings.md:307](tokamak-3d-viz/findings.md#L307) | [`analysis.py`](tokamak-3d-viz/src/tokviz/analysis.py) · `find_locked_bands` |
| **VR11** | «Класифікувати можна в будь-якому порядку» | спростовано | [findings.md:312](tokamak-3d-viz/findings.md#L312) | [`analysis.py`](tokamak-3d-viz/src/tokviz/analysis.py) · `classify` |
| **VR12** | «Дефіцит ширини острова — дискретність засіву» | спростовано | [findings.md:317](tokamak-3d-viz/findings.md#L317) | [`analysis.py`](tokamak-3d-viz/src/tokviz/analysis.py) · `measured_island_width` |
| **VR13** | «Взаємної узгодженості $\nu$ досить для захоплення» | спростовано | [findings.md:338](tokamak-3d-viz/findings.md#L338) | [`analysis.py`](tokamak-3d-viz/src/tokviz/analysis.py) · `find_locked_bands` |
| **VR14** | Обвідна $\psi_N^{m/2}$ не відтворює поле реальних котушок | спростовано | [findings.md:327](tokamak-3d-viz/findings.md#L327) | [`run_coil_compare.py`](tokamak-3d-viz/src/run_coil_compare.py) · `fit_exponent` |
| **VR15** | Ширина острова з $\psi$ як імпульсом: завищення в $\sqrt q$ | спростовано; код виправлено | [findings.md:345](tokamak-3d-viz/findings.md#L345) | [`perturbation.py`](tokamak-3d-viz/src/tokviz/perturbation.py) · `island_width_psin` |
| **VR16** | «Дефіцит ширини — стохастичний шар сепаратриси» | спростовано | [findings.md:361](tokamak-3d-viz/findings.md#L361) | [`test_physics.py`](tokamak-3d-viz/tests/test_physics.py) · `test_island_width_matches_traced_separatrix` |

## Візуалізація: гіпотези (VC) — 7

| ID | Суть | Статус | Реєстр | Код |
|---|---|---|---|---|
| **VC1** | Обвідна $\psi_N^{m/2}$ близька до поля RMP-котушок (→VR14) | закрито: спростовано | [findings.md:385](tokamak-3d-viz/findings.md#L385) | [`run_coil_compare.py`](tokamak-3d-viz/src/run_coil_compare.py) · `resonant_harmonic` |
| **VC2** | Паразитні 1,3 % не дають видимих островів (→VE26) | закрито: підтверджено | [findings.md:392](tokamak-3d-viz/findings.md#L392) | [`run_parasitic.py`](tokamak-3d-viz/src/run_parasitic.py) · `main` |
| **VC3** | Шість обраних ракурсів достатні для лекції | відкрито | [findings.md:410](tokamak-3d-viz/findings.md#L410) | — |
| **VC4** | Кадри читатимуться при проєкції у великій залі | відкрито | [findings.md:413](tokamak-3d-viz/findings.md#L413) | — |
| **VC5** | Немонотонність частки хаосу зникає з числом зерен (→VE33) | закрито: підтверджено | [findings.md:401](tokamak-3d-viz/findings.md#L401) | [`run_analysis.py`](tokamak-3d-viz/src/run_analysis.py) · `main` |
| **VC6** | Обернена задача на сліді важча за екскурсійну (→VE30) | закрито: підтверджено | [findings.md:372](tokamak-3d-viz/findings.md#L372) | [`run_footprint.py`](tokamak-3d-viz/src/run_footprint.py) · `main` |
| **VC7** | Понад три моди обумовленість помітно гіршає (→VE31) | закрито: підтверджено | [findings.md:373](tokamak-3d-viz/findings.md#L373) | [`run_mode_scan.py`](tokamak-3d-viz/src/run_mode_scan.py) · `main` |

## Візуалізація: відкрите (VQ) — 5

| ID | Суть | Статус | Реєстр | Код |
|---|---|---|---|---|
| **VQ1** | Кріостат моделі тороїдальний; в ITER — циліндр із куполом | відкрите | [findings.md:420](tokamak-3d-viz/findings.md#L420) | — |
| **VQ2** | Конфлікт CC BY і CC BY-SA при перевипуску (←Q13) | відкрите | [findings.md:423](tokamak-3d-viz/findings.md#L423) | — |
| **VQ3** | Чи додати виміряний перетин W7-X поруч із синтетичним | відкрите | [findings.md:426](tokamak-3d-viz/findings.md#L426) | — |
| **VQ4** | Чи надійний тест пласкості $\nu$ при новому засіві | відкрите | [findings.md:429](tokamak-3d-viz/findings.md#L429) | [`analysis.py`](tokamak-3d-viz/src/tokviz/analysis.py) · `find_locked_bands` |
| **VQ5** | U1: 21/30 хаотичних проти 10–13/30 у таблиці — неузгоджено | відкрите | [findings.md:443](tokamak-3d-viz/findings.md#L443) | [`run_coil_compare.py`](tokamak-3d-viz/src/run_coil_compare.py) |

## Бекенд котушок, пункти без номерів (U) — 3

| ID | Суть | Статус | Реєстр | Код |
|---|---|---|---|---|
| **U1** | «Хаос бекенда котушок спричиняє гребінка бічних смуг» | спростовано | [COIL-BACKEND.md:91](tokamak-3d-viz/docs/COIL-BACKEND.md#L91) | [`coilfield.py`](tokamak-3d-viz/src/tokviz/coilfield.py) · `CoilPerturbation` |
| **U2** | Хаос бекенда котушок: фізика чи похибка реконструкції поля? | відкрите | [COIL-BACKEND.md:95](tokamak-3d-viz/docs/COIL-BACKEND.md#L95) | [`coilfield.py`](tokamak-3d-viz/src/tokviz/coilfield.py) · `CoilPerturbation` |
| **U3** | Тримати $\vect A$, а не $\Bvec$: $\nabla\cdot\Bvec$ з $3,19\cdot10^{-2}$ до $2,50\cdot10^{-6}$ | встановлено (виміряно) | [COIL-BACKEND.md:35](tokamak-3d-viz/docs/COIL-BACKEND.md#L35) | [`coilfield.py`](tokamak-3d-viz/src/tokviz/coilfield.py) · `divergence_report` |

