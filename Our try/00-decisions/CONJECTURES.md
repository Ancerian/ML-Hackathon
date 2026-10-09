# Реєстр здогадок / Conjecture register

**Правило входу:** усе, що **ще не перевірено**. Кожен запис має: що саме стверджується,
**що його спростує**, і **скільки коштує перевірка**.

**Правило виходу:** перевірено → переїжджає в [ESTABLISHED.md](ESTABLISHED.md),
у розділ A/B або в C (спростоване). **Нічого не зникає мовчки.**

**Правило цитування:** запис звідси **не можна** використовувати в `05-concept/`
чи в матеріалах для учасників без позначки «гіпотеза».

Останнє оновлення: **2026-10-01** (C5 закрито → R13; перед тим того ж дня половину C4 — GS_score — спростовано → R12); до того **2026-09-30** (C1 закрито → R10, C2 → R11); до того **2026-09-29** (узгодження реєстрів; було 2026-09-23; правки — у
[CHANGELOG.md](CHANGELOG.md)). Історія: висунуто ~14 гіпотез, **13 спростовано** (було 8; 2026-09-29 додано R9 — колишня C8; 2026-09-30 — R10, колишня C1, і R11, колишня C2; 2026-10-01 — R12, половина C4, і R13, колишня C5).

---

## Активні здогадки

| # | Здогадка | Що її спростує | Вартість перевірки |
|---|---|---|---|
| **C1** | ~~Три кластери α (E2) відповідають механізмам …~~ → **закрито 2026-09-30:** спростовано, тепер **R10** в [ESTABLISHED.md](ESTABLISHED.md); повний запис — у розділі «Закриті здогадки» нижче | — | — |
| **C2** | ~~Спектрально зважена втрата, виведена з кривої S(k), покращує реальні моделі~~ → **закрито 2026-09-30:** спростовано, тепер **R11** в [ESTABLISHED.md](ESTABLISHED.md); повний запис — у розділі «Закриті здогадки» нижче | — | — |
| **C3** | Число обумовленості з E14 (1/√λ у O-точці) **корисне для стратифікації тесту** | Розподіл λ по корпусу виявиться вузьким ⇒ стратифікувати нема за чим | ~1 год: порахувати λ_min по всіх кадрах, подивитися розкид. ⚠️ *2026-09-29 (N5):* сама E14 не має відтворюваного скрипта — спершу відновити тест 1/√λ <br>**2026-09-30 (побічно з перевірки C1):** тест E14 відновлено (`03-deep-dives/D1-psi-to-scalars/c1/e14_curvature.py`). λ на осі на 28 розрядах DIII-D: 0,13–3,6 (міжквартильний розмах 0,73–1,58) — **широкий**, тож головний спростовувач C3 не спрацював. Але α осі між терцилями λ змінюється слабко й непослідовно (DIII-D 0,47 → 0,54 → 0,54; MAST 0,52 → 0,54 → 0,55; демо 0,59 → 0,53 → 0,51). Вердикту немає |
| **C4** | Додавання `GS_score` і `Calibration` **змінить упорядкування** семи наявних бейзлайнів | Упорядкування збігається з тим, що дає `S` ⇒ нові члени нічого не додають | ~півдня: реалізувати `S′`, прогнати `experiments.py`. **Частковий результат (N6, 2026-09; не вердикт):** `manual/code/ch11_gs_residual_baseline.py` на 36 кадрах трьох демо-розрядів DIII-D дає медіани g(EFIT) = **0,0103**, g(прогноз PCA+Ridge) = **0,0174**, g(PCA-50 від EFIT) = **0,0170**, хоча R² прогнозу лише 0,877, а 1 − R²(PCA-50) = 2,26·10⁻⁷. GS-нев'язка відрізняє обидва від EFIT, але **ледь розрізняє** слабкий прогноз і майже точну L²-проєкцію (узгоджено з E8). Це ризик для C4; упорядкування семи бейзлайнів ще не прогнано. Демо-розряди, можливо, входять у навчальні<br>**Попередня реєстрація 2026-10-01 (перевірка запущена за згодою користувача):** перевіряється лише половина — **GS_score** (Calibration потребує смуг невизначеності; разом із C6). GS_score кадру = clip(1 − (g_pred − g_true)/(g_ref − g_true), 0, 1), g_ref = медіана g(істина + 1 % шуму) на 28 розрядах; ваги metric-design без Calibration + скан w_r 0,05–0,30; 7 бейзлайнів `experiments.py` (+ PCA+Ridge) на розбитті C2, мережі — 3 зерна; критерій — стійка перестановка пари (≥ 95 % бутстреп-повторень по розрядах). Деталі — `04-novelty/c4/THEORY.md` (не редагується після запуску)<br>**Результат 2026-10-01: половину C4 (GS_score) спростовано → R12.** При головній вазі точковий порядок змінюється (τ Кендалла 0,857: MLP (sklearn) над UNet_Lite і Conv Decoder), але жодна перестановка не стійка (74–75 % бутстреп-повторень; при w_r = 0,30 — 94 %). Побічно — E38 (карти мереж не є рівновагами). **C4 лишається активною лише в частині Calibration** («додавання Calibration змінить упорядкування») — перевіряти разом із C6 |
| **C5** | ~~Обернена задача Рівня 3 (спектр (m,n)) трактабельна за 48 год зі starter kit~~ → **закрито 2026-10-01:** спростовано, тепер **R13** в [ESTABLISHED.md](ESTABLISHED.md); повний запис — у розділі «Закриті здогадки» нижче | — | — |
| **C6** | Одночасні конформні смуги на полі 65×65 — **правильна форма** UQ-члена | Емпіричне покриття виявиться недосяжним без абсурдної ширини | ~день |
| **C7** | Deep Ritz на задачі плазми буде **seed-нестабільним** (обиратиме різні гілки), а на Біні — стабільним | Обидва стабільні ⇒ неопуклість не проявляється на доступних масштабах | ~день, потрібні обидва солвери |
| **C8** | ~~Вилучення LCFS … систематичний сигнал~~ → **переміщено 2026-09-29:** спростовано як артефакт, тепер **R9** в [ESTABLISHED.md](ESTABLISHED.md); повний запис — у розділі «Закриті здогадки» нижче | — | — |

### Що означає «Рівень 3» (означення додано 2026-09-29)

У корпусі (`03-deep-dives/D3-poincare-inverse/findings.md`, `multimode.py`, C5) «Рівень 3»
вживається без означення, а Рівні 1 і 2 **ніде не визначені**. Тому означення нижче —
**ретроспективне**, відновлене з контексту, а не цитата з раніших документів:

> **Рівень 3** — обернена задача на даних перетину Пуанкаре: за спостережуваною структурою
> силових ліній (проколи/екскурсії або слід на диверторі) відновити **спектр збурення (m, n)** —
> амплітуди й фази мод. Це «кандидат у дослідницький трек» з
> `02-field-map/F-poincare-and-topology.md` §6 і напрям, обраний основним у пілоті D3.

Про Рівні 1–2 нічого не стверджується; якщо вони потрібні путівнику, їх треба визначити окремо.

## Закриті здогадки (лишаються для історії)

Правило виходу виконано: запис переїхав у [ESTABLISHED.md](ESTABLISHED.md), тут — лише слід.

| # | Здогадка | Що її спростує | Вартість перевірки |
|---|---|---|---|
| ~~**C5**~~ → **R13** | **ЗАКРИТО 2026-10-01 — спростовано → R13** ([ESTABLISHED.md](ESTABLISHED.md), розд. C; доказ `03-deep-dives/D3-poincare-inverse/c5/`, критерії — `c5/THEORY.md`, записані до запуску). Базовий багатостартовий МНК: **0/40** успіхів на виправленому Tokamap, **0/3** на стенді `tokamak-3d-viz`; хибних мінімумів немає, але жоден старт не дійшов до рівня шуму (χ²/χ²_true: медіана 110 на A, 303–503 на B). Причина — скляний ландшафт екскурсії (E40). Початкове твердження (закреслено): ~~Обернена задача Рівня 3 (спектр (m,n)) **тракта́бельна за 48 год** зі starter kit~~ | *До 2026-10-01 спростовувача не було («—»).* Заданий у попередній реєстрації: на будь-якому стенді < 50 % успіхів або > 25 % екземплярів із хибним мінімумом | ⚠️ **ПІЛОТ ВИКОНАНО 23.09, гіпотеза НЕ закрита.** Одномодова версія **занадто легка** (SNR 57 при 3%); багатомодова **непротестовна**, бо наше узагальнення мапи **не симплектичне** (R8). **Блокер: потрібен валідний багатомодовий симплектичний стенд — рекомендовано `pyoculus`.** Деталі: `03-deep-dives/D3-poincare-inverse/findings.md`.<br>**Оновлення 2026-09-29 (результати `tokamak-3d-viz`, 24–28.09): блокер знято, гіпотезу НЕ закрито — предмет змінився.** VE21: стенд без відображення, `B = ∇×(ψ_total ∇φ)` — ∇·B = 0 **точно** для будь-якого числа мод (перевірено з трьома одночасними модами 2/1, 5/2, 3/1; інтегратор DOP853 не симплектичний, дрейф 10⁻⁷ за 200 обертів). VE22: самі амплітуди (дві рознесені моди) — обумовленість **1,2**, тривіально. VE23: три упаковані моди з амплітудами **і фазами** — обумовленість **17,9** (фазово-усереднене спостережуване) проти **5,8** (фазово-роздільне); помилка фаз при 10 % шуму 94,6 % проти 14,2 %. VE24: перевага роздільності = фазова інформація (2,2× обумовленості) + обсяг даних (3,7× точності фаз). VE25: затиснута мода 5/2 найгірша лише без фазової інформації. VE30 (колишня VC6): слід на стінці обумовлений гірше (9,3) за роздільну екскурсію (4,1), зважений до краю. VE31 (колишня VC7): від 2 до 6 мод найслабший напрямок 0,245 → 0,087, обумовленість 4,1 → 11,5 — плавно, без порогу. **Висновок `tokamak-3d-viz/docs/INVERSE.md` §3:** задача трактабельна, її складність задається **вибором спостережуваного, а не фізикою** — це ручка дизайну змагання. Відкрите: нелінійність (обумовленість локальна), вакуумне поле без відгуку плазми, один розряд/кадр, слід без трасування многовидів. Джерела: `tokamak-3d-viz/findings.md`, `docs/INVERSE.md`, `docs/CAMPAIGN.md`<br>**Оновлення 2026-09-29 (перевірки путівника, зареєстровано за згодою користувача):** (1) **внутрішній блокер теж має розв'язання — E35:** зламаність `multimode.py` (R8) спричинена тим, що поправка до T′ бере V′(T) (коеф. a_k·m_k) замість первісної h(T) (a_k/m_k); з виправленням max|det J − 1| = 9,5·10⁻⁸ (a = 0,2), 3,6·10⁻⁷ (a = 0,5). Отже, **симплектичний багатомодовий стенд Tokamap тепер можливий** (окрім поля без відображення з `tokamak-3d-viz`, VE21) — `03-deep-dives/D3-poincare-inverse/multimode_fixed_check.py`; ідентифіковність на виправленій мапі ще не прогнано. (2) **Обумовленість залежить від вибірки:** для того самого набору 2/1, 5/2, 3/1 з фазами усереднене спостережуване дає κ = **17,9** у `docs/INVERSE.md` (22 рівні × 6 фаз, 28 обертів) і **10,7** у скані мод `docs/CAMPAIGN.md` §4 (16 × 5, 24 оберти); у скані усереднена κ немонотонна по K: 22,1 → 10,7 → 28,7 (K = 2, 3, 4). Порівнювати числа κ можна лише за однакової вибірки; висновок «складність задає спостережуване» це не скасовує (роздільне в скані монотонне 4,1 → 11,5), але абсолютні κ — не властивість задачі. Див. `tokamak-3d-viz/findings.md` VE23, VE31<br>**Попередня реєстрація 2026-10-01 (нелінійна перевірка запущена; план затверджено користувачем):** спростовувача досі не було (колонка «—»), тепер він є: базовий багатостартовий нелінійний МНК відновлює 3 амплітуди + 3 фази з фазово-роздільної екскурсії при 3 % шуму. Успіх — усі амплітуди в межах 10 %, усі фази в межах 20°; хибний мінімум — старт поза допуском з χ² ≤ 1,5·χ²_true. Стенди: A — виправлений Tokamap (E35; 40 екземплярів × 16 стартів), B — `tokamak-3d-viz` (3 × 6, плюс скан пар фаз). **Підтверджено:** A ≥ 80 % успіхів і ≤ 10 % хибних, B 3/3 і без хибних; **спростовано:** на будь-якому стенді < 50 % успіхів або > 25 % хибних (B: ≤ 1/3 або хоч один хибний); інакше — «уточнено». Бюджет «48 год» не є критерієм, час повідомляється. Деталі — `03-deep-dives/D3-poincare-inverse/c5/THEORY.md` (не редагується після запуску; SHA-256 1ea23e22…81e1a4) |
| ~~**C2**~~ → **R11** | **ЗАКРИТО 2026-09-30 — спростовано → R11** ([ESTABLISHED.md](ESTABLISHED.md), розд. C; доказ `03-deep-dives/D1-psi-to-scalars/c2/`, критерії — `c2/THEORY.md`, записані до навчання). PCA+Ridge на вагу не реагує (\|Δ\| < 10⁻⁴); UNet_Lite (3 зерна): усі ваги точково гірші за пласку MSE за Consistency (−0,002…−0,017), усі 95 % ДІ містять 0; S(k) не краща за контролі H¹ і перемішану. Побічно — E37. Початкове твердження (закреслено): ~~Спектрально зважена втрата, виведена з кривої S(k), **покращує реальні моделі**~~ | Навчити бейзлайн із нею і без — різниці в `S′` немає | ~півдня, потрібен цикл навчання <br>**Попередня реєстрація 2026-09-30 (перевірка запущена за згодою користувача):** S′ не реалізовано (Calibration потребує невизначеності), тож спростовувач уточнено: різниці немає в **Consistency** (головна) і композитному **S** офіційного скорера. Моделі — PCA+Ridge (вага через зважену PCA) і UNet_Lite (вага у втраті, 3 зерна); дані — розбиття `SCORE.md` (навчання 0–35, валідація 36–39, тест 60–67); контролі — пласка MSE, H¹ з тим самим діапазоном ваг, перемішана вага; S(k) переміряно на 28 розрядах у робочій точці моделей. Правило ваги, бутстреп і критерії — `03-deep-dives/D1-psi-to-scalars/c2/THEORY.md` (не редагується після запуску) |
| ~~**C1**~~ → **R10** | **ЗАКРИТО 2026-09-30 — спростовано → R10** ([ESTABLISHED.md](ESTABLISHED.md), розд. C; доказ `03-deep-dives/D1-psi-to-scalars/c1/`, критерії — `c1/THEORY.md`, записані до запуску). Розбиття E2 стабільне лише на 3 демо-розрядах (96,7 % бутстрепу по розрядах); на 28 розрядах DIII-D — **0 %**, MAST — 3,3 %, 12 однонульових рівноваг Серфона–Фрайдберга через скорер — 0,3 %; для гладкого збурення кластерів немає (усі α ≈ 0,69–1,08). Конкурентна H-alt теж спростована. Натомість встановлено механізм для осі (E36). Початкове твердження (закреслено): ~~Три кластери α (E2) відповідають механізмам: **положення екстремуму / контур крізь сідло / інтеграл**~~ | Синтетика з контрольованою геометрією дає інші кластери, або кластери зникають при зміні вибірки розрядів | ~2 год. **Базовіше за все інше** — від цього залежить, чи D1 є теорією, чи каталогом <br>**Попередня реєстрація 2026-09-30 (перевірка запущена, план затверджено користувачем):** передбачення α за механізмами P1–P7, конкурентна гіпотеза **H-alt** («положення екстремуму ≈ ½ проти значення/інтеграла ≳ 0,8», два кластери), операційні означення (локальний нахил, режимний α, 1-D k-means + силует > 0,5, бутстреп по розрядах) і критерії «підтверджено / спростовано / уточнено / не вирішено» — у `03-deep-dives/D1-psi-to-scalars/c1/THEORY.md` (не редагується після запуску). Уточнення вартості: ~1–1,5 дня (синтетика двох рівнів + 25 розрядів DIII-D з HF), а не ~2 год |
| ~~**C8**~~ → **R9** | **ЗАКРИТО 2026-09-29 — спростовано як артефакт → R9** ([ESTABLISHED.md](ESTABLISHED.md), розд. C; доказ `04-novelty/c8_sign_check.py`). Провали виникають лише в рядках зі знаком −1: `skeleton()` викликає `extract_lcfs` без `axis_sign` → «first-success» fallback скорера; зі знаком, поверненим у конвенцію MAST, 0 провалів при всіх k. Нових даних не знадобилося. Початкове твердження (закреслено): ~~Вилучення LCFS, що провалилося на 3/8 і 2/8 кадрів MAST при перенесенні (див. E16), — **систематичний сигнал**, а не шум.~~ *Історична позначка:* ⚠️ **ПОТРЕБУЄ ДЖЕРЕЛА ДАНИХ (N3, 2026-09-29):** як записано, нездійсненна — `efit_psirz` для MAST на HF не публікується, окрім 3 демо-розрядів (`manual/tools/COORD_NOTES.md`, розд. 4) | *(історичне, неактуально після R9)* *Було:* ~~Більша вибірка показує випадковий розподіл провалів по k~~ → **Переписано 2026-09-29:** на вибірці MAST з EFIT-рівновагами з **FAIR-MAST (CC BY-SA 4.0)** або іншого джерела MAST EFIT (значно більше за 3 демо-розряди) провали вилучення LCFS розподілені по k випадково | *(історичне)* *Було:* ~~≈2 год: повторити на повному потоці MAST, не на 3 демо-розрядах~~ → **Переписано 2026-09-29:** ≈2 год обчислень **плюс** невідома вартість отримання даних: завантаження й конвертація EFIT із FAIR-MAST (Tier B, `01-landscape/datasets.md`) і ліцензійне обтяження BY-SA (Q13; журнал перевірки V-C4). Без джерела даних не перевіряється |

## Здогадки, відкладені свідомо

| # | Здогадка | Чому відкладено |
|---|---|---|
| **C9** | Перенесення між машинами можна покращити фізично коваріантним поданням (коефіцієнти функцій Гріна замість пікселів) | Це задача **живого** змагання Sophelio. Беремо як мотивацію, не як власну роботу |
| **C10** | Контамінація міток: частина похибки будь-якої ML-моделі — це успадковане зміщення EFIT, а не похибка моделі | Потребує незалежної істини, якої немає. Записано як відкрите питання поля |

---

## Як читати співвідношення 8 із 14

*(2026-09-29: тепер 9 — додано R9, колишня C8; вона теж не потрапила ні в концепт, ні в матеріали
для учасників. Знаменник «~14» не перераховувався. 2026-09-30: тепер 10 — додано R10, колишня C1, потім 11 — R11, колишня C2; у концепті їх не було,
але E2, на якій вона стояла, цитується в методичці й путівнику — там додано обмеження.)*

Це **не** показник поганої роботи — це показник того, що перевірка працює. Жодна з восьми
спростованих гіпотез не потрапила в концепт чи в матеріали для учасників. Дві з них
(R2 про Біна, R7 про H5) дали **кращі результати після спростування**, ніж містили до нього.

Небезпека не в тому, що гіпотези хибні, а в тому, що хибна гіпотеза **потрапляє в документ
без позначки**. Саме для цього існують два реєстри.

---

## English summary

**Entry rule:** everything **not yet verified**. Each entry states the claim, **what would falsify
it**, and **what testing costs**. **Exit rule:** once tested it moves to `ESTABLISHED.md` — into the
measured, the read, or the refuted section. **Nothing disappears silently.** **Citation rule:** no
entry here may be used in the concept or in participant-facing material without being marked as a
conjecture.

**Four active conjectures** (C3, the Calibration half of C4, C6, C7; "Five" until C5 was closed as refuted → R13 on 2026-10-01; "Six" until C2 was closed as refuted → R11 on 2026-09-30; this line said "Eight" until 2026-09-29, when C8 was closed as refuted → R9, and "Seven" until 2026-09-30, when C1 was closed as refuted → R10). **C1** — whether the three amplification-exponent clusters correspond to extremum-position / saddle-contour / integral mechanisms — was tested on 2026-09-30 against criteria fixed in advance (`03-deep-dives/D1-psi-to-scalars/c1/THEORY.md`): the partition holds on the 3 demo shots (96.7 % of shot-bootstrap resamples) but on **0 %** of resamples over 28 DIII-D shots, 3.3 % on MAST and 0.3 % on 12 single-null Cerfon–Freidberg equilibria; smooth perturbations show no clusters at all. **Refuted → R10.** What survives is the axis mechanism (E36): α = 1 below η* = λh²/2, ≈ 0.4–0.5 above it.

Others: whether a spectrally weighted loss actually improves models (C2 — **tested 2026-09-30 and refuted, R11**: PCA+Ridge is insensitive to it, UNet_Lite gets slightly worse in every weighting, all CIs contain 0; side result E37 — UNet_Lite reaches R²ψ = 0.976 yet Consistency 0.179, below PCA+Ridge at R²ψ = 0.090); whether the 1/√λ
conditioning number has enough spread across the corpus to stratify by (C3); whether the two new
metric terms **reorder** the existing baselines, without which they add nothing (C4 — the GS_score half was **tested 2026-10-01 and refuted, R12**: the point ranking moves, τ = 0.857, but no swap holds in ≥ 95 % of bootstrap replicates; side result E38 — map-output networks have R²ψ 0.96–0.98 yet GS-inconsistency 0.81–0.96, worse than truth + 1 % noise; the Calibration half stays open, with C6); whether the
Level-3 inverse problem is tractable in 48 hours (C5 — **tested 2026-10-01 and refuted, R13**: it had no falsifier until then; a pre-registered multi-start least-squares baseline recovered 3 amplitudes + 3 phases on 0/40 Tokamap and 0/3 `tokamak-3d-viz` instances at 3 % noise, with no spurious minima but no start reaching the noise level; side result E40 — the excursion misfit landscape is glassy); whether simultaneous conformal bands on a
65×65 field are achievable without absurd width (C6); and whether Deep Ritz is seed-unstable on the
non-convex plasma problem but stable on convex Bean (C7). Formerly also: whether the LCFS-extraction
failures seen under cross-machine projection are systematic or noise (C8) — **closed 2026-09-29 as
refuted (R9)**: a sign-handling artefact (failures only where sign −1 was chosen; `skeleton()` passes no
`axis_sign`; with the sign restored, 0 failures at every k), testable without new data.

**Updates 2026-09-29.** **C4** has a partial result (not a verdict): the GS residual separates
predictions from EFIT (0.0103) but barely separates a weak PCA+Ridge prediction (0.0174, R² 0.877)
from a near-exact PCA-50 projection (0.0170). **C5**: the blocker is removed — the visualisation
stand defines the field as B = ∇×(ψ_total∇φ), divergence-free exactly for any number of modes —
but the conjecture is **not closed; its subject changed**: amplitudes alone are trivial
(condition number 1.2), difficulty comes from phases and is set by the choice of observable
(17.9 phase-averaged vs 5.8 phase-resolved), degrading smoothly with mode count and with a
wall-footprint observable (VE21–VE25, VE30, VE31). **"Level 3"** is now defined explicitly, and
retroactively, as recovering the (m, n) perturbation spectrum from Poincaré-type data; Levels 1–2
were never defined. Later on 2026-09-29 (guide checks, registered with user approval): C5's
*internal* blocker also has a fix — the multi-mode Tokamap must use the antiderivative h(T), not V′(T)
(E35; max|det J − 1| ≤ 3.6 × 10⁻⁷), so a symplectic multi-mode Tokamap stand is now possible; and the
condition numbers depend on sampling (17.9 in INVERSE, 22 levels × 6 phases, 28 turns, vs 10.7 in the
mode scan, 16 × 5, 24 turns, for the same three modes; non-monotone 22.1 → 10.7 → 28.7 over K = 2–4).
**C8** (superseded the same day — closed as R9) was infeasible as written — MAST `efit_psirz` is published on HF only for
three demo shots — so its falsifier and cost now require FAIR-MAST (CC BY-SA 4.0) or another MAST
EFIT source (**needs a data source**). **C3** depends on E14, which had no reproducible script — **restored 2026-09-30** (`c1/e14_curvature.py`); the axis curvature λ spans 0.13–3.6 over 28 DIII-D shots, so the main falsifier of C3 (a narrow distribution) did not fire.

**Two conjectures deliberately deferred:** physically covariant representations for cross-machine
transfer (that is the live competition's problem), and EFIT label contamination (needs independent
ground truth that does not exist).

**On the 8-refuted-of-14 ratio** (this line said "7-refuted-of-13" until 2026-09-29, out of step
with the Ukrainian text and with R1–R8; since R9 was added on 2026-09-29 the count is nine, denominator not recounted): that is not a sign of poor work but of verification
functioning. None of the eight reached the concept or participant-facing material, and two of them produced
**better results after refutation** than they contained before it. The danger is never that a
hypothesis is wrong — it is that a wrong hypothesis enters a document unmarked. That is what the
two registers exist to prevent.
