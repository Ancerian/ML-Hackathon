# Конкурентний аналіз: змагання і бенчмарки / Competitive landscape

Станом на **2026-09-22**. Маркери достовірності: `[V]` — джерело відкрито й прочитано,
`[S]` — лише зі зведення пошуку, `[?]` — суперечливо або непідтверджено.

**Головний висновок:** у ніші є **рівно одне живе гучне змагання** — Fusion Equilibrium
Challenge. І є **одна структурна діра**: жоден фузійний бенчмарк не розрахований на
24–48-годинний студентський формат. Усе наявне — це 98 ГБ – 15 ТБ або угоди про
передачу даних. Це і є наша точка входу.

---

## 1. Fusion Equilibrium Challenge (Sophelio / NeurIPS 2026) — головний орієнтир

- Сайт: <https://fusion-equilibrium-challenge.sophelio.io/> `[V]`
- Стаття: <https://arxiv.org/abs/2609.01750> `[V]`
- Таблиця лідерів: <https://www.codabench.org/competitions/17739/> `[V]`
- Дані: <https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge> `[V]`

**Організатори** `[V]`: Sophelio (Craig Michoski, Tapan Ganatma Nakkina) спільно з
DIII-D / General Atomics, UKAEA MAST-U (FAIR-MAST), UT Austin Institute for Fusion Studies,
EPFL Swiss Plasma Center, MIT PSFC. Перше фузійне змагання, прийняте до NeurIPS Competition Track.

### Задача `[V]`

Для кожної мітки часу EFIT передбачити повну карту полоїдального потоку **ψ(R,Z) на сітці 65×65**
плюс два скаляри, яких карта потоку не містить: **q₉₅** і **β_N**.

Вхід **навмисно виключає всю магнітну діагностику**: струми котушок полоїдального поля
(DIII-D: 18 F-котушок + ECOILA + bcoil; MAST: 10 P-котушок + соленоїд, TF, EFPS),
профілі томсонівського розсіяння Tₑ/nₑ, струм плазми Iₚ, геометрія машини.

**Мотивація:** нейтронне опромінення в машинах класу SPARC/ARC деградує магнітні сенсори
(дрейф інтеграторів, наведена ЕРС), тож реконструкція без магнітної діагностики —
реакторно-релевантна, а не штучна гра.

### Метрика `[V]`

```
S = 0,55·R²ψ + 0,15·R²_{q95,βN} + 0,10·(1 − D_LCFS) + 0,20·Consistency
G_ratio = S(MAST) / S(DIII-D)          зі шлюзом S(DIII-D) ≥ 0,85
```

- `D_LCFS` — **симетрична відстань Гаусдорфа** між передбаченою й істинною останньою
  замкненою поверхнею, нормована на середній великий радіус.
- `Consistency` — середнє R² по **семи** похідних скалярах, **перерахованих з поданого ψ**
  тим самим функціоналом, що й з істинного. Незалежної скалярної «голови» немає — це
  виправлення геймабельності v1.

> **Примітка про 7 vs 8 — ПЕРЕВІРЕНО, відповідь: СІМ.** `[V]`
> Сторінка Evaluation на Codabench має заголовок дослівно «Consistency — seven ψ-derived
> scalars» і пояснює, **чому восьмий прибрано**: колонки `dsep` на DIII-D і MAST — це
> **різні фізичні величини** (у DIII-D це зазор сепаратриса↔лімітер з a-файлу, у MAST —
> баланс дивертора δR_sep), тож один функціонал не міг узгоджено оцінювати обидві машини.
> Картка датасету на HF і стаття (§1.5) теж кажуть **сім**. Єдине «вісім» — у спливній
> підказці на головній сторінці, і це застаріла маркетингова копія.
> Локальний `fusion_scoring/common.py` незалежно підтверджує:
> `R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li`.

### Два челенджі `[V]`

| | Challenge 1 | Challenge 2 |
|---|---|---|
| Постановка | навчання і тест на DIII-D | навчання на DIII-D, **zero-shot** тест на MAST |
| Метрика | `S` | `G_ratio` |
| Шлюз | — | S(DIII-D) ≥ 0,85 (піднято 2026-08-18) |
| Приз | $500 | $500 |

### Терміни `[V]`

| Дата | Подія |
|---|---|
| 2026-07-27 → **2026-10-18** | Phase 1, публічна таблиця лідерів, 5 подань/день, 100 усього |
| 2026-10-19 | відкриття сліпого тесту |
| **2026-10-26** | закриття Phase 2 (3 сліпі подання) |
| грудень 2026 | сесія на NeurIPS |

**Стан:** 176 учасників / 684 подання `[V]` (станом на 2026-09-22).
**Топові бали публічно недоступні** `[V]`: `/api/competitions/17739/results/` повертає
«You are not a competition admin or superuser», `/api/leaderboards/?competition=17739` — порожній
масив, а сторінка результатів рендериться на клієнті. Щоб їх побачити, потрібен обліковий запис
Codabench. Публічно читаються лише колонки таблиці (`d3d_S`, `g_ratio`, `mast_S`, `d3d_r2_psi`,
`d3d_r2_qb`, `d3d_dlcfs`, `d3d_consistency`) і дати фаз.

> ⚠️ **Зміна правил, що знецінює старі числа.** 2026-08-18 шлюз Challenge 2 піднято
> з `R²ψ > 0,6` до **`S(DIII-D) ≥ 0,85`**, бо старий шлюз обмежував лише член 0,55·R²ψ,
> і навмисно ослаблений вхід по DIII-D роздував відношення приблизно втричі.
> **Будь-яке значення `G_ratio` до 18.08 не порівнянне з поточними.**

### Опубліковані бейзлайни організаторів `[V]` (демо-підвибірка, ~264 зразки / 3 розряди)

| Бейзлайн | Результат |
|---|---|
| PCA + Ridge | SSIM = 0,84 |
| MLP на PCA-цілях (~41 тис. параметрів) | MSE = 0,005 |
| Згортковий декодер (~5 млн параметрів) | MSE = 0,239 (перенавчається) |
| **Наївне перенесення відображення котушок** | **SSIM 0,83 (DIII-D) → 0,10 (MAST)** |

### Що організатори самі називають невирішеним `[V]`

1. **Перенесення між машинами практично не розв'язане.** Дослівно: *«наївне перенесення
   відображення котушок у нашому пілоті падає з SSIM 0,83 до 0,10 — фактично провал,
   і це чесніший стрес-тест, ніж розбиття всередині машини»*.
2. Центральне наукове питання: чи схоплюють навчені моделі *«універсальну фізику рівняння
   Града–Шафранова, а не інженерні дрібниці окремої установки»*.
3. **Визнані обмеження даних:** лише дивертовані зрізи (лімітовані плазми виключені —
   межа з карти потоку «систематично перевищує»); фільтрація за наявністю Томсона зміщує
   покриття режимів; невизначеність істини EFIT не охарактеризована.

**Наш висновок:** клонувати задачу не можна (живе змагання). Дані — брати
(CC BY 4.0, стрімінг без авторизації). Незайняті осі — у `saturated-vs-headroom.md`.

---

## 2. TokaMark (UKAEA + IBM + STFC, 2026) — найкращий структурний шаблон

- Стаття: <https://arxiv.org/abs/2602.10132> `[V]` · Код: <https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark_baseline> `[V]`
- Дані: 11 573 розряди MAST, 39 сигналів, частоти 0,2 кГц – 500 кГц. **CC BY 4.0** `[V]`

**14 задач у 4 групах** `[V]`: G1 миттєва реконструкція рівноваги (3), G2 короткочасна
динаміка магнітних (3), G3 динаміка кінетичних профілів (3), G4 довгострокове прогнозування
МГД (5).

**Метрика:** ієрархічна NRMSE/NMAE, агрегована зразок → вікно → сигнал → задача → розряд,
нормована на глобальну емпіричну σ. **Розбиття і випадкове, і темпоральне** — це те, чого
майже ніхто не робить.

**Результати (випадкове розбиття, CNN+LSTM)** `[V]`: G1 **0,1439** · G2 **0,1343** ·
G3 **0,3428** · G4 **0,3255**.

**Що вони самі називають провалом** `[V]`: темпоральне розбиття *«виявило значні збої
узагальнення»* для G3–G4 — **прогноз м'якого рентгену поза розподілом NRMSE > 3,0**.
Систематичні пропуски: ~38–55% (рівновага), ~46% (Томсон).

**Чому це для нас важливо:** (а) таксономія задач + ієрархічна нормована метрика + явне
розділення випадкового і темпорального розбиття — готовий шаблон для нашого концепту;
(б) є **прецедент одноосібної похідної роботи**: N. Gupta, *Benchmarking Sensor Robustness
in Plasma Diagnostic Models: A Systematic Evaluation on TokaMark*,
<https://arxiv.org/abs/2607.11915> `[V]` — тобто бенчмарк реально перевикористовний
командою наших розмірів.

---

## 3. DisruptionBench і екосистема прогнозу зриву

- Стаття: Spangher et al., *J. Fusion Energy* **44**:26 (2025),
  <https://link.springer.com/article/10.1007/s10894-025-00495-2> `[S]` (платний доступ)
- Код: <https://github.com/MIT-PSFC/DisruptionBench> — **архівовано 2025-07-21, read-only** `[V]`

**Критично:** репозиторій **не постачає даних**. Ви маєте подати власні оцінки
розривності словником. Дані C-Mod / DIII-D / EAST (~30 000 розрядів) **не завантажуються
відкрито**. Це **скоринговий харнес, а не датасет** — і це жорстка перешкода для студентів.

**Результат:** CCNN найкращий, **AUC до 0,974 на Alcator C-Mod** (intra-machine) `[S]`.

**Лінія робіт:** Rea et al. 2018 (*PPCF* 60:084008, random forest, DIII-D) →
Kates-Harbeck, Svyatkovskiy, Tang 2019 (*Nature* 568:526, FRNN, JET→DIII-D,
<https://www.nature.com/articles/s41586-019-1116-4> `[V]`) → Zhu et al. 2021
(*Nucl. Fusion* 61:026007, HDL, <https://arxiv.org/abs/2007.01401> `[V]`) → CCNN
(<https://arxiv.org/abs/2312.01286> `[V]`).

**Свіже (2026)** `[S]`: прогноз без експериментів через синтетичну аугментацію діагностик
(<https://arxiv.org/abs/2606.08462>); дистиляція знань на EAST
(<https://arxiv.org/abs/2607.04241>); horizon-aware тривоги
(<https://arxiv.org/abs/2609.24443>).

**Наш висновок:** сюди не йдемо. Обґрунтування — `02-field-map/E-disruption-why-not.md`.

---

## 4. Інші змагання і хакатони в ніші

| Назва | Організатор | Рік | Задача | Дані | Масштаб | Стан |
|---|---|---|---|---|---|---|
| **AI for Fusion Energy Challenge** `[V]` | IAEA + ITU AI for Good, з MIT PSFC, J-TEXT, HL-2A | 2023 | перенесення прогнозу зриву між машинами | **потрібна підписана угода про передачу даних** | не розкрито | завершено, результати не опубліковано |
| **ConStellaration** `[V]` | Proxima Fusion + HuggingFace | 2025, стаття на NeurIPS 2025 | оптимізація межі стеларатора, 3 задачі | >150 000 QI-рівноваг, відкрито на HF | 77+ учасників, **без призу** | таблиця лідерів жива, переможців не оголошено `[?]` |
| **Columbia Perturbed Equilibrium Hackathon** `[V]` | Columbia FRC (Nikolas Logan) | 2025-07-28→30 | **код-спринт, не ML**: відкрита Julia-реалізація GPEC | внутрішні коди | студенти Columbia + індустрія | завершено. **Найближчий прецедент університетського фузійного хакатону** |
| **ML4CFD** `[S]` | NeurIPS 2024 | 2024 | сурогат CFD для аеропрофілю | згенеровано OpenFOAM | **>240 команд** | завершено; переможець **перевершив сам OpenFOAM** за сукупними метриками |
| **RealPDE** `[V]` | NeurIPS 2026 | 2026 | Sim2Real + адаптація під час тесту для PDE | парні реальні PIV + CFD | — | живе, **призовий фонд $21 000** |

**Чого немає взагалі** `[V]`: жодного ML-хакатону від ITER; жодного публічного дата-челенджу
від PPPL; жодного відкритого змагання EUROfusion (вони фінансують 15 проєктів, але змагань
не проводять).

**Калібрування призів:** флагманське фузійне змагання платить **$1 000 разом**. RealPDE —
$21 000. Тобто працює престиж (співавторство, доповідь на NeurIPS), а не гроші.
**Ми можемо конкурувати доступом і менторством, а не касою.**

---

## 5. Загальні SciML-бенчмарки як шаблон

| Бенчмарк | Що взяти |
|---|---|
| **PDEBench** `[V]` <https://github.com/pdebench/PDEBench> | канонічна структура «дані + код + бейзлайни + `metrics.py`». Пряма і обернена задачі. Офіційної таблиці лідерів у репозиторії **немає** |
| **The Well** (PolymathicAI) `[V]` <https://github.com/PolymathicAI/the_well> | 16 датасетів, 15 ТБ, BSD-3. **Автори самі пишуть**, що їхні бейзлайни *«не слід вважати SOTA»* — тобто запас великий |
| **ML4CFD ретроспектива** `[S]` <https://arxiv.org/abs/2506.08516> | **найкраще джерело уроків** із дизайну фізичного змагання: багатокритеріальна оцінка = точність + фізична достовірність + обчислювальна ефективність + узагальнення поза розподілом |

**Мета-бенчмарки 2026, що критикують стан справ** `[S]` — корисні як аргументація
у вступі нашого концепту: PhysicsBench (<https://arxiv.org/abs/2608.24056>);
*Diagnosing Failure Modes of Neural Operators* (<https://arxiv.org/abs/2601.11428>);
*A Diagnostic Software Suite for Auditing Learned PDE Simulators*
(<https://arxiv.org/abs/2606.18200>); *Common Task Framework for Critical Evaluation of SciML*
(<https://arxiv.org/abs/2510.23166>).

---

## English summary

**One live high-profile competition exists in this niche** — the Fusion Equilibrium Challenge
(Sophelio, NeurIPS 2026): reconstruct ψ(R,Z) on a 65×65 grid plus q₉₅ and β_N from PF-coil
currents and Thomson scattering only, no magnetics. Phase 1 closes 2026-10-18; 176 participants,
684 submissions; $1,000 total prize. Composite metric
`S = 0.55·R²ψ + 0.15·R²_{q95,βN} + 0.10·(1−D_LCFS) + 0.20·Consistency`, cross-machine
`G_ratio = S(MAST)/S(DIII-D)` with an S ≥ 0.85 eligibility gate. Its own pilot shows naive
cross-machine transfer collapsing from SSIM 0.83 to 0.10 — the flagship open problem.

**TokaMark** (UKAEA/IBM/STFC, CC BY 4.0, 11,573 MAST shots, 14 tasks) is the best structural
template: task taxonomy, hierarchical normalised metric, and explicit random-vs-temporal splits.
It reports catastrophic out-of-distribution failure (NRMSE > 3.0) under temporal splitting, and a
single-author derivative paper already exists — proof the benchmark is reusable at our scale.

**DisruptionBench** ships no data and is archived; the underlying C-Mod/DIII-D/EAST corpus is not
openly downloadable. Disruption prediction is therefore both saturated and inaccessible.

**Structural gap we can occupy:** no fusion benchmark is sized for a 24–48 hour student event.
Everything current is 98 GB–15 TB or requires data-sharing agreements. Prestige, not money, drives
participation ($1,000 for the flagship vs $21,000 for RealPDE), so access and mentorship are
credible competitive advantages for a university event.
