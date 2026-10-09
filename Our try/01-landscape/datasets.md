# Дані: що реально доступно / Datasets by access friction

Станом на **2026-09-22**. Маркери: `[V]` відкрито й перевірено, `[S]` зі зведення пошуку,
`[?]` непідтверджено.

Впорядковано за **тертям доступу**, бо для хакатону це важливіше за обсяг.
Датасет, що потребує угоди про передачу даних, для студентів не існує.

---

## Tier A — нульове тертя: без авторизації, пряме завантаження

| Датасет | Обсяг | Ліцензія | Доступ | Придатність для нас |
|---|---|---|---|---|
| **Sophelio Fusion Equilibrium Challenge** `[V]`<br><https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge> | **103,86 ГБ, 9 121 файл Parquet** (виміряно 2026-09-22): 7 041 `diii_d_train` + 874 `diii_d_public_test` + 1 206 `mast_public_test` | **CC BY 4.0** (код starter — MIT) | `load_dataset(...)`, **стрімінг працює без авторизації**, Parquet | **Основне джерело реальних даних.** Найнижче тертя у фузійній ніші взагалі |
| **Open Density Limit Database** (MIT PSFC) `[V]`<br><https://github.com/MIT-PSFC/open_density_limit_database> | Alcator C-Mod, рядок = 10 мс; явно підмножина | дані CC BY, код MIT | звичайний `git clone`, CSV **і** HDF5 | **Найменший фузійний датасет, який ми знайшли.** Містить `demo.ipynb` із лінійним SVM і ROC. Ідеальний **вхідний поріг для базового треку** |
| **ConStellaration** (Proxima + HF) `[V]`<br><https://huggingface.co/datasets/proxima-fusion/constellaration> | >150 000 QI-рівноваг стеларатора | **MIT** `[V]` — не Creative Commons | HF datasets API | Стеларатори, не токамаки. Резерв. MIT — це **софтверна** ліцензія на дані, що юридично незграбно для європейського права на бази даних; зберігати повідомлення про ліцензію |
| **OpenSTEP** (UKAEA) `[V]`<br><https://github.com/ukaea/OpenSTEP> | сценарій STEP SPP-001: рівновага, профілі, транспорт, стінка | **CC BY 4.0** | GitHub; HDF5 через IMAS, NetCDF5 | Симуляція, не експеримент. Резерв для синтетики |
| **The Well** (PolymathicAI) `[V]` | 15 ТБ, 16 датасетів (включно з МГД) | BSD-3-Clause | CLI або HF-стрімінг | Завеликий. Цінний як **шаблон**, не як дані |

## Tier B — відкрито, але потрібне знання інструментів

| Датасет | Обсяг | Ліцензія | Доступ |
|---|---|---|---|
| **FAIR-MAST** (UKAEA) `[V]`<br><https://mastapp.site/> · <https://github.com/ukaea/fair-mast> | **11 573 розряди**, кампанії M05–M09 | софт MIT; **дані CC BY-SA 4.0** ⚠️ | Postgres + FastAPI JSON + GraphQL; публічний S3 (`s3.echo.stfc.ac.uk`), Zarr v2 + Parquet, `--no-sign-request` — **без авторизації** |
| **TokaMark** `[V]` | 11 573 розряди MAST, 39 сигналів | **CC BY 4.0** | GitHub + бекенд FAIR-MAST |
| **ITPA HDB5** `[V]` <https://osf.io/drwcq> | v5.2.3, 2026-03-12; табличний, 11 установок | **CC BY 4.0** `[V]` — підтверджено через OSF API (HTML-сторінка рендериться JS, тому ліцензію було важко знайти) | пряме завантаження з OSF; програмно через `fusionflux` (PyPI, SHA-256) |

> ⚠️ **CC BY-SA 4.0 у FAIR-MAST — вірусна, і це підтверджено детально** `[V]`.
> §3(b) поширює ShareAlike на **Adapted Material**. Критично для даних: **§4(b) каже, що коли
> ви реалізуєте sui generis права на базу даних щодо суттєвої частини, похідна база
> **Є** Adapted Material** — включно для цілей ShareAlike. Тобто перерозмічений набір ψ,
> відфільтрована підмножина чи навчальний корпус, похідні від даних MAST, — **підпадають**.
> Проста *агрегація* в більшу незалежну базу (§4(c)) не підпадає, але межа залежить від фактів.
> **Ваги моделі, навченої на цих даних, — справді невирішене питання**; жодні рекомендації CC
> не трактують навчені параметри як визначений випадок. Не вважайте, що ви чисті.

## Tier C — практично недоступно студентам

| Датасет | Причина |
|---|---|
| **Сирий архів DIII-D** (<https://d3dfusion.org/>) `[V]` | «відкритий, але контрольований»: потрібен обліковий запис / колаборація. Зріз на HF від Sophelio — **єдиний** безтертєвий шматок DIII-D |
| **JET** `[V]` | дані **не можуть бути публічними через юридичні обмеження**, «за обґрунтованим запитом». Працював 1983–2023 |
| **KSTAR** `[V]` | публічного датасету **не знайдено**. ML-роботи використовують внутрішні дані кампаній |
| **EFIT-AI** (>6 млн рівноваг) `[S]` | заявлено «буде опубліковано», **публічної точки завантаження не знайдено** `[?]`. Якщо відкриють — найбільший корпус рівноваг у світі. Варто написати листа в GA |
| **Дані DisruptionBench** (~30 тис. розрядів) `[V]` | не випущені, лише харнес |
| **Дані IAEA-челенджу** (C-Mod / J-TEXT / HL-2A) `[V]` | потрібна підписана угода |

---

## Синтетичні генератори — основа нашого sim-to-real

Це рішення **D5**: власного генератора не пишемо.

| Інструмент | Що дає | Установка | Примітка |
|---|---|---|---|
| **FreeGS** `[V]`<br><https://github.com/freegs-plasma/freegs> | free-boundary GS, Пікар + мультисітка, чистий Python | `pip install FreeGS` | **Найпростіший для навчання.** Містить функції Гріна через еліптичні інтеграли |
| **FreeGSNKE** `[V]`<br><https://github.com/FusionComputingLab/freegsnke> · <https://docs.freegsnke.com/> | Ньютон–Крилов (збігається краще за Пікара), лінеаризований і нелінійний еволюційні солвери, 4-й порядок | pip | **Найкращий для серйозної роботи.** Саме ним зроблено роботу про множинні розв'язки |
| **TokaLab** `[V]`<br><https://tokalab.github.io/> · <https://github.com/TokaLab> | віртуальний токамак: VirtualLab, SimPla (рівновага), **SynDiag (синтетична діагностика)** | MATLAB + Python | **Істина відома за побудовою.** Знімає ризик «день 1 витрачено на дані» |
| **TokaMaker** (Open FUSION Toolkit) `[V]`<br><https://arxiv.org/abs/2311.07719> | FEM на неструктурованих трикутниках, статичний і нестаціонарний | — | Явно спроєктований для навчання |
| **DESC** `[V]` <https://github.com/PlasmaControl/DESC> | 3D стеларатор, **JAX, повністю диференційовний** | `pip install desc-opt` | Еталон «солвера, крізь який можна пропустити градієнт» |

---

## Що вже лежить у нас локально

| Шлях | Що це |
|---|---|
| `fusion equilibrium challenge/starter/parquet_data/` | **6 демо-розрядів (3 DIII-D, 3 MAST), 67 МБ** — реальні дані без завантаження 98 ГБ. Досить для D1 |
| `fusion equilibrium challenge/starter/.venv/` | Python 3.12 із закріпленими залежностями + torch 2.5.1 (MPS) |
| `fusion equilibrium challenge/starter/my_experiments/baseline_pca_ridge.joblib` | навчений PCA+Ridge (1,7 МБ), 50 компонент |
| `fusion equilibrium challenge/downloaded_huggingface/hf_dataset/` | демо-розряди в розкладці Hub для `--source local` |

---

## ⚠️ Конфлікт ліцензій, який треба розв'язати ДО будь-якого перевипуску

**Знахідка перевірки V-C3/V-C4** `[V]`. Датасет Sophelio FEC — **CC BY 4.0 (без share-alike)**,
але його конфігурація `mast_public_test` містить **1 206 розрядів MAST**, а головна сторінка
челенджу вказує UKAEA / MAST-U і програму FAIR-MAST серед партнерів.

Якщо ці дані MAST походять з архіву FAIR-MAST під **CC BY-SA 4.0**, то перевипуск їх нижче
за течією як простий **CC BY 4.0 не дозволений самою BY-SA** — для цього потрібен окремий
дозвіл від UKAEA.

Можливі пояснення: (а) UKAEA дала Sophelio окремий дозвіл; (б) дані MAST прийшли шляхом,
не покритим ліцензією каталогу; (в) **це нерозв'язаний ліцензійний дефект**.

**Що робимо:** поки не з'ясовано — **трактуємо частину MAST як обтяжену BY-SA**
і не перевипускаємо її. Частина DIII-D цим не зачеплена. Питання додано до
`00-decisions/open-questions.md`.

## Ліцензійна гігієна: що зробити перед публікацією будь-чого

1. **Sophelio CC BY 4.0** `[V]` — атрибуція обов'язкова, **формулювання встановлено**:
   використовувати bibtex-блок із картки датасету (`fusion_equilibrium_challenge`, автори
   Michoski, Waller, Sammuli, Boyes, Clark, Smith, Nakkina, Hatch, Nazikian; 2026;
   Sophelio and General Atomics) **плюс подяку DOE**: Office of Science, Office of Fusion
   Energy Sciences, DIII-D National Fusion Facility, нагорода **DE-FC02-04ER54698** і
   **DE-SC0024426, DE-SC0024499, DE-SC0024409, DE-SC0024571**, плюс повний disclaimer із README.
   Зверніть увагу: список авторів у bibtex **датасету** відрізняється від списку авторів
   **статті** (там ведучий — Nakkina) — не плутати.
2. **FAIR-MAST CC BY-SA 4.0** — share-alike. **Не змішувати** з CC BY-4.0-джерелами в одному
   похідному файлі, доки не вирішено, під якою ліцензією виходимо.
3. **HDB5 — CC BY 4.0, підтверджено** `[V]`. Але поле `copyright_holders` на OSF **порожнє**,
   тож для атрибуції треба або спитати супровідників, кого називати, або спиратися на
   канонічну цитату: Verdoolaege G. et al., *Nucl. Fusion* **61**(7), 076006 (2021),
   doi:10.1088/1741-4326/abdb91. Також на вузлі увімкнено `access_requests_enabled` —
   перевірити, що потрібні файли лежать у публічному компоненті.
4. **`fusionsimulator.io`** — README самосуперечливий («MIT» і «All rights reserved» водночас).
   **Нічого похідного не публікуємо до відповіді автора** (Q10 в `open-questions.md`).
5. **Код starter kit — справді MIT** `[V]`. GitHub показує `NOASSERTION`, бо до тексту MIT
   дописано розділ «NOTE ON SCOPE»; це ламає автоматичний класифікатор, але не ліцензію.
   Локальна перевірка `LICENSE`: «MIT License, Copyright (c) 2026 Sophelio and the Fusion
   Equilibrium Challenge authors», і окремо зазначено, що **MIT покриває лише код, не дані**.
6. **Канонічний репозиторій starter kit** — `github.com/Sophelio/fusion-equilibrium-challenge-starter`
   `[V]`. Є живий форк `github.com/RimmaShaf/...` (створений 2026-08-19, застарілий) — він
   **віддає 200**, тож наївна перевірка «чи не 404?» його помилково благословить. Не цитувати.
   Наш локальний клон указує саме на канонічний репозиторій — перевірено.
7. Хеші файлів будь-якого нашого релізу — для контролю цілісності.

---

## English summary

Ranked by **access friction**, which matters more than size for a hackathon.

**Tier A (no auth, direct download):** the Sophelio challenge corpus on HuggingFace
(CC BY 4.0, streaming, no auth — the lowest-friction fusion data available today, and our
primary real-data source); the MIT PSFC Open Density Limit Database (smallest fusion dataset
found, ships a working SVM notebook — ideal entry-track on-ramp); ConStellaration; OpenSTEP;
The Well.

**Tier B (open, tooling required):** FAIR-MAST (11,573 shots, public S3, **CC BY-SA 4.0 —
share-alike propagates to derived datasets**, so do not mix with CC BY sources); TokaMark
(CC BY 4.0); ITPA HDB5 (**licence unconfirmed — do not use until resolved**).

**Tier C (effectively unavailable):** raw DIII-D, JET (legal restrictions), KSTAR (no public
release), EFIT-AI (announced, no endpoint found), DisruptionBench corpus, IAEA challenge data.

**Synthetic generators for the sim-to-real half:** FreeGS (teaching), FreeGSNKE (serious work,
Newton–Krylov — the code used for the multiple-solutions result), TokaLab (virtual tokamak with
synthetic diagnostics and known ground truth), TokaMaker, DESC (differentiable). Per decision D5
we write none of our own.

**Already local:** 6 demo shots (67 MB), a pinned venv, and a trained PCA+Ridge model — enough
to start deep-dive D1 today.
