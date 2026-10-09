# Бриф для авторів путівника гіпотезами

Корінь: `/Users/illia_nakonecnyi/Desktop/ТОКАМАК ПРОДЖЕКТ`. Путівник: `guide/`. План: `/Users/illia_nakonecnyi/.claude/plans/pasted-content-id-8601-role-humming-donut.md` §7 (прочитай §7.1–7.4).

## Мета і читач
- Читач — **студенти ФМФ КПІ** (3–4 курс, знають Python і базову фізику плазми з методички `manual/main.pdf`).
- Для кожної гіпотези/твердження розповідаємо **історію**:
  1. що стверджує (фізика, простими словами + формула);
  2. звідки взялась ідея (мотивація, документ, дата);
  3. що ми зробили (експеримент/скрипт, параметри);
  4. що вийшло (числа з `results.json`/реєстру/повторного запуску);
  5. що з цього випливло (статус, наступна ідея — стрілки «спростування → нова ідея»);
  6. **де це в коді** — фрагмент коду з поясненням кожного блоку;
  7. як студенту відтворити / розвинути (команда запуску, що змінити).
- Тон: чесний, з помилками й спростуваннями як нормальною частиною науки (правило реєстрів: «нічого не зникає мовчки»).

## Джерела істини (узгоджені 2026-09-29, затверджено користувачем, G4)
- `Our try/00-decisions/{ESTABLISHED,CONJECTURES,DECISIONS,open-questions,CHANGELOG}.md`, `Our try/99-bibliography/verification-log.md` (ID з префіксом `V-`), `Our try/0x-*/**`.
- `tokamak-3d-viz/findings.md` (VE1–VE33, VR1–VR16, VC1–VC7, VQ1–VQ4), `tokamak-3d-viz/docs/*.md` (ANALYSIS, CAMPAIGN, INVERSE, COIL-BACKEND, PHYSICS, CHANGES-2026-09-29-island-width).
- `Tokamak-GS-solver/CHANGES-2026-09-29.md`; `manual/tools/COORD_NOTES.md`; `analysis/JET3D_NOTES.md`.
- Статуси й числа — **тільки** з цих файлів або з повторного запуску коду. Нічого не вигадувати. Якщо реєстри суперечать один одному — вказати, не «виправляти» мовчки.

## Формат (LaTeX, XeLaTeX)
- Файл розділу: `guide/chapters/gNN.tex`, починається з `\chapter{…}\label{ch:gNN}`. Мітки: `sec:gNN-…`, `fig:gNN-…`, `eq:gNN-…`.
- Преамбула = преамбула методички + `listings`. Позначення — `guide/notation.sty` (це посилання на `manual/notation.sty`): використовуй ті самі макроси. Десяткова кома, СІ, українська (правопис 2019), англійський термін у дужках при першій появі.
- **Код:** лише командою `\code{<шлях від guide/src>}{<перший рядок>}{<останній рядок>}{<підпис>}`, наприклад `\code{ourtry/04-novelty/gs_residual_probe.py}{38}{46}{(оператор $\GSop$)}`. Префікси: `ourtry/` → `Our try/`, `viz/` → `tokamak-3d-viz/`, `gs/` → `Tokamak-GS-solver/`, `fec/` → `fusion equilibrium challenge/`, `manualcode/` → `manual/code/`. Код НЕ копіювати в .tex — лише через `\code`, щоб він завжди збігався з файлом. Фрагмент 8–30 рядків; після кожного — пояснення по блоках (рядок N–M: що робить і навіщо).
- Рамки (визначені в преамбулі методички): `established{ID}`, `conjecture{ID}`, `refuted{ID}`, `practicum[назва]`, `derivation[тема]`, `outofcorpus[тема]`; для короткої картки пункту — `itemcard{ID}{статус}`.
- Посилання на методичку: «\cite{Manual}, розд. N» (номери розділів методички з 2026-09-29, після вставки розд. 5 про JET: 1 пастка, 2 частинки, 3 рівновага, 4 геометрія, 5 анатомія JET, 6 перенесення, 7 потоки, 8 стійкість, 9 стінка, 10 зриви, 11 нагрів, 12 моделювання; до цього 5–11 були на одиницю меншими).
- Бібліографія: проєктні ключі вже є в `guide/bib/refs_project.tex` (RegEst, RegConj, RegDec, RegQ, RegLog, RegVerif, RegViz, Manual, FEC). Зовнішні джерела (Escande & Momo 2024, Ham & Farrell 2024, Pentland 2025, McClenaghan 2024, Burby–Tang–Maulik 2021, Ding et al. arXiv:2511.19114, Prigozhin 1996, Bartolucci 2021, Balescu–Vlad–Spineanu 1998, Lao 1985 тощо) — бери дані з `Our try/99-bibliography/refs.bib` і verification-log; додавай у **свій** файл `guide/bib/refs_gNN.tex` (`\bibitem{ключ} …` у стилі ДСТУ-подібному, як у методичці). Ключ унікальний: перед додаванням `grep -h '\\bibitem' guide/bib/refs_*.tex`.
- Рисунки: лише власні графіки проєкту (наприклад `Our try/03-deep-dives/*/*.png`, `tokamak-3d-viz/data/analysis/*.png`) — копіюй у `guide/figures/` (стискай `sips -Z 1600 -s format jpeg`), підпис «Власний розрахунок, код: …». Жодних рисунків зі статей.
- Обсяг: орієнтовно 3–5 с. на розділ; покажчик — 4–6 с.

## Перевірка перед здачею
1. `cd guide && latexmk -xelatex -interaction=nonstopmode -outdir=build/<твій_id> main.tex` (НЕ пиши main.pdf у корені — паралельні автори) — без помилок; перевір свої сторінки візуально (`pdftoppm` + Read).
2. `python3 tools/check_code_refs.py chapters/gNN.tex` — усі `\code` вказують на існуючі файли й діапазони; кожен діапазон починається з потрібної функції/класу (скрипт показує перший рядок — переконайся очима).
3. `python3 ../manual/tools/lint_notation.py chapters/gNN.tex` — 0 порушень.
4. Кожне число звірене з файлом/запуском; кожен статус — з узгодженим реєстром.

## Звіт координатору
Сторінки, список пунктів (ID), покритих розділом, фрагменти коду (файл:рядки), перевірені числа (звідки), розбіжності/сумніви.
Не редагуй інші файли, крім своїх `chapters/gNN.tex`, `bib/refs_gNN.tex`, нових файлів у `figures/` (з префіксом `gNN_`).


## ОНОВЛЕННЯ (координатор): код тепер через minted
- `\code` переписано на `minted` (listings під XeLaTeX зсував нумерацію при не-ASCII над фрагментом). Синтаксис `\code{шлях}{a}{b}{підпис}` не змінився.
- Збірка: `cd guide && latexmk -outdir=build/<id> main.tex` — `latexmkrc` уже вмикає `-shell-escape` і PATH до Pygments; minted підставляє код на другому проході (latexmk робить це сам).
- Не використовуй `lstlisting`/`\lstinline` — лише `\code` або `\mintinline{python}{...}` для коротких вставок.
