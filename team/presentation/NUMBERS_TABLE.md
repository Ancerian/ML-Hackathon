# Таблиця всіх числових значень проєкту (Numbers Cheat Sheet)

**Лабораторія:** AI-лабораторія ім. В. М. Горшкова, Фізико-математичний факультет КПІ ім. Ігоря Сікорського  
**Проєкт:** ML-Hackathon — Магнітна рівновага токамака і динаміка силових ліній  
**Гілка:** `presentation-prep` | **Дата:** 2026-10-10  
**Призначення:** Єдине джерело правди для всіх чисел, які озвучуються на захисті, наводяться на слайдах або в документах. Жодне число не береться з голови.

---

## 1. Головні метрики моделей на тестовому спліті (8 розрядів #060–#067, 1521 кадрів)

> **Походження моделей:** Усі 4 порівнювані моделі (`PCA+Ridge`, `Linear Regression`, `MLP (sklearn)`, `UNet_Lite`) є стандартними архітектурами бейзлайнів зі стартер-коду організаторів (`experiments.py`, `experiments_torch.py`). Власної моделі, що обходить бейзлайни за $S$, не заявляємо. Нейромережеві бейзлайни вищі за PCA+Ridge на 60–67, але розкид по розрядах великий; $S$ не відрізняє фізичність.

| Параметр / Величина | Точне числове значення | 95% Bootstrap CI | Контекст у доповіді | Файл-джерело | Команда або скрипт відтворення |
|:---|:---:|:---:|:---|:---|:---|
| **$S$ (UNet_Lite)** | **0.6476** | [0.6289 .. 0.6671] | Ранг 1 за офіційною метрикою $S$ (60–67) | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$S$ (MLP sklearn)** | **0.6437** | [0.6239 .. 0.6622] | Ранг 2 за офіційною метрикою $S$ (60–67) | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$S$ (PCA+Ridge)** | **0.1926** | [0.1053 .. 0.5985] | Ранг 3 за $S$ на 60–67 (провал на #061, #063) | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$S$ (Linear Regr)** | **0.1924** | [0.1052 .. 0.5982] | Ранг 4 за офіційною метрикою $S$ (60–67) | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$\Delta S$ (UNet – MLP)** (60–67) | **-0.0015** | **[-0.0121 .. +0.0089]** | **0 в CI: UNet та MLP нерозрізненні за $S$** | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$\Delta S$ (UNet – PCA)** (60–67) | **+0.2662** | **[+0.1457 .. +0.3920]** | Перевага CNN за $S$ на 60–67 через викиди #061, #063 | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$\Delta S$ (MLP – PCA)** (60–67) | **+0.2677** | **[+0.1540 .. +0.3869]** | Перевага MLP за $S$ на 60–67 | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$\Delta S$ (UNet – PCA)** (36–39) | **+0.0419** | **[+0.0030 .. +0.0922]** | На 36–39 різниця невелика ($0.713$ vs $0.738$) | [`team/notes/TEST_SNOOPING.md`](../notes/TEST_SNOOPING.md) | `"team/scripts/eval_snooping_audit.py"` |
| **$\Delta S$ (MLP – PCA)** (36–39) | **+0.0489** | **[+0.0071 .. +0.1117]** | На 36–39 бали близькі ($0.740$ vs $0.738$) | [`team/notes/TEST_SNOOPING.md`](../notes/TEST_SNOOPING.md) | `"team/scripts/eval_snooping_audit.py"` |
| **$S'\text{-gate}$ (UNet_Lite)** | **0.4202** | [0.3719 .. 0.4710] | Ранг 2 за діагностичним шлюзом GS | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$S'\text{-gate}$ (MLP sklearn)** | **0.5331** | [0.4003 .. 0.6360] | Ранг 1 за діагностичним шлюзом GS | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$S'\text{-gate}$ (PCA+Ridge)** | **0.3614** | [0.2290 .. 0.4910] | Ранг 3 за діагностичним шлюзом | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$S'\text{-gate}$ (Linear Regr)** | **0.3609** | [0.2288 .. 0.4906] | Ранг 4 за діагностичним шлюзом | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$\Delta S'\text{-gate}$ (UNet – MLP)** | **-0.1129** | **[-0.2044 .. -0.0006]** | 0 не в CI: інверсія рангів через штраф за $g$ | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$\Delta S'\text{-gate}$ (UNet – PCA)** (60–67) | **+0.0588** | **[-0.0471 .. +0.1608]** | **0 в CI: за шлюзом GS UNet і PCA нерозрізненні** | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$\Delta S'\text{-gate}$ (UNet – PCA)** (36–39) | **-0.1884** | **[-0.2335 .. -0.1181]** | **0 не в CI: на 36–39 UNet гірший за PCA за $S'$** | [`team/notes/TEST_SNOOPING.md`](../notes/TEST_SNOOPING.md) | `"team/scripts/eval_snooping_audit.py"` |

---

## 2. Фізичні компоненти та нев'язки Ґреда–Шафранова

| Параметр / Величина | Точне числове значення | Контекст у доповіді | Файл-джерело | Команда або скрипт відтворення |
|:---|:---:|:---|:---|:---|
| **$g(\psi)$ (UNet_Lite)** | **0.8627** | Медіанна відносна нев'язка ГШ (високочастотний шум других похідних) | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$g(\psi)$ (MLP)** | **0.5027** | Медіанна відносна нев'язка ГШ (гладкість завдяки PCA-базису) | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$g(\psi)$ (PCA+Ridge)** | **0.0331** | Вкрай низька нев'язка через сильну лінійну фільтрацію | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$g(\psi)$ (Linear Regr)** | **0.0332** | Вкрай низька нев'язка | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$g_{\mathrm{ref}}$ (Опорний поріг)** | **0.6328** (розряди 0–35) vs **0.650** (E6) | Нев'язка карти $\psi$ з 1% синтетичним білим шумом ($\sigma = 0.01 \cdot \mathrm{std}(\psi)$). За README організаторів (§5.3, E6) істина EFIT дає $g=0.009$, а та сама карта з 1% білого шуму — $g=0.650$ (стрибок у 70 разів). Наше значення $g_{\mathrm{ref}} = 0.6328$ обчислено як медіана за розрядами #000–#035 із тим самим 1% шумом. Розбіжність $0.6328$ vs $0.650$ зумовлена вибіркою кадрів або статистичною реалізацією генератора шуму [?] (обидва числа узгоджені на рівні $0.63\text{--}0.65$). | [`Our try/04-novelty/c4/gref.json`](../../Our%20try/04-novelty/c4/gref.json) | `"fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/c4/gref.py"` |
| **$g_{\mathrm{clean}}$ (Чистий EFIT)** | **0.009** (E6) / **0.0097** (тест C4) | Чисельна нев'язка ідеальної реконструкції EFIT без шуму | [`Our try/04-novelty/c4/gref.json`](../../Our%20try/04-novelty/c4/gref.json) | `"fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/c4/gref.py"` |
| **$\mathcal{G}_{\mathrm{GS}}$ (UNet_Lite)** | **0.7213** | Мультиплікативний гейт за нев'язку (штраф $28\%$) | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$\mathcal{G}_{\mathrm{GS}}$ (MLP)** | **0.9614** | Мультиплікативний гейт за нев'язку (штраф лише $4\%$) | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$\mathcal{P}_{\mathrm{topo}}$ (UNet_Lite)** | **0.9217** | Множник топологічної чистоти магнітних осей | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$\mathcal{P}_{\mathrm{topo}}$ (MLP)** | **0.8532** | Множник топологічної чистоти | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$R^2_\psi$ (UNet_Lite)** | **0.976** | Коефіцієнт детермінації карти потоку (гачок доповіді) | [`Our try/04-novelty/t1/results.json`](../../Our%20try/04-novelty/t1/results.json) | `"fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/t1/run_t1.py"` |

---

## 3. Результати дослідницьких задач новизни (T2 – T12)

| Задача | Досліджувана величина | Числове значення (Point / CI) | Статус / Вердикт | Файл-джерело | Команда відтворення |
|:---:|:---|:---:|:---|:---|:---|
| **T2** | Зниження $g$ регуляризацією ГШ ($\beta = 10^{-3}$) | **-76.9%** ($0.1589 \pm 0.0045$, $R^2_\psi = 0.9398$) | **ПІДТВЕРДЖЕНО** (Категорія A) | [`Our try/04-novelty/t2/results.json`](../../Our%20try/04-novelty/t2/results.json) | `"fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/t2/run_t2.py"` |
| **T3** | $\Delta\mathrm{Consistency}$ (Multi-Task, $\lambda=0.1$) | **-0.0203** (95% CI [-0.0902 .. +0.0703]) | **СПРОСТОВАНО** (Категорія A) | [`Our try/04-novelty/t3/results.json`](../../Our%20try/04-novelty/t3/results.json) | `"fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/t3/run_t3.py"` |
| **T3** | $\Delta\mathrm{Consistency}$ (Multi-Task, $\lambda=0.5$) | **-0.0702** (95% CI [-0.1496 .. +0.0943]) | **СПРОСТОВАНО** (Категорія A) | [`Our try/04-novelty/t3/results.json`](../../Our%20try/04-novelty/t3/results.json) | `"fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/t3/run_t3.py"` |
| **T4** | Валідні рівноваги на Брату (MDN проти MSE) | **98.4%** проти **2.8%** | **ПІДТВЕРДЖЕНО** (Категорія B) | [`Our try/04-novelty/t4/results.json`](../../Our%20try/04-novelty/t4/results.json) | `"fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/t4/run_t4.py"` |
| **T5** | Одночасне конформне покриття (номінал 90%) | **87.7%** ($\kappa = 0.220$) | **ПІДТВЕРДЖЕНО З УТОЧН.** (Категорія B) | [`Our try/04-novelty/t5/results.json`](../../Our%20try/04-novelty/t5/results.json) | `"fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/t5/run_t5.py"` |
| **T6** | Точність HénonNet vs RF у 2-модовому режимі | **65%** проти **0%** | **СПРОСТОВАНО** (Категорія B) | [`Our try/04-novelty/t6/results.json`](../../Our%20try/04-novelty/t6/results.json) | `"fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/t6/run_t6.py"` |
| **T7** | Співвідношення варіабельності $\mathrm{IQR}(g)_{\mathrm{Ritz}} / \mathrm{IQR}(g)_{\mathrm{PINN}}$ | **[13.5 .. 32.7]** (10 seeds) | **СПРОСТОВАНО** (Категорія A) | [`Our try/04-novelty/t7/results.json`](../../Our%20try/04-novelty/t7/results.json) | `"fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/t7/run_t7.py"` |
| **T8** | Кількість фіктивних критичних осей на кадр | **0.08** проти **1.81** (**-95.4%**) | **ПІДТВЕРДЖЕНО** (Категорія B) | [`Our try/04-novelty/t8/results.json`](../../Our%20try/04-novelty/t8/results.json) | `"fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/t8/run_t8.py"` |
| **T10** | Колапс покриття при відмові зондів ($Cov_M$) | **3.2%** проти **63.6%** | **СПРОСТОВАНО** (Категорія B) | [`Our try/04-novelty/t10/results.json`](../../Our%20try/04-novelty/t10/results.json) | `"fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/t10/run_t10.py"` |
| **T10** | Виграш аугментації за відмов датчиків | **+82.7%** $R^2$ | **ПІДТВЕРДЖЕНО** (Категорія B) | [`Our try/04-novelty/t10/results.json`](../../Our%20try/04-novelty/t10/results.json) | `"fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/t10/run_t10.py"` |
| **T11** | Збереження топології DIII-D → MAST ($1\mathrm{O}/0\mathrm{X}$) | **100%** ($R^2_\psi = \mathbf{0.6880}$ vs $\mathbf{-0.9960}$) | **ПІДТВЕРДЖЕНО** (Категорія B) | [`Our try/04-novelty/t11/results.json`](../../Our%20try/04-novelty/t11/results.json) | `"fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/t11/run_t11.py"` |
| **T12** | Збереження фазового об'єму інтегратора | $|\det J - 1| \le \mathbf{10^{-10}}$ | **ВИКОНАНО** (Категорія B) | [`Our try/04-novelty/t12/results.json`](../../Our%20try/04-novelty/t12/results.json) | `"fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/t12/run_t12.py"` |
| **T12** | Швидкість трасування силових ліній на JAX | **968.7** ліній/с | **ВИКОНАНО** (Категорія B) | [`Our try/04-novelty/t12/results.json`](../../Our%20try/04-novelty/t12/results.json) | `"fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/t12/run_t12.py"` |
| **T13** | Релаксація ГШ ($dt=2\cdot 10^{-4}, n=8$) | $g$ за оператором релаксації падає ($-55.5\%$, $0.8618 \to 0.3838$), за незалежною 5-точковою схемою майже не змінюється ($0.9802 \to 0.9329$, $-4.8\%$); приріст $S'\text{-gate}$ ($0.4202 \to 0.6195$) специфічний для оператора | **ПІДТВЕРДЖЕНО (оператор-залежно)** (Категорія A) | [`Our try/04-novelty/t13/results.json`](../../Our%20try/04-novelty/t13/results.json) | `"fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/t13/run_t13.py"` |

---

## 4. Верифікація конвеєра та характеристики датасету

| Параметр / Характеристика | Числове значення | Фізичний / технічний зміст | Файл-джерело | Команда відтворення |
|:---|:---:|:---|:---|:---|
| **Harness Self-Check `perfect`** | $S = 1.000000$, $S' = 1.000000$ | Похибка $< 10^{-9}$ на еталонному полі | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py --mode perfect` |
| **Harness Self-Check `zeros`** | $S = 0.000000$, $S' = 0.000000$ | Похибка $< 10^{-9}$ на нульовому полі | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py --mode zeros` |
| **Загальна кількість розрядів** | **9 121** розрядів | Повний відкритий датасет DIII-D | [`Our try/00-decisions/ESTABLISHED.md`](../../Our%20try/00-decisions/ESTABLISHED.md) ([E26]) | Відкритий каталог датасету |
| **Загальний обсяг даних** | **103.86 ГБ** | Обсяг вихідних HDF5/файлів розрядів | [`Our try/00-decisions/ESTABLISHED.md`](../../Our%20try/00-decisions/ESTABLISHED.md) ([E26]) | Статистика датасету |
| **Розмір тестового спліту** | **8** розрядів (#060–#067) | Публічний контрольний спліт хакатону | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | Каталог спліту |
| **Кількість кадрів тесту** | **1 521** кадрів | Сумарна кількість 2D кадрів у 8 розрядах | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `eval_submission.py` |
| **Репліки бутстрепу** | **1 000** реплік | Кількість ітерацій випадкового семплювання | [`team/LEADERBOARD.md`](../LEADERBOARD.md) | `eval_submission.py` |

---

## 5. Аудит Test Snooping (розряди #036–#039) та розкид PCA+Ridge

| Параметр / Характеристика | Числове значення | Фізичний / технічний зміст | Файл-джерело | Команда відтворення |
|:---|:---:|:---|:---|:---|
| **$g(\psi)$ UNet (36–39 vs 60–67)** | **0.8557** проти **0.8627** | Узгодженість нев'язки між сплітами (не чистий контроль: 36–39 є VAL) | [`team/notes/TEST_SNOOPING.md`](../notes/TEST_SNOOPING.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" team/scripts/eval_snooping_audit.py` |
| **$S$ UNet на 36–39** | **0.7130** [0.6239 .. 0.7491] | Офіційний бал на 4 валідаційних розрядах (1039 кадрів) | [`team/notes/TEST_SNOOPING.md`](../notes/TEST_SNOOPING.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" team/scripts/eval_snooping_audit.py` |
| **$S$ MLP на 36–39** | **0.7404** [0.6498 .. 0.7737] | Офіційний бал на 4 валідаційних розрядах | [`team/notes/TEST_SNOOPING.md`](../notes/TEST_SNOOPING.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" team/scripts/eval_snooping_audit.py` |
| **$S$ PCA+Ridge на 36–39** | **0.7384** [0.6113 .. 0.7753] | Офіційний бал PCA+Ridge на 36–39 (без атипових провалів) | [`team/notes/TEST_SNOOPING.md`](../notes/TEST_SNOOPING.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" team/scripts/eval_snooping_audit.py` |
| **$S$ PCA+Ridge на #061, #063** | **0.0836**, **0.0794** | Два провальні розряди тесту, що обвалюють спліт 60–67 до 0.1926 | [`team/notes/TEST_SNOOPING.md`](../notes/TEST_SNOOPING.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$S$ PCA+Ridge на #064, #067** | **0.5776**, **0.6330** | Високі розряди тесту, де PCA працює на рівні нелінійних мереж | [`team/notes/TEST_SNOOPING.md`](../notes/TEST_SNOOPING.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" eval_submission.py` |
| **$\Delta S$ на 36–39 (UNet – MLP)** | **-0.0071** [-0.0201 .. +0.0057] | 0 входить у CI: нерозрізненість за $S$ повторюється | [`team/notes/TEST_SNOOPING.md`](../notes/TEST_SNOOPING.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" team/scripts/eval_snooping_audit.py` |
| **$\Delta S'\text{-gate}$ на 36–39** | **-0.2195** [-0.2503 .. -0.1742] | 0 не входить у CI: інверсія на користь MLP зберігається | [`team/notes/TEST_SNOOPING.md`](../notes/TEST_SNOOPING.md) | `"fusion equilibrium challenge/starter/.venv/bin/python" team/scripts/eval_snooping_audit.py` |
