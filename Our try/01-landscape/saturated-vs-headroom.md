# Де стеля, а де простір / Saturated vs. headroom

Станом на **2026-09-22**. Це найважливіший файл блоку `01-landscape/`: він визначає,
за що взагалі є сенс змагатися.

**Правило:** задача придатна для хакатону, якщо **найкращий опублікований результат помітно
нижчий за очевидну стелю**. Інакше студентська команда за 48 годин не зрушить нічого,
і таблиця лідерів перетвориться на лотерею з шуму.

---

## НАСИЧЕНЕ — сюди не йдемо

| Задача | Доказ насичення |
|---|---|
| **Прогноз зриву всередині машини** | CCNN дає **AUC 0,974** на Alcator C-Mod `[S]`; AUC > 0,95 на EAST за об'єднаного навчання; AUC > 0,9 на JET із моделей, навчених на DIII-D. Залишковий розрив — **операційний** (частота хибних тривог за фіксованого часу попередження), а не за AUC. Студентська команда цього не зрушить. Додатково: дані закриті (див. `datasets.md`, Tier C) |
| **Реконструкція ψ всередині DIII-D** | Прогін starter kit уже дає **S = 0,8143** на 20 відкладених розрядах `[V]`; шлюз Challenge 2 виставлено на **0,85**, тобто організатори очікують, що це долається рутинно. R²ψ — поблажлива метрика на гладких полях: **PCA+Ridge уже дає SSIM 0,84** `[V]` |
| **1D-задачі PDEBench** (адвекція, Бюргерс, дифузія-реакція) | довго оптимізовані бейзлайни FNO/U-Net; приріст інкрементальний `[S]` |
| **Сурогати аеропрофілю в стилі ML4CFD** | переможець **уже перевершив еталонний OpenFOAM** за сукупними метриками `[S]` |

> **Наш власний доказ на цю тему.** У `fusion equilibrium challenge/SCORE.md` зафіксовано:
> `experiments.py` дає Ridge R²ψ = 0,396 на внутрішньому розбитті, але **на 8 справді інших
> розрядах та сама модель отримує 0,064**. Внутрішнє розбиття лестить кожній моделі.
> Це наш власний, відтворений результат — і він точно ілюструє, чому «насичене» треба
> міряти на чесному розбитті за розрядами.

---

## Є ПРОСТІР — кандидати для хакатону

Впорядковано за силою доказу, що запас реальний.

| Напрямок | Доказ запасу | Чи наше? |
|---|---|---|
| **Перенесення між машинами (zero-shot), DIII-D → MAST** | Наївне перенесення: **SSIM 0,83 → 0,10** `[V]`. Ціль `G_ratio ≈ 1,0`; ніхто близько не підійшов. **Флагманська невирішена задача ніші** | Так, але це задача живого змагання — беремо як *мотивацію*, не як нашу оцінювану задачу |
| **Фізична узгодженість (GS-резидуал)** | Ding et al. `[V]`: суто дані дають **0,25% L²-похибки за великих фізичних резидуалів**. Мала похибка розв'язку **не тягне** малий резидуал. Жоден бенчмарк цього не штрафує | **Так — це наша вісь №1** (див. `04-novelty/`) |
| **Калібрування невизначеності** | Емпіричне покриття номінальних інтервалів у фузійній ML-літературі практично не публікують `[S]`. EFIT-Prime розділяє алеаторну/епістемічну, але каліброваності не доводить | **Так — наша вісь №2** |
| **Обумовленість ψ → похідні скаляри** | LCFS — лінія рівня через сідлову точку, де ∇ψ = 0. Відображення необмежене в L². Ніхто не міряв показники підсилення | **Так — глибокий розбір D1** |
| **Неєдиність розв'язку GS** | Ham–Farrell 2024 `[V]`, Pentland et al. 2025 `[V]` — множинні рівноваги з **однаковими магнітними вимірами**. Уся ML-література припускає однозначність | **Так — глибокий розбір D2**, найвища новизна і найвищий ризик |
| **Темпоральний зсув / зсув кампанії** | TokaMark: темпоральне розбиття дає **NRMSE > 3,0** для м'якого рентгену `[V]`. Майже ніхто не звітує темпоральні розбиття | Резерв — гарна номінація |
| **Робастність до пропусків і відмов сенсорів** | ~38–55% пропусків (рівновага) і ~46% (Томсон) у даних MAST `[V]`. SOTA — **одноосібна стаття** (arXiv:2607.11915) `[V]`. Поле тонке | Так, дешева номінація |
| **Compute-light** | «Бейдж» Sophelio (<2 год на одному GPU/CPU) — **самодекларований і неоцінюваний** `[V]`. Ніхто не робить ефективність повноцінною метрикою | Так, і лягає на кластер Mac mini |
| **Регуляризація зроблена правильно** | L-крива / GCV / принцип Морозова у фузійній спільноті **не практикуються** `[S]`. Порядок базису p′/FF′ — це незадекларований параметр регуляризації | Так — чиста обчислювальна математика |
| **Закони подібності з чесною валідацією** | FusionFlux `[V]`: random forest б'є степеневий закон на **29% за групованої CV**, але **програє на всіх 13 мітках бази і всіх 11 установках**. Протокол валідації сам є відкритою задачею — на крихітних табличних даних | Резерв, чудова педагогічна пастка |

---

## Висновок для концепту

Три осі, де ми одночасно **(а)** маємо доказ незайнятості, **(б)** маємо код, і
**(в)** граємо на сильні сторони фізмату:

1. **GS-резидуал як оцінювана вісь** — бо ніхто не штрафує за порушення рівняння,
   а `Tokamak-GS-solver/src/numerical/compute.py` уже містить оператор Δ* і функції Гріна.
2. **Калібрування UQ** — бо ніша порожня, а математика (конформні передбачення,
   гаусові апостеріори) саме наша.
3. **Обумовленість і неєдиність** — бо це два питання, на які фізмат відповідає
   краще за ML-спільноту, і обидва дають публікабельний результат.

Перенесення між машинами лишається **найсильнішою мотивацією у вступі**, але
оцінюваною задачею його не робимо: це предмет живого змагання.

---

## English summary

**Rule:** a task is hackathon-worthy only if the best published result sits visibly below the
obvious ceiling.

**Saturated — avoid:** intra-machine disruption prediction (CCNN AUC 0.974; remaining gap is
operational false-alarm rate, not AUC — and the data is closed); intra-machine ψ reconstruction on
DIII-D (starter kit already at S = 0.8143, Challenge-2 gate set at 0.85, PCA+Ridge already at
SSIM 0.84); PDEBench 1D tasks; ML4CFD-style airfoil surrogates (winner already beat OpenFOAM).

Our own reproduced evidence reinforces this: in `SCORE.md`, Ridge scores R²ψ = 0.396 on an
internal split but **0.064 on 8 genuinely unseen discharges** — internal splits flatter every model.

**Real headroom:** cross-machine zero-shot transfer (SSIM 0.83 → 0.10); **physics consistency via
the GS residual** (Ding et al.: 0.25% L² error alongside large physics residuals — small solution
error does not imply small residual, and no benchmark penalises it); **UQ calibration** (empirical
coverage essentially never reported); **ψ → derived-scalar conditioning** (the LCFS is a level set
through a saddle point where ∇ψ = 0, so the map is unbounded in L²); **GS non-uniqueness**
(multiple equilibria with identical magnetic measurements — the entire ML literature assumes a
single-valued map); temporal/campaign shift; sensor-dropout robustness; compute-light efficiency
as a first-class scored axis; regularisation-parameter selection done properly.

**Concept conclusion:** our three axes are the **GS residual**, **UQ calibration**, and
**conditioning/non-uniqueness** — each with evidence of being unoccupied, existing code to build
on, and a natural fit to a Physics & Mathematics faculty. Cross-machine transfer stays as the
motivating narrative, not as our scored task, since it belongs to a live competition.
