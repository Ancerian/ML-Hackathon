# Картки моделей та пакет публікації / Model Cards & Publication Package

**Платформи публікації:** Hugging Face Hub / Zenodo  
**Версія вихідного коду:** Git commit [`280e427`](https://github.com/Ancerian/ML-Hackathon/commit/280e427)  
**Репозиторій коду:** [github.com/Ancerian/ML-Hackathon](https://github.com/Ancerian/ML-Hackathon) (Ліцензія MIT)  
**Хеш спліту даних (C2 / SCORE.md Benchmark Split):** `8417b94e7b9dbfbb7c4112248e0115cd1d4a463f12f3e159fd359678be98c7b4`  
**Політика даних:** Сирі експериментальні дані **НЕ редистриб'ютуються**. Публікуються виключно навчені ваги моделей, конфігурації архітектур та верифіковані матриці оцінки, дозволені ліцензією CC BY 4.0.

---

## 1. Обов'язкова юридична атрибуція та подяка грантодавцям

### 1.1. Атрибуція датасету (Creative Commons Attribution 4.0 International)
Будь-яке використання моделей, навчених на даних токамака DIII-D у рамках даного пакету, вимагає наступного бібліографічного посилання:

```bibtex
@dataset{fusion_equilibrium_challenge_2026,
  title     = {The Fusion Equilibrium Challenge: Predicting Plasma Shape from Control Inputs},
  author    = {Michoski, Craig and Waller, Matthew and Sammuli, Brian and Boyes, William 
               and Clark, Mitchell and Smith, Sterling and Nakkina, Tapan Ganatma 
               and Hatch, David and Nazikian, Raffi},
  year      = {2026},
  publisher = {Hugging Face},
  note      = {Sophelio and General Atomics. Licensed under CC BY 4.0}
}
```

### 1.2. Обов'язкова подяка Міністерству енергетики США (DOE Acknowledgment)
Згідно з умовами доступу до експериментальних даних DIII-D National Fusion Facility:
> *"This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Fusion Energy Sciences, using the DIII-D National Fusion Facility, a DOE Office of Science user facility, under Award **DE-FC02-04ER54698**, and under Awards **DE-SC0024426**, **DE-SC0024499**, **DE-SC0024409**, and **DE-SC0024571**."*

### 1.3. Юридичний дисклеймер DOE (Mandatory Disclaimer)
> *"This report and model checkpoint package was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof."*

---

## 2. Специфікація чекпоінтів моделей

### Модель 1: `UNet_Lite_DIIID_C2`
- **Опис:** Полегшена 2D-згорткова U-Net архітектура для прямої реконструкції полоїдального магнітного потоку $\psi(R, Z)$ на сітці $65 \times 65$ за котушковими струмами та давачами.
- **Вхідні дані:** Вектор котушкових струмів та магнітних вимірювань ($d = 50$).
- **Вихід:** 2D-матриця потоку $\psi \in \mathbb{R}^{65 \times 65}$.
- **Метрики (на відкладених розрядах 60–67):**  
  $R^2_\psi = 0.9758$, $\mathrm{Consistency} = 0.1791$, $S = 0.6634$, медіанна нев'язка ГШ $g = 0.8565$.
- **Конформне покриття (Simultaneous 90%):** Емпіричне покриття $87.7\%$ (95% CI $[64.9\%, 97.7\%]$), відносна ширина $\kappa = 0.220$.
- **Ліцензія ваг:** CC BY 4.0.
- **Хеш тренувального спліту:** `8417b94e7b9dbfbb7c4112248e0115cd1d4a463f12f3e159fd359678be98c7b4`.

### Модель 2: `ConvDecoder_DIIID_C2`
- **Опис:** Повністю згортковий декодер із лінійним проєктором для синтезу профілю $\psi$.
- **Вхід:** Діагностичний вектор $x \in \mathbb{R}^{50}$.
- **Вихід:** Матриця $\psi \in \mathbb{R}^{65 \times 65}$.
- **Метрики:** $R^2_\psi = 0.9668$, $\mathrm{Consistency} = 0.1542$, $g = 0.8105$.
- **Конформне покриття:** Емпіричне покриття $79.6\%$, півширина $W = 0.0999$.
- **Ліцензія ваг:** CC BY 4.0.

### Модель 3: `FNO_GS_Reg_DIIID` (Задача T2)
- **Опис:** Фур'є-нейрооператор (Fourier Neural Operator, 4 spectral modes, channel width 32), навчений із диференційовною фізичною регуляризацією нев'язкою Град-Шафранова $\mathcal{L} = \mathcal{L}_{\mathrm{data}} + \beta \|\Delta^*\psi - \hat{J}_\phi\|^2$ при оптимальному $\beta = 10^{-3}$.
- **Метрики (Парето-оптимальна точка):**  
  $R^2_\psi = 0.9398 \pm 0.0130$ (проти $0.9381$ для $\beta = 0$), нев'язка $g = 0.1589 \pm 0.0045$ (**$-76.9\%$ зниження фізичної нев'язки** порівняно з нерегуляризованим оператором $g = 0.6877$).
- **Ліцензія ваг:** CC BY 4.0.
- **Скрипт відтворення:** `Our try/04-novelty/t2/run_t2.py`.

### Модель 4: `TopoReg_ConvDecoder_DIIID` (Задача T8)
- **Опис:** Згортковий декодер із топологічним регуляризатором просторового градієнта $\mathcal{L}_{\mathrm{topo}} = 0.20 \|\nabla\psi_{\mathrm{pred}} - \nabla\psi_{\mathrm{true}}\|^2$.
- **Метрики:**  
  Зниження середньої кількості фіктивних критичних точок $\nabla\psi = 0$ всередині плазми з $1.81$ до **$0.08$** на кадр (**$-95.4\%$**). Зростання частки канонічних кадрів (1 O-точка, 0 X-точок всередині LCFS) з $45.8\%$ до **$93.3\%$**. Покращення $R^2_\psi = 0.8806$ проти $0.8729$.
- **Ліцензія ваг:** CC BY 4.0.
- **Скрипт відтворення:** `Our try/04-novelty/t8/run_t8.py`.

### Модель 5: `MDN_Bratu_BranchResolver` (Задача T4)
- **Опис:** Mixture Density Network ($K=2$ компонент гаусової суміші) для апроксимації багатозначних операторів рівноваги у точках зворотної біфуркації (модель Братý 1D, $\lambda \le \lambda^* = 3.51383$).
- **Метрики:**  
  Частка фізично валідних розв'язків ($g < 0.10$): **$98.4\%$** (проти лише $2.8\%$ для стандартного $L^2$-регресора з нев'язкою $g = 1.0979$). Збалансоване покриття обох гілок розв'язку ($48.4\%$ нижня / $51.6\%$ верхня) без колапсу мод.
- **Ліцензія ваг:** MIT / CC BY 4.0 (чиста синтетика).
- **Скрипт відтворення:** `Our try/04-novelty/t4/run_t4.py`.

### Модель 6: `HenonNet_Tokamap_Inverse` (Задача T6)
- **Опис:** Симплектична інверсна нейромережа з циклічним кодуванням фаз $(\cos\phi, \sin\phi)$ для відновлення спектральних амплітуд і фаз збурень магнітного поля Токомапу за фазово-роздільною екскурсією.
- **Метрики:**  
  Час інференсу **$0.21$ мс** на екземпляр ($\times 23\,635$ швидше за диференціальну еволюцію). Відсоток відновлення: $95.0\%$ у 1-модовому режимі, $65.0\%$ у складному 2-модовому режимі $2/1 + 3/1$ (де класичний нелінійний МНК зазнає повного краху з $0.0\%$ успіху через склоподібність ландшафту).
- **Ліцензія ваг:** MIT (чиста симуляція Гамільтонової динаміки).
- **Скрипт відтворення:** `Our try/04-novelty/t6/run_t6.py`.

---

## 3. Обмеження та правила експлуатації (Intended & Unintended Use)

- **Призначення:**  
  Наукові дослідження в галузі магнітного утримання термоядерної плазми, порівняльний аналіз фізично інформованих нейромереж, верифікація UQ та топологічних інваріантів.
- **Заборонене використання (Unintended Use):**  
  Категорично забороняється використання даних моделей для безпосереднього зворотного зв'язку в реальному часі в системах керування положенням плазми або систем захисту від зривів (disruption mitigation) на діючих термоядерних установках без сертифікованого класичного перевірочного контуру (certifier, Zheng et al. 2025), оскільки навіть моделі з $R^2_\psi > 0.97$ можуть генерувати нефізичні дефекти градієнта та зміщення осі при OOD збуреннях.
