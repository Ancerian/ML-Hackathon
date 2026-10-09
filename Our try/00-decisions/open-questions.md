# Відкриті питання / Open questions

Питання, на які ми **не можемо** відповісти самі. Згруповано за адресатом.
Статус: `відкрите` / `в роботі` / `закрите`.

---

## До спеціалістів з фізики плазми (зустріч з деканом, консультанти)

| # | Питання | Навіщо | Статус |
|---|---|---|---|
| Q1 | Чи прийнятно для JET-подібної машини трактувати реконструкцію ψ **без магнітної діагностики** як реалістичний сценарій, а не штучну гру? | Це вся мотивація задачі. Аргумент Sophelio — нейтронна деградація сенсорів у SPARC/ARC. Треба, щоб наші консультанти це підтвердили або спростували | відкрите |
| Q2 | Наскільки сильно **виродження p′ vs FF′** псує реконструкцію без внутрішніх діагностик (MSE, поляриметрія)? Чи є в них досвід/числа? | Визначає, чи має сенс номінація «мінімальний набір діагностик» | відкрите |
| Q3 | Чи справді **неєдиність розв'язку GS** (Ham–Farrell 2024, Pentland 2025) зустрічається в робочих режимах, чи лише в екзотичних? | Від цього залежить, чи D2 — реальна задача, чи математичний курйоз | відкрите |
| Q4 | Реалістичні значення **сталої часу зв'язування τ** і **запасу ентальпії ΔH** для NbTi/Nb₃Sn у режимі ITER/JT-60SA | Без них розділ про надпровідність — вправа з вигаданими числами | відкрите |
| Q5 | Наскільки **вакуумна камера екранує** швидке dB/dt на котушках під час струмового гасіння? Порядок множника | Якщо екранування — фактор 10+, оцінки за Біном і зв'язуванням втрачають сенс | відкрите |

## До організаційної частини (декан, AI-лабораторія)

| # | Питання | Навіщо | Статус |
|---|---|---|---|
| Q6 | Чи є доступ до **кластера AI-лабораторії** (Mac mini, Apple Silicon) для учасників і для еталонних замірів? Скільки вузлів, яка конфігурація? | Номінація compute-light і будь-який замір затримки без цього неможливі | відкрите |
| Q7 | Точні **дати, місце, розмір команди, склад журі** | Усе це `[уточнюється]` в референсному протоколі й досі невідоме | відкрите |
| Q8 | Чи потрібна **платформа зі скорингом** (Codabench / власна), чи достатньо ручної перевірки подань? | Codabench вимагає підготовки бандла; ручна перевірка не масштабується за >10 команд | відкрите |
| Q9 | Хто **реально** виділяється на побудову starter kit і на скільки годин? | План на 8–9 днів аналізу + 2 тижні інфраструктури тримається на цій відповіді | відкрите |

## До зовнішніх сторін

| # | Питання | Адресат | Статус |
|---|---|---|---|
| Q10 | Умови поширення **даних, згенерованих `fusionsimulator.io`**. README репозиторію містить водночас «MIT License» і «Copyright 2026 Daniel Burgess. All rights reserved» | Daniel Burgess (Columbia Fusion Research Center) | відкрите — **не публікувати нічого похідного до відповіді** |
| ✔ Q11 | Ліцензія **ITPA HDB5** | **ЗАКРИТО: CC BY 4.0**, підтверджено через OSF API. Але поле `copyright_holders` порожнє — атрибуція через канонічну цитату Verdoolaege et al. (2021) | закрите |
| ✔ Q12 | Атрибуція для скорера й даних Sophelio | **ЗАКРИТО.** Код — справді MIT (перевірено локально; GitHub показує `NOASSERTION` через дописаний розділ «NOTE ON SCOPE»). Дані — CC BY 4.0. Формулювання атрибуції — у `01-landscape/datasets.md` | закрите |
| **Q13** | ⚠️ **Правова підстава переходу 1 206 розрядів MAST із CC BY-SA 4.0 (FAIR-MAST) у випуск CC BY 4.0 (Sophelio).** Share-alike поширюється на похідні бази через §4(b), тож перевипуск як простий CC BY **не дозволений самою BY-SA** | Sophelio та/або UKAEA | **відкрите — блокує перевипуск частини MAST.** До з'ясування трактуємо її як обтяжену BY-SA |

---

## Питання, які ми закрили самі

| # | Питання | Відповідь |
|---|---|---|
| ✔ | Чи є локально код симулятора `tok-sym-core`? | **Ні.** У папці лише протокол і TODO, написані так, ніби код є. Рішення: не спиратися на нього (D5) |
| ✔ | Чи насичений прогноз зриву? | **Так**, intra-machine AUC ≈ 0,974; і дані закриті. Деталі — `01-landscape/saturated-vs-headroom.md` |
| ✔ | Скільки скалярів у члені Consistency метрики Sophelio — 7 чи 8? | Локальний `fusion_scoring/common.py` каже **сім** (`R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li`); `dsep` прибрано у v3. Код надійніший за сайт |

---

## English summary

Open questions blocking parts of the work, by addressee:

- **Plasma physicists**: is magnetics-free reconstruction physically realistic (Q1); severity of the p′/FF′ degeneracy without internal diagnostics (Q2); whether GS non-uniqueness occurs in operational regimes (Q3); realistic τ and ΔH for NbTi/Nb₃Sn (Q4); vacuum-vessel shielding factor for fast dB/dt (Q5).
- **Organisers**: cluster access (Q6), dates/venue/jury (Q7), scoring platform (Q8), actual staffing (Q9).
- **External**: redistribution terms for `fusionsimulator.io`-generated data — the README is self-contradictory, so **nothing derived from it gets published until answered** (Q10); HDB5 licence (Q11); exact attribution wording for reusing the Sophelio scorer and data (Q12).

Already closed: no local simulator code; disruption prediction is saturated with closed data; the Consistency term averages **seven** scalars (per the scoring code, which supersedes the website).
