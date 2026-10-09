# E. Прогноз зриву: чому ми туди не йдемо / Why we are not doing disruption prediction

Одна сторінка, щоб рішення **D1** було задокументоване, а не забуте через місяць.

---

## Контекст

Прикріплений `tokamak-hackathon-protocol.md` і `TODO-simulator-baseline.md` описують
повноцінний хакатон із раннього попередження зриву: основна задача з метрикою F2,
сім додаткових завдань Д1–Д7, оцінювальний скрипт, референсні моделі. Робота якісна.
Ми її **не використовуємо як задачу** з трьох причин.

## Причина 1: задача насичена

| Доказ | Значення |
|---|---|
| CCNN у DisruptionBench `[S]` | **AUC 0,974** на Alcator C-Mod (intra-machine) |
| Об'єднане навчання на EAST `[S]` | AUC > 0,95 |
| Перенесення DIII-D → JET `[S]` | AUC > 0,9 |

Залишковий розрив у полі — **операційний**: контроль частоти хибних тривог за фіксованого
часу попередження, а не зростання AUC. Це вимагає доступу до реальної машини й до
операційних вимог, а не кращої моделі. **Студентська команда за 48 годин цього не зрушить.**

## Причина 2: даних немає

| Джерело | Стан |
|---|---|
| **DisruptionBench** `[V]` | **Харнес без даних.** Архівовано 2025-07-21, read-only. Ви маєте подати власні оцінки розривності словником |
| C-Mod / DIII-D / EAST (~30 тис. розрядів) | не випущені |
| **JET** `[V]` | «не можуть бути публічними через юридичні обмеження» |
| **KSTAR** `[V]` | публічного датасету не знайдено |
| Дані IAEA-челенджу 2023 `[V]` | потрібна підписана угода про передачу даних |

Тобто студенти опинилися б на іграшкових даних — або на наших власних синтетичних,
результати за якими стосуються **нашої моделі, а не фізики**.

## Причина 3: симулятора в нас немає

`TODO-simulator-baseline.md` написано так, ніби репозиторій `fusion-sim/` із крейтом
`tok-sym-core` доступний. **Його немає в проєкті.** Крім того:

- README симулятора самосуперечливий щодо ліцензії: водночас «MIT License» і
  «Copyright 2026 Daniel Burgess. All rights reserved» (питання **Q10**);
- сам TODO оцінює доведення движка до придатного стану у **2,5–3 тижні однієї людини** —
  це весь наш горизонт;
- TODO містить нерозв'язані фізичні проблеми, які прямо загрожують якості задачі:
  фоновий ризик ≈ 0,005 с⁻¹ дає ~10% зривів «з нізвідки» на 20-секундному розряді
  (розділ 4.1), а завжди присутній прекурсор на `locked_mode` робить тривіальний поріг
  майже оптимальним (розділ 4.2). Тобто задача ризикує бути **або випадковою, або тривіальною**.

## Що ми звідти беремо

Протокол не марний. Ми **перевикористовуємо**:

1. **Вступний розділ «Як працює токамак»** — він написаний добре і саме для аудиторії
   без попередніх знань із фізики плазми. Піде у вступ нашого концепту майже як є.
2. **Наратив конвеєра** рівновага → стійкість → зрив — як пояснення, **навіщо** взагалі
   потрібна реконструкція ψ. Рішення **D2**: наратив так, код ні.
3. **Структуру номінацій** (Д1–Д7 + власне завдання) — як шаблон організації
   дослідницького треку.
4. **Ідею оцінювального скрипта з перевірками коректності подання** (монотонний час,
   score ∈ [0,1], покриття всіх розрядів) — це гігієна, яку варто перенести.
5. **Розділ про надпровідність (Д7)** — як вихідну точку для `D-superconductivity.md`,
   з істотною переробкою (див. той файл).

## Умова перегляду

Це рішення переглядається, якщо з'явиться **відкритий датасет зривів із мітками
*t*_ТГ і причинами**, доступний без угоди про передачу даних.
Найімовірніший кандидат — випуск EFIT-AI або розширення FAIR-MAST.

---

## English summary

The attached disruption-prediction protocol is good work, but we are not using it as a task.

**It is saturated**: CCNN reaches AUC 0.974 intra-machine on C-Mod, >0.95 on EAST, >0.9 for
DIII-D→JET transfer. The remaining gap is operational — false-alarm rate at fixed warning time —
which needs machine access and operational requirements, not a better model.

**The data does not exist for us**: DisruptionBench is an archived scoring harness that **ships no
data**; the C-Mod/DIII-D/EAST corpus is unreleased; JET is legally restricted; KSTAR has no public
dataset; the 2023 IAEA challenge required a signed data-sharing agreement.

**We have no simulator**: the TODO assumes a `fusion-sim`/`tok-sym-core` repository that is not in
the project; the upstream README is self-contradictory on licensing; the TODO itself budgets
2.5–3 person-weeks just to make the engine usable — our entire horizon; and it documents unresolved
physics that threatens task quality (a background risk floor producing ~10% causeless disruptions,
and an always-present locked-mode precursor that would make a trivial threshold near-optimal).

**What we do reuse**: the excellent zero-prerequisites tokamak primer; the
equilibrium → stability → disruption pipeline as *narrative* justification for why ψ matters
(decision D2: narrative yes, code no); the nominations structure as a template for the research
track; the submission-validation hygiene from the scoring script; and the superconductivity section
as the starting point for `D-superconductivity.md`.

**Revisit condition**: an openly downloadable disruption dataset with *t*_TQ labels and causes,
requiring no data-sharing agreement — most plausibly an EFIT-AI release or a FAIR-MAST extension.
