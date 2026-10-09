# Другий бекенд збурення: поле реальних котушок

`src/tokviz/coilfield.py` · `src/tokviz/rmp_coils.py`

Після спростування VC1 (обвідна ψ_N^(m/2) не описує профіль реального масиву)
стенд отримав **другий, паралельний** бекенд. Аналітичний лишився без змін — усі
попередні результати спираються на нього і відтворюються (регресія 13/13).

---

## 1. Чому не можна було просто тримати B на сітці

Перша реалізація обчислювала B за Біо–Савáром на сітці, розкладала за
тороїдальними гармоніками й інтерполювала сплайнами. Виміряна відносна `|∇·B|`:

| | ядро (ψ_N < 0,5) | край (ψ_N < 0,95) |
|---|---|---|
| сирий Біо–Савар | 5·10⁻⁷ | 1,5·10⁻⁷ |
| сітка 65, 5 гармонік | 4,4·10⁻⁴ | **2,12** |
| сітка 129, 8 гармонік | 4,4·10⁻⁵ | 4,1·10⁻³ |

На краю дивергенція **перевищувала сам масштаб поля**. Причина — провідники
стоять усередині розрахункової області, і біля них поле майже сингулярне.

**Найнебезпечніше тут не величина помилки, а її вигляд.** Поле, що не зберігає
потік, збиває ψ уздовж лінії, руйнує збіжність Біркгофа — і класифікатор чесно
доповідає «хаос». Я отримав 19 хаотичних орбіт із 34 уже на 3 кА і жодної
захопленої смуги. Це виглядало як переконлива фізика дискретного масиву з
бічними смугами. Насправді це була помилка представлення.

Це той самий клас провалу, що й у R8 батьківського реєстру: втрата збереження
потоку через підстановку замість структурної побудови. Різниця лише в тому, що
там вона проявилась як `det J = 1,7`, а тут маскувалася під фізику.

## 2. Виправлення: тримати векторний потенціал

Поле зберігається як **A**, розкладене за тороїдальними гармоніками й
сплайноване в (R, Z); **B береться як аналітичний ротор**:

    B_R   = (1/R) ∂A_Z/∂φ − ∂A_φ/∂Z
    B_φ   = ∂A_R/∂Z − ∂A_Z/∂R
    B_Z   = A_φ/R + ∂A_φ/∂R − (1/R) ∂A_R/∂φ

Мішані похідні сплайнів комутують точно, тож `∇·B = ∇·(∇×A) = 0` **структурно**,
незалежно від грубості сітки — рівно з тієї самої причини, з якої аналітичний
бекенд пише своє поле як `curl(ψ ∇φ)`.

| | тримаючи B | тримаючи A |
|---|---|---|
| відносна `\|∇·B\|` | 3,19·10⁻² | **2,50·10⁻⁶** |

Перевірки (у `tests/test_physics.py`):

* `curl A` проти Біо–Савара: **2,6·10⁻¹¹**
* Біо–Савар проти аналітичного поля кругової петлі: **до 4 знаків**
* реконструкція поля проти сирого Біо–Савара всередині плазми: медіана **0,1 %**

## 3. Фізична різниця між бекендами

| | аналітичний | котушки |
|---|---|---|
| δB_φ | **тотожно нуль** (збурення — функція потоку) | є |
| тороїдальний спектр | одне n на моду | гребінка n = 1, 5, 7, 11, 13 |
| полоїдальний спектр | одне m на моду | широкий |
| вартість виклику | ~9 сплайнів | ~96 сплайнів |
| побудова | миттєва | ~85 с |

δB_φ не косметичний: він входить у знаменник рівнянь силової лінії
`dR/dφ = R B_R/B_φ`, і аналітичний бекенд не може його відтворити за жодної
амплітуди. Заради цього `FieldLines` переписано так, щоб працювати з **B**, а не
лише з похідними ψ; аналітичний шлях при цьому дає тотожно ті самі рівняння.

Гребінка бічних смуг реальна: масив із 6 котушок, що жене n = 1, дає n = 1 ± 6k.
Виміряна потужність у векторному потенціалі: n=1 : 1,000; n=5 : 0,126;
n=7 : 0,041; n=11 : 0,029.

## 4. Що НЕ вийшло — відкрито

**Бекенд котушок дає набагато більше хаосу за аналітичний і не показує островів.**

| струм (лише n = 1) | δB/B_pol | регулярні | острівні | хаотичні |
|---|---|---|---|---|
| 3·10² А | 1,8·10⁻⁴ | 20 | 0 | 10 |
| 1·10³ А | 5,9·10⁻⁴ | 18 | 0 | 12 |
| 3·10³ А | 1,8·10⁻³ | 17 | 0 | 13 |

Частка хаосу **майже не росте з амплітудою** — 33 % → 43 % при десятикратному
зростанні. Це підлога, а не відгук на збурення. Аналітичний бекенд при
δB/B_pol ~ 10⁻³ дає одиниці відсотків хаосу і чіткі захоплені смуги.

**Мою гіпотезу про причину спростовано прямим тестом.** Я припустив, що хаос
породжує гребінка бічних смуг. Залишив лише n = 1: 21 хаотичних із 30. Повна
гребінка: 26 із 30. Смуги погіршують, але не є причиною.

Дві правдоподібні причини лишаються нерозділеними:

1. **Фізична.** Поле котушок має широкий полоїдальний спектр, тож жене
   **всі** резонанси n = 1 (q = 1, 2, 3, 4, 5) одночасно, а не два. Пʼять
   ланцюгів, що перекриваються, руйнують острівні ядра при набагато меншому
   δB/B, ніж два. Це узгоджується з тим, що реальні RMP-експерименти бачать
   стохастизацію краю при помірних струмах.
2. **Числова.** Залишкова похибка реконструкції поля (медіана 0,1 %, максимум
   9 % біля провідників) широкосмугова і діє як шум на кожному кроці, що
   занижує збіжність Біркгофа й читається як хаос.

**Поки це не розділено, бекенд котушок придатний для геометрії силових ліній і
для порівняння профілів, але НЕ для статистики островів і хаосу.** Для
статистики лишається аналітичний бекенд.

Що розділило б: прогін із суттєво дрібнішою сіткою й більшим числом гармонік —
якщо підлога хаосу впаде, причина числова; якщо ні, фізична.

---

## English summary

After VC1 was refuted, the stand gained a **second, parallel** perturbation
backend driven by a real coil array. The analytic backend is untouched and all
earlier results still reproduce (13/13 regression).

**Holding B on a grid does not work.** With the conductors inside the domain, a
65×65 grid with five toroidal harmonics gave a relative `|div B|` of 2.12 at the
plasma edge — larger than the field itself. The danger is not the size of the
error but its appearance: a field that does not preserve flux breaks psi
conservation along a line, collapses the weighted-Birkhoff convergence, and the
classifier faithfully reports chaos. It looked like convincing physics.

**The fix is structural:** hold the VECTOR POTENTIAL and take B as its analytic
curl. Spline mixed partials commute, so `div(curl A) = 0` to roundoff whatever
the grid. Relative `|div B|` went from 3.19e-2 to **2.50e-6**. Validated: curl A
against Biot-Savart to 2.6e-11, Biot-Savart against the analytic circular loop
to four digits, field reconstruction to 0.1% median inside the plasma.

**The backends differ physically, not cosmetically.** The analytic perturbation
is a flux function, so its `dB_phi` is identically zero; a coil array's is not,
and `dB_phi` enters the denominator of `dR/dphi = R B_R/B_phi`. `FieldLines` was
rewritten to work through B for this reason, with the analytic path reducing to
exactly its previous equations.

**What did not work, and is left open:** the coil backend produces far more
chaos than the analytic one and shows no islands, with a chaotic fraction that
barely responds to amplitude (33% → 43% over a tenfold current increase). That
is a floor, not a response. My hypothesis that the sideband comb caused it was
refuted by direct test — n=1 alone still gives 21 chaotic orbits of 30. Two
candidate causes remain unseparated: the coil field's broad poloidal spectrum
driving all five n=1 resonances at once, or residual field-reconstruction error
acting as broadband noise. Until they are separated, the coil backend is usable
for field-line geometry and profile comparisons but **not** for island and chaos
statistics.
