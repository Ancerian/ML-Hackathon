# tokamak-3d-viz

**Наукова 3D-візуалізація токамака реакторного класу з плазмою, керованою
реальними даними розряду DIII-D.**

Blender 5.2.1 LTS (з відмінностями для 4.2 LTS) · EEVEE + Cycles/Metal ·
Python 3.12 для фізики

Лабораторія ШІ імені В. М. Горшкова, ФМФ, КПІ ім. Ігоря Сікорського

---

## Що це

Відтворюваний конвеєр, який перетворює реальну реконструкцію рівноваги EFIT на
набір анотованих 4K-кадрів: зовнішній вигляд машини, розріз, плазма, **перетин
Пуанкаре з магнітними островами**, q-профіль і дивертор.

Плазма не вигадана. Її топологію взято з **розряду DIII-D 203702, кадр 75
(t = 1760 мс)**. Усі числа на кадрі читаються з `manifest.json`, який згенерував
конвеєр, тож зображення й документація не можуть розійтися.

Перевірки, які конвеєр проходить (виміряно, не заявлено):

| Перевірка | Результат |
|---|---|
| Магнітна вісь проти `efit_r_axis`/`efit_z_axis` | **ΔR = 0,0002 мм** (комірка сітки 26,6 мм) |
| Калібрування B_φ на осі | **1,927 Т** проти реальних 1,9–2,1 Т у DIII-D |
| q контурний проти q з трасування | збіг **0,001–0,16 %** |
| Дрейф ψ за 200 тороїдальних обертів | **1·10⁻⁷** |
| Острови | **2 на q = 2, 3 на q = 3** — як і мусить давати ланцюг m/n |

---

## Швидкий старт

```bash
# 0. залежності (фізика; Blender має власний Python)
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1. вкажіть, де лежать parquet розрядів (у репозиторії їх НЕМАЄ, див. ATTRIBUTION.md)
export TOKVIZ_DATA_DIR=/шлях/до/parquet_data

# 2. фізика: рівновага, калібрування F, трасування, острови, геометрія
python src/run_bake.py --out data/bake01

# 3. сцена + рендер (EEVEE для look-dev, Cycles для фіналу)
blender --background --python src/blender/build_scene.py -- \
        --bake data/bake01 --out tokamak.blend --render both

# 4. наукові анотації
python src/annotate.py --bake data/bake01 --render-dir render
```

Швидкий прохід для перевірки (без важкого трасування):

```bash
python src/run_bake.py --out data/bake01 --skip-trace
blender --background --python src/blender/build_scene.py -- \
        --bake data/bake01 --out /tmp/t.blend --render eevee \
        --res 960 540 --samples-eevee 24 --only-camera 02_cutaway
```

---

## Структура

```
docs/BLUEPRINT.md           ← повний виробничий план (укр. + English summary)
docs/PHYSICS.md             ← конвенції, викладки, валідація, застереження
docs/API-VERIFIED-5.2.md    ← перевірений API Blender і відмінності 4.2 → 5.x
findings.md                 ← виміряні результати у двореєстровому стилі проєкту

src/tokviz/                 ← фізика (numpy/scipy, поза Blender)
  equilibrium.py    ψ, поле, q-профіль, прямокутний кут θ*
  perturbation.py   резонансні моди, ширина острова, Чириков
  fieldline.py      трасування DOP853, перетин Пуанкаре
  islands.py        пошук O-/X-точок, зерна всередині островів
  surfaces.py       обертання профілів, котушки, D-переріз
  export.py         запис .npz + manifest.json
src/run_bake.py             ← драйвер конвеєра фізики

src/blender/tokblend/       ← побудова сцени (bpy)
  bl.py             сумісність 4.2/5.x, помічники
  materials.py      шейдери: об'ємна плазма, метали, поверхні потоку
  machine.py        камера, котушки, кріостат, дивертор, плитки, патрубки
  plasma.py         поверхні потоку, силові лінії, проколи
  cutaway.py        оснастка розрізу (GN / Boolean / шейдерне відсічення)
  cameras.py        шість кадрів, світло, допоміжні лампи для EEVEE
  render.py         налаштування рушіїв, рендер кадрів
  compositor.py     Glare (EEVEE втратив Bloom у 4.2)
src/blender/build_scene.py  ← точка входу для Blender
src/annotate.py             ← шкала ψ_N, q-профіль, підписи, застереження

tools/introspect_api.py     ← регенерує перевірений API-довідник
tests/test_physics.py       ← перевірки фізики
```

---

## Головні застереження

1. **В осесиметрії перетин Пуанкаре — це точно контури ψ.** Острови внесено
   **свідомо** як приписане збурення; вони не є властивістю реконструкції EFIT
   і їх не можна подавати як виміряні.
2. **Вакуумний перетин переоцінює стохастичність** — плазма екранує резонансні
   компоненти. Це друкується на кадрі.
3. **Чириков S ≥ 1 — евристика, не теорема.**
4. **Геометрія — реакторний композит**, а не ITER і не DIII-D у натуральну
   величину. Рівномірний масштаб s = 3,0 зберігає аспектне відношення, а отже
   q(ψ) і топологію.
5. **Кольорова шкала «гаряче ядро» — це конвенція.** Справжня фотографія у
   видимому світлі виглядає навпаки.

---

## English summary

A reproducible pipeline that turns a real EFIT equilibrium reconstruction into
an annotated 4K still set of a reactor-class tokamak: exterior, cutaway, plasma,
**Poincaré section with magnetic islands**, q-profile and divertor.

The plasma is not invented — its topology comes from **DIII-D discharge 203702,
frame 75 (t = 1760 ms)** — and every number on the image is read from the
`manifest.json` the pipeline wrote, so the render and the documentation cannot
drift apart.

Physics runs outside Blender (numpy/scipy/pyarrow/scikit-image) and bakes to
`.npz`; Blender 5.2 ships numpy but no scipy, so it only ever loads arrays.
Measured validation: the magnetic axis matches EFIT to 0.0002 mm; F(ψ),
calibrated against the dataset's own q95 because the dataset ships no `fpol`,
yields B_φ = 1.927 T on axis against DIII-D's real 1.9–2.1 T; contour-integral
and traced safety factors agree to 0.001–0.16%; ψ drifts by ~10⁻⁷ over 200
toroidal turns; and the section shows exactly 2 islands at q = 2 and 3 at q = 3,
as an m/n chain must.

Read `docs/BLUEPRINT.md` for the full production plan, `docs/PHYSICS.md` for the
derivations and caveats, and `docs/API-VERIFIED-5.2.md` for the Blender
4.2 → 5.x differences that break scripts outright.

**Caveats that travel with the images:** in strict axisymmetry a Poincaré
section is exactly the ψ contours, so the islands are a *prescribed*
perturbation and never a measurement; a vacuum section *overestimates*
stochasticity; Chirikov S ≥ 1 is a heuristic, not a theorem; and the geometry is
a reactor-class composite reached by a uniform scaling that leaves q(ψ) and the
field-line topology invariant — it is neither ITER nor DIII-D at true size.
