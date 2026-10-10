# Нові презентаційні рендери

Файли в `submission/assets/renders/reimagined/` перерендерені локально для цієї презентації з наявної сцени `tokamak-3d-viz/tokamak.blend` та її baked-даних `tokamak-3d-viz/data/bake01/`. Скрипт відтворення — `submission/render_assets.py`.

- Blender 5.2.2; прозорий фон PNG-композицій, після чого кадри зведені на теплий світлий фон `#f7f7f2`. У матеріалах металу зменшено металевість; рендери не містять вбудованих титрів.
- `tokamak-cutaway-paper.jpg` і `tokamak-cutaway-paper.mp4` — розріз камери `CAM_02_cutaway` та короткий 3-секундний орбітальний проліт.
- `tokamak-fieldlines-paper.jpg` і `tokamak-fieldlines-paper.mp4` — лінії поля з камери `CAM_03_plasma` та короткий 3-секундний орбітальний проліт.
- `tokamak-surfaces-paper.jpg` — магнітні поверхні з камери `CAM_05_qprofile`.
- `tokamak-poincare-paper.jpg` — переріз Пуанкаре з камери `CAM_04_poincare`.

Це пояснювальні візуалізації зі сцени DIII-D #203702, а не зображення тестових розрядів #060–#067 і не прямі кадри вимірювань. Атрибуція рівноважних даних: **Fusion Equilibrium Challenge (Sophelio / General Atomics), CC BY 4.0**. Код сцени `tokamak-3d-viz` — MIT; див. `tokamak-3d-viz/ATTRIBUTION.md` і `tokamak-3d-viz/LICENSE`.
