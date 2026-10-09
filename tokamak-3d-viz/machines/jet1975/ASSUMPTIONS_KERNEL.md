# JET 1975 — припущення геометричного ядра (фаза Ф2)

`dimensions.yaml` і `profiles/*.csv` містять **лише джерельні** значення. Щоб побудувати 3D-модель,
ядру (`src/tokviz/machine/`) потрібні ще розміщення, зазори й розміри, яких у джерелі немає.
Усі вони зібрані тут. Ядро **читає їх саме з цього файлу**: з блоку `yaml` нижче, шлях `kernel.<id>`.
Кожне таке значення потрапляє в `manifest.json` із `confidence: assumed` (або `digitized`, якщо
число взято з нотатки в YAML), тож у кожному компоненті видно, на яких припущеннях він стоїть.
Дельта-варіант (наприклад, `machines/jet1983/deltas.yaml`) може перевизначити будь-яке з них
через `path: kernel.<id>`.

## Параметричні припущення (читаються ядром)

```yaml
kernel:
  # --- toroidal layout -------------------------------------------------------------
  - {id: tf_phi0_deg, value: 0.0, unit: deg, confidence: assumed, note: "TF coil i at phi = phi0 + i*360/32. Coils sit over the bellows (Table I.3-1 gives Rce 'at the bellows shields' azimuth'), so rigid sector i is centred between coils i and i+1"}
  - {id: hport_sector_in_octant, value: 2, unit: index, source: JET1975, page: 297, confidence: assumed, note: "octant = half sector + 3 full sectors + half sector (welds in the middle of the outermost rigid sectors, p.299); the horizontal port is in the MIDDLE rigid sector (p.297) -> index 2 when sector 4j is the weld sector"}
  - {id: vport_large_sector_in_octant, value: 1, unit: index, confidence: assumed, note: "vertical 14x83 cm port: one top + one bottom per octant (YAML ports.vertical_ports_per_octant, assumed); put in the rigid sector before the horizontal-port sector"}
  - {id: vport_small_sector_in_octant, value: 3, unit: index, confidence: assumed, note: "vertical 7x28 cm port: one top + one bottom per octant, in the rigid sector after the horizontal-port sector"}
  - {id: iron_limb_phi0_deg, value: 16.875, unit: deg, confidence: assumed, note: "limb/arm azimuth not given. phi = 16.875 + 45 j puts the limbs 11.25 deg before / 33.75 deg after each horizontal port (over the vertical-port sectors); the symmetric choice (octant welds, 22.5 deg from each port) collides with the six NBI injectors of the 1975 layout, this one clears them"}
  # --- vessel / bellows -----------------------------------------------------------------
  - {id: vessel_bellows_angle_deg, value: 3.0, unit: deg, confidence: assumed, note: "toroidal extent of one double-bellows section; not printed (plan Fig. IV.1-9 has only a 1 m scale bar). Rigid sector then spans 8.25 deg (0.64 m at the horizontal port, > 0.46 m opening)"}
  - {id: bellows_convolutions, value: 6, unit: count, confidence: assumed, note: "convolutions per bellows; not given in the source"}
  - {id: bellows_ply_depth_fraction, value: 0.4, unit: "-", confidence: assumed, note: "each of the two concentric plies (p.295) oscillates over 40 % of the 0.12-0.18 m wall envelope: outer ply from the exterior skin inwards, inner ply from the plasma-side skin outwards; 1.7 mm wall (p.306) as thin-shell thickness"}
  # --- TF coils ----------------------------------------------------------------------------
  - {id: tf_wedge_contact_gap, value: 0.001, unit: m, confidence: assumed, note: "the wedged inner legs bear on each other along the radial planes phi = +-pi/32; a 1 mm gap keeps neighbouring meshes from being coplanar"}
  - {id: tf_support_pad_R_range, value: [2.9, 3.4], unit: m, source: JET1975, page: 325, figure: "Fig. IV.2-1", confidence: digitized, note: "from the note of tf_coils.support_pad_Z_outer ('R ~ 2.9-3.4 m')"}
  - {id: tf_support_pad_embed, value: 0.05, unit: m, confidence: assumed, note: "pad block extends 5 cm into the casing below the casing contour"}
  # --- PF coils ------------------------------------------------------------------------------
  - {id: pf_coil1_interpack_gap, value: 0.020, unit: m, source: JET1975, page: 402, figure: "Fig. IV.4-13", confidence: drawing, note: "'10+10' gaps inside the 497 mm pitch -> pack height 0.477 m (0.327 x 0.477 = 0.156 m2 vs 0.157 table)"}
  # --- iron ----------------------------------------------------------------------------------
  - {id: iron_foot_extra_width, value: 0.20, unit: m, confidence: assumed, note: "foot plate toroidal width = limb width + 0.20 m (CSV gives it +-0.10 m radially, 'assumed width')"}
  # --- structure ---------------------------------------------------------------------------
  - {id: structure_clearance, value: 0.003, unit: m, confidence: assumed, note: "clearance between ring/shell blocks and TF casing sides, and between the ring collar and TF casing / iron centre piece"}
  - {id: ring_collar_R_split, value: 2.0, unit: m, confidence: assumed, note: "ring inside R = 2.0 m is modelled axisymmetric (collar, lifted clear of the TF casing contour and kept under the iron centre piece, which the digitized ring top 3.04 +-0.15 overlaps by 6 cm); outside, 32 wedge blocks between TF coils"}
  - {id: ring_port_opening_margin, value: 0.05, unit: m, confidence: assumed, note: "radial margin of the ring-block openings around the vertical ports (p.376: 'a large opening for the vertical ports')"}
  - {id: shell_plate_thickness, value: 0.02, unit: m, confidence: assumed, note: "simplified mechanical shell: bolted Al-alloy plates between TF coils, flush with the casing outer contour"}
  - {id: shell_port_band_half_height, value: 0.62, unit: m, confidence: assumed, note: "shell plates stop at |Z| = 0.62 m to leave the horizontal-port band (0.96 m + duct wall) free"}
  - {id: inner_cylinder_radial_clearance, value: 0.0015, unit: m, confidence: assumed, note: "inner cylinder fitted between PF coil 1 (R 1.085) and the TF inner leg (R 1.1165): modelled 28.5 mm thick instead of the 40 mm average (p.374) - README contradiction 9"}
  # --- plasma / limiters -----------------------------------------------------------------
  - {id: plasma_min_wall_clearance, value: 0.02, unit: m, confidence: assumed, note: "minimum LCFS-to-vessel-inner-wall distance required of the fitted D; triangularity is reduced if violated"}
  - {id: outer_limiter_sectors_in_octant, value: [1, 3], unit: index, confidence: assumed, note: "16 outer plates = 2 per octant, in the two rigid sectors without the horizontal port"}
  - {id: rail_limiter_sectors_in_octant, value: [0, 2], unit: index, confidence: assumed, note: "upper/lower rail plates, 2 per octant (count assumed = 16 like the outer rail), in the sectors without vertical ports"}
  - {id: limiter_module_thickness, value: 0.07, unit: m, confidence: assumed, note: "radial depth of a discrete limiter module (type limiter_module, used by delta variants) when its params give none; 0.07 keeps a module retracted to R_front 4.31 inside the 4.389 m plasma-side wall"}
  # --- ports -----------------------------------------------------------------------------------
  - {id: port_hduct_wall, value: 0.02, unit: m, confidence: assumed, note: "horizontal port duct wall"}
  - {id: port_hduct_R_end, value: 5.5, unit: m, confidence: assumed, note: "horizontal stub flange face radius: outside the TF casing (4.99) and PF coil 4 (5.37), inside the iron limbs (5.99)"}
  - {id: port_flange_margin, value: 0.10, unit: m, confidence: assumed, note: "horizontal port flange overhang"}
  - {id: port_flange_thickness, value: 0.04, unit: m, confidence: assumed, note: ""}
  - {id: port_vduct_wall, value: 0.008, unit: m, confidence: assumed, note: "vertical duct wall; the 14 cm port between two TF coils leaves ~1 cm per side at R 2.55"}
  - {id: port_vflange_margin, value: 0.05, unit: m, confidence: assumed, note: ""}
  - {id: port_vflange_thickness, value: 0.03, unit: m, confidence: assumed, note: ""}
  - {id: port_vlarge_flange_Z, value: 3.22, unit: m, source: JET1975, page: 80, figure: "Fig. I.3-2 item 10", confidence: digitized, note: "from the note of ports.vertical_port_large_R_range ('flange top Z ~ +-3.22')"}
  - {id: port_vsmall_flange_Z, value: 3.24, unit: m, source: JET1975, page: 80, figure: "Fig. I.3-2 item 11", confidence: digitized, note: "from the note of ports.vertical_port_small_R_range ('top Z ~ +-3.24')"}
  # --- NBI ----------------------------------------------------------------------------------------
  - {id: nbi_port_octant, value: 0, unit: index, confidence: assumed, note: "the NBI port is the horizontal port of octant 0"}
  - {id: nbi_plan_side, value: 1.0, unit: "-", confidence: assumed, note: "+1: beams aimed along +phi (co-direction not specified)"}
  - {id: nbi_standoff_R, value: 5.6, unit: m, confidence: assumed, note: "beam ducts start where the beam line reaches R = 5.6 m (outside TF casing and PF coil 4); the converging beam path inside is not meshed"}
  - {id: nbi_plan_columns, value: 2, unit: count, confidence: assumed, note: "six injectors read as 2 columns (25 deg apart in plan) x 3 rows (27 deg apart in the side view); 3 x 2 would need a 50 deg plan fan that cannot pass between two iron limbs 45 deg apart"}
  - {id: nbi_tank_width_over_height, value: 1.0, unit: "-", confidence: assumed, note: "tank width not printed; taken equal to the digitized height 1.21 m"}
  # --- scale figure -------------------------------------------------------------------------------
  - {id: human_height, value: 1.80, unit: m, confidence: assumed, note: "scale figure"}
  - {id: human_R, value: 9.0, unit: m, confidence: assumed, note: "standing outside the iron limbs (7.38 m)"}
  - {id: human_phi_deg, value: 200.0, unit: deg, confidence: assumed, note: "between limbs, away from the NBI"}
```

## Непараметричні рішення ядра

| що | рішення | чому |
|---|---|---|
| Плазма (G2) | D-форма Міллера `R = R0 + a cos(t + asin δ sin t)`, `Z = κ a sin t`, `R0 = 2,96`, `a = 1,25`, `κ = b/a = 1,68`. δ підігнано до **форми** Табл. IV.2-4: у таблиці лише верхній зовнішній квадрант, тому внутрішній край підгонки закріплено на `Ri(plasma) = 1,71` і підганяються `(a_t, κ_t, δ)`. Далі δ переноситься на проєктну D. Результат перевіряється: мінімальний зазор до плазмової обшивки ≥ `plasma_min_wall_clearance`, інакше δ зменшується | сирі точки таблиці виходять за `b` і перетинають стінку (README, суперечність 5) |
| TF, клин | бічні грані внутрішньої ноги лежать на радіальних площинах φ = ±π/32: півширина `min(0,169, R tan(π/32))` | «wedge-shaped», 0,26 «at wedge»: ширина 0,26 виходить при R = 1,32 м, тобто всередині ноги (1,1165…1,49) |
| TF, відповідність контурів | кожна точка отвору з'єднана з точкою, де її зовнішня нормаль перетинає контур корпусу | чотирикутні перерізи без перекосу |
| TF, носик R196 | заокруглення носика клина в горизонтальному перерізі **не моделюється** | дрібна деталь, на ≤ 1 см |
| Порти | модель «stub + отвір у стінці»: у стінці сектора вирізано отвір із косяком між обшивками, окремий компонент `ports` дає патрубок із фланцем. Прапорець `vessel.params.openings` вимикає отвори | патрубки видно ззовні, отвори — зсередини камери |
| Сильфони | коруговані поверхні між профілями, дві концентричні оболонки; це відкриті тонкі оболонки з `thickness_m = 0,0017` | для модифікатора solidify у Blender |
| Екрани сильфонів («fixed poloidal limiters», Re 4,33 / Ri 1,70) | **не моделюються** | форми в джерелі немає |
| Зварні шви октантів, ребра подвійної стінки | не моделюються | немає розмірів |
| Верхній і нижній рейкові лімітери | оцифрована позиція (2,3; ±1,95) лежить усередині плазми (b = 2,10). Тому пластини поставлено конформно до LCFS: їхня грань лежить на LCFS там, де промінь з (R0, 0) через оцифровану точку перетинає LCFS. Пластина має 1,00 м полоїдально, 0,60 м тороїдально, 0,01 м завтовшки | суперечність малюнка з Табл. I.3-1 |
| Кільця | колар (R < 2,0) осесиметричний. Його точки, що потрапляють у контур корпусу TF, підняті над корпусом, а ті, що заходять під центральний блок заліза, опущені під нього. Зовнішня частина — 32 клинові блоки між котушками з паралельними пазами (півширина 0,169 + 0,003). У блоках над вертикальними портами вирізано отвори | README: кільце несе клинові блоки між котушками. Положення кільця по Z припущене (±0,15) |
| Оболонка | спрощена: пластини 2 см між сусідніми котушками вздовж зовнішнього контуру корпусу, при R ≥ 3,66 і \|Z\| ≥ 0,62 | деталей плит/лотків у джерелі немає |
| Ярмо | стовпи, плечі й опори — прямокутні бруски з тороїдальною шириною з YAML. Плечі біля осі обрізано по клину 45°. Колона, маточина й центральні блоки осесиметричні (маса блока 95 т проти 90 т надрукованих). Полігони `model_*` (осесиметрична модель потоку, p.394) у 3D **не використовуються** | |
| NBI | «45° до магнітної осі» (p.301) трактується так: центральний пучок перетинає коло R = R0 під кутом 45°, тобто радіус дотику R0 cos 45° = 2,09 м (біля порту це 27,7° від радіального напряму). 25° і 27° — кроки між інжекторами: 2 стовпці в плані × 3 ряди. Шлях пучка всередині R < 5,6 м не моделюється | варіант 3 × 2 дає віяло 50°, яке не проходить між стовпами ярма |
| Азимут стовпів ярма | φ = 16,875° + 45° j, а не на зварних швах октантів | інакше стовпи перетинаються з інжекторами NBI; джерела для азимута немає |
| Масштаб | 1:1, метри. Екземпляри подано як трансформації, не копії | |
