# JET 1983 — припущення геометричного ядра (варіант «як побудовано»)

`deltas.yaml` містить **лише джерельні** зміни відносно `jet1975` (плюс `assumed_params`, які там уже позначені).
Щоб побудувати нові вузли 1983 року, ядру потрібні ще розміщення й розміри, яких у джерелах немає.
Вони зібрані тут. Ядро читає блок `yaml` нижче (`kernel.<id>`) **поверх** `jet1975/ASSUMPTIONS_KERNEL.md`:
записи з тим самим `id` замінюють базові, решта успадковується.
У `manifest.json` кожне таке значення має `confidence: assumed` і `origin: machines/jet1983/ASSUMPTIONS_KERNEL.md`.

## Параметричні припущення (читаються ядром)

```yaml
kernel:
  # --- toroidal frame: follow the octant convention of the deltas --------------------------
  - {id: tf_phi0_deg, value: -5.625, unit: deg, confidence: assumed, note: "1983 deltas assume octant k spans phi = 45(k-1)..45k with the main horizontal port at 45(k-1)+22.5 and bellows at 5.625 + 11.25 m (vessel_octant_layout_1983). The kernel puts rigid sector i between TF coils i and i+1 and the octant weld in the middle of sector 4j, so phi0 = -5.625 makes the weld at phi = 45 j, the port at 45 j + 22.5 and the TF coils (over the bellows) at 5.625 + 11.25 m. The 1975 base keeps phi0 = 0 (whole machine rotated by 5.625 deg; nothing physical depends on the origin)"}
  - {id: iron_limb_phi0_deg, value: 0.0, unit: deg, confidence: assumed, note: "limbs at the octant welds (phi = 45 j), 22.5 deg from each main port. The 1975 choice (16.875 = 11.25 deg before each port) was made only to clear the six 1975 NBI injectors, which are absent in 1983; with it the pumping chambers (D 2 m, assumed) of octants 1/5 would hit a limb (axis 1.27 m from the limb plane at R 6.5 vs 1.0 + 0.505 m). Symmetric layout: axis-to-limb 2.49 m"}
  # --- limiter modules ------------------------------------------------------------------------
  - {id: limiter_module_thickness, value: 0.035, unit: m, confidence: assumed, note: "radial depth of the 0.40 x 0.80 m modules. The 1975 kernel value 0.07 puts the back of a Ni module (front 4.31) at 4.38 m, outside the plasma-side wall at the module corners (wall R 4.352 at |Z| = 0.40); 0.035 leaves 7 mm there"}
  # --- restraint rings --------------------------------------------------------------------------
  - {id: restraint_bridge_insulation_gap, value: 0.005, unit: m, confidence: assumed, note: "gap between an insulated bridge (across a bellows) and the thick skin segments of the neighbouring rigid sectors, each side. Bridge = same band section as the skins, between the parallel bellows flanges |y| < d - gap"}
  # --- pumping chambers (octants 1, 5) ------------------------------------------------------------
  - {id: pump_chamber_spool_length, value: 0.10, unit: m, confidence: assumed, note: "short spool between the main-port flange (kernel port_hduct_R_end 5.5) and the chamber cylinder; the chamber axis is at R = 5.5 + 0.10 + D/2 (D from the deltas envelope, itself assumed from a photo +-50 %)"}
  - {id: pump_chamber_door_frame, value: 0.10, unit: m, confidence: assumed, note: "radial frame around the 1200 mm i.d. door (text) -> door boss radius 0.70 m"}
  - {id: pump_chamber_door_boss, value: 0.10, unit: m, confidence: assumed, note: "door boss protrudes 0.10 m beyond the chamber cylinder, on the side opposite the torus port (text: opposite the torus port)"}
  - {id: pump_chamber_turbo_pump_DH, value: [0.5, 0.6], unit: m, confidence: assumed, note: "two turbomolecular pumps at the base of each chamber (text); diameter x height not given; placed under the chamber floor at +-D/4 toroidally"}
  # --- NBI port adaptors / rotary valves (octants 4, 8) ------------------------------------------------
  - {id: rotary_valve_body_DH, value: [0.9, 1.5], unit: m, confidence: assumed, note: "rotor = cylinder with vertical axis (text); body size not stated. Must contain the 0.5 x 1.1 m race-track opening (text) -> D 0.9 m, H 1.5 m; mounted on the outer face of the Middle Port Adaptor of the same octant (text: attached to the outside of each port adaptor)"}
  # --- gas introduction (octants 2, 6) -------------------------------------------------------------------
  - {id: gas_inlet_module_DL, value: [0.3, 0.4], unit: m, confidence: assumed, note: "gas introduction module (text: two modules at octants 2 and 6, port and size NOT stated): cylinder D 0.3 m, 0.4 m long, on the main-horizontal-port flange of the octant (a port that carries no pump chamber and no NBI adaptor in 1983)"}
  # --- limiter ports ---------------------------------------------------------------------------------------
  - {id: limiter_port_sectors_in_octant, value: [3], unit: index, confidence: assumed, note: "one limiter port per octant (deltas limiter_port, assumed) on the outer midplane of the rigid sector after the main-port sector (Fig. 86: next to the main horizontal port); sector 3 of the octant = phi 45(k-1) + 33.75"}
  - {id: limiter_port_diameter, value: 0.30, unit: m, confidence: assumed, note: "circular stub, size unknown (deltas); 0.30 m i.d. passes between the TF casings (gap ~0.56 m at R 4.6)"}
  - {id: limiter_port_R_end, value: 5.0, unit: m, confidence: assumed, note: "flange face just outside the TF casing outer leg (R 4.99); inside PF coil 4 (R 5.11, and |Z| of coil 4 >= 0.535 anyway)"}
```

## Непараметричні рішення ядра (1983)

| що | рішення | чому |
|---|---|---|
| Розкладка октанта | Компонент `vessel` читає запис `vessel_octant_layout_1983` (тип `vessel_sector`). 40 жорстких деталей: 8 × (торцева «a» + торцева «b») + 8 × по три повних (вертикальний великий порт / головний горизонтальний / вертикальний малий). Торці жорстких деталей — площини, паралельні площині сильфона/TF на відстані `d = 0,075 м` (половина `bellows_toroidal_length`). Тому сектори клиновидні, а сильфон має сталу тороїдальну довжину 0,15 м | інтерпретація з deltas (паралельні фланці сильфонів, 1975 p.295, p.303). Ширини збігаються з таблицею deltas: 9,292° / 4,646° / 1,958° при R 4,389 і 6,085° / 3,043° / 5,165° при R 1,664 |
| Шов октанта | площина φ = 45 j між торцевими деталями «a» (кінець октанта j−1) і «b» (початок октанта j). Губчастий шов (lip weld) не моделюється | немає розмірів |
| Сильфони 1983 | компонент `bellows` читає `vessel_bellows_1983`: 32 шт., паралельні фланці, `n = round((0,15 − 2·0,004)/0,010) = 14` гофрів, товщина 2 мм (override). Два концентричні шари, кожен коливається на 40 % зазору подвійної стінки (`bellows_ply_depth_fraction` бази) | висоту гофра 120 мм двічі в зазор 0,12–0,18 м не вмістити. Фланці 4 мм не моделюються окремо — вони збігаються з торцями жорстких деталей |
| PF-1 | 8 котушок, крок 0,533 м: `pf_coils` читає overrides `coil1_count`/`coil1_pitch_height`. Запис `pf_coil1_stack_1983` лише перевіряється (Z-центри збігаються до 1e-6 м), окремої геометрії не має (стан `realized`) | немає подвійної котушки |
| Кільця жорсткості | для кожного Z = ±1: 32 сегменти потовщеної обшивки (один на крок 11,25°, між фланцями сильфонів; біля шва октанта сегмент охоплює обидві торцеві деталі) + 32 ізольовані містки над сильфонами. Переріз: зовнішня обшивка при \|Z − Z₀\| ≤ 0,10 м, товщина 0,09 м радіально назовні | «up to 90 mm thick outer skins + insulated bridges across the bellows» (1983 p.24). До отвору TF при Z = 0,9 лишається 19 мм |
| Насосні камери | вертикальний циліндр D × H з рядка `envelope` (Ø2 × 4 м), вісь на R = 6,6 м у площині головного порту, Z центровано на екваторі. Проставка до фланця порту, двері Ø1,2 м (+ рамка) на протилежному боці, 2 турбонасоси знизу | розміри з фото ±50 %, позначено `envelope_assumed` |
| Адаптери NBI | прямокутна коробка 1,0 (рад.) × 1,4 (верт.) × 1,0 (тор.) м від фланця порту (R 5,5 → 6,5) | «much smaller than the pumping chambers» |
| Поворотні клапани | циліндр Ø0,9 × 1,5 м на зовнішній грані адаптера. Прохідний отвір 0,5 × 1,1 м не вирізано (записано в `meta.opening_m`). **Статус**: `status: not_installed`, `installed: 1984-01`, `hide_in_strict_variant: true` | 1983 p.40: встановлення в січні 1984 |
| Напуск газу | циліндр Ø0,3 × 0,4 м на фланці головного порту октантів 2 і 6 | порт і розмір не вказано |
| Лімітерний порт | круглий патрубок Ø0,30 м від зовнішньої обшивки до R = 5,0 м з фланцем, 8 шт. (сектор 3 кожного октанта, екватор). **Отвір у стінці не вирізано** | розміри невідомі |
| Лімітерні модулі | 12 модулів з deltas (φ, R_front — assumed там). Глибина 0,035 м (див. вище) | |
| Записи без геометрії | `vessel_bake_operation_1983`, `first_wall_1983`, `pf_coil_data_1982`, `iron_core_1982`, `mechanical_structure_1982`, `octant_port_topology_1983` → стан `record_only` (у `octant_port_topology_1983` — `realized` через `ports`). `vessel_bellows_length_estimate` → `realized` через `bellows` | факти для manifest |
| Нумерація октантів | у deltas і в мітках нових вузлів 1983 року октанти **1…8**; у мітках `vessel`/`ports`, успадкованих від 1975, — індекс ядра **0…7** (октант k = індекс k−1) | сумісність з базою |
