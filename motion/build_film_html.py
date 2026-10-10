#!/usr/bin/env python3
"""motion/build_film_html.py — Compiles team/presentation/film/index.html.

Reads data from team/presentation/film/data/ and assembles a self-contained,
cinematic 1920x1080 Canvas2D film with embedded datasets, pure render functions,
zero-CDN fonts/libraries, and keyboard/sync controls.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILM_DIR = ROOT / "team/presentation/film"
DATA_DIR = FILM_DIR / "data"

def load_json(name: str):
    p = DATA_DIR / name
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def build():
    contours_gt = load_json("contours_gt.json")
    contours_unet = load_json("contours_unet.json")
    meta = load_json("equilibrium_meta.json")
    poincare_chaos = load_json("poincare_keyframe_chaos.json")
    poincare_sweep = load_json("poincare_sweep_compact.json")
    sources = load_json("../sources.json")

    # Serialize data compactly for inlining
    raw_gt_str = json.dumps(contours_gt.get("contours", []), separators=(',', ':'))
    raw_unet_str = json.dumps(contours_unet.get("contours", []), separators=(',', ':'))
    raw_chaos_str = json.dumps(poincare_chaos, separators=(',', ':'))
    raw_sweep_str = json.dumps(poincare_sweep.get("frames", []), separators=(',', ':'))
    raw_sources_str = json.dumps(sources.get("registry", []), separators=(',', ':'))

    html_content = f'''<!DOCTYPE html>
<html lang="uk">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
  <title>TokaBench-GS: Магнітна рівновага токамака (Кіно-доповідь)</title>
  <style>
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      user-select: none;
      -webkit-user-select: none;
    }}
    html, body {{
      width: 100%;
      height: 100%;
      background: #04060B;
      color: #F2F5FA;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    #stage-container {{
      position: relative;
      width: 1920px;
      height: 1080px;
      transform-origin: center center;
      background: #04060B;
      box-shadow: 0 0 80px rgba(0, 0, 0, 0.9);
      overflow: hidden;
    }}
    canvas#filmCanvas {{
      position: absolute;
      top: 0; left: 0;
      width: 1920px;
      height: 1080px;
      z-index: 10;
    }}
    .video-layer {{
      position: absolute;
      top: 0; left: 0;
      width: 1920px;
      height: 1080px;
      object-fit: cover;
      opacity: 0;
      transition: opacity 0.8s cubic-bezier(0.16, 1, 0.3, 1);
      z-index: 1;
      pointer-events: none;
    }}
    .video-layer.active {{
      opacity: 0.38;
    }}
    /* HUD & Navigation indicators */
    #hud-overlay {{
      position: absolute;
      top: 24px; left: 96px; right: 96px;
      height: 48px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 20px;
      color: #8A94A8;
      font-family: "SF Mono", "JetBrains Mono", Menlo, Monaco, Consolas, monospace;
      font-variant-numeric: tabular-nums;
      z-index: 50;
      opacity: 0;
      transition: opacity 0.4s ease;
      pointer-events: none;
    }}
    #hud-overlay.visible {{
      opacity: 0.9;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      background: rgba(125, 249, 255, 0.1);
      border: 1px solid rgba(125, 249, 255, 0.25);
      border-radius: 4px;
      color: #7DF9FF;
      font-size: 16px;
    }}
    #debug-panel {{
      position: absolute;
      bottom: 24px; left: 96px; right: 96px;
      padding: 12px 20px;
      background: rgba(4, 6, 11, 0.85);
      border: 1px solid rgba(255, 179, 71, 0.3);
      border-radius: 6px;
      color: #FFB347;
      font-family: monospace;
      font-size: 16px;
      display: none;
      justify-content: space-between;
      z-index: 60;
    }}
    /* Preload image cache */
    .hidden-asset {{
      display: none;
    }}
  </style>
</head>
<body>

  <div id="stage-container">
    <!-- Video background elements (1080p final clips) -->
    <video id="vid-act1" class="video-layer" src="assets/act1_cutaway_final.mp4" muted loop playsinline preload="auto"></video>
    <video id="vid-act2" class="video-layer" src="assets/act2_surfaces_final.mp4" muted loop playsinline preload="auto"></video>
    <video id="vid-act3" class="video-layer" src="assets/act3_fieldlines_final.mp4" muted loop playsinline preload="auto"></video>
    <video id="vid-act4a" class="video-layer" src="assets/act4a_poincare_single_final.mp4" muted loop playsinline preload="auto"></video>
    <video id="vid-act4b" class="video-layer" src="assets/act4b_poincare_chaos_final.mp4" muted loop playsinline preload="auto"></video>

    <!-- Main Canvas -->
    <canvas id="filmCanvas" width="1920" height="1080"></canvas>

    <!-- Subtle Presenter HUD -->
    <div id="hud-overlay">
      <div id="hud-mode"><span class="badge" id="mode-badge">PRESENTER</span> <span id="scene-title">S0 Вступ: Парадокс R²</span></div>
      <div id="hud-beat">БІТ <span id="beat-num">1</span> / <span id="beat-total">2</span></div>
      <div id="hud-time">00:00 / 12:00</div>
    </div>

    <!-- Debug HUD -->
    <div id="debug-panel">
      <span id="dbg-fps">FPS: 60 (p95: 16.2 ms)</span>
      <span id="dbg-scene">Scene: S0, Beat: 1, t: 0.00s</span>
      <span id="dbg-sources">Sources: verified</span>
    </div>
  </div>

  <!-- Offscreen 2D Residual Maps -->
  <img id="img-residual-gt" class="hidden-asset" src="data/residual_gt.png" alt="GT residual">
  <img id="img-residual-unet" class="hidden-asset" src="data/residual_unet.png" alt="UNet residual">

  <!-- Embedded Data -->
  <script>
    const GT_CONTOURS = {raw_gt_str};
    const UNET_CONTOURS = {raw_unet_str};
    const POINCARE_CHAOS = {raw_chaos_str};
    const POINCARE_SWEEP = {raw_sweep_str};
    const SOURCES_REGISTRY = {raw_sources_str};
  </script>

  <!-- Film Engine Core -->
  <script>
    (function() {{
      "use strict";

      // -------------------------------------------------------------
      // 1. ENGINE CONFIG & URL PARAMETERS
      // -------------------------------------------------------------
      const urlParams = new URLSearchParams(window.location.search);
      const isShort = urlParams.get("short") === "1";
      const isFilmModeInit = urlParams.get("mode") === "film";
      const isDebug = urlParams.get("debug") === "1";
      const captureTime = urlParams.get("capture") !== null ? parseFloat(urlParams.get("t") || "0.0") : null;

      const canvas = document.getElementById("filmCanvas");
      const ctx = canvas.getContext("2d", {{ alpha: false }});
      const stage = document.getElementById("stage-container");

      const hudOverlay = document.getElementById("hud-overlay");
      const modeBadge = document.getElementById("mode-badge");
      const sceneTitleEl = document.getElementById("scene-title");
      const beatNumEl = document.getElementById("beat-num");
      const beatTotalEl = document.getElementById("beat-total");
      const hudTimeEl = document.getElementById("hud-time");
      const debugPanel = document.getElementById("debug-panel");

      if (isDebug) debugPanel.style.display = "flex";

      // -------------------------------------------------------------
      // 2. TIMELINE & BEAT DEFINITIONS (12:00 TOTAL = 720 SECONDS)
      // -------------------------------------------------------------
      // 10 Scenes, total 720s in standard mode (300s in short mode)
      const SCENES = [
        {{
          id: "S0", title: "Вступ: Парадокс R²", startTime: 0, duration: 30,
          beats: [
            {{ id: "b0_1", time: 0, duration: 18, text: "Коефіцієнт детермінації карти потоку R²ψ = 0.977.", note: "Уявіть ситуацію: згорткова нейромережа видає R² = 0.976. Будь-який стандартний Data Science пайплайн каже: задача розв'язана." }},
            {{ id: "b0_2", time: 18, duration: 12, text: "Це фізика?", note: "Але ми запитали себе: чи фізичний цей розв'язок? Що станеться, якщо пустити його в реальну плазму?" }}
          ]
        }},
        {{
          id: "S1", title: "Токамак DIII-D", startTime: 30, duration: 45, videoId: "vid-act1",
          beats: [
            {{ id: "b1_1", time: 0, duration: 22, text: "Рівновага плазми — це розв'язок рівняння.", note: "Плазма утримується балансом сили Лоренца і градієнта тиску: j x B = ∇p." }},
            {{ id: "b1_2", time: 22, duration: 23, text: "Геометрія тора DIII-D та магнітний шнур.", note: "В осесиметричному торі цей баланс строго описується двовимірним рівнянням Ґреда–Шафранова." }}
          ]
        }},
        {{
          id: "S2", title: "Рівновага і Ґрад–Шафранов", startTime: 75, duration: 120, videoId: "vid-act2",
          beats: [
            {{ id: "b2_1", time: 0, duration: 30, text: "Вкладені поверхні потоку ψ від осі до сепаратриси.", note: "Полоїдальний потік ψ(R, Z) задає магнітні поверхні, вкладені навколо еліптичної O-точки." }},
            {{ id: "b2_2", time: 30, duration: 30, text: "Сепаратриса (LCFS) та гіперболічна X-точка (Bp = 0).", note: "Зовнішня межа утримання проходить через гіперболічну X-точку, де полоїдальне поле строго нульове." }},
            {{ id: "b2_3", time: 60, duration: 30, text: "Рівняння ГШ: Δ*ψ = −μ₀R²p′(ψ) − FF′(ψ).", note: "Оператор Δ*ψ містить другі просторові похідні в просторі Соболєва H²." }},
            {{ id: "b2_4", time: 90, duration: 30, text: "Нев'язка g(ψ): чи існує пара фізичних профілів?", note: "Нев'язка g перевіряє рівновагу суто з карти ψ без знання тиску й струму (E6)." }}
          ]
        }},
        {{
          id: "S3", title: "R² ≠ Фізика", startTime: 195, duration: 120,
          beats: [
            {{ id: "b3_1", time: 0, duration: 40, text: "Розряд #062 кадр 147: Істина проти UNet_Lite.", note: "Погляньте на розряд #062 (медіанний за нев'язкою). Ліворуч істина EFIT, праворуч — UNet_Lite." }},
            {{ id: "b3_2", time: 40, duration: 40, text: "Нев'язка g: істина 0.0065 проти UNet 0.8618 (поріг 0.6328).", note: "Нев'язка істини 0.0065. У UNet вона вибухає до 0.8618 через високочастотний шум других похідних." }},
            {{ id: "b3_3", time: 80, duration: 40, text: "Істина + 1% шуму дає g≈0.65: мережі не є рівновагами!", note: "Істина з 1% шуму дає g≈0.65 (E6). Мережі з R² 0.98 мають g 0.86 — це взагалі не рівноваги (E38)!" }}
          ]
        }},
        {{
          id: "S4", title: "Шлюз Ґреда–Шафранова", startTime: 315, duration: 75,
          beats: [
            {{ id: "b4_1", time: 0, duration: 25, text: "Реалізували Задачу 1 організаторів: нелінійний шлюз.", note: "Адитивний член у R12 не переупорядковував моделі. Організатори запропонували шлюз GS (Задача 1)." }},
            {{ id: "b4_2", time: 25, duration: 25, text: "Штраф за g: UNet отримує штраф 28%, MLP зберігає гладкість.", note: "UNet штрафується шлюзом (G=0.721), тоді як MLP на базі PCA зберігає низьку нев'язку (g=0.503)." }},
            {{ id: "b4_3", time: 50, duration: 25, text: "Чутливість до g_ref: стабільний поділ фізичних моделей.", note: "Ми перевірили стійкість при варіації g_ref ±20%: розподіл моделей залишається стабільним." }}
          ]
        }},
        {{
          id: "S5", title: "Результати з довірчими інтервалами", startTime: 390, duration: 90,
          beats: [
            {{ id: "b5_1", time: 0, duration: 30, text: "За офіційною метрикою S моделі статистично нерозрізненні.", note: "UNet (0.6476) та MLP (0.6437): 95% CI парної різниці містить нуль [-0.012 .. +0.009]. Ранги нерозрізнені!" }},
            {{ id: "b5_2", time: 30, duration: 30, text: "Шлюз GS інвертує ранги: MLP 0.533 проти UNet 0.420.", note: "За діагностичним шлюзом S'-gate нуль не входить у CI: MLP виходить на 1 місце завдяки фізичності." }},
            {{ id: "b5_3", time: 60, duration: 30, text: "Розкид по розрядах: 8 імпульсів мають високу нестабільність.", note: "Увага: вибірка з 8 розрядів чутлива. PCA+Ridge провалюється на #061 і #063, але не гірший на #067 і 36–39." }}
          ]
        }},
        {{
          id: "S6", title: "Силові лінії тора", startTime: 480, duration: 45, videoId: "vid-act3",
          beats: [
            {{ id: "b6_1", time: 0, duration: 22, text: "Трасування ліній — канонічна гамільтонова система.", note: "Завдяки ∇·B = 0 фазовий потік зберігається точно: dR/dφ = -∂H/∂Z, dZ/dφ = ∂H/∂R (E19)." }},
            {{ id: "b6_2", time: 22, duration: 23, text: "Коефіцієнт запасу q = m/n та інваріантні тори KAM.", note: "У незбуреному полі лінії намотуються на гладкі тори з частотою 1/q." }}
          ]
        }},
        {{
          id: "S7", title: "Переріз Пуанкаре та хаос", startTime: 525, duration: 105, videoId: "vid-act4b",
          beats: [
            {{ id: "b7_1", time: 0, duration: 25, text: "Одиночна мода 2/1: острів без хаосу, KAM-тори цілі.", note: "При одиночній моді виникає резонансний острів 2/1 бурштинового кольору. Хаосу немає (E20)." }},
            {{ id: "b7_2", time: 25, duration: 25, text: "Додається мода 3/1: взаємодія резонансних поверхонь.", note: "Введення другої моди руйнує останні інваріантні тори між резонансами q=2 та q=1.5." }},
            {{ id: "b7_3", time: 50, duration: 30, text: "Ключовий кадр A=4.8e-3: початок перекриття (S≈1.0–1.1).", note: "При амплітуді A=4.8e-3 критерій Чирикова досягає 1.0025: сепаратриси починають перекриватися." }},
            {{ id: "b7_4", time: 80, duration: 25, text: "Ширина острова порядку 3 см; FTLE k = 2.02.", note: "Ширина острова 2/1 порядку 3 см за аналітичною формулою. k = FTLE(хаос)/FTLE(регул) = 2.02 (#203702)." }}
          ]
        }},
        {{
          id: "S8", title: "Спростоване як перевірена наука", startTime: 630, duration: 50,
          beats: [
            {{ id: "b8_1", time: 0, duration: 20, text: "Реєстр гіпотез R1–R13: відхилені гіпотези організаторів.", note: "13 спростованих гіпотез (R1–R13) — це не поразка, а захист від хибних висновків." }},
            {{ id: "b8_2", time: 20, duration: 15, text: "R2 і R7 після спростування дали більше, ніж до нього.", note: "Спростування прямого перенесення (R7) привело до розуміння інваріантності канонічного скелета." }},
            {{ id: "b8_3", time: 35, duration: 15, text: "Перевірили свій результат T13: покращення специфічне для оператора.", note: "Релаксація знижує g за власним оператором (0.86→0.38), але не за 5-точковою схемою (0.98→0.93)!" }}
          ]
        }},
        {{
          id: "S9", title: "Фінал і висновки", startTime: 680, duration: 40,
          beats: [
            {{ id: "b9_1", time: 0, duration: 22, text: "R² не замінює фізику; шлюз GS виявляє нефізичні карти.", note: "Три головні уроки хакатону: диференціальний контроль нев'язки, фізичний шлюз та наукова чесність." }},
            {{ id: "b9_2", time: 22, duration: 18, text: "Blender-сцена і моделі: автори проєкту (CC BY 4.0).", note: "Дякуємо за увагу! Відкритий код TokaBench-GS доступний у нашому репозиторії." }}
          ]
        }}
      ];

      // Short version maps only S0, S3, S5, S7, S9 (compressed to 300s)
      const SHORT_SCENE_IDS = ["S0", "S3", "S5", "S7", "S9"];
      const ACTIVE_SCENES = isShort ? SCENES.filter(s => SHORT_SCENE_IDS.includes(s.id)) : SCENES;

      // Calculate cumulative beat timeline
      let allBeats = [];
      let totalFilmDuration = 0;
      ACTIVE_SCENES.forEach((sc, sIdx) => {{
        sc.beats.forEach((b, bIdx) => {{
          const dur = isShort ? (b.duration * (300 / 720)) : b.duration;
          allBeats.push({{
            sceneIndex: sIdx,
            sceneId: sc.id,
            sceneTitle: sc.title,
            videoId: sc.videoId,
            beatIndex: bIdx,
            beatTotal: sc.beats.length,
            beatId: b.id,
            text: b.text,
            note: b.note,
            globalStart: totalFilmDuration,
            duration: dur,
            globalEnd: totalFilmDuration + dur
          }});
          totalFilmDuration += dur;
        }});
      }});

      let currentBeatIdx = 0;
      let currentGlobalTime = 0.0;
      let isPlayingFilm = isFilmModeInit;
      let isPresenterMode = !isFilmModeInit;
      let beatAnimProgress = 0.0; // 0 to 1 during transition
      let lastTimestamp = performance.now();
      let frameTimes = [];

      // Broadcast channel for sync with external Presenter Notes window
      const syncChannel = new BroadcastChannel("tokabench_film_sync");

      function postSyncState() {{
        const b = allBeats[currentBeatIdx] || allBeats[0];
        const nextB = allBeats[currentBeatIdx + 1] || null;
        syncChannel.postMessage({{
          type: "STATE_UPDATE",
          currentBeat: currentBeatIdx,
          totalBeats: allBeats.length,
          globalTime: currentGlobalTime,
          totalDuration: totalFilmDuration,
          sceneId: b.sceneId,
          sceneTitle: b.sceneTitle,
          text: b.text,
          note: b.note,
          nextText: nextB ? nextB.text : "Кінець доповіді",
          nextNote: nextB ? nextB.note : "",
          isPresenter: isPresenterMode,
          isPlaying: isPlayingFilm
        }});

        if (notesWindow && !notesWindow.closed) {{
          try {{
            const doc = notesWindow.document;
            const m = Math.floor(currentGlobalTime / 60);
            const s = Math.floor(currentGlobalTime % 60);
            const tClock = doc.getElementById("t-clock");
            const tBox = doc.getElementById("notes-timer");
            const scEl = doc.getElementById("notes-scene");
            const prEl = doc.getElementById("notes-prompt");
            const bdEl = doc.getElementById("notes-body");
            const nxEl = doc.getElementById("notes-next");
            if (tClock) tClock.textContent = (m < 10 ? "0" + m : m) + ":" + (s < 10 ? "0" + s : s);
            if (tBox) {{
              if (m >= 12) {{
                tBox.style.borderColor = "#ef4444";
                tBox.style.color = "#ef4444";
              }} else if (m >= 10) {{
                tBox.style.borderColor = "#eab308";
                tBox.style.color = "#eab308";
              }} else {{
                tBox.style.borderColor = "#22c55e";
                tBox.style.color = "#22c55e";
              }}
            }}
            if (scEl) scEl.textContent = b.sceneId + ": " + b.sceneTitle + " (Біт " + (currentBeatIdx + 1) + " / " + allBeats.length + ")";
            if (prEl) prEl.textContent = b.text;
            if (bdEl) bdEl.textContent = b.note;
            if (nxEl) nxEl.textContent = nextB ? (nextB.sceneId + ": " + nextB.text) : "(Кінець доповіді)";
          }} catch(e) {{}}
        }}
      }}

      // -------------------------------------------------------------
      // 3. LETTERBOX RESIZE & RETINA RESOLUTION
      // -------------------------------------------------------------
      function resizeStage() {{
        const winW = window.innerWidth;
        const winH = window.innerHeight;
        const scale = Math.min(winW / 1920, winH / 1080);
        stage.style.transform = `translate(-50%, -50%) scale(${{scale}})`;
        stage.style.position = "absolute";
        stage.style.top = "50%";
        stage.style.left = "50%";
      }}
      window.addEventListener("resize", resizeStage);
      resizeStage();

      // -------------------------------------------------------------
      // 4. PRECOMPILED PATH2D FOR CONTOURS & POINCARE
      // -------------------------------------------------------------
      // Precompile GT Contours
      const gtRaw = GT_CONTOURS.contours || GT_CONTOURS || [];
      const gtPaths = gtRaw.map(c => {{
        const p2d = new Path2D();
        const r = c.r;
        const z = c.z;
        if (!r || r.length === 0) return {{ psi_n: c.psi_n, path: p2d }};
        // Map (R, Z) where R in [0.84, 2.54], Z in [-1.6, 1.6] to 1920x1080 stage coords
        // Center around (960, 540)
        for (let i = 0; i < r.length; i++) {{
          const cx = 960 + (r[i] - 1.69) * 440;
          const cy = 540 - z[i] * 400;
          if (i === 0) p2d.moveTo(cx, cy);
          else p2d.lineTo(cx, cy);
        }}
        p2d.closePath();
        return {{ psi_n: c.psi_n, path: p2d }};
      }});

      // Precompile UNet Contours
      const unetRaw = UNET_CONTOURS.contours || UNET_CONTOURS || [];
      const unetPaths = unetRaw.map(c => {{
        const p2d = new Path2D();
        const r = c.r;
        const z = c.z;
        if (!r || r.length === 0) return {{ psi_n: c.psi_n, path: p2d }};
        for (let i = 0; i < r.length; i++) {{
          const cx = 960 + (r[i] - 1.69) * 440;
          const cy = 540 - z[i] * 400;
          if (i === 0) p2d.moveTo(cx, cy);
          else p2d.lineTo(cx, cy);
        }}
        p2d.closePath();
        return {{ psi_n: c.psi_n, path: p2d }};
      }});

      // -------------------------------------------------------------
      // 5. PROCEDURAL FILM GRAIN NOISE CANVAS (12 FPS, 3.5% OPACITY)
      // -------------------------------------------------------------
      const grainCanvas = document.createElement("canvas");
      grainCanvas.width = 256;
      grainCanvas.height = 256;
      const grainCtx = grainCanvas.getContext("2d");
      let lastGrainTime = 0;

      function updateGrain(now) {{
        if (now - lastGrainTime < 83) return; // 12 fps
        lastGrainTime = now;
        const imgData = grainCtx.createImageData(256, 256);
        const data = imgData.data;
        for (let i = 0; i < data.length; i += 4) {{
          const v = (Math.random() * 255) | 0;
          data[i] = v;
          data[i + 1] = v;
          data[i + 2] = v;
          data[i + 3] = 18; // ~7% alpha max
        }}
        grainCtx.putImageData(imgData, 0, 0);
      }}

      // -------------------------------------------------------------
      // 6. VIDEO LAYER CONTROLLER
      // -------------------------------------------------------------
      const videoElements = document.querySelectorAll(".video-layer");
      function syncVideo(activeId) {{
        videoElements.forEach(v => {{
          if (v.id === activeId) {{
            if (!v.classList.contains("active")) {{
              v.classList.add("active");
              v.currentTime = 0;
              v.play().catch(() => {{}});
            }}
          }} else {{
            if (v.classList.contains("active")) {{
              v.classList.remove("active");
              v.pause();
            }}
          }}
        }});
      }}

      // -------------------------------------------------------------
      // 7. EASING FUNCTIONS
      // -------------------------------------------------------------
      const ease = {{
        outCubic: t => 1 - Math.pow(1 - t, 3),
        inOutCubic: t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2,
        outExpo: t => t === 1 ? 1 : 1 - Math.pow(2, -10 * t),
        lerp: (a, b, t) => a + (b - a) * t,
        clamp: (v, min, max) => Math.max(min, Math.min(max, v))
      }};

      // -------------------------------------------------------------
      // 8. SCENE RENDERERS (PURE FUNCTIONS: renderS0 - renderS9)
      // -------------------------------------------------------------

      // S0: Cold Open (R2 = 0.977 -> "Це фізика?")
      function renderS0(t, ctx, beat) {{
        // Black background
        ctx.fillStyle = "#04060B";
        ctx.fillRect(0, 0, 1920, 1080);

        const progress = ease.clamp(t / 20.0, 0.0, 1.0);
        const cameraDrift = 1.0 + 0.03 * Math.sin(t * 0.1);

        ctx.save();
        ctx.translate(960, 540);
        ctx.scale(cameraDrift, cameraDrift);
        ctx.translate(-960, -540);

        // Draw contours progressively
        ctx.lineWidth = 2.5;
        gtPaths.forEach((cp, idx) => {{
          const delay = idx * 0.07;
          const cpProg = ease.clamp((progress - delay) / 0.35, 0.0, 1.0);
          if (cpProg <= 0) return;

          const alpha = ease.outCubic(cpProg);
          ctx.strokeStyle = idx === 0 
            ? `rgba(125, 249, 255, ${{0.9 * alpha}})` 
            : `rgba(125, 249, 255, ${{0.35 * alpha}})`;
          
          if (idx === 0) {{
            ctx.shadowColor = "#7DF9FF";
            ctx.shadowBlur = 24 * alpha;
          }} else {{
            ctx.shadowBlur = 0;
          }}
          ctx.stroke(cp.path);
        }});
        ctx.restore();

        // Counter: R2 = 0.977
        const countProg = ease.clamp(t / 8.0, 0.0, 1.0);
        const currentR2 = (0.976962 * ease.outExpo(countProg)).toFixed(3);
        const popScale = countProg >= 0.95 ? 1.0 + 0.04 * Math.sin((countProg - 0.95) * 20 * Math.PI) : 1.0;

        ctx.save();
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.translate(960, 540);
        ctx.scale(popScale, popScale);

        ctx.fillStyle = "#F2F5FA";
        ctx.font = "bold 160px 'SF Mono', 'JetBrains Mono', monospace";
        ctx.fillText(`R²ψ = ${{currentR2}}`, 0, -20);

        // Subtitle badge
        ctx.fillStyle = "#8A94A8";
        ctx.font = "40px ui-sans-serif, system-ui, sans-serif";
        ctx.fillText("Коефіцієнт детермінації карти полоїдального потоку", 0, 80);

        // Beat 2 question: "Це фізика?"
        if (t >= 15.0) {{
          const qProg = ease.clamp((t - 15.0) / 3.0, 0.0, 1.0);
          ctx.fillStyle = `rgba(255, 179, 71, ${{ease.outCubic(qProg)}})`;
          ctx.font = "bold 64px ui-sans-serif, system-ui, sans-serif";
          ctx.fillText("Це фізика?", 0, 180);
        }}
        ctx.restore();

        // Footer caption
        renderCaption(ctx, "DIII-D розряд #062, сітка 65×65 (bench/results/unet_lite.json#official.r2_psi)");
      }}

      // S1: Tokamak Cutaway Overview
      function renderS1(t, ctx, beat) {{
        // Video layer provides background; Canvas provides crisp typographic hierarchy
        const titleProg = ease.clamp(t / 4.0, 0.0, 1.0);

        ctx.save();
        renderVignette(ctx);

        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillStyle = "#F2F5FA";
        ctx.font = "bold 96px ui-sans-serif, system-ui, sans-serif";

        const textY = 380 - (1.0 - ease.outCubic(titleProg)) * 60;
        ctx.globalAlpha = ease.outCubic(titleProg);
        ctx.fillText("Рівновага плазми —", 96, textY);
        ctx.fillText("це розв'язок рівняння.", 96, textY + 110);

        ctx.fillStyle = "#7DF9FF";
        ctx.font = "44px 'SF Mono', 'JetBrains Mono', monospace";
        ctx.fillText("j × B = ∇p  (Магнітогідродинаміка)", 96, textY + 250);

        ctx.restore();
        renderCaption(ctx, "Геометрія вакуумної камери токамака DIII-D (#203702)");
      }}

      // S2: Equilibrium and Grad-Shafranov PDE
      function renderS2(t, ctx, beat) {{
        ctx.fillStyle = "#04060B";
        ctx.fillRect(0, 0, 1920, 1080);

        // Draw magnetic surfaces on the left half (x ~ 540)
        ctx.save();
        ctx.translate(480, 540);
        const drift = 1.0 + 0.015 * Math.sin(t * 0.2);
        ctx.scale(drift * 0.9, drift * 0.9);
        ctx.translate(-960, -540);

        gtPaths.forEach((cp, idx) => {{
          const alpha = 0.25 + 0.5 * Math.sin(t * 0.8 + idx * 0.3);
          ctx.strokeStyle = idx === 0 ? "#7DF9FF" : `rgba(125, 249, 255, ${{alpha}})`;
          ctx.lineWidth = idx === 0 ? 3.0 : 1.8;
          ctx.stroke(cp.path);
        }});

        // Pulse O-point
        ctx.fillStyle = "#FFC857";
        ctx.beginPath();
        const oR = 7 + 2 * Math.sin(t * 3.0);
        ctx.arc(960 + (1.7697 - 1.69) * 440, 540 - (-0.05) * 400, oR, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // Right side: Equation & Term Highlights (x >= 960)
        ctx.save();
        ctx.textAlign = "left";
        ctx.textBaseline = "top";

        ctx.fillStyle = "#8A94A8";
        ctx.font = "34px ui-sans-serif, sans-serif";
        ctx.fillText("РІВНЯННЯ ҐРЕДА–ШАФРАНОВА (E6)", 1000, 240);

        // Large Math Equation
        ctx.fillStyle = "#F2F5FA";
        ctx.font = "bold 52px 'SF Mono', monospace";
        ctx.fillText("Δ*ψ = −μ₀R²p′(ψ) − FF′(ψ)", 1000, 310);

        // Term Breakdown with highlighting
        const termStep = Math.min(3, Math.floor(t / 25));
        
        ctx.font = "36px ui-sans-serif, sans-serif";
        
        // Term 1: Δ*ψ
        ctx.fillStyle = termStep >= 1 ? "#FF6B4A" : "#8A94A8";
        ctx.fillText("• Δ*ψ : оператор других похідних у H²", 1000, 420);
        ctx.font = "30px 'SF Mono', monospace";
        ctx.fillText("  (підсилює високі частоти як k²)", 1000, 470);

        // Term 2: Pressure p'(ψ)
        ctx.font = "36px ui-sans-serif, sans-serif";
        ctx.fillStyle = termStep >= 2 ? "#7DF9FF" : "#8A94A8";
        ctx.fillText("• p′(ψ) : градієнт кінетичного тиску", 1000, 540);

        // Term 3: Poloidal current FF'(ψ)
        ctx.fillStyle = termStep >= 3 ? "#FFC857" : "#8A94A8";
        ctx.fillText("• FF′(ψ) : тороїдальний магнітний потік", 1000, 610);

        // Residual definition
        ctx.fillStyle = "#F2F5FA";
        ctx.font = "bold 36px ui-sans-serif, sans-serif";
        ctx.fillText("Нев'язка g(ψ) перевіряє рівновагу", 1000, 720);
        ctx.fillStyle = "#7DF9FF";
        ctx.fillText("лише з карти ψ — без знання тиску!", 1000, 770);

        ctx.restore();
        renderCaption(ctx, "Баланс сил у торі: j_φ = −Δ*ψ / (μ₀R). Джерело: README §5.3, E6");
      }}

      // S3: R² != Physics (Side-by-side comparison on frame #062:147)
      function renderS3(t, ctx, beat) {{
        ctx.fillStyle = "#04060B";
        ctx.fillRect(0, 0, 1920, 1080);

        const gtImg = document.getElementById("img-residual-gt");
        const unetImg = document.getElementById("img-residual-unet");

        // Split Layout: Left Ground Truth, Right UNet_Lite
        const panelW = 560;
        const panelH = 560;
        const topY = 140;

        // Panel 1: Ground Truth (Left: 280)
        ctx.save();
        ctx.strokeStyle = "rgba(125, 249, 255, 0.4)";
        ctx.lineWidth = 2;
        ctx.strokeRect(280, topY, panelW, panelH);
        if (gtImg && gtImg.complete) {{
          ctx.drawImage(gtImg, 280, topY, panelW, panelH);
        }}

        ctx.textAlign = "center";
        ctx.fillStyle = "#F2F5FA";
        ctx.font = "bold 42px ui-sans-serif, sans-serif";
        ctx.fillText("ІСТИНА (Ground Truth)", 560, topY - 25);

        ctx.font = "bold 52px 'SF Mono', monospace";
        ctx.fillStyle = "#7DF9FF";
        ctx.fillText("g = 0.0065", 560, topY + panelH + 50);
        ctx.font = "30px ui-sans-serif, sans-serif";
        ctx.fillStyle = "#8A94A8";
        ctx.fillText("Фізично гладкий розв'язок", 560, topY + panelH + 90);
        ctx.restore();

        // Panel 2: UNet_Lite (Right: 1080)
        ctx.save();
        ctx.strokeStyle = "rgba(255, 107, 74, 0.4)";
        ctx.lineWidth = 2;
        ctx.strokeRect(1080, topY, panelW, panelH);
        if (unetImg && unetImg.complete) {{
          ctx.drawImage(unetImg, 1080, topY, panelW, panelH);
        }}

        ctx.textAlign = "center";
        ctx.fillStyle = "#F2F5FA";
        ctx.font = "bold 42px ui-sans-serif, sans-serif";
        ctx.fillText("UNet_Lite (R² = 0.977)", 1360, topY - 25);

        ctx.font = "bold 52px 'SF Mono', monospace";
        ctx.fillStyle = "#FF6B4A";
        ctx.fillText("g = 0.8618", 1360, topY + panelH + 50);
        ctx.font = "30px ui-sans-serif, sans-serif";
        ctx.fillStyle = "#FFB347";
        ctx.fillText("Високочастотний шум других похідних", 1360, topY + panelH + 90);
        ctx.restore();

        // Central Warning Threshold g_ref = 0.6328
        ctx.save();
        ctx.textAlign = "center";
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "32px 'SF Mono', monospace";
        ctx.fillText("Поріг 1% шуму: g_ref = 0.6328", 960, 855);

        // Unified scale legend bar at bottom
        const barX = 660, barY = 885, barW = 600, barH = 14;
        const grad = ctx.createLinearGradient(barX, 0, barX + barW, 0);
        grad.addColorStop(0, "#000004");
        grad.addColorStop(0.2, "#320A5A");
        grad.addColorStop(0.4, "#781C6D");
        grad.addColorStop(0.6, "#BB3754");
        grad.addColorStop(0.8, "#ED6925");
        grad.addColorStop(1.0, "#FCFFA4");
        ctx.fillStyle = grad;
        ctx.fillRect(barX, barY, barW, barH);

        ctx.font = "24px monospace";
        ctx.fillStyle = "#8A94A8";
        ctx.fillText("0.0", barX, barY + 36);
        ctx.fillText("0.6328 (поріг)", barX + barW * (0.6328 / 1.2), barY + 36);
        ctx.fillText("1.20 (inferno)", barX + barW, barY + 36);

        // Dashed line at g_ref
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        const refMarkerX = barX + barW * (0.6328 / 1.2);
        ctx.beginPath();
        ctx.moveTo(refMarkerX, barY - 10);
        ctx.lineTo(refMarkerX, barY + barH + 10);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();

        renderCaption(ctx, "Розряд #062 кадр 147 (медіана тесту). Нев'язка GS обчислена з ψ на сітці 65×65 (E6, E38)");
      }}

      // S4: GS Gate Function
      function renderS4(t, ctx, beat) {{
        ctx.fillStyle = "#04060B";
        ctx.fillRect(0, 0, 1920, 1080);

        renderVignette(ctx);

        ctx.save();
        ctx.textAlign = "left";
        ctx.fillStyle = "#F2F5FA";
        ctx.font = "bold 64px ui-sans-serif, sans-serif";
        ctx.fillText("Задача 1 організаторів: Шлюз GS замість доданку", 120, 160);

        ctx.fillStyle = "#7DF9FF";
        ctx.font = "36px 'SF Mono', monospace";
        ctx.fillText("G(g) = 1.0  при g ≤ g_ref,   exp(−(g − g_ref) / g_ref)  при g > g_ref", 120, 230);

        // Plot gate curve
        const plotX = 240, plotY = 340, plotW = 1440, plotH = 480;
        ctx.strokeStyle = "rgba(138, 148, 168, 0.3)";
        ctx.lineWidth = 2;
        ctx.strokeRect(plotX, plotY, plotW, plotH);

        // Axis labels
        ctx.fillStyle = "#8A94A8";
        ctx.font = "28px monospace";
        ctx.fillText("g = 0.0", plotX - 20, plotY + plotH + 40);
        ctx.fillText("g_ref = 0.6328", plotX + plotW * (0.6328 / 1.2) - 80, plotY + plotH + 40);
        ctx.fillText("g = 1.20", plotX + plotW - 40, plotY + plotH + 40);
        ctx.fillText("G = 1.0", plotX - 100, plotY + 20);
        ctx.fillText("G = 0.0", plotX - 100, plotY + plotH);

        // Draw curve
        ctx.beginPath();
        ctx.strokeStyle = "#7DF9FF";
        ctx.lineWidth = 4;
        const g_ref = 0.6328;
        for (let px = 0; px <= plotW; px += 4) {{
          const g_val = (px / plotW) * 1.2;
          const g_mult = g_val <= g_ref ? 1.0 : Math.exp(-(g_val - g_ref) / g_ref);
          const py = plotY + plotH - g_mult * (plotH - 40) - 20;
          if (px === 0) ctx.moveTo(plotX + px, py);
          else ctx.lineTo(plotX + px, py);
        }}
        ctx.stroke();

        // Model Points on Curve
        const models = [
          {{ name: "PCA+Ridge (g=0.033)", g: 0.0331, color: "#7DF9FF", desc: "G = 1.000 (без штрафу)", above: true }},
          {{ name: "MLP (g=0.503)", g: 0.5027, color: "#FFC857", desc: "G = 0.961 (штраф 4%)", above: true }},
          {{ name: "UNet_Lite (g=0.863)", g: 0.8627, color: "#FF6B4A", desc: "G = 0.721 (штраф 28%)", above: false }}
        ];

        models.forEach(m => {{
          const px = plotX + (m.g / 1.2) * plotW;
          const g_mult = m.g <= g_ref ? 1.0 : Math.exp(-(m.g - g_ref) / g_ref);
          const py = plotY + plotH - g_mult * (plotH - 40) - 20;

          ctx.fillStyle = m.color;
          ctx.beginPath();
          ctx.arc(px, py, 12, 0, Math.PI * 2);
          ctx.fill();

          ctx.font = "bold 30px ui-sans-serif, sans-serif";
          if (m.above) {{
            ctx.fillText(m.name, px - 20, py - 60);
            ctx.font = "24px monospace";
            ctx.fillStyle = "#8A94A8";
            ctx.fillText(m.desc, px - 20, py - 25);
          }} else {{
            ctx.fillText(m.name, px + 15, py - 40);
            ctx.font = "24px monospace";
            ctx.fillStyle = "#8A94A8";
            ctx.fillText(m.desc, px + 15, py - 15);
          }}
        }});

        ctx.restore();
        renderCaption(ctx, "Реалізація Задачі 1 організаторів (team/docs/METRIC_S_PRIME.md v2.2)");
      }}

      // S5: Results & Confidence Intervals Forest Plot
      function renderS5(t, ctx, beat) {{
        ctx.fillStyle = "#04060B";
        ctx.fillRect(0, 0, 1920, 1080);

        renderVignette(ctx);

        ctx.save();
        ctx.textAlign = "left";
        ctx.fillStyle = "#F2F5FA";
        ctx.font = "bold 64px ui-sans-serif, sans-serif";
        ctx.fillText("Оцінювання моделей: Офіційний S проти S′-gate", 120, 150);

        // Forest plot layout
        const pY1 = 300;
        const pY2 = 600;
        const barX = 400;
        const scaleW = 1000;

        // Subhead 1: Official S
        ctx.font = "40px ui-sans-serif, sans-serif";
        ctx.fillStyle = "#8A94A8";
        ctx.fillText("Офіційна метрика S (8 контрольних розрядів #060–#067):", 120, pY1 - 40);

        // UNet_Lite S: 0.6476 [0.6289..0.6671]
        renderCiBar(ctx, 120, pY1 + 20, "UNet_Lite", 0.6476, 0.6289, 0.6671, "#7DF9FF", barX, scaleW);
        // MLP S: 0.6437 [0.6239..0.6622]
        renderCiBar(ctx, 120, pY1 + 100, "MLP (sklearn)", 0.6437, 0.6239, 0.6622, "#FFC857", barX, scaleW);
        // PCA+Ridge S: 0.1926 [0.1053..0.5985]
        renderCiBar(ctx, 120, pY1 + 180, "PCA+Ridge", 0.1926, 0.1053, 0.5985, "#8A94A8", barX, scaleW);

        // Highlight: CI overlap text
        ctx.fillStyle = "#FFB347";
        ctx.font = "bold 34px ui-sans-serif, sans-serif";
        ctx.fillText("ΔS = −0.0015 [−0.012 .. +0.009] : 0 входить у CI (моделі нерозрізнені)", 400, pY1 + 250);

        // Subhead 2: S'-gate
        ctx.font = "40px ui-sans-serif, sans-serif";
        ctx.fillStyle = "#8A94A8";
        ctx.fillText("Діагностична метрика S′-gate (із фізичним шлюзом GS):", 120, pY2 + 20);

        // MLP S': 0.5331 [0.4003..0.6360]
        renderCiBar(ctx, 120, pY2 + 80, "MLP (sklearn)", 0.5331, 0.4003, 0.6360, "#FFC857", barX, scaleW);
        // UNet S': 0.4202 [0.3719..0.4710]
        renderCiBar(ctx, 120, pY2 + 160, "UNet_Lite", 0.4202, 0.3719, 0.4710, "#FF6B4A", barX, scaleW);

        // Highlight: Rank Inversion text
        ctx.fillStyle = "#7DF9FF";
        ctx.font = "bold 34px ui-sans-serif, sans-serif";
        ctx.fillText("ΔS′-gate = −0.1129 [−0.204 .. −0.001] : інверсія рангів на користь фізичності", 400, pY2 + 230);

        ctx.restore();
        renderCaption(ctx, "Парний бутстреп 1000 реплік (team/LEADERBOARD.md). Застереження сплітів: TEST_SNOOPING.md");
      }}

      function renderCiBar(ctx, labelX, y, label, val, lo, hi, color, startX, width) {{
        ctx.fillStyle = "#F2F5FA";
        ctx.font = "32px 'SF Mono', monospace";
        ctx.fillText(label, labelX, y + 8);

        // Scale maps [0.0, 1.0] to [startX, startX + width]
        const xVal = startX + val * width;
        const xLo = startX + lo * width;
        const xHi = startX + hi * width;

        // Line
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(xLo, y);
        ctx.lineTo(xHi, y);
        ctx.stroke();

        // Caps
        ctx.beginPath();
        ctx.moveTo(xLo, y - 8); ctx.lineTo(xLo, y + 8);
        ctx.moveTo(xHi, y - 8); ctx.lineTo(xHi, y + 8);
        ctx.stroke();

        // Center dot
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(xVal, y, 7, 0, Math.PI * 2);
        ctx.fill();

        // Number readout
        ctx.fillStyle = color;
        ctx.font = "26px monospace";
        ctx.fillText(`${{val.toFixed(4)}} [${{lo.toFixed(4)}}..${{hi.toFixed(4)}}]`, xHi + 20, y + 8);
      }}

      // S6: Field Lines Overview
      function renderS6(t, ctx, beat) {{
        // Video layer `act3_fieldlines_final.mp4` provides primary motion
        renderVignette(ctx);

        ctx.save();
        ctx.textAlign = "left";
        ctx.fillStyle = "#F2F5FA";
        ctx.font = "bold 80px ui-sans-serif, sans-serif";
        ctx.fillText("Тороїдальні силові лінії", 120, 240);

        ctx.font = "48px 'SF Mono', monospace";
        ctx.fillStyle = "#7DF9FF";
        ctx.fillText("q = m / n  (Коефіцієнт запасу стійкості)", 120, 340);

        ctx.font = "38px ui-sans-serif, sans-serif";
        ctx.fillStyle = "#8A94A8";
        ctx.fillText("Гамільтонова система силових ліній (E19):", 120, 430);
        ctx.fillText("dR/dφ = −∂H/∂Z,   dZ/dφ = ∂H/∂R", 120, 480);
        ctx.fillText("Нульовий дрейф H та точне збереження фазового об'єму.", 120, 540);

        ctx.restore();
        renderCaption(ctx, "JAX/XLA гамільтонів трасувальник ліній (E19, E20, T12)");
      }}

      // S7: Poincare Section & Chirikov Stochasticity
      function renderS7(t, ctx, beat) {{
        ctx.fillStyle = "#04060B";
        ctx.fillRect(0, 0, 1920, 1080);

        // Native Vector Poincare Render from POINCARE_CHAOS & POINCARE_SWEEP
        ctx.save();
        // Camera centered at (680, 540)
        const pCenterX = 680, pCenterY = 540;
        ctx.translate(pCenterX, pCenterY);
        ctx.scale(1.0, 1.0);

        // Draw points from POINCARE_CHAOS
        const lines = POINCARE_CHAOS.lines || [];
        lines.forEach((l, idx) => {{
          const isIslandCore = idx < 20;
          ctx.fillStyle = isIslandCore ? "rgba(255, 200, 87, 0.75)" : "rgba(255, 107, 74, 0.55)";
          l.pts.forEach(p => {{
            // R in [1.2, 2.3], Z in [-1.0, 1.0] -> canvas offset
            const px = (p[0] - 1.75) * 580;
            const py = -p[1] * 580;
            ctx.fillRect(px, py, 2.2, 2.2);
          }});
        }});
        ctx.restore();

        // Right side: Info Panel (x >= 1160)
        ctx.save();
        ctx.textAlign = "left";
        ctx.fillStyle = "#F2F5FA";
        ctx.font = "bold 48px ui-sans-serif, sans-serif";
        ctx.fillText("ПЕРЕРІЗ ПУАНКАРЕ", 1160, 200);

        ctx.font = "38px 'SF Mono', monospace";
        ctx.fillStyle = "#FFC857";
        ctx.fillText("Моди 2/1 + 3/1", 1160, 270);

        ctx.font = "30px ui-sans-serif, sans-serif";
        ctx.fillStyle = "#8A94A8";
        ctx.fillText("Амплітуда збурення:", 1160, 360);
        ctx.fillStyle = "#F2F5FA";
        ctx.font = "bold 34px monospace";
        ctx.fillText("A = 4.8 × 10⁻³", 1160, 405);

        ctx.fillStyle = "#8A94A8";
        ctx.font = "30px ui-sans-serif, sans-serif";
        ctx.fillText("Критерій Чирикова (S):", 1160, 470);
        ctx.fillStyle = "#FF6B4A";
        ctx.font = "bold 38px monospace";
        ctx.fillText("S ≈ 1.0–1.1  (S = 1.0025)", 1160, 520);
        ctx.font = "24px ui-sans-serif, sans-serif";
        ctx.fillText("Початок перекриття островів", 1160, 560);

        ctx.fillStyle = "#8A94A8";
        ctx.font = "30px ui-sans-serif, sans-serif";
        ctx.fillText("Ширина острова 2/1:", 1160, 630);
        ctx.fillStyle = "#7DF9FF";
        ctx.font = "bold 34px monospace";
        ctx.fillText("порядку 3 см", 1160, 675);
        ctx.font = "24px ui-sans-serif, sans-serif";
        ctx.fillText("(аналітична оцінка)", 1160, 710);

        ctx.fillStyle = "#8A94A8";
        ctx.font = "30px ui-sans-serif, sans-serif";
        ctx.fillText("FTLE відношення (k):", 1160, 770);
        ctx.fillStyle = "#FFC857";
        ctx.font = "bold 34px monospace";
        ctx.fillText("k = 2.02 (сцена #203702)", 1160, 815);

        ctx.restore();
        renderCaption(ctx, "вакуумні моди 2/1 і 3/1 на рівновазі DIII-D #203702; S≈1.0–1.1: початок перекриття островів; стохастичність ймовірна, кількісно не підтверджена");
      }}

      // S8: Refuted Hypotheses Ledger
      function renderS8(t, ctx, beat) {{
        ctx.fillStyle = "#04060B";
        ctx.fillRect(0, 0, 1920, 1080);

        renderVignette(ctx);

        ctx.save();
        ctx.textAlign = "left";
        ctx.fillStyle = "#F2F5FA";
        ctx.font = "bold 64px ui-sans-serif, sans-serif";
        ctx.fillText("Книга обліку: Негативні результати як сильна наука", 120, 140);

        const rows = [
          {{ id: "R2", text: "Адитивний лосс на скаляри: не усуває неліпшицевість осі (E1)" }},
          {{ id: "R4", text: "Чистий L2-регресор на Брату: колапсує до нефізичного середнього (E4)" }},
          {{ id: "R5", text: "Глобальний МНК на Токамапі: застрягає в скляному ландшафті (E40)" }},
          {{ id: "R7", text: "Прямий декартовий перенос DIII-D → MAST: колапсує без топології (E16)" }},
          {{ id: "R12", text: "Адитивний член нев'язки S - λg: не забезпечує стійкого переупорядкування" }},
          {{ id: "P-R14", text: "Гіпотеза C7 (Deep Ritz стійкіший за PINN): спростовано, IQR вищий у 13–32 рази" }}
        ];

        let startY = 230;
        rows.forEach((r, idx) => {{
          const rowProg = ease.clamp((t - idx * 2.5) / 1.5, 0.0, 1.0);
          if (rowProg <= 0) return;

          ctx.fillStyle = "#F2F5FA";
          ctx.font = "bold 32px 'SF Mono', monospace";
          ctx.fillText(`[${{r.id}}]`, 140, startY + idx * 75);

          ctx.font = "30px ui-sans-serif, sans-serif";
          ctx.fillStyle = "#8A94A8";
          ctx.fillText(r.text, 300, startY + idx * 75);

          // Falling STAMP "СПРОСТОВАНО"
          if (rowProg >= 0.6) {{
            const stampProg = (rowProg - 0.6) / 0.4;
            const sScale = 1.0 + (1.0 - stampProg) * 0.8;
            ctx.save();
            ctx.translate(1600, startY + idx * 75 - 10);
            ctx.scale(sScale, sScale);
            ctx.rotate(-0.06);

            ctx.strokeStyle = "rgba(255, 107, 74, 0.85)";
            ctx.lineWidth = 3;
            ctx.strokeRect(-120, -22, 240, 44);

            ctx.fillStyle = "#FF6B4A";
            ctx.textAlign = "center";
            ctx.font = "bold 26px 'SF Mono', monospace";
            ctx.fillText("СПРОСТОВАНО", 0, 8);
            ctx.restore();
          }}
        }});

        // T13 Self-Audit Beat (t >= 25s)
        if (t >= 25.0) {{
          const t13Prog = ease.clamp((t - 25.0) / 2.0, 0.0, 1.0);
          ctx.fillStyle = `rgba(125, 249, 255, ${{ease.outCubic(t13Prog)}})`;
          ctx.font = "bold 36px ui-sans-serif, sans-serif";
          ctx.fillText("Самоаудит T13: за оператором релаксації g падає (0.862 → 0.384),", 140, 720);
          ctx.fillText("але за незалежною 5-точковою схемою майже не змінюється (0.980 → 0.933):", 140, 770);
          ctx.fillStyle = `rgba(255, 179, 71, ${{ease.outCubic(t13Prog)}})`;
          ctx.fillText("«покращення специфічне для оператора» (Our try/04-novelty/t13/results.json).", 140, 820);
        }}

        ctx.restore();
        renderCaption(ctx, "Реєстр зафіксованих негативних результатів R1–R13 (README §6.3, INDEX.md)");
      }}

      // S9: Finale Bookend
      function renderS9(t, ctx, beat) {{
        ctx.fillStyle = "#04060B";
        ctx.fillRect(0, 0, 1920, 1080);

        const fadeOut = ease.clamp((t - 30.0) / 10.0, 0.0, 1.0);

        ctx.save();
        ctx.globalAlpha = 1.0 - fadeOut;

        // Subtle background contour watermark on the right
        ctx.save();
        ctx.translate(1650, 480);
        ctx.scale(0.85, 0.85);
        ctx.translate(-960, -540);
        if (gtPaths.length > 0) {{
          ctx.strokeStyle = `rgba(125, 249, 255, ${{0.15 * (1.0 - fadeOut)}})`;
          ctx.lineWidth = 2.5;
          ctx.stroke(gtPaths[0].path);
        }}
        ctx.restore();

        ctx.textAlign = "left";
        ctx.fillStyle = "#F2F5FA";
        ctx.font = "bold 64px ui-sans-serif, sans-serif";
        ctx.fillText("Висновки для дослідницької спільноти", 120, 180);

        const takeaways = [
          "1. Високий R²ψ ≈ 0.98 не гарантує рівноваги: потрібен диференціальний контроль нев'язки g.",
          "2. Офіційна метрика S сліпа до других похідних: шлюз GS відновлює фізичну ієрархію моделей.",
          "3. Відкриті перевірки та спростовані гіпотези — надійний фундамент для майбутніх TokaBench."
        ];

        takeaways.forEach((text, i) => {{
          const p = ease.clamp((t - i * 4.0) / 3.0, 0.0, 1.0);
          ctx.fillStyle = `rgba(242, 245, 250, ${{ease.outCubic(p)}})`;
          ctx.font = "34px ui-sans-serif, sans-serif";
          ctx.fillText(text, 120, 320 + i * 110);
        }});

        // Attribution
        ctx.fillStyle = "#8A94A8";
        ctx.font = "30px ui-sans-serif, sans-serif";
        ctx.fillText("Blender-сцена і моделі: автори проєкту (CC BY 4.0); Three.js (MIT), шрифти (OFL)", 120, 750);
        ctx.fillText("AI-лабораторія ім. В. М. Горшкова, Фізико-математичний факультет КПІ ім. Ігоря Сікорського, 2026", 120, 800);

        ctx.restore();
        renderCaption(ctx, "ML-Hackathon: Токамак DIII-D. Репозиторій: presentation-prep");
      }}

      // Common helpers
      function renderVignette(ctx) {{
        const vGrad = ctx.createRadialGradient(960, 540, 400, 960, 540, 1100);
        vGrad.addColorStop(0, "rgba(4, 6, 11, 0.0)");
        vGrad.addColorStop(1, "rgba(4, 6, 11, 0.85)");
        ctx.fillStyle = vGrad;
        ctx.fillRect(0, 0, 1920, 1080);
      }}

      function renderCaption(ctx, text) {{
        ctx.save();
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillStyle = "#8A94A8";
        ctx.font = "36px ui-sans-serif, sans-serif";
        ctx.fillText(text, 96, 1032);
        ctx.restore();
      }}

      // -------------------------------------------------------------
      // 9. MAIN RENDER PIPELINE
      // -------------------------------------------------------------
      function renderFrame(globalTime) {{
        // Find active beat and active scene
        let currentB = allBeats[0];
        for (let i = 0; i < allBeats.length; i++) {{
          if (globalTime >= allBeats[i].globalStart && globalTime < allBeats[i].globalEnd) {{
            currentB = allBeats[i];
            currentBeatIdx = i;
            break;
          }}
          if (i === allBeats.length - 1 && globalTime >= allBeats[i].globalEnd) {{
            currentB = allBeats[i];
            currentBeatIdx = i;
          }}
        }}

        const sc = ACTIVE_SCENES[currentB.sceneIndex];
        const sceneTime = globalTime - sc.startTime;

        // Video background sync
        syncVideo(currentB.videoId || null);

        // Call pure scene renderer
        switch (currentB.sceneId) {{
          case "S0": renderS0(sceneTime, ctx, currentB); break;
          case "S1": renderS1(sceneTime, ctx, currentB); break;
          case "S2": renderS2(sceneTime, ctx, currentB); break;
          case "S3": renderS3(sceneTime, ctx, currentB); break;
          case "S4": renderS4(sceneTime, ctx, currentB); break;
          case "S5": renderS5(sceneTime, ctx, currentB); break;
          case "S6": renderS6(sceneTime, ctx, currentB); break;
          case "S7": renderS7(sceneTime, ctx, currentB); break;
          case "S8": renderS8(sceneTime, ctx, currentB); break;
          case "S9": renderS9(sceneTime, ctx, currentB); break;
          default:   renderS0(sceneTime, ctx, currentB); break;
        }}

        // Film grain overlay
        updateGrain(performance.now());
        ctx.save();
        ctx.globalCompositeOperation = "screen";
        ctx.globalAlpha = 0.035;
        const pat = ctx.createPattern(grainCanvas, "repeat");
        ctx.fillStyle = pat;
        ctx.fillRect(0, 0, 1920, 1080);
        ctx.restore();

        // Update HUD
        const mins = Math.floor(globalTime / 60);
        const secs = Math.floor(globalTime % 60);
        const totMins = Math.floor(totalFilmDuration / 60);
        const totSecs = Math.floor(totalFilmDuration % 60);
        hudTimeEl.textContent = `${{String(mins).padStart(2, '0')}}:${{String(secs).padStart(2, '0')}} / ${{String(totMins).padStart(2, '0')}}:${{String(totSecs).padStart(2, '0')}}`;
        sceneTitleEl.textContent = `${{currentB.sceneId}} ${{currentB.sceneTitle}}`;
        beatNumEl.textContent = currentB.beatIndex + 1;
        beatTotalEl.textContent = currentB.beatTotal;
        modeBadge.textContent = isPresenterMode ? "PRESENTER" : "FILM";

        if (isDebug) {{
          debugPanel.querySelector("#dbg-scene").textContent = `Scene: ${{currentB.sceneId}}, Beat: ${{currentB.beatIndex + 1}}, t: ${{globalTime.toFixed(2)}}s`;
        }}
      }}

      // -------------------------------------------------------------
      // 10. ANIMATION LOOP & CLOCK
      // -------------------------------------------------------------
      function loop(now) {{
        const dt = (now - lastTimestamp) / 1000.0;
        lastTimestamp = now;

        // FPS tracking
        frameTimes.push(dt);
        if (frameTimes.length > 60) frameTimes.shift();
        if (isDebug && frameTimes.length > 10) {{
          const avgDt = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
          const fps = Math.round(1.0 / avgDt);
          const sorted = [...frameTimes].sort((a, b) => a - b);
          const p95 = (sorted[Math.floor(sorted.length * 0.95)] * 1000).toFixed(1);
          debugPanel.querySelector("#dbg-fps").textContent = `FPS: ${{fps}} (p95: ${{p95}} ms)`;
        }}

        if (isPlayingFilm) {{
          currentGlobalTime += dt;
          if (currentGlobalTime >= totalFilmDuration) {{
            currentGlobalTime = totalFilmDuration;
            isPlayingFilm = false;
          }}
          postSyncState();
        }}

        renderFrame(currentGlobalTime);
        requestAnimationFrame(loop);
      }}

      // -------------------------------------------------------------
      // 11. KEYBOARD NAVIGATION & PRESENTER CONTROLS
      // -------------------------------------------------------------
      window.addEventListener("keydown", e => {{
        // Arrow Right: Next Beat
        if (e.key === "ArrowRight") {{
          e.preventDefault();
          if (currentBeatIdx < allBeats.length - 1) {{
            currentBeatIdx++;
            currentGlobalTime = allBeats[currentBeatIdx].globalStart;
            postSyncState();
          }}
        }}
        // Arrow Left: Previous Beat
        else if (e.key === "ArrowLeft") {{
          e.preventDefault();
          if (currentBeatIdx > 0) {{
            currentBeatIdx--;
            currentGlobalTime = allBeats[currentBeatIdx].globalStart;
            postSyncState();
          }}
        }}
        // Space: Toggle Play / Pause in Film mode
        else if (e.code === "Space") {{
          e.preventDefault();
          isPlayingFilm = !isPlayingFilm;
          isPresenterMode = !isPlayingFilm;
          postSyncState();
        }}
        // 'F': Toggle Fullscreen
        else if (e.key === "f" || e.key === "F") {{
          e.preventDefault();
          if (!document.fullscreenElement) {{
            document.documentElement.requestFullscreen().catch(() => {{}});
          }} else {{
            document.exitFullscreen().catch(() => {{}});
          }}
        }}
        // 'N': Open Presenter Notes Window
        else if (e.key === "n" || e.key === "N") {{
          e.preventDefault();
          openNotesWindow();
        }}
        // '[' and ']': Step ±5s
        else if (e.key === "[") {{
          e.preventDefault();
          currentGlobalTime = Math.max(0, currentGlobalTime - 5);
        }}
        else if (e.key === "]") {{
          e.preventDefault();
          currentGlobalTime = Math.min(totalFilmDuration, currentGlobalTime + 5);
        }}
        // Digits 0..9 jump to scene
        else if (e.key >= "0" && e.key <= "9") {{
          const scNum = parseInt(e.key, 10);
          const targetScene = ACTIVE_SCENES[scNum];
          if (targetScene) {{
            const targetBeat = allBeats.find(b => b.sceneId === targetScene.id);
            if (targetBeat) {{
              currentBeatIdx = allBeats.indexOf(targetBeat);
              currentGlobalTime = targetBeat.globalStart;
              postSyncState();
            }}
          }}
        }}
        // Home / End
        else if (e.key === "Home") {{
          e.preventDefault();
          currentBeatIdx = 0;
          currentGlobalTime = 0;
          postSyncState();
        }}
        else if (e.key === "End") {{
          e.preventDefault();
          currentBeatIdx = allBeats.length - 1;
          currentGlobalTime = allBeats[currentBeatIdx].globalStart;
          postSyncState();
        }}
      }});

      // Show HUD momentarily on mouse movement
      let hudTimer = null;
      window.addEventListener("mousemove", () => {{
        hudOverlay.classList.add("visible");
        clearTimeout(hudTimer);
        hudTimer = setTimeout(() => hudOverlay.classList.remove("visible"), 2500);
      }});

      // -------------------------------------------------------------
      // 12. PRESENTER NOTES POPUP WINDOW
      // -------------------------------------------------------------
      let notesWindow = null;
      function openNotesWindow() {{
        if (notesWindow && !notesWindow.closed) {{
          notesWindow.focus();
          return;
        }}
        notesWindow = window.open("", "PresenterNotes", "width=850,height=700");
        if (!notesWindow) return;

        notesWindow.document.body.innerHTML = `
          <div id="notes-timer" style="font-family: monospace; font-size: 54px; font-weight: bold; padding: 16px 24px; border-radius: 8px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center; background: #04060B; border: 2px solid #22c55e; color: #22c55e;">
            <span id="t-clock">00:00</span>
            <span style="font-size: 22px;">ЛІМІТ: 12:00</span>
          </div>
          <div id="notes-scene" style="font-size: 20px; color: #7DF9FF; margin-bottom: 8px; text-transform: uppercase;">СЦЕНА S0</div>
          <div id="notes-prompt" style="font-size: 32px; font-weight: bold; margin-bottom: 16px; color: #FFFFFF;">Завантаження...</div>
          <div id="notes-body" style="font-size: 24px; color: #A0ABC0; background: rgba(255, 255, 255, 0.05); padding: 18px; border-radius: 6px; margin-bottom: 24px; line-height: 1.5;">...</div>
          <div style="border-top: 1px solid rgba(255, 255, 255, 0.15); padding-top: 16px; color: #64748b; font-size: 18px;">
            <strong>НАСТУПНИЙ БІТ:</strong>
            <div id="notes-next" style="margin-top: 6px; color: #8A94A8;">...</div>
          </div>
        `;
        notesWindow.document.body.style.background = "#0A0F1C";
        notesWindow.document.body.style.color = "#F2F5FA";
        notesWindow.document.body.style.fontFamily = "system-ui, sans-serif";
        notesWindow.document.body.style.padding = "24px";
        notesWindow.document.body.style.margin = "0";
        notesWindow.document.title = "Нотатки доповідача (12:00 Таймер)";
        postSyncState();
      }}

      // -------------------------------------------------------------
      // 13. HEADLESS CAPTURE MODE SUPPORT (?capture=1&t=SECONDS)
      // -------------------------------------------------------------
      if (captureTime !== null) {{
        currentGlobalTime = captureTime;
        // Immediate synchronous render for headless screenshots
        renderFrame(currentGlobalTime);
        // Also schedule on next tick
        setTimeout(() => {{
          renderFrame(currentGlobalTime);
          console.log(`[CAPTURE_DONE] t=${{captureTime}} rendered.`);
        }}, 50);
      }} else {{
        // Start live animation loop
        requestAnimationFrame(loop);
      }}

    }})();
  </script>
</body>
</html>
'''

    out_path = FILM_DIR / "index.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ Generated {out_path} ({len(html_content):,} bytes)")

if __name__ == "__main__":
    build()
