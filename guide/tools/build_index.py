#!/usr/bin/env python3
"""Будує тіло таблиці «Покажчик пунктів реєстрів» (guide/appendix/index_rows.tex).

ID і статуси ПАРСЯТЬСЯ з реєстрів (щоб покажчик був повним і ID збігалися);
короткий зміст («суть»), посилання на код і розділ путівника задано вручну в SUM нижче.
Скрипт падає (код 1), якщо:
  * у реєстрі є ID, якого немає в SUM (пункт загубився б мовчки), або навпаки;
  * посилання на код указує на неіснуючий файл або функцію/клас, якої у файлі немає.

Використання:  python3 tools/build_index.py          (з каталогу guide/)
"""
from __future__ import annotations

import re
import sys
import pathlib

GUIDE = pathlib.Path(__file__).resolve().parent.parent
ROOT = GUIDE.parent
PREFIX = {"ourtry/": "Our try/", "viz/": "tokamak-3d-viz/", "gs/": "Tokamak-GS-solver/",
          "fec/": "fusion equilibrium challenge/", "manualcode/": "manual/code/"}

REG = {
    "est": ROOT / "Our try/00-decisions/ESTABLISHED.md",
    "conj": ROOT / "Our try/00-decisions/CONJECTURES.md",
    "dec": ROOT / "Our try/00-decisions/DECISIONS.md",
    "oq": ROOT / "Our try/00-decisions/open-questions.md",
    "ver": ROOT / "Our try/99-bibliography/verification-log.md",
    "viz": ROOT / "tokamak-3d-viz/findings.md",
    "coil": ROOT / "tokamak-3d-viz/docs/COIL-BACKEND.md",
}


# ---------------------------------------------------------------------------
# парсери: повертають {ID: (файл, рядок, статус)}
# ---------------------------------------------------------------------------
def _lines(key):
    return REG[key].read_text(encoding="utf-8").splitlines()


def parse_est():
    out, sec = {}, None
    for i, ln in enumerate(_lines("est"), 1):
        if ln.startswith("## A."):
            sec = "A"
        elif ln.startswith("## B."):
            sec = "B"
        elif ln.startswith("## C."):
            sec = "C"
        elif ln.startswith("## English"):
            sec = None
        m = re.match(r"^\| \*\*([ER]\d+)\*\* \|", ln)
        if not m or sec is None:
            continue
        iid = m.group(1)
        if sec == "A":
            st = "встановлено (виміряно)"
            if "Статус: виправлено" in ln:
                st = "виправлено частково" if "(частково)" in ln else "виправлено"
            elif "Відтворено 2026-09-30" in ln:
                st = "встановлено; відтворено 2026-09-30"
            elif "Обмеження 2026-09-30" in ln:
                st = r"встановлено; лише 3 розряди ($\to$R10)"
            elif "Уточнення 2026-09-30" in ln:
                st = r"встановлено; уточнено ($\to$E36)"
            elif "Без відтворюваного скрипта" in ln:
                st = "встановлено; без скрипта"
            elif "Уточнення 2026-09-29 (N4)" in ln:
                st = "встановлено; уточнено (N4)"
            elif "Було 0,016" in ln:
                st = "встановлено; число уточнено"
            elif "Примітка 2026-09-29 (k = 50" in ln:
                st = "встановлено; примітка ($k=50$)"
            elif "Застереження 2026-09-29" in ln:
                st = "встановлено; із застереженням"
        elif sec == "B":
            st = "встановлено (прочитано)"
        else:
            st = "спростовано"
            if "Повторилося" in ln:
                st = "спростовано; рецидив виправлено"
            elif "Корінь знайдено" in ln:
                st = r"спростовано; корінь знайдено ($\to$E35)"
            elif "колишня C8" in ln:
                st = r"спростовано (артефакт; $\leftarrow$C8)"
            elif "колишня C1" in ln:
                st = r"спростовано ($\leftarrow$C1)"
            elif "половина C4" in ln:
                st = r"спростовано ($\leftarrow$половина C4)"
            elif "колишня C2" in ln:
                st = r"спростовано ($\leftarrow$C2)"
            elif "Механізм відтворено 2026-09-30" in ln:
                st = r"спростовано; механізм відтворено ($\to$E36)"
        out[iid] = ("ESTABLISHED.md", i, st)
    return out


def parse_conj():
    out, deferred = {}, False
    for i, ln in enumerate(_lines("conj"), 1):
        if ln.startswith("## Здогадки, відкладені"):
            deferred = True
        m = re.match(r"^\| \*\*(C\d+)\*\* \|", ln)
        if not m:
            continue
        if deferred:
            st = "відкладено"
        elif "переміщено 2026-09-29" in ln:
            st = r"закрито: спростовано ($\to$R9)"
        elif "закрито 2026-09-30" in ln:
            st = {"C1": r"закрито: спростовано ($\to$R10)", "C2": r"закрито: спростовано ($\to$R11)"}[m.group(1)]
        elif "закрито 2026-10-01" in ln and m.group(1) == "C5":
            st = r"закрито: спростовано ($\to$R13)"
        elif "2026-09-30 (побічно" in ln:
            st = r"активна; E14 відновлено, $\lambda$ широкий"
        elif "НЕ закрита" in ln:
            st = "активна; пілот, не закрито"
        elif "ПОТРЕБУЄ ДЖЕРЕЛА" in ln:
            st = "активна; потрібні дані"
        elif "Результат 2026-10-01: половину C4" in ln:
            st = r"активна лише Calibration (GS $\to$R12)"
        elif "Частковий результат" in ln:
            st = "активна; частковий результат"
        elif "без відтворюваного" in ln.lower() or "не має відтворюваного" in ln:
            st = "активна; залежить від E14"
        else:
            st = "активна"
        out[m.group(1)] = ("CONJECTURES.md", i, st)
    return out


def parse_dec():
    out = {}
    for i, ln in enumerate(_lines("dec"), 1):
        m = re.match(r"^\| \*\*(D\d+)\*\* \|", ln)
        if m:
            st = "прийнято; закрито" if "ЗАКРИТО" in ln else "прийнято"
            out[m.group(1)] = ("DECISIONS.md", i, st)
    return out


def parse_oq():
    out = {}
    for i, ln in enumerate(_lines("oq"), 1):
        m = re.match(r"^\| (?:✔ )?(?:\*\*)?(Q\d+)(?:\*\*)? \|", ln)
        if not m:
            continue
        last = ln.rstrip(" |").split("|")[-1].strip()
        if "закрите" in last:
            st = "закрите"
        elif "блокує" in last:
            st = "відкрите; блокує перевипуск"
        elif "не публікувати" in last:
            st = "відкрите; не публікувати похідне"
        else:
            st = "відкрите"
        out[m.group(1)] = ("open-questions.md", i, st)
    return out


VERDICT = [("ПІДТВЕРДЖЕНО (попередньо)", "підтверджено (попередньо)"),
           ("ПІДТВЕРДЖЕНО, з", "підтверджено із застереженнями"),
           ("СПРОСТОВАНО", "спростовано"),
           ("ЧАСТКОВО ІСНУЄ", "частково існує"),
           ("ФОРМУЛЮВАННЯ БУЛО ХИБНЕ", "чернетку спростовано")]


def parse_ver():
    out, cur = {}, None
    for i, ln in enumerate(_lines("ver"), 1):
        m = re.match(r"^### (V-[AD]\d)\.", ln)
        if m:
            cur = m.group(1)
            out[cur] = ("verification-log.md", i, "перевірено (власні виміри)")
            continue
        if cur and ln.startswith("**ВЕРДИКТ"):
            for k, v in VERDICT:
                if k in ln:
                    out[cur] = (out[cur][0], out[cur][1], v)
                    break
            cur = None
        m = re.match(r"^\| \*\*(V-[BC]\d)\*\* \|", ln)
        if m:
            out[m.group(1)] = ("verification-log.md", i, "перевірено [V]" if "[V]" in ln else "перевірено")
    return out


def parse_viz():
    out, sec = {}, None
    for i, ln in enumerate(_lines("viz"), 1):
        if ln.startswith("## ВСТАНОВЛЕНО (виміряно"):
            sec = "VE"
        elif ln.startswith("## ВСТАНОВЛЕНО (прочитано"):
            sec = "VEr"
        elif ln.startswith("## СПРОСТОВАНІ"):
            sec = "VR"
        elif ln.startswith("## ГІПОТЕЗИ"):
            sec = "VC"
        elif ln.startswith("## Відкрите"):
            sec = "VQ"
        elif ln.startswith("## English"):
            sec = None
        m = re.match(r"^\*\*(V[ERCQ]\d+a?)[\.\s]", ln)
        if not m or sec is None:
            continue
        iid = m.group(1)
        if sec in ("VE", "VEr"):
            st = "встановлено (виміряно)" if sec == "VE" else "встановлено (прочитано)"
        elif sec == "VR":
            st = "спростовано"
            if "Виправлено 2026-09-29" in ln or iid == "VR15":
                st = "спростовано; код виправлено"
        elif sec == "VC":
            if "ЗАКРИТО, спростовано" in ln:
                st = "закрито: спростовано"
            elif "ЗАКРИТО, підтверджено" in ln:
                st = "закрито: підтверджено"
            else:
                st = "відкрито"
            if iid in out and out[iid][2].startswith("закрито"):
                continue          # перша (закривальна) згадка важливіша за початкове формулювання
        else:
            st = "відкрите"
        if iid in out and sec != "VC":
            continue
        out[iid] = ("findings.md", i, st)
    return out


# U1–U3 у COIL-BACKEND.md не мають номерів у джерелі — прив'язуємо до якорів тексту
U_ANCHORS = {
    "U1": ("Мою гіпотезу про причину спростовано", "спростовано"),
    "U2": ("Дві правдоподібні причини лишаються нерозділеними", "відкрите"),
    "U3": ("## 2. Виправлення: тримати векторний потенціал", "встановлено (виміряно)"),
}


def parse_coil():
    out = {}
    lines = _lines("coil")
    for iid, (anchor, st) in U_ANCHORS.items():
        hits = [i for i, ln in enumerate(lines, 1) if anchor in ln]
        if not hits:
            sys.exit(f"якір {iid} не знайдено в COIL-BACKEND.md: {anchor!r}")
        out[iid] = ("COIL-BACKEND.md", hits[0], st)
    return out


# ---------------------------------------------------------------------------
# ручна частина: ID -> (суть ≤ 12 слів, код "шлях:функція" або None, розділи)
# розділи: список номерів путівника (1..10); порожній список = «лише покажчик»
# ---------------------------------------------------------------------------
D1P = "ourtry/03-deep-dives/D1-psi-to-scalars/conditioning.py"
C1P = "ourtry/03-deep-dives/D1-psi-to-scalars/c1/"
C2P = "ourtry/03-deep-dives/D1-psi-to-scalars/c2/"
C4P = "ourtry/04-novelty/c4/"
D2P = "ourtry/03-deep-dives/D2-gs-nonuniqueness/branch_mean.py"
D3P = "ourtry/03-deep-dives/D3-poincare-inverse/"
NOV = "ourtry/04-novelty/"
VZ = "viz/src/"
TV = "viz/src/tokviz/"

SUM = {
    # --- ESTABLISHED A/B -------------------------------------------------
    "E1": (r"Відображення $\psi\to$ скаляри не ліпшицеве в $L^2$", D1P + ":scalars_of", [2]),
    "E2": (r"Показники $\alpha$ розпадаються на три кластери", D1P + ":main", [2]),
    "E3": (r"Хвіст PCA шкідливіший за білий шум тієї ж енергії", D1P + ":pert_pca_tail", [2]),
    "E4": (r"$L^2$-середнє двох гілок не є розв'язком", D2P + ":rel_residual_bratu", [3]),
    "E5": (r"Точка складки Братý $\lambda^*$ відтворена до останньої цифри", D2P + ":lam_star", [3]),
    "E6": (r"GS-нев'язка обчислювана лише з поданого $\psi$", NOV + "gs_residual_probe.py:gs_inconsistency", [4]),
    "E7": (r"GS-член інваріантний до масштабу $\psi$", NOV + "gs_residual_probe.py:gs_inconsistency", [4]),
    "E8": (r"GS-член сліпий до гладкої похибки й PCA-усічення (PCA-5: 0,0093)", NOV + "pca5_check.py", [4]),
    "E9": (r"Скелет істини канонічний: 1 O-точка, 0 сідел", NOV + "topology_probe.py:critical_points", [5]),
    "E10": (r"Скелет стійкий до 3\,\% шуму, руйнується на 10\,\%", NOV + "topology_probe.py:critical_points", [5]),
    "E11": (r"Сума індексів зберігається й у хаосі; інформативна лише кількість", NOV + "topology_probe.py:_winding", [5]),
    "E12": (r"На істині $\chi_{\mathrm{E}}(\psi\le t)\equiv1$ на всіх рівнях", NOV + "topology_ec_curve.py:ec_curve", [5]),
    "E13": (r"Крива Ейлера менш чутлива за лічильник критичних точок", NOV + "topology_ec_curve.py:ec_curve", [5]),
    "E14": (r"Зміщення осі під шумом $\propto1/\sqrt{\lambda}$ (відтворено для $R$)", C1P + "local_models.py:part_a", [2]),
    "E15": (r"Топологічний критерій машинно-незалежний (DIII-D і MAST)", NOV + "h5_transfer_topology.py:skeleton", [6]),
    "E16": (r"Топологія переживає перенесення, значення $\psi$ --- ні", NOV + "h5_transfer_topology.py:skeleton", [6]),
    "E17": (r"Харнес скорера цілий: perfect $\to1$, zeros $\to0$", "fec/starter/local_score.py:score_shot", []),
    "E18": (r"В осесиметрії перетин Пуанкаре --- рівно контури $\psi$", TV + "fieldline.py:invariant_drift", [7]),
    "E19": (r"Імпульс --- тороїдальний потік, гамільтоніан --- полоїдальний", TV + "perturbation.py:island_width_psin", [7, 8]),
    "E20": (r"Одна резонансна мода не дає хаосу", None, [7]),
    "E21": (r"Розв'язок GS може бути неєдиним на реальній геометрії", None, [3]),
    "E22": (r"Мала $L^2$-похибка не тягне малого фізичного резидуалу", None, [4]),
    "E23": (r"Жоден фузійний бенчмарк не оцінює резидуал PDE", None, [4]),
    "E24": (r"Стан Біна опуклий і однозначний; задача плазми --- ні", None, [3]),
    "E25": (r"Прогноз зриву насичений і має закриті дані", None, [1]),
    "E26": (r"Корпус Sophelio: 103,86~ГБ, 9\,121 розряд", None, [1]),
    "E27": (r"Consistency усереднює сім скалярів, не вісім", None, [1]),
    "E28": (r"Ліцензії HDB5, ConStellaration, Sophelio, FAIR-MAST", None, [1, 6]),
    "E29": (r"Код starter kit справді MIT", None, [1]),
    "E30": (r"Tokamap реалізовано; він точно симплектичний", D3P + "tokamap.py:jacobian_det", [7]),
    "E31": (r"Спостережуване на одній фазі непридатне; «540$\times$» порівнює різні стенди", D3P + "identifiability.py:profile", [7]),
    "E32": (r"Відновлення однієї амплітуди тривіальне (SNR 57)", D3P + "identifiability.py:main", [7]),
    "E33": (r"Функція Гріна GS-solver брала $k$ замість $m=k^2$", "gs/src/numerical/compute.py:GreenFunction", [4, 10]),
    "E34": (r"GS-solver не запускався з README; полагоджено частково", "gs/src/numerical/grad_shafranov.py:GSsolverFreeBoundary", [10]),
    "E35": (r"Корінь R8: поправка $T'$ бере первісну $h(T)$, а не $V'(T)$", D3P + "multimode_fixed_check.py:step_fixed", [7]),
    "E37": (r"UNet: $R^2_\psi=0{,}976$, але Consistency 0,179 < PCA+Ridge 0,270", C2P + "evaluate.py:main", [2]),
    "E38": (r"Мережі: $R^2_\psi\approx0{,}97$, але $g=0{,}81$--$0{,}96$ --- не рівноваги", C4P + "evaluate.py:main", [4]),
    "E39": (r"Вкладені поверхні без осесиметрії можливі (Landreman 2026; перевірено власним кодом)", "ourtry/99-bibliography/landreman2026/verify_landreman.py:check", [7]),
    "E40": (r"Ландшафт нев'язки екскурсії «скляний»: базовий фіт застрягає біля істини", D3P + "c5/stand_tokamap.py:observable", [7]),
    "E36": (r"Похибка осі: нахил 1 при $\eta<\eta^*=\lambda h^2/2$, $\approx\tfrac12$ вище", C1P + "local_models.py:part_a", [2]),
    # --- ESTABLISHED C (спростоване) --------------------------------------
    "R1": (r"«Ніхто не нав'язує інтегральну крайову умову жорстко»", None, [1]),
    "R2": (r"«Бін і вільна межа плазми --- одна математика»", None, [3]),
    "R3": (r"«Симплектична нейромережа для силових ліній --- наша ідея»", None, [7]),
    "R4": (r"«$\psi$ як канонічний імпульс»; повторилося в коді viz", TV + "perturbation.py:island_width_psin", [7, 8]),
    "R5": (r"«Крива Ейлера краща за лічильник критичних точок»", NOV + "topology_ec_curve.py:ec_curve", [5]),
    "R6": (r"«$\alpha=\tfrac12$ --- універсальний закон» (механізм відтворено)", C1P + "local_models.py:part_a", [2]),
    "R7": (r"«Провал перенесення має топологічну складову» (H5)", NOV + "h5_transfer_topology.py:skeleton", [6]),
    "R8": (r"«Tokamap переноситься на суму мод підстановкою»", D3P + "multimode.py:det_J", [7]),
    "R9": (r"«Провали LCFS MAST --- систематичний сигнал» (колишня C8)", NOV + "c8_sign_check.py", [6]),
    "R11": (r"«Спектральна вага з $S(k)$ покращує моделі» (колишня C2)", C2P + "train.py:unet", [2]),
    "R12": (r"«GS\_score змінить упорядкування бейзлайнів» (половина C4)", C4P + "evaluate.py:main", [4]),
    "R13": (r"«Обернена задача Рівня~3 трактабельна за 48~год» (колишня C5)", D3P + "c5/fit_common.py:run_one", [7, 10]),
    "R10": (r"«Три кластери $\alpha$ --- три механізми» (колишня C1)", C1P + "real_sweep.py:run_machine", [2]),
    # --- CONJECTURES -------------------------------------------------------
    "C1": (r"Три кластери $\alpha$ відповідають трьом механізмам ($\to$R10)", C1P + "real_sweep.py:run_machine", [2, 10]),
    "C2": (r"Спектрально зважена втрата покращує реальні моделі ($\to$R11)", C2P + "train.py:unet", [2, 10]),
    "C3": (r"Число обумовленості $1/\sqrt{\lambda}$ придатне для стратифікації тесту", None, [2, 10]),
    "C4": (r"Калібрування змінить упорядкування бейзлайнів (GS-половина $\to$R12)", C4P + "evaluate.py:main", [4, 10]),
    "C5": (r"Обернена задача Рівня~3 трактабельна за 48~год ($\to$R13)", VZ + "run_inverse.py:observable", [7, 10]),
    "C6": (r"Одночасні конформні смуги --- правильна форма UQ-члена", None, [4, 10]),
    "C7": (r"Deep Ritz нестабільний на плазмі, стабільний на Біні", None, [3, 10]),
    "C8": (r"Провали вилучення LCFS на MAST --- систематичний сигнал ($\to$R9)", NOV + "h5_transfer_topology.py:skeleton", [6, 10]),
    "C9": (r"Коваріантне подання покращить перенесення між машинами", None, [6, 10]),
    "C10": (r"Частина похибки моделей --- успадковане зміщення EFIT", None, [6, 10]),
    # --- open questions ----------------------------------------------------
    "Q1": (r"Чи реалістична реконструкція $\psi$ без магнітної діагностики", None, [10]),
    "Q2": (r"Наскільки виродження $p'$/$FF'$ псує реконструкцію", None, [10]),
    "Q3": (r"Чи трапляється неєдиність GS у робочих режимах", None, [3, 10]),
    "Q4": (r"Реалістичні $\tau$ і $\Delta H$ для NbTi/Nb$_3$Sn", None, [10]),
    "Q5": (r"Наскільки камера екранує швидке $\dd B/\dd t$", None, [10]),
    "Q6": (r"Доступ до кластера AI-лабораторії", None, [10]),
    "Q7": (r"Дати, місце, розмір команди, журі", None, [10]),
    "Q8": (r"Чи потрібна платформа зі скорингом", None, [10]),
    "Q9": (r"Хто реально будує starter kit і скільки годин", None, [10]),
    "Q10": (r"Умови поширення даних \texttt{fusionsimulator.io}", None, [1, 10]),
    "Q11": (r"Ліцензія ITPA HDB5 (CC BY 4.0)", None, [1]),
    "Q12": (r"Атрибуція скорера й даних Sophelio", None, [1]),
    "Q13": (r"Перевипуск MAST із CC BY-SA у CC BY --- підстава?", None, [1, 6, 10]),
    # --- decisions ---------------------------------------------------------
    "D1": (r"Ядро --- реконструкція $\psi$, а не прогноз зриву", None, [1]),
    "D2": (r"Наскрізний конвеєр --- лише в наративі", None, [1]),
    "D3": (r"Два треки: базовий і дослідницький", None, [1]),
    "D4": (r"Дані --- гібрид sim-to-real", None, [1]),
    "D5": (r"Синтетика з FreeGS/FreeGSNKE/TokaLab, без власного генератора", None, [1]),
    "D6": (r"Метрика $S'$: Sophelio + GS-резидуал + калібрування UQ", None, [1, 4]),
    "D7": (r"Артефакти двомовні", None, [1]),
    "D8": (r"Надпровідність --- лише як опуклий контроль (Бін) у D2", None, [1, 3]),
    "D9": (r"Квенч магніта --- не оцінювана задача", None, [1]),
    # --- verification log --------------------------------------------------
    "V-A1": (r"Deep Ritz до рівняння GS не застосовували (пошук arXiv)", None, []),
    "V-A2": (r"Жорстка інтегральна умова вже є (McClenaghan 2024; $\to$R1)", None, []),
    "V-A3": (r"Патологію «середнього гілок» для GS не квантифіковано", None, []),
    "V-A4": (r"Жоден фузійний ML-бенчмарк не оцінює резидуал GS ($\to$E23)", None, []),
    "V-D1": (r"Симплектична мережа для силових ліній --- HénonNet ($\to$R3)", None, []),
    "V-D2": (r"Топологічна вісь: новизна застосування, не методу", None, []),
    "V-D3": (r"Канонічні змінні: «$\psi$ як імпульс» хибне ($\to$R4)", None, [8]),
    "V-D4": (r"Власні виміри топології ($\to$E9--E13, R5)", None, []),
    "V-B1": (r"Обсяг корпусу: 103,86~ГБ ($\to$E26)", None, []),
    "V-B2": (r"Кількість розрядів: 9\,121 ($\to$E26)", None, []),
    "V-B3": (r"Consistency: сім скалярів ($\to$E27)", None, []),
    "V-B4": (r"Канонічний репозиторій starter kit --- Sophelio", None, []),
    "V-B5": (r"Топові бали недоступні без облікового запису", None, []),
    "V-C1": (r"ITPA HDB5 --- CC BY 4.0 ($\to$Q11)", None, []),
    "V-C2": (r"ConStellaration --- MIT", None, []),
    "V-C3": (r"Sophelio FEC --- CC BY 4.0", None, []),
    "V-C4": (r"FAIR-MAST --- CC BY-SA 4.0 ($\to$Q13)", None, []),
    # --- viz: VE -----------------------------------------------------------
    "VE1": (r"Пошук осі збігається з EFIT до 0,0002~мм", TV + "equilibrium.py:find_axis", [8]),
    "VE2": (r"LCFS набору даних --- справді контур $\psi$", TV + "equilibrium.py:set_boundary_from_contour", [8]),
    "VE3": (r"$F$ з $q_{95}$ дає $B_\varphi=1{,}927$~Тл без знання $B_0$", TV + "equilibrium.py:calibrate_F", [8]),
    "VE4": (r"$q$ контурним інтегралом і трасуванням збігаються до 0,16\,\%", TV + "fieldline.py:q_from_tracing", [8]),
    "VE5": (r"Інтегратор зберігає $\psi$ до $10^{-7}$ за 200 обертів", TV + "fieldline.py:invariant_drift", [8]),
    "VE6": (r"Растеризація $\cos m\theta^*$ на 65$\times$65 дає фальшиві острови", TV + "perturbation.py:Perturbation", [8]),
    "VE7": (r"Аналітичні похідні збурення збігаються зі скінченними різницями", "viz/tests/test_physics.py:test_perturbation_derivatives_match_finite_difference", [8]),
    "VE8": (r"Ланцюг $m/n$ дає рівно $m$ островів у розрізі", TV + "islands.py:classify_fixed_points", [8]),
    "VE9": (r"Чириков $S=0{,}458<1$ (було 0,714); поверхні виживають", TV + "perturbation.py:chirikov", [8]),
    "VE10": (r"Перший рендер Cycles на Metal: $\sim$100~с компіляції", None, []),
    "VE11": (r"Продуктивність Cycles на M4: GPU лише $\sim$1,2$\times$", None, []),
    "VE11a": (r"Час рендера з тестової сцени занижено вдвічі", None, []),
    "VE12": (r"Blender не має імпортера STEP/IGES", None, []),
    "VE13": (r"Зміни API Blender 4.2$\to$5.x, що ламають скрипти", None, []),
    "VE14": (r"EEVEE: об'ємна емісія не освітлює поверхні", None, []),
    "VE15": (r"ITER-2024: перша стінка --- вольфрам замість берилію", None, []),
    "VE16": (r"Число обертання трасуванням відтворює $1/q$ до 0,03\,\%", TV + "analysis.py:rotation_number", [8]),
    "VE17": (r"Хаотичні орбіти групуються рівно на резонансах", TV + "analysis.py:classify", [8]),
    "VE18": (r"Виправлена маятникова ширина збігається з виміряною до 0--5\,\%", TV + "analysis.py:measured_island_width", [8]),
    "VE19": (r"Хаос існує задовго до $S=1$; перехід через 1 не перевірено", TV + "analysis.py:classify", [8]),
    "VE20": (r"Згущення засіву завищує частку хаосу в 1,6--1,9 раза", VZ + "run_analysis.py:main", [8]),
    "VE21": (r"$\Bvec=\nabla\times(\psi\nabla\varphi)$ бездивергентне для будь-якого числа мод", TV + "fieldline.py:FieldLines", [7]),
    "VE22": (r"Відновлення самих амплітуд тривіальне (обумовленість 1,2)", VZ + "run_inverse.py:observable", [7]),
    "VE23": (r"Складність оберненої задачі задає спостережуване, а не фізика", VZ + "run_inverse.py:observable", [7]),
    "VE24": (r"Перевага роздільності = фазова інформація + обсяг даних", VZ + "run_inverse.py:main", [7]),
    "VE25": (r"Затиснута мода 5/2 гірша лише без фазової інформації", VZ + "run_inverse.py:main", [7]),
    "VE26": (r"Паразитні гармоніки стенда не дають видимих островів", VZ + "run_parasitic.py:main", [9]),
    "VE27": (r"Біо--Савар перевірено проти аналітичної петлі", TV + "rmp_coils.py:biot_savart", [9]),
    "VE28": (r"Поле масиву $4\cdot10^{-23}$~Тл --- через точку симетрії", TV + "rmp_coils.py:icoil_array", [9]),
    "VE29": (r"Інстанси Geometry Nodes не успадковують матеріали", None, []),
    "VE30": (r"Слід на стінці зважений до краю (VC6)", VZ + "run_footprint.py:observable", [7]),
    "VE31": (r"Обумовленість гіршає з числом мод плавно (VC7)", VZ + "run_mode_scan.py:main", [7]),
    "VE32": (r"Гістограма ударів непридатна для скінченних різниць", TV + "footprint.py:connection_length_profile", []),
    "VE33": (r"Немонотонність частки хаосу при 40 зернах --- шум (VC5)", VZ + "run_analysis.py:main", [8]),
    # --- viz: VR -----------------------------------------------------------
    "VR1": (r"«Геометричного кута досить для розміщення островів»", TV + "equilibrium.py:theta_star_field", [8]),
    "VR2": (r"«Паразитні гармоніки інтерполяції можна ігнорувати»", TV + "perturbation.py:Perturbation", [8]),
    "VR3": (r"«Об'ємну сітку плазми можна різати разом з машиною»", None, []),
    "VR4": (r"«Камери можна кадрувати за $R_0$»", None, []),
    "VR5": (r"«Спільна ціль TRACK\_TO для всіх камер»", None, []),
    "VR6": (r"«Можу навести замкнену форму кривої Princeton D»", TV + "surfaces.py:d_shape", []),
    "VR7": (r"«\texttt{angle2 == 0} означає зсув $0^\circ$»", None, []),
    "VR8": (r"«Число обертання --- просте середнє за вікном»", TV + "analysis.py:rotation_number", [8]),
    "VR9": (r"«Острів розпізнається за екскурсією $\psi_N$»", TV + "analysis.py:classify", [8]),
    "VR10": (r"«Захоплення $\nu$ --- близькість до раціонального»", TV + "analysis.py:find_locked_bands", [8]),
    "VR11": (r"«Класифікувати можна в будь-якому порядку»", TV + "analysis.py:classify", [8]),
    "VR12": (r"«Дефіцит ширини острова --- дискретність засіву»", TV + "analysis.py:measured_island_width", [8]),
    "VR13": (r"«Взаємної узгодженості $\nu$ досить для захоплення»", TV + "analysis.py:find_locked_bands", [8]),
    "VR14": (r"Обвідна $\psi_N^{m/2}$ не відтворює поле реальних котушок", VZ + "run_coil_compare.py:fit_exponent", [9]),
    "VR15": (r"Ширина острова з $\psi$ як імпульсом: завищення в $\sqrt q$", TV + "perturbation.py:island_width_psin", [8]),
    "VR16": (r"«Дефіцит ширини --- стохастичний шар сепаратриси»", "viz/tests/test_physics.py:test_island_width_matches_traced_separatrix", [8]),
    # --- viz: VC / VQ ------------------------------------------------------
    "VC1": (r"Обвідна $\psi_N^{m/2}$ близька до поля RMP-котушок ($\to$VR14)", VZ + "run_coil_compare.py:resonant_harmonic", [9]),
    "VC2": (r"Паразитні 1,3\,\% не дають видимих островів ($\to$VE26)", VZ + "run_parasitic.py:main", [9]),
    "VC3": (r"Шість обраних ракурсів достатні для лекції", None, [10]),
    "VC4": (r"Кадри читатимуться при проєкції у великій залі", None, [10]),
    "VC5": (r"Немонотонність частки хаосу зникає з числом зерен ($\to$VE33)", VZ + "run_analysis.py:main", [8]),
    "VC6": (r"Обернена задача на сліді важча за екскурсійну ($\to$VE30)", VZ + "run_footprint.py:main", [7]),
    "VC7": (r"Понад три моди обумовленість помітно гіршає ($\to$VE31)", VZ + "run_mode_scan.py:main", [7]),
    "VQ1": (r"Кріостат моделі тороїдальний; в ITER --- циліндр із куполом", None, [10]),
    "VQ2": (r"Конфлікт CC BY і CC BY-SA при перевипуску ($\leftarrow$Q13)", None, [10]),
    "VQ3": (r"Чи додати виміряний перетин W7-X поруч із синтетичним", None, [10]),
    "VQ4": (r"Чи надійний тест пласкості $\nu$ при новому засіві", TV + "analysis.py:find_locked_bands", [8, 10]),
    "VQ5": (r"U1: 21/30 хаотичних проти 10--13/30 у таблиці --- неузгоджено", VZ + "run_coil_compare.py", [9]),
    # --- COIL-BACKEND (без номерів у джерелі) -----------------------------
    "U1": (r"«Хаос бекенда котушок спричиняє гребінка бічних смуг»", TV + "coilfield.py:CoilPerturbation", [9]),
    "U2": (r"Хаос бекенда котушок: фізика чи похибка реконструкції поля?", TV + "coilfield.py:CoilPerturbation", [9, 10]),
    "U3": (r"Тримати $\vect A$, а не $\Bvec$: $\nabla\cdot\Bvec$ з $3{,}19\cdot10^{-2}$ до $2{,}50\cdot10^{-6}$", TV + "coilfield.py:divergence_report", [9]),
}

GROUPS = [
    ("Батьківський реєстр встановленого (E)", "est", "E"),
    ("Спростоване батьківського реєстру (R)", "est", "R"),
    ("Гіпотези (C)", "conj", "C"),
    ("Відкриті питання (Q)", "oq", "Q"),
    ("Рішення (D)", "dec", "D"),
    ("Журнал верифікації (V-)", "ver", "V-"),
    ("Візуалізація: встановлене (VE)", "viz", "VE"),
    ("Візуалізація: спростоване (VR)", "viz", "VR"),
    ("Візуалізація: гіпотези (VC)", "viz", "VC"),
    ("Візуалізація: відкрите (VQ)", "viz", "VQ"),
    ("Бекенд котушок, пункти без номерів (U)", "coil", "U"),
]


def sort_key(iid):
    m = re.match(r"^(V-[A-D]|VE|VR|VC|VQ|[ERCQDU])(\d+)(a?)$", iid)
    pre, num, suf = m.group(1), int(m.group(2)), m.group(3)
    order = {"V-A": 0, "V-D": 1, "V-B": 2, "V-C": 3}
    return (order.get(pre, 0), num, suf)


def check_code(ref):
    path, _, func = ref.partition(":")
    real = path
    for k, v in PREFIX.items():
        if path.startswith(k):
            real = v + path[len(k):]
            break
    f = ROOT / real
    if not f.exists():
        return f"файл не існує: {real}"
    if func:
        txt = f.read_text(encoding="utf-8", errors="replace")
        if not re.search(rf"^\s*(def|class)\s+{re.escape(func)}\b", txt, re.M):
            return f"у {real} немає def/class {func}"
    return None


def tex_path(s):
    return r"\nolinkurl{" + s + "}"


def main():
    parsed = {}
    for fn in (parse_est, parse_conj, parse_dec, parse_oq, parse_ver, parse_viz, parse_coil):
        parsed.update(fn())

    bad = 0
    miss = sorted(set(parsed) - set(SUM))
    extra = sorted(set(SUM) - set(parsed))
    if miss:
        print("У реєстрах є, у SUM немає:", miss); bad += 1
    if extra:
        print("У SUM є, у реєстрах немає:", extra); bad += 1
    for iid, (_, code, _) in SUM.items():
        if code:
            err = check_code(code)
            if err:
                print(f"{iid}: {err}"); bad += 1
    if bad:
        sys.exit(1)

    rows, counts = [], {}
    for title, reg, pre in GROUPS:
        ids = [i for i in parsed if REG_OF(i) == (reg, pre)]
        ids.sort(key=sort_key)
        counts[title] = len(ids)
        rows.append(r"\multicolumn{6}{l}{\rule{0pt}{2.6ex}\bfseries\sffamily " + title
                    + f" --- {len(ids)}}}\\\\*")
        for iid in ids:
            fname, line, st = parsed[iid]
            summ, code, chs = SUM[iid]
            where = tex_path(fname) + rf"\newline р.~{line}"
            codec = tex_path(code) if code else "---"
            chap = ", ".join(rf"\ref{{ch:g{c:02d}}}" for c in chs) if chs else r"\emph{лише тут}"
            rows.append(rf"\textbf{{{iid}}} & {summ} & {st} & {where} & {codec} & {chap}\\")
    out = GUIDE / "appendix/index_rows.tex"
    out.write_text("% АВТОГЕНЕРОВАНО tools/build_index.py — не редагувати вручну\n"
                   + "\n".join(rows) + "\n", encoding="utf-8")
    total = sum(counts.values())
    for k, v in counts.items():
        print(f"{v:4d}  {k}")
    print(f"{total:4d}  усього -> {out.relative_to(GUIDE)}")


def REG_OF(iid):
    if iid.startswith("V-"):
        return ("ver", "V-")
    for pre in ("VE", "VR", "VC", "VQ"):
        if iid.startswith(pre):
            return ("viz", pre)
    p = iid[0]
    return {"E": ("est", "E"), "R": ("est", "R"), "C": ("conj", "C"), "Q": ("oq", "Q"),
            "D": ("dec", "D"), "U": ("coil", "U")}[p]


if __name__ == "__main__":
    main()
