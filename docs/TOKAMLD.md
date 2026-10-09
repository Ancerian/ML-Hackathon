# Документація пакету `tokamld`

**tokamld** — відкрита Python/PyTorch бібліотека для розрахунку магнітної рівноваги токомаків, фізичних операторів Град-Шафранова, топологічного аналізу та конформного UQ.

---

## 1. Архітектура модулів

### `tokamld.gs` (Grad-Shafranov Operators)
- `delta_star(psi, R, Z)`: Диференційовний оператор $\Delta^*\psi = \frac{\partial^2\psi}{\partial R^2} - \frac{1}{R}\frac{\partial\psi}{\partial R} + \frac{\partial^2\psi}{\partial Z^2}$ (підтримує `torch.Tensor` з autograd та `numpy.ndarray`).
- `residual(psi, R, Z, pprime, ffprime)`: Повна нев'язка $\Delta^*\psi + \mu_0 R^2 p'(\psi) + FF'(\psi)$.
- `gs_inconsistency(psi, R, Z, mask_coarse, mask_f)`: Відносна нерівноважність поля $g(\psi) = \min_{p', FF'} \|\Delta^*\psi + \dots\| / \|\Delta^*\psi\|$ всередині LCFS через МНК.
- `gate(g, score, g_ref=0.6328)`: Неперервний фізичний шлюз $\mathcal{G}_{\mathrm{GS}}$ (Специфікація $S'$ v2.1).

### `tokamld.topology` (Critical Points & Topology)
- `find_critical_points(psi, R, Z, inside, min_npix=3)`: Знаходження критичних точок з індексом Пуанкаре-Хопфа (O-точки $+1$, X-точки $-1$).
- `count_critical_points(psi, R, Z, inside, min_npix=3)`: Підрахунок $(N_{\mathrm{O}}, N_{\mathrm{X}})$.
- `check_canonical_topology(psi, R, Z, inside)`: Перевірка канонічної рівноваги: $N_{\mathrm{O}} = 1, N_{\mathrm{X}} = 0 \implies N_{\mathrm{spurious}} = 0$.

### `tokamld.conformal` (Simultaneous Conformal Prediction)
- `SimultaneousConformalBands(alpha=0.90)`: Клас для побудови конформних смуг із спільною гарантією покриття для всього 2D-поля за калібрувальною вибіркою (C6, P2.5).

### `tokamld.fieldline` (Fieldlines, Green's Function & Island Width)
- `green_function(R0, Z0, R, Z, mu)`: Функція Гріна кругового витка струму з виправленим аргументом $m = k^2$ для еліптичних інтегралів (E33).
- `magnetic_island_width(psi_tilde, q_res, dq_dpsi)`: Фізично коректна ширина магнітного острова $W = 4\sqrt{\tilde{\psi} q / |dq/d\psi|}$ (R4, VR15).
- `FieldLineIntegrator`: Інтегратор силових ліній магнітного поля та побудова перерізів Пуанкаре.

---

## 2. Верифікаційні регресійні тести

1. **Функція Гріна (E33):** перевірено проти прямої числової квадратури Біо-Савара (похибка $< 10^{-10}$).
2. **Ширина острова (VR15):** точне співпадіння з аналітичною формулою Гамільтоніана без помилкового $\sqrt{q}$.
3. **Аналітична рівновага Серфона-Фрайдберга:** перевірено точний розв'язок Соловйова $\Delta^*(R^4/8) = R^2$ (похибка дискретизації $< 10^{-3}$).
4. **Істина + 1% шуму на 25 розрядах:** підтверджено $g \approx 0.6368 \approx 0.63$ (E6/E38).
5. **Репродукція скорера:** $S'$ на `perfect` $\to 1.0$, `zeros` $\to 0.0$, `PCA+Ridge` $\to 0.361368$.
