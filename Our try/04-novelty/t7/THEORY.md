# ПРЕДРЕЄСТРАЦІЯ: Гіпотеза C7 (T7) — Порівняння стабільності Deep Ritz та PINN

**Дата:** 2026-10-09  
**Статус:** ЗАФІКСОВАНО ДО ЗАПУСКУ ЕКСПЕРИМЕНТУ  
**Гіпотеза C7:**  
Варіаційний підхід Deep Ritz (мінімізація функціоналу Діріхле $E[\psi] = \int \frac{1}{2R} |\nabla \psi|^2 - J_\phi \psi \, dR dZ$) має принципово відмінну стабільність за ініціалізацією (seed) порівняно з формулюванням Physics-Informed Neural Networks (PINN, мінімізація нев'язки оператора другого порядку $\|\Delta^* \psi + \mu_0 R^2 p' + FF'\|^2_{L^2}$):
1. На неопуклих / витягнутих формах плазми з вільною межею PINN страждає від відомих режимів відмови (Krishnapriyan et al., 2021: pathology of PINN optimization across second-order differential operators, stiffness, gradient pathologies).
2. Deep Ritz оперує інтегралом від градієнтів першого порядку $|\nabla \psi|^2$, що згладжує оптимізаційний ландшафт і зменшує розкид метрик за випадковими початковими вагами.

---

## 1. Показник стабільності та метрики

Для кожного методу (Deep Ritz vs PINN) тренується ансамбль із **$N_{\mathrm{seeds}} = 10$ випадкових зерен (seeds: 0..9)** при **строго однаковому бюджеті обчислень**:
- Архітектура мережі: MLP з синусною/GELU активацією, 4 приховані шари по 64 нейрони;
- Кількість кроків оптимізації: 1000 ітерацій Adam (learning rate $10^{-3}$);
- Вибірка точок колокації: 2000 точок в області $[R_{\min}, R_{\max}] \times [Z_{\min}, Z_{\max}]$.

### Метрики:
1. **$R^2_\psi$** відносно аналітичного еталону Cerfon–Freidberg;
2. **Відносна нев'язка Град-Шафранова $g(\psi)$** (`tokamld.gs.residual`);
3. **Показник стабільності (Seed Dispersion):**
   - Інтерквартильний розмах (IQR) та стандартне відхилення $\sigma(g)$ і $\sigma(R^2_\psi)$ за 10 зернами;
   - Коефіцієнт варіації $\mathrm{CV}(g) = \sigma(g) / \mu(g)$.

---

## 2. Критерій спростування гіпотези C7 (Falsification Threshold — До запуску)

Гіпотеза C7 **СПРОСТОВУЄТЬСЯ**, якщо:
1. Розкид нев'язки $\mathrm{CV}(g)$ для Deep Ritz виявиться **не меншим** (або статистично нерозрізненим), ніж для PINN на витягнутих конфігураціях:
   $$\frac{\mathrm{IQR}(g)_{\mathrm{DeepRitz}}}{\mathrm{IQR}(g)_{\mathrm{PINN}}} \ge 1.0$$
   або медіанна якість Deep Ritz $R^2_\psi$ деградує нижче $0.80$ через проблеми апроксимації граничних умов.

Гіпотеза C7 **ПІДТВЕРДЖУЄТЬСЯ**, якщо:
1. Deep Ritz демонструє стабільніший розв'язок за зернами:
   $$\mathrm{IQR}(g)_{\mathrm{DeepRitz}} < 0.6 \cdot \mathrm{IQR}(g)_{\mathrm{PINN}}$$
   та розкид $\sigma(R^2_\psi)_{\mathrm{DeepRitz}} < \sigma(R^2_\psi)_{\mathrm{PINN}}$.

---

## 3. Тестові конфігурації (3 форми Cerfon–Freidberg)

1. **Конфігурація 1 (Круглий переріз / Baseline):** $\kappa = 1.0, \delta = 0.0, \epsilon = 0.3$.
2. **Конфігурація 2 (Витягнутий токомак DIII-D / Elongated D-shape):** $\kappa = 1.7, \delta = 0.33, \epsilon = 0.32$.
3. **Конфігурація 3 (Неопуклий Бін / Bean / Single-Null X-point):** $\kappa = 2.0, \delta = 0.5, \epsilon = 0.32$.

---

## 4. Фізичні функціонали втрат

1. **Deep Ritz Loss:**
   $$\mathcal{L}_{\mathrm{Ritz}}(\theta) = \frac{1}{N_{\Omega}} \sum_{i=1}^{N_{\Omega}} \left[ \frac{1}{2 R_i} \left( \left(\frac{\partial \psi_\theta}{\partial R}\right)^2 + \left(\frac{\partial \psi_\theta}{\partial Z}\right)^2 \right) - \mu_0 J_\phi(R_i, \psi_\theta) \psi_\theta \right] + \lambda_{\mathrm{bdy}} \frac{1}{N_{\partial\Omega}} \sum_{j=1}^{N_{\partial\Omega}} (\psi_\theta - \psi_{\mathrm{exact}})^2$$
2. **PINN Loss:**
   $$\mathcal{L}_{\mathrm{PINN}}(\theta) = \frac{1}{N_{\Omega}} \sum_{i=1}^{N_{\Omega}} \left| \Delta^* \psi_\theta(R_i, Z_i) - \mu_0 R_i J_\phi(R_i, \psi_\theta) \right|^2 + \lambda_{\mathrm{bdy}} \frac{1}{N_{\partial\Omega}} \sum_{j=1}^{N_{\partial\Omega}} (\psi_\theta - \psi_{\mathrm{exact}})^2$$
   де $\Delta^* \psi = \frac{\partial^2 \psi}{\partial R^2} - \frac{1}{R} \frac{\partial \psi}{\partial R} + \frac{\partial^2 \psi}{\partial Z^2}$.

Вага граничних умов фіксується однаковою для обох: $\lambda_{\mathrm{bdy}} = 100.0$.
