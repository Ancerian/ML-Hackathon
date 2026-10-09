# Перевірка Landreman 2026 (arXiv:2609.26742 v2)

Незалежна чисельна перевірка статті **M. Landreman, «Analytic toroidal 3D MHD equilibria and
steady Euler flows with invariant surfaces»**, arXiv:2609.26742 v2 (28.09.2026). PDF —
`library/ARXIV_Landreman_2026_Analytic_toroidal_3D_MHD_equilibria.pdf`. Виконано 2026-10-01.
Реєстр: `00-decisions/ESTABLISHED.md` **E39**; журнал правок — `CHANGELOG.md` №115–120.

## Вердикт

**Математичних помилок не знайдено.** Усі перевірені рівняння обох сімейств виконуються до
похибки скінченних різниць. Дві примітки, яких у статті немає (не помилки):

1. **Сімейство 1 (ι = 2)** — лінійний образ рівноваги Солов'ева, B = A·B₀(A⁻¹x). Тому саме **поле B
   має точну неперервну симетрію** — «еліптичне обертання» v = ((a/b)y, −(b/a)x, 0), яке не є
   евклідовою ізометрією (`hidden_sym.py`: найменше сингулярне число 3,7·10⁻¹²). Її порушують |B|, струм і тиск (до 2026-10-02 — «тиск її порушує», неповно; `symmetry_recheck.py`). Тиск
   (max |v·∇p|/(|v||∇p|) = 0,62). Усі силові лінії замкнені. Контрприклад до гіпотези Греда
   формально коректний (у евклідовому формулюванні), але слабкий.
2. **Сімейство 2 (шир ι)** — афінної симетрії B не знайдено. Шир у числовому прикладі (3.27)
   мізерний: ι 2,2869 → 2,2878 (+0,04 %); на параметрах рис. 2 — +9,3 % / −1,4 % / −2,6 %.

## Що перевірено

| Що | Результат | Скрипт |
|---|---|---|
| ∇·B = 0, (∇×B)×B = ∇p, B·∇ψ = 0; 3 + 4 набори параметрів, по 40 точок у Ω_δ | 10⁻¹⁰–10⁻¹² (ϵ = 0,7: 5·10⁻⁷ — точки ближче до краю U_ϵ) | `verify_landreman.py` |
| Параметризації (2.11), (2.15), (3.12–3.13); якобіан (2.13) = −ab | до 10⁻¹⁵ | те саме |
| Стелараторна симетрія, два періоди поля (обидва сімейства) | точно | те саме |
| β_V = 2/57 при ϵ = ½, δ = 1/64; ⟨\|B\|²⟩_V (2.27) Монте-Карло | 0,035088; 0,8899 ± 0,0011 проти 0,8906 | те саме |
| ι = 2 трасуванням силової лінії | 2,000000 | те саме |
| ι(0) = h₀H(ϵ) = 2,28690; ι(δ) ≈ 2,2878 для (3.27) | 2,286900; 2,28779 за (3.24) і 2,28777 трасуванням | те саме |
| Афінні поля v = Mx + c з [v, B] = 0 | сімейство 1 — є (див. вище); сімейство 2 — немає | `hidden_sym.py` |
| Діапазон ι на параметрах рис. 2 | див. вище | `iota_fig2.py` |

Побічно: числа статті узгоджені з нашим словником E19. При ϵ → 0 маємо Φ_t = πψ, тобто
ψ_t = ψ/2, і ι = dψ_p/dψ_t = 2 — як у статті. Помилковий словник R4 дав би ½.

## Відтворення (≈10 с)

```bash
cd "../../../fusion equilibrium challenge/starter"
.venv/bin/python "../../Our try/99-bibliography/landreman2026/verify_landreman.py"
.venv/bin/python "../../Our try/99-bibliography/landreman2026/hidden_sym.py"
.venv/bin/python "../../Our try/99-bibliography/landreman2026/iota_fig2.py"
```

Збережені виводи — `*.out.txt`.

---

## English summary

Independent numerical check of Landreman, arXiv:2609.26742 v2. **No mathematical errors found**:
∇·B = 0, force balance and B·∇ψ = 0 hold to 10⁻¹⁰–10⁻¹² for both families; the field-line and
surface parameterisations, the Jacobian −ab, stellarator symmetry, two field periods, β_V = 2/57,
ι = 2 and the sheared-ι values 2.28690 / 2.2878 are all reproduced. Two caveats the paper does not
state: family 1 is a linear image of a Solov'ev field, so **B itself keeps an exact continuous
non-isometric symmetry** (an elliptic rotation) broken by |B|, the current and the pressure ("only by the pressure" until 2026-10-02), and all its lines are
closed — a formally valid but weak counterexample to Grad; family 2 has no affine symmetry, but its
shear in example (3.27) is only +0.04 % (figure-2 parameters: +9.3 %, −1.4 %, −2.6 %). As a side
check, the paper's ι = 2 is consistent with our canonical dictionary E19 and not with the R4 error.
