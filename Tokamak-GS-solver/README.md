# Tokamak-GS-solver — лише наші виправлення

Оригінальний розв'язувач рівняння Ґреда–Шафранова (вільна межа + PINN): **https://github.com/ZINZINBIN/Tokamak-GS-solver**.
Оригінал не має файлу ліцензії, тому його код тут **не копіюється**. Тут лежать лише:

- `CHANGES-2026-09-29.md` — що саме ми виправили і чому (команди README, функція Гріна з аргументом `k` замість `m = k²`, тести);
- `0001-repair-free-boundary-and-green-function.patch` — наш коміт `8c5aa4f` у форматі `git format-patch`.

## Як отримати виправлену версію

```bash
git clone https://github.com/ZINZINBIN/Tokamak-GS-solver.git
cd Tokamak-GS-solver
git checkout 7be12f61d26795c2fc34b57bc770fccf9e5ca7e2     # база, до якої застосовано патч
git am ../0001-repair-free-boundary-and-green-function.patch
python -m pytest tests -q                                  # 22 тести
```

Що виправлено і що лишилось відкритим — `CHANGES-2026-09-29.md`; у реєстрі проєкту це E33 (функція Гріна) і E34 (вільна межа, частково).
