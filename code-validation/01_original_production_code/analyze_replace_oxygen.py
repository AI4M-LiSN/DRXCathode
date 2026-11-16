#!/usr/bin/env python3
"""统计 Replace_R30_{0..99}.vasp 中的两个比值。

ratio1 = (n_Cl + n_F) / (n_Cl + n_F + n_O)
ratio2 = n_F / (n_Cl + n_F)
"""

from pathlib import Path

import numpy as np
from ase.io import read


N_FILES = 100
RADIUS = 30


def analyze(path: Path) -> tuple[float, float]:
    symbols = read(str(path), format="vasp").get_chemical_symbols()
    n_O, n_Cl, n_F = symbols.count("O"), symbols.count("Cl"), symbols.count("F")
    r1 = (n_Cl + n_F) / (n_O + n_Cl + n_F)
    r2 = n_F / (n_Cl + n_F) if (n_Cl + n_F) else float("nan")
    return r1, r2


def report(name: str, values: np.ndarray) -> None:
    ddof = 1 if values.size > 1 else 0
    print(f"  {name:<26} "
          f"mean={values.mean():.4f}  "
          f"std={values.std(ddof=ddof):.4f}  "
          f"var={values.var(ddof=ddof):.6f}  "
          f"median={np.median(values):.4f}  "
          f"min={values.min():.4f}  max={values.max():.4f}")


def main() -> None:
    r1, r2 = [], []
    for j in range(N_FILES):
        f = Path(f"Replace_R{RADIUS}_{j}.vasp")
        if not f.exists():
            print(f"[missing] {f.name}")
            continue
        a, b = analyze(f)
        r1.append(a)
        r2.append(b)
        print(f"{j:>3}  ratio1={a:.4f}  ratio2={b:.4f}")

    r1, r2 = np.array(r1), np.array(r2)
    print(f"\\n=== N = {r1.size} ===")
    report("(Cl+F)/(Cl+F+O)", r1)
    report("F/(Cl+F)",        r2)


if __name__ == "__main__":
    main()
