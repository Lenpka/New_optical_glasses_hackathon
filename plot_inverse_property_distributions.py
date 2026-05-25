"""Распределения Tg, Density, n_d, ν_d для inverse design и recovery.

    python plot_inverse_property_distributions.py
    python plot_inverse_property_distributions.py --show

Выход: output/property_distributions/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plots import load_inverse_design_property_datasets, plot_inverse_recovery_design_distributions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Распределения свойств (design / recovery / SCHOTT)",
    )
    parser.add_argument("--data", type=Path, default=None, help="merged_data.parquet")
    parser.add_argument("--sciglass-zip", type=Path, default=None)
    parser.add_argument("--schott-xlsx", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    datasets, counts = load_inverse_design_property_datasets(
        data_path=args.data,
        sciglass_zip=args.sciglass_zip,
        schott_xlsx=args.schott_xlsx,
    )
    print("Выборки:")
    for k, n in counts.items():
        print(f"  {k}: {n:,} строк")

    paths = plot_inverse_recovery_design_distributions(
        datasets,
        output_dir=args.output,
        show=args.show,
    )
    print("\nСохранено:")
    for name, p in paths.items():
        print(f"  {name}: {p}")


if __name__ == "__main__":
    main()
