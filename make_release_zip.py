"""Собрать ZIP для передачи моделей и glass_tool сторонним пользователям.

    python make_release_zip.py
    python make_release_zip.py --output dist/glass_tool_release.zip
"""

from __future__ import annotations

import argparse
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SCIGLASS = Path(
    r"C:/Users/user/AppData/Local/GlassPy/GlassPy/data/select_SciGK.csv.zip"
)

# Исходники, без которых glass_tool не запустится
CODE_FILES = [
    "glass_tool.py",
    "glass_tool_app.py",
    "glass_tool/__init__.py",
    "glass_tool/core.py",
    "glass_tool/cli.py",
    "match_schott_sciglass.py",
    "match_schott_sciglass_v2.py",
    "match_schott_sciglass_v3.py",
    "gan_glass_design.py",
    "inverse_glass_design.py",
    "requirements-ml.txt",
    "ИНСТРУКЦИЯ_ИНСТРУМЕНТ.md",
    "LICENSE",
]

# Данные и обученные модели
DATA_FILES = [
    "merged_data.parquet",
    "schott-optical-glass-overview-excel-format-en 202501113.xlsx",
]

GAN_FILES = [
    "output/gan_design/checkpoints/generator.pt",
    "output/gan_design/condition_stats.json",
]

FORWARD_GLOB = "output/inverse_design_2h/forward_models/*.joblib"

README_RELEASE = """# Glass Tool — релиз для пользователей

## Быстрый старт

1. Распакуйте архив.
2. Python 3.10+:
   ```
   python -m venv .venv
   .venv\\Scripts\\activate
   pip install -r requirements-ml.txt
   ```
3. Проверка:
   ```
   python glass_tool.py check
   ```
4. Примеры:
   ```
   python glass_tool.py recover --nd 1.85 --vd 25
   python glass_tool.py recover --glass "N-SF11"
   python glass_tool.py generate --nd 1.90 --vd 22 -n 300 --out candidates.csv
   streamlit run glass_tool_app.py
   ```

## Содержимое

- `glass_tool.py` — CLI
- `data/select_SciGK.csv.zip` — база SciGlass (восстановление), если включена в архив
- `merged_data.parquet` — обучение surrogate / GAN
- `output/gan_design/` — генератор cWGAN-GP
- `output/inverse_design_2h/forward_models/` — модели ND, NUD, Density, Tg

Подробнее: ИНСТРУКЦИЯ_ИНСТРУМЕНТ.md

Собрано: {built_at}
"""


def _add_file(zf: zipfile.ZipFile, root: Path, rel: str, arc_prefix: str = "") -> bool:
    path = root / rel
    if not path.is_file():
        return False
    arc = f"{arc_prefix}{rel}".replace("\\", "/")
    zf.write(path, arcname=arc)
    return True


def build_zip(output: Path, include_sciglass: bool, sciglass_path: Path) -> dict[str, list[str]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    added: list[str] = []
    missing: list[str] = []

    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zf:
        zf.writestr(
            "README_RELEASE.md",
            README_RELEASE.format(built_at=built_at),
        )
        added.append("README_RELEASE.md")

        for rel in CODE_FILES:
            if _add_file(zf, PROJECT_DIR, rel):
                added.append(rel)
            else:
                if rel != "LICENSE":
                    missing.append(rel)

        for rel in DATA_FILES:
            if _add_file(zf, PROJECT_DIR, rel):
                added.append(rel)
            else:
                missing.append(rel)

        for rel in GAN_FILES:
            if _add_file(zf, PROJECT_DIR, rel):
                added.append(rel)
            else:
                missing.append(rel)

        for p in sorted((PROJECT_DIR / "output/inverse_design_2h/forward_models").glob("*.joblib")):
            rel = p.relative_to(PROJECT_DIR).as_posix()
            if _add_file(zf, PROJECT_DIR, rel):
                added.append(rel)

        if include_sciglass and sciglass_path.is_file():
            zf.write(
                sciglass_path,
                arcname="data/select_SciGK.csv.zip",
            )
            added.append(f"data/select_SciGK.csv.zip (from {sciglass_path})")
        else:
            zf.writestr(
                "data/ПОЛОЖИТЕ_СЮДА_select_SciGK.csv.zip.txt",
                "Скопируйте select_SciGK.csv.zip (SciGlass / GlassPy) в:\n"
                "  data/select_SciGK.csv.zip\n\n"
                "Или укажите путь при запуске:\n"
                "  python glass_tool.py recover --sciglass-zip ПУТЬ\\select_SciGK.csv.zip ...\n",
            )
            missing.append("data/select_SciGK.csv.zip (не найден — см. data/*.txt)")

    return {"added": added, "missing": missing, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Собрать ZIP-релиз glass_tool")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "dist" / "glass_tool_release.zip",
    )
    parser.add_argument(
        "--sciglass-zip",
        type=Path,
        default=DEFAULT_SCIGLASS,
        help="Путь к select_SciGK.csv.zip",
    )
    parser.add_argument(
        "--no-sciglass",
        action="store_true",
        help="Не включать SciGlass (меньший архив)",
    )
    args = parser.parse_args()

    include_sg = not args.no_sciglass and args.sciglass_zip.is_file()
    report = build_zip(args.output, include_sg, args.sciglass_zip)

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Готово: {args.output.resolve()}")
    print(f"Размер: {size_mb:.1f} MB")
    print(f"Файлов в архиве: {len(report['added'])}")
    if report["missing"]:
        print("Не найдено / не включено:")
        for m in report["missing"]:
            print(f"  - {m}")
    if not include_sg and not args.no_sciglass:
        print(
            f"\nSciGlass не найден: {args.sciglass_zip}\n"
            "Положите zip в data/ или передайте --sciglass-zip"
        )


if __name__ == "__main__":
    main()
