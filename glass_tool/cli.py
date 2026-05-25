"""Командная строка для glass_tool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from glass_tool.core import (
    ToolPaths,
    check_environment,
    format_generate_text,
    format_recover_text,
    list_schott_glasses,
    recover_composition,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _json_default(obj: object) -> object:
    import numpy as np

    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(type(obj))


def cmd_check(args: argparse.Namespace) -> int:
    paths = ToolPaths(
        sciglass_zip=args.sciglass_zip,
        schott_xlsx=args.schott_xlsx,
        gan_dir=args.gan_dir,
        forward_models=args.forward_models,
        merged_data=args.data,
    )
    c = check_environment(paths)
    print("Проверка окружения:")
    for k, v in c.items():
        mark = "OK" if v else "—"
        print(f"  [{mark}] {k}")
    if c["ready_recover"]:
        print("\nВосстановление состава: готово")
    else:
        print("\nВосстановление: нужен SciGlass zip (--sciglass-zip)")
    if c["ready_generate"]:
        print("Генерация (GAN): готово")
    else:
        print("Генерация: нужны GAN checkpoint, forward_models, merged_data.parquet")
    return 0 if c["ready_recover"] else 1


def cmd_recover(args: argparse.Namespace) -> int:
    paths = ToolPaths(
        sciglass_zip=args.sciglass_zip,
        schott_xlsx=args.schott_xlsx,
    )
    result = recover_composition(
        nd=args.nd,
        vd=args.vd,
        density=args.density,
        tg=args.tg,
        glass_name=args.glass,
        k_neighbors=args.neighbors,
        paths=paths,
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        print(f"JSON: {args.out}")
    if not args.quiet:
        print(format_recover_text(result))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from glass_tool.core import generate_compositions

    paths = ToolPaths(
        gan_dir=args.gan_dir,
        forward_models=args.forward_models,
        merged_data=args.data,
    )
    result = generate_compositions(
        args.nd,
        args.vd,
        n_samples=args.n,
        top_k=args.top,
        pb_free_only=not args.allow_pb,
        min_nd=args.min_nd,
        paths=paths,
    )
    if args.out:
        import pandas as pd

        df = pd.DataFrame(result["candidates"])
        out = Path(args.out)
        if out.suffix.lower() == ".json":
            out.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
        else:
            df.to_csv(out, index=False)
        print(f"Сохранено: {out}")
    if not args.quiet:
        print(format_generate_text(result))
    return 0


def cmd_list_glasses(args: argparse.Namespace) -> int:
    paths = ToolPaths(schott_xlsx=args.schott_xlsx)
    names = list_schott_glasses(paths, limit=args.limit)
    if not names:
        print("Каталог SCHOTT не найден.")
        return 1
    print(f"Примеры марок SCHOTT (первые {len(names)}):")
    for n in names:
        print(f"  {n}")
    return 0


def cmd_interactive(_: argparse.Namespace) -> int:
    print()
    print("=" * 60)
    print("  Инструмент составов оптических стекол")
    print("=" * 60)
    print("  1 — Восстановить состав по свойствам (SciGlass)")
    print("  2 — Восстановить по марке SCHOTT")
    print("  3 — Сгенерировать кандидаты (cWGAN-GP)")
    print("  4 — Проверить окружение")
    print("  0 — Выход")
    print()

    paths = ToolPaths()
    while True:
        try:
            choice = input("Выбор [0-4]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if choice == "0":
            return 0
        if choice == "4":
            cmd_check(argparse.Namespace(
                sciglass_zip=paths.sciglass_zip,
                schott_xlsx=paths.schott_xlsx,
                gan_dir=paths.gan_dir,
                forward_models=paths.forward_models,
                data=paths.merged_data,
            ))
            continue

        if choice == "1":
            try:
                nd = float(input("n_d: ").replace(",", "."))
                vd = float(input("ν_d (Abbe): ").replace(",", "."))
                d_in = input("ρ (кг/м³, Enter — авто): ").strip()
                t_in = input("T_g (°C, Enter — авто): ").strip()
                density = float(d_in.replace(",", ".")) if d_in else None
                tg = float(t_in.replace(",", ".")) if t_in else None
                r = recover_composition(nd=nd, vd=vd, density=density, tg=tg, paths=paths)
                print(format_recover_text(r))
            except Exception as e:
                print(f"Ошибка: {e}")
            continue

        if choice == "2":
            name = input("Марка SCHOTT (например N-SF11): ").strip()
            try:
                r = recover_composition(glass_name=name, paths=paths)
                print(format_recover_text(r))
            except Exception as e:
                print(f"Ошибка: {e}")
            continue

        if choice == "3":
            c = check_environment(paths)
            if not c["ready_generate"]:
                print("Генерация недоступна. Запустите пункт 4.")
                continue
            try:
                nd = float(input("Целевой n_d: ").replace(",", "."))
                vd = float(input("Целевой ν_d: ").replace(",", "."))
                n = int(input("Число генераций [300]: ").strip() or "300")
                r = generate_compositions(nd, vd, n_samples=n, paths=paths)
                print(format_generate_text(r))
            except Exception as e:
                print(f"Ошибка: {e}")
            continue

        print("Неизвестный пункт.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="glass_tool",
        description="Восстановление и генерация составов оптических стекол",
    )
    _defaults = ToolPaths()
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sciglass-zip", type=Path, default=_defaults.sciglass_zip)
    common.add_argument("--schott-xlsx", type=Path, default=_defaults.schott_xlsx)
    common.add_argument("--gan-dir", type=Path, default=_defaults.gan_dir)
    common.add_argument("--forward-models", type=Path, default=_defaults.forward_models)
    common.add_argument("--data", type=Path, default=_defaults.merged_data)

    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("check", parents=[common], help="Проверить наличие данных и моделей")
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser(
        "recover", parents=[common], help="Восстановить состав (соседи SciGlass)"
    )
    sp.add_argument("--nd", type=float, help="Показатель преломления n_d")
    sp.add_argument("--vd", type=float, help="Число Аббе ν_d")
    sp.add_argument("--density", type=float, help="Плотность, кг/м³")
    sp.add_argument("--tg", type=float, help="Температура стеклования, °C")
    sp.add_argument("--glass", type=str, help="Марка из каталога SCHOTT")
    sp.add_argument("--neighbors", type=int, default=20, help="Число соседей (top-k)")
    sp.add_argument("--out", type=Path, help="Сохранить JSON")
    sp.add_argument("-q", "--quiet", action="store_true")
    sp.set_defaults(func=cmd_recover)

    sp = sub.add_parser(
        "generate", parents=[common], help="Сгенерировать кандидаты (cWGAN-GP)"
    )
    sp.add_argument("--nd", type=float, required=True)
    sp.add_argument("--vd", type=float, required=True)
    sp.add_argument("-n", type=int, default=300, help="Число генераций")
    sp.add_argument("--top", type=int, default=20, help="Сколько лучших показать")
    sp.add_argument("--min-nd", type=float, default=1.75)
    sp.add_argument("--allow-pb", action="store_true", help="Не фильтровать PbO")
    sp.add_argument("--out", type=Path, help="CSV или JSON")
    sp.add_argument("-q", "--quiet", action="store_true")
    sp.set_defaults(func=cmd_generate)

    sp = sub.add_parser("list-glasses", parents=[common], help="Список марок SCHOTT")
    sp.add_argument("--limit", type=int, default=40)
    sp.set_defaults(func=cmd_list_glasses)

    sp = sub.add_parser(
        "interactive",
        aliases=["ui"],
        parents=[common],
        help="Интерактивное меню в терминале",
    )
    sp.set_defaults(func=cmd_interactive)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
