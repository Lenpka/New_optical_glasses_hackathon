"""Conditional WGAN-GP для генерации составов оптических стекол (SciGlass).

    python gan_glass_design.py
    python gan_glass_design.py --epochs 800 --n-generate 5000

Обучает cWGAN-GP на бессвинцовых высокопреломляющих стёклах (ND>1.70, PbO<=1),
генерирует кандидаты под целевые n_d / nu_d, фильтрует surrogate-моделями.

Выход: output/gan_design/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from inverse_glass_design import (
    ANOMALY_OXIDES,
    DEFAULT_DATA,
    OXIDE_FEATURES,
    PROPERTY_MAP,
    TARGETS,
    build_distance_model,
    comp_to_str,
    filter_design_space,
    load_sciglass_df,
    oxide_columns,
    predict_properties,
)
from match_schott_sciglass import classify_glass_family, setup_logging

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "output" / "gan_design"
DEFAULT_FORWARD_DIR = PROJECT_DIR / "output" / "inverse_design_2h" / "forward_models"

LATENT_DIM = 64
COND_DIM = 2  # ND300, NUD300 (нормированные)


def load_forward_models(models_dir: Path, oxide_cols: list[str]) -> dict[str, Any]:
    models = {}
    for target in TARGETS:
        hits = list(models_dir.glob(f"{target}_*.joblib"))
        if not hits:
            raise FileNotFoundError(f"Нет модели {target} в {models_dir}")
        p = hits[0]
        name = p.stem.split("_", 1)[1]
        models[target] = {"model": joblib.load(p), "model_name": name, "features": oxide_cols}
    return models


def prepare_training_tensors(
    df: pd.DataFrame,
    oxide_cols: list[str],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    nd = pd.to_numeric(df[PROPERTY_MAP["ND300"]], errors="coerce")
    nud = pd.to_numeric(df[PROPERTY_MAP["NUD300"]], errors="coerce")
    valid = nd.notna() & nud.notna()
    sub = df.loc[valid].copy()

    X = sub[oxide_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    if "PBO" in oxide_cols:
        X[:, oxide_cols.index("PBO")] = 0.0
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    X = X / row_sum * 100.0

    nd_v = nd.loc[valid].to_numpy(dtype=np.float32)
    nud_v = nud.loc[valid].to_numpy(dtype=np.float32)
    stats = {
        "nd_min": float(nd_v.min()),
        "nd_max": float(nd_v.max()),
        "nud_min": float(nud_v.min()),
        "nud_max": float(nud_v.max()),
    }
    nd_n = (nd_v - stats["nd_min"]) / (stats["nd_max"] - stats["nd_min"] + 1e-6)
    nud_n = (nud_v - stats["nud_min"]) / (stats["nud_max"] - stats["nud_min"] + 1e-6)
    C = np.stack([nd_n, nud_n], axis=1).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(C), stats


def normalize_condition(nd: float, nud: float, stats: dict[str, float]) -> np.ndarray:
    nd_n = (nd - stats["nd_min"]) / (stats["nd_max"] - stats["nd_min"] + 1e-6)
    nud_n = (nud - stats["nud_min"]) / (stats["nud_max"] - stats["nud_min"] + 1e-6)
    return np.array([nd_n, nud_n], dtype=np.float32)


class Generator(nn.Module):
    def __init__(self, n_oxides: int, pbo_idx: int | None, latent: int = LATENT_DIM, cond: int = COND_DIM):
        super().__init__()
        self.pbo_idx = pbo_idx
        self.net = nn.Sequential(
            nn.Linear(latent + cond, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, n_oxides),
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, c], dim=1)
        logits = self.net(x)
        comp = torch.softmax(logits, dim=1) * 100.0
        if self.pbo_idx is not None:
            comp = comp.clone()
            comp[:, self.pbo_idx] = 0.0
            comp = comp / comp.sum(dim=1, keepdim=True).clamp(min=1e-6) * 100.0
        return comp


class Critic(nn.Module):
    def __init__(self, n_oxides: int, cond: int = COND_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_oxides + cond, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, c], dim=1))


def gradient_penalty(
    critic: Critic,
    real: torch.Tensor,
    fake: torch.Tensor,
    c: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch = real.size(0)
    alpha = torch.rand(batch, 1, device=device)
    alpha = alpha.expand_as(real)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = critic(interp, c)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((grads.view(batch, -1).norm(2, dim=1) - 1) ** 2).mean()


def train_cwgan_gp(
    compositions: torch.Tensor,
    conditions: torch.Tensor,
    oxide_cols: list[str],
    output_dir: Path,
    *,
    epochs: int = 400,
    batch_size: int = 256,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    lr: float = 1e-4,
    seed: int = 42,
) -> tuple[Generator, dict[str, float]]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Устройство: %s, samples=%s", device, len(compositions))

    pbo_idx = oxide_cols.index("PBO") if "PBO" in oxide_cols else None
    n_ox = len(oxide_cols)

    G = Generator(n_ox, pbo_idx).to(device)
    D = Critic(n_ox).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    ds = TensorDataset(compositions, conditions)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    history = {"d_loss": [], "g_loss": []}
    for epoch in range(1, epochs + 1):
        d_epoch, g_epoch, n_batches = 0.0, 0.0, 0
        for real_x, real_c in loader:
            real_x = real_x.to(device)
            real_c = real_c.to(device)
            bs = real_x.size(0)

            for _ in range(n_critic):
                z = torch.randn(bs, LATENT_DIM, device=device)
                fake_x = G(z, real_c).detach()
                d_real = D(real_x, real_c).mean()
                d_fake = D(fake_x, real_c).mean()
                gp = gradient_penalty(D, real_x, fake_x, real_c, device)
                d_loss = d_fake - d_real + lambda_gp * gp
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                d_epoch += float(d_loss.item())

            z = torch.randn(bs, LATENT_DIM, device=device)
            fake_x = G(z, real_c)
            g_loss = -D(fake_x, real_c).mean()
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            g_epoch += float(g_loss.item())
            n_batches += 1

        history["d_loss"].append(d_epoch / max(n_batches * n_critic, 1))
        history["g_loss"].append(g_epoch / max(n_batches, 1))
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            logger.info(
                "Epoch %s/%s  D=%.4f  G=%.4f",
                epoch, epochs, history["d_loss"][-1], history["g_loss"][-1],
            )

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(G.state_dict(), ckpt_dir / "generator.pt")
    torch.save(D.state_dict(), ckpt_dir / "critic.pt")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["d_loss"], label="critic")
    ax.plot(history["g_loss"], label="generator")
    ax.set_xlabel("epoch")
    ax.set_title("cWGAN-GP training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "gan_training_loss.png", dpi=150)
    plt.close(fig)

    stats = {
        "epochs": epochs,
        "n_samples": len(compositions),
        "device": str(device),
        "final_d_loss": history["d_loss"][-1],
        "final_g_loss": history["g_loss"][-1],
    }
    return G, stats


@torch.no_grad()
def generate_compositions(
    G: Generator,
    stats: dict[str, float],
    oxide_cols: list[str],
    *,
    nd_targets: list[float],
    nud_targets: list[float],
    n_per_target: int,
    device: torch.device,
) -> pd.DataFrame:
    G.eval()
    rows = []
    for nd in nd_targets:
        for nud in nud_targets:
            c_np = normalize_condition(nd, nud, stats)
            c = torch.from_numpy(c_np).unsqueeze(0).expand(n_per_target, -1).to(device)
            z = torch.randn(n_per_target, LATENT_DIM, device=device)
            comps = G(z, c).cpu().numpy()
            for i in range(n_per_target):
                comp = dict(zip(oxide_cols, comps[i]))
                rows.append({
                    "target_ND300": nd,
                    "target_NUD300": nud,
                    **{f"oxide_{k}": v for k, v in comp.items()},
                })
    return pd.DataFrame(rows)


def score_and_filter(
    generated: pd.DataFrame,
    oxide_cols: list[str],
    models: dict[str, Any],
    design_df: pd.DataFrame,
    dist_threshold: float,
) -> pd.DataFrame:
    nn, scaler, _ = build_distance_model(design_df, oxide_cols)
    ox_limits = {}
    for ox in ANOMALY_OXIDES:
        if ox in oxide_cols:
            vals = pd.to_numeric(design_df[ox], errors="coerce").fillna(0)
            ox_limits[ox] = float(vals.quantile(0.99))

    rows = []
    for _, r in generated.iterrows():
        comp_v = np.array([float(r[f"oxide_{c}"]) for c in oxide_cols])
        comp = dict(zip(oxide_cols, comp_v))
        props = predict_properties(comp_v, models, oxide_cols)
        dist = float(
            nn.kneighbors(scaler.transform(comp_v.reshape(1, -1)))[0][0, 0]
        )
        anomaly = any(comp.get(ox, 0) > ox_limits.get(ox, np.inf) for ox in ANOMALY_OXIDES)
        feasible = (
            props["ND300"] >= 1.75
            and props["TG"] >= 450
            and comp.get("PBO", 0) <= 0.01
            and comp.get("SIO2", 0) >= 5
            and dist <= max(dist_threshold, 3.0)
            and not anomaly
        )
        row_s = pd.Series(comp)
        rows.append({
            "composition": comp_to_str(pd.Series(comp)),
            "target_ND300": r["target_ND300"],
            "target_NUD300": r["target_NUD300"],
            "ND300_pred": props["ND300"],
            "NUD300_pred": props["NUD300"],
            "DENSITY_pred": props["DENSITY"],
            "TG_pred": props["TG"],
            "PbO": comp.get("PBO", 0),
            "glass_family": classify_glass_family(row_s, oxide_cols),
            "distance_to_training": dist,
            "feasible": feasible,
            "anomaly_oxide": anomaly,
            **{f"oxide_{k}": v for k, v in comp.items()},
        })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(
            ["feasible", "ND300_pred", "distance_to_training"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        df["gan_rank"] = np.arange(1, len(df) + 1)
    return df


def plot_generated_vs_real(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    nd_col = PROPERTY_MAP["ND300"]
    nud_col = PROPERTY_MAP["NUD300"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        pd.to_numeric(real_df[nd_col], errors="coerce"),
        pd.to_numeric(real_df[nud_col], errors="coerce"),
        s=8, alpha=0.25, c="#348ABD", label="SciGlass train",
    )
    if len(gen_df):
        ax.scatter(
            gen_df["ND300_pred"],
            gen_df["NUD300_pred"],
            s=25, alpha=0.7, c="#A60628", marker="x", label="GAN + surrogate",
        )
    ax.axvline(1.80, color="gray", ls="--", lw=1)
    ax.set_xlabel("n_d")
    ax.set_ylabel("nu_d")
    ax.set_title("cWGAN-GP: real vs generated (filtered)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "gan_nd_vd_scatter.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="cWGAN-GP glass composition generator")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--forward-models", type=Path, default=DEFAULT_FORWARD_DIR)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-generate", type=int, default=3000, help="Всего генераций (делится по целям)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Только scoring уже сгенерированных (checkpoints + gan_raw_generated.csv)",
    )
    args = parser.parse_args()

    setup_logging()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)

    df = load_sciglass_df(args.data)
    oxide_cols = oxide_columns(df)
    design_df = filter_design_space(df, oxide_cols)

    cond_path = args.output / "condition_stats.json"
    if cond_path.exists():
        cond_stats = json.loads(cond_path.read_text(encoding="utf-8"))
    else:
        _, _, cond_stats = prepare_training_tensors(design_df, oxide_cols)
        cond_path.write_text(json.dumps(cond_stats, indent=2), encoding="utf-8")

    train_stats: dict[str, Any] = {"epochs": args.epochs, "score_only": args.score_only}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pbo_idx = oxide_cols.index("PBO") if "PBO" in oxide_cols else None
    G = Generator(len(oxide_cols), pbo_idx).to(device)

    if args.score_only:
        G.load_state_dict(torch.load(args.output / "checkpoints" / "generator.pt", map_location=device))
        raw_path = args.output / "gan_raw_generated.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Нет {raw_path} — сначала запустите полный пайплайн")
        generated = pd.read_csv(raw_path)
        logger.info("Score-only: %s сгенерированных составов", len(generated))
    else:
        X, C, cond_stats = prepare_training_tensors(design_df, oxide_cols)
        cond_path.write_text(json.dumps(cond_stats, indent=2), encoding="utf-8")
        G, train_stats = train_cwgan_gp(
            X, C, oxide_cols, args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        G.load_state_dict(torch.load(args.output / "checkpoints" / "generator.pt", map_location=device))

    nd_targets = [1.80, 1.85, 1.90, 1.95, 2.00, 2.05]
    nud_targets = [18.0, 22.0, 26.0, 30.0, 35.0]
    if not args.score_only:
        n_per = max(1, args.n_generate // (len(nd_targets) * len(nud_targets)))
        generated = generate_compositions(
            G, cond_stats, oxide_cols,
            nd_targets=nd_targets,
            nud_targets=nud_targets,
            n_per_target=n_per,
            device=device,
        )
        generated.to_csv(args.output / "gan_raw_generated.csv", index=False)

    models = load_forward_models(args.forward_models, oxide_cols)
    _, _, dist_threshold = build_distance_model(design_df, oxide_cols)
    scored = score_and_filter(generated, oxide_cols, models, design_df, dist_threshold)

    top100 = scored.head(100)
    top20 = scored[
        (scored["PbO"] <= 0.01) & (scored["ND300_pred"] > 1.80) & (scored["feasible"])
    ].head(20)

    top100.to_csv(args.output / "top_100_candidates.csv", index=False)
    top20.to_csv(args.output / "top_20_lead_free_high_n.csv", index=False)
    scored.to_csv(args.output / "all_gan_candidates.csv", index=False)

    plot_generated_vs_real(design_df, scored[scored["feasible"]], args.output)

    report = {
        **train_stats,
        **cond_stats,
        "n_generated": len(generated),
        "n_feasible": int(scored["feasible"].sum()) if len(scored) else 0,
        "max_nd_pred": float(scored["ND300_pred"].max()) if len(scored) else None,
        "nd_targets": nd_targets,
        "nud_targets": nud_targets,
        "forward_models_dir": str(args.forward_models.resolve()),
        "output_dir": str(args.output.resolve()),
    }
    (args.output / "gan_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("cWGAN-GP — готово")
    print("=" * 60)
    n_train = cond_stats.get("n_samples", "—")
    print(f"  Обучающая выборка: {n_train} составов (ND>1.70, PbO<=1)")
    print(f"  Epochs: {args.epochs}, feasible: {report['n_feasible']}")
    print(f"  Max ND_pred: {report['max_nd_pred']}")
    print(f"  Выход: {args.output}")
    logger.info("Сохранено в %s", args.output)


if __name__ == "__main__":
    main()
