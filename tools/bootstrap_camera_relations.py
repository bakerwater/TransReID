#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap stability analysis for TransReID camera relationships.

Input:
  outputs/stage1/market/train_features_neutral.pt

Outputs:
  camera_pair_bootstrap.csv       Pair-wise bootstrap CI
  camera_discrepancy_ci.csv       Global identity-cluster bootstrap CI
  camera_pair_rank_stability.csv Mean rank/std and Top-3 probabilities
  pairwise_order_stability.csv    Pair-of-pairs ordering stability
  bootstrap_summary.json         Run metadata and consistency check
  *.png                          CI, rank and Top-3 plots

Run from the TransReID repository root:
python tools/bootstrap_camera_relations.py \
  --train_features outputs/stage1/market/train_features_neutral.pt \
  --output_dir outputs/stage1/market/bootstrap \
  --n_bootstrap 2000 \
  --seed 42
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch


def load_features(path):
    try:
        pack = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        pack = torch.load(path, map_location="cpu")

    for key in ("features", "pids", "camids"):
        if key not in pack:
            raise KeyError(f"Missing key '{key}' in {path}")

    features = pack["features"]
    pids = pack["pids"]
    camids = pack["camids"]

    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(pids, torch.Tensor):
        pids = pids.detach().cpu().numpy()
    if isinstance(camids, torch.Tensor):
        camids = camids.detach().cpu().numpy()

    features = np.asarray(features, dtype=np.float32)
    pids = np.asarray(pids, dtype=np.int64).reshape(-1)
    camids = np.asarray(camids, dtype=np.int64).reshape(-1)

    if features.ndim != 2:
        raise ValueError(f"Expected 2-D features, got {features.shape}")
    if not (len(features) == len(pids) == len(camids)):
        raise ValueError("features, pids and camids have inconsistent lengths")
    if not np.isfinite(features).all():
        raise ValueError("Feature matrix contains NaN or Inf")
    if np.any(camids < 0):
        raise ValueError("Negative camera IDs found")

    # Exclude invalid identity labels, if present.
    valid = pids >= 0
    removed = int((~valid).sum())
    features, pids, camids = features[valid], pids[valid], camids[valid]

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Zero-norm feature vector found")
    features = features / norms

    return features, pids, camids, removed, pack.get("metadata", {})


def build_identity_camera_centroids(features, pids, camids):
    """Normalize image features, average per (PID, camera), normalize again."""
    groups = {}
    for i, (pid, cam) in enumerate(zip(pids, camids)):
        groups.setdefault((int(pid), int(cam)), []).append(i)

    pid_camera = {}
    for (pid, cam), indices in groups.items():
        center = features[np.asarray(indices, dtype=np.int64)].mean(axis=0)
        norm = np.linalg.norm(center)
        if norm < 1e-12:
            continue
        pid_camera.setdefault(cam, {})[pid] = center / norm

    return pid_camera


def build_pair_data(pid_camera, camera_ids):
    pairs = []
    for i in range(len(camera_ids)):
        for j in range(i + 1, len(camera_ids)):
            cam_a, cam_b = int(camera_ids[i]), int(camera_ids[j])
            a = pid_camera.get(cam_a, {})
            b = pid_camera.get(cam_b, {})
            common = sorted(set(a).intersection(b))
            if not common:
                print(f"WARNING: Cam{cam_a + 1}-Cam{cam_b + 1}: no shared IDs")
                continue

            distances = np.asarray([
                np.clip(1.0 - np.dot(a[pid], b[pid]), 0.0, 2.0)
                for pid in common
            ], dtype=np.float64)

            pairs.append({
                "cam_a": cam_a,
                "cam_b": cam_b,
                "label": f"Cam{cam_a + 1}-Cam{cam_b + 1}",
                "pids": np.asarray(common, dtype=np.int64),
                "distances": distances,
                "n_shared": len(common),
                "point": float(distances.mean()),
            })

    if not pairs:
        raise RuntimeError("No camera pairs with shared identities")
    return pairs


def ci95(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pairwise_bootstrap(pairs, n_bootstrap, rng):
    """Independent resampling within each camera pair."""
    rows = []
    for k, pair in enumerate(pairs):
        d = pair["distances"]
        n = len(d)
        boot_means = np.empty(n_bootstrap, dtype=np.float64)

        for b in range(n_bootstrap):
            boot_means[b] = rng.choice(d, size=n, replace=True).mean()

        lo, hi = ci95(boot_means)
        rows.append({
            "camera_pair": pair["label"],
            "n_shared_identities": n,
            "point_estimate": pair["point"],
            "bootstrap_mean": float(boot_means.mean()),
            "bootstrap_std": float(boot_means.std(ddof=1)),
            "ci95_lower": lo,
            "ci95_upper": hi,
            "ci95_width": hi - lo,
        })
        print(
            f"[Pair-wise {k + 1}/{len(pairs)}] {pair['label']}: "
            f"{pair['point']:.6f}, CI [{lo:.6f}, {hi:.6f}]"
        )
    return rows


def average_ranks(values):
    """Return 1-based ranks; ties receive their average rank."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    return ranks


def global_identity_bootstrap(pairs, all_pids, n_bootstrap, rng):
    """
    Resample all training identities with replacement.
    A shared identity multiplicity vector is used for every camera pair.
    """
    pid_to_idx = {int(pid): i for i, pid in enumerate(all_pids)}
    pair_indices = [
        np.asarray([pid_to_idx[int(pid)] for pid in p["pids"]], dtype=np.int64)
        for p in pairs
    ]

    n_ids = len(all_pids)
    n_pairs = len(pairs)
    boot = np.full((n_bootstrap, n_pairs), np.nan, dtype=np.float64)

    for b in range(n_bootstrap):
        sampled = rng.integers(0, n_ids, size=n_ids)
        multiplicity = np.bincount(sampled, minlength=n_ids)

        for j, pair in enumerate(pairs):
            idx = pair_indices[j]
            weights = multiplicity[idx]
            denominator = weights.sum()
            if denominator > 0:
                boot[b, j] = np.dot(
                    weights.astype(np.float64), pair["distances"]
                ) / denominator

        if (b + 1) % max(1, n_bootstrap // 10) == 0:
            print(f"[Global identity bootstrap] {b + 1}/{n_bootstrap}")

    valid = np.isfinite(boot).all(axis=1)
    boot = boot[valid]
    if len(boot) < 0.95 * n_bootstrap:
        raise RuntimeError(
            f"Only {len(boot)}/{n_bootstrap} valid global bootstrap replicates"
        )

    ranks = np.vstack([average_ranks(row) for row in boot])

    ci_rows = []
    rank_rows = []
    for j, pair in enumerate(pairs):
        values = boot[:, j]
        lo, hi = ci95(values)
        r = ranks[:, j]
        rank_lo, rank_hi = ci95(r)

        ci_rows.append({
            "camera_pair": pair["label"],
            "n_shared_identities": pair["n_shared"],
            "point_estimate": pair["point"],
            "global_bootstrap_mean": float(values.mean()),
            "global_bootstrap_std": float(values.std(ddof=1)),
            "ci95_lower": lo,
            "ci95_upper": hi,
            "ci95_width": hi - lo,
        })
        rank_rows.append({
            "camera_pair": pair["label"],
            "point_estimate": pair["point"],
            "mean_rank": float(r.mean()),
            "rank_std": float(r.std(ddof=1)),
            "rank_ci95_lower": rank_lo,
            "rank_ci95_upper": rank_hi,
            "top3_similar_probability": float(np.mean(r <= 3)),
            "top3_most_different_probability": float(np.mean(r >= n_pairs - 2)),
        })

    # Stability of every pair-of-camera-pairs ordering.
    order_rows = []
    point = np.asarray([p["point"] for p in pairs])
    for i in range(n_pairs):
        for j in range(i + 1, n_pairs):
            delta = point[i] - point[j]
            if abs(delta) < 1e-15:
                continue
            if delta < 0:
                preserved = boot[:, i] < boot[:, j]
                original_order = f"{pairs[i]['label']} < {pairs[j]['label']}"
            else:
                preserved = boot[:, i] > boot[:, j]
                original_order = f"{pairs[j]['label']} < {pairs[i]['label']}"

            order_rows.append({
                "camera_pair_a": pairs[i]["label"],
                "camera_pair_b": pairs[j]["label"],
                "distance_a": float(point[i]),
                "distance_b": float(point[j]),
                "original_order": original_order,
                "order_preserved_probability": float(preserved.mean()),
                "order_reversed_probability": float(1.0 - preserved.mean()),
            })

    return boot, ranks, ci_rows, rank_rows, order_rows


def parse_camera_id(label):
    match = re.search(r"(\d+)$", str(label).strip())
    if not match:
        raise ValueError(f"Cannot parse camera label: {label}")
    value = int(match.group(1))
    return value - 1 if str(label).lower().startswith("cam") else value


def read_existing_discrepancy(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValueError(f"Existing discrepancy CSV is empty: {path}")

    col_ids = [parse_camera_id(x) for x in rows[0][1:]]
    matrix = {}
    for row in rows[1:]:
        if not row:
            continue
        row_id = parse_camera_id(row[0])
        for col_id, value in zip(col_ids, row[1:]):
            if value.strip():
                matrix[(row_id, col_id)] = float(value)
    return matrix


def check_existing_matrix(pairs, path, tolerance):
    if not path or not Path(path).is_file():
        return {"checked": False, "reason": f"File not found or not supplied: {path}"}

    existing = read_existing_discrepancy(path)
    comparisons = []
    missing = []
    for p in pairs:
        key = (p["cam_a"], p["cam_b"])
        reverse = (p["cam_b"], p["cam_a"])
        if key in existing:
            old = existing[key]
        elif reverse in existing:
            old = existing[reverse]
        else:
            missing.append(p["label"])
            continue

        comparisons.append({
            "camera_pair": p["label"],
            "recomputed": p["point"],
            "existing": old,
            "absolute_difference": abs(p["point"] - old),
        })

    if not comparisons:
        return {"checked": True, "reason": "No overlapping camera pairs"}

    diffs = [x["absolute_difference"] for x in comparisons]
    mismatches = [x for x in comparisons if x["absolute_difference"] > tolerance]
    return {
        "checked": True,
        "path": str(path),
        "pairs_compared": len(comparisons),
        "mean_absolute_difference": float(np.mean(diffs)),
        "max_absolute_difference": float(np.max(diffs)),
        "tolerance": tolerance,
        "num_pairs_over_tolerance": len(mismatches),
        "pairs_over_tolerance": mismatches,
        "missing_pairs": missing,
    }


def make_plots(out_dir, pairs, ci_rows, rank_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable: skipping plots")
        return

    ci = {x["camera_pair"]: x for x in ci_rows}
    ordered = sorted(pairs, key=lambda x: x["point"])
    labels = [x["label"] for x in ordered]
    points = np.asarray([x["point"] for x in ordered])
    low = np.asarray([ci[x["label"]]["ci95_lower"] for x in ordered])
    high = np.asarray([ci[x["label"]]["ci95_upper"] for x in ordered])
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, max(5, len(labels) * 0.38)))
    ax.errorbar(points, y, xerr=np.vstack([points - low, high - points]),
                fmt="o", capsize=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean cosine discrepancy")
    ax.set_title("Global identity-bootstrap 95% confidence intervals")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "camera_discrepancy_ci_forest.png", dpi=220)
    plt.close(fig)

    rank_sorted = sorted(rank_rows, key=lambda x: x["mean_rank"])
    labels = [x["camera_pair"] for x in rank_sorted]
    means = np.asarray([x["mean_rank"] for x in rank_sorted])
    stds = np.asarray([x["rank_std"] for x in rank_sorted])
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, max(5, len(labels) * 0.38)))
    ax.errorbar(means, y, xerr=stds, fmt="o", capsize=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean rank (1 = smallest discrepancy)")
    ax.set_title("Camera-pair rank stability (mean ± std)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "camera_pair_rank_stability.png", dpi=220)
    plt.close(fig)

    x = np.arange(len(labels))
    width = 0.38
    similar = [r["top3_similar_probability"] for r in rank_sorted]
    different = [r["top3_most_different_probability"] for r in rank_sorted]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, similar, width, label="Top-3 most similar")
    ax.bar(x + width / 2, different, width, label="Top-3 most different")
    ax.set_xticks(x, labels, rotation=55, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Bootstrap probability")
    ax.set_title("Top-3 camera-pair membership stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "camera_pair_top3_probability.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--existing_discrepancy",
        default="outputs/stage1/market/analysis/camera_discrepancy.csv"
    )
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    if args.n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates; recommend 2000.")
    if args.tolerance < 0:
        raise ValueError("--tolerance must be non-negative.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading features:", args.train_features)
    features, pids, camids, removed, metadata = load_features(args.train_features)
    camera_ids = np.sort(np.unique(camids))
    all_pids = np.sort(np.unique(pids))

    print("Feature shape:", features.shape)
    print("Unique identities:", len(all_pids))
    print("Camera IDs:", camera_ids.tolist())
    print("Invalid PID images removed:", removed)

    print("\nBuilding identity-camera centroids...")
    pid_camera = build_identity_camera_centroids(features, pids, camids)
    print("Building per-identity camera-pair distances...")
    pairs = build_pair_data(pid_camera, camera_ids)
    print("Number of camera pairs:", len(pairs))

    # Independent reproducible random streams for the two bootstrap procedures.
    seed_seq = np.random.SeedSequence(args.seed)
    rng_pair, rng_global = [
        np.random.default_rng(s) for s in seed_seq.spawn(2)
    ]

    print("\nRunning pair-wise bootstrap...")
    pair_rows = pairwise_bootstrap(pairs, args.n_bootstrap, rng_pair)
    write_csv(out_dir / "camera_pair_bootstrap.csv", pair_rows)

    print("\nRunning global identity-cluster bootstrap...")
    boot, ranks, ci_rows, rank_rows, order_rows = global_identity_bootstrap(
        pairs, all_pids, args.n_bootstrap, rng_global
    )
    write_csv(out_dir / "camera_discrepancy_ci.csv", ci_rows)
    write_csv(out_dir / "camera_pair_rank_stability.csv", rank_rows)
    write_csv(out_dir / "pairwise_order_stability.csv", order_rows)

    np.save(out_dir / "global_bootstrap_discrepancies.npy", boot)
    np.save(out_dir / "global_bootstrap_ranks.npy", ranks)

    consistency = check_existing_matrix(
        pairs, args.existing_discrepancy, args.tolerance
    )

    summary = {
        "input_features": args.train_features,
        "metadata": metadata,
        "n_images_used": int(len(features)),
        "n_invalid_pid_images_removed": int(removed),
        "n_unique_identities": int(len(all_pids)),
        "camera_ids_internal_zero_based": [int(x) for x in camera_ids],
        "n_camera_pairs": int(len(pairs)),
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "mean_pairwise_order_preserved_probability": (
            float(np.mean([x["order_preserved_probability"] for x in order_rows]))
            if order_rows else None
        ),
        "existing_discrepancy_consistency_check": consistency,
        "notes": [
            "Rank 1 means smallest discrepancy / most similar camera pair.",
            "Top-3 most different means the three largest discrepancies.",
            "Intervals quantify uncertainty over sampled training identities.",
            "Discrepancy is a composite cross-camera appearance difference, not a pure camera-style measure."
        ]
    }
    with (out_dir / "bootstrap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    make_plots(out_dir, pairs, ci_rows, rank_rows)

    print("\n=== Existing discrepancy consistency check ===")
    print(json.dumps(consistency, indent=2, ensure_ascii=False))
    print("\nTop-3 most similar pairs:")
    for x in sorted(rank_rows, key=lambda r: r["mean_rank"])[:3]:
        print(
            f"{x['camera_pair']}: mean_rank={x['mean_rank']:.3f}, "
            f"rank_std={x['rank_std']:.3f}, "
            f"P(top3)={x['top3_similar_probability']:.3f}"
        )
    print("\nTop-3 most different pairs:")
    for x in sorted(rank_rows, key=lambda r: r["mean_rank"], reverse=True)[:3]:
        print(
            f"{x['camera_pair']}: mean_rank={x['mean_rank']:.3f}, "
            f"rank_std={x['rank_std']:.3f}, "
            f"P(top3)={x['top3_most_different_probability']:.3f}"
        )
    print("\nBootstrap analysis finished. Output:", out_dir)


if __name__ == "__main__":
    main()
