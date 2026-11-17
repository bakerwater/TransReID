#!/usr/bin/env python3
"""
Compute:
1) train-set camera discrepancy / affinity / confidence matrices using shared identities;
2) directed query-camera -> gallery-camera retrieval metrics (mAP, Rank-1/5, mINP).

Recommended:
- train features: train_features_neutral.pt (SIE multiplier set to zero)
- test features:  test_features_native.pt (normal baseline inference)
- optional test_features_neutral.pt for a diagnostic comparison.

Run from the TransReID repository root:
python tools/analyze_camera_relations.py \
  --train_features outputs/stage1/market/train_features_neutral.pt \
  --test_native outputs/stage1/market/test_features_native.pt \
  --test_neutral outputs/stage1/market/test_features_neutral.pt \
  --output_dir outputs/stage1/market/analysis
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def torch_load(path):
    # Compatible with older PyTorch and newer versions where weights_only defaults to True.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_pack(path):
    pack = torch_load(path)
    feats = pack["features"].float()
    feats = F.normalize(feats, p=2, dim=1).cpu()
    pids = pack["pids"].cpu().numpy().astype(np.int64)
    camids = pack["camids"].cpu().numpy().astype(np.int64)
    return pack, feats, pids, camids


def write_matrix_csv(path, matrix, camera_ids):
    labels = [f"Cam{int(c) + 1}" for c in camera_ids]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["camera"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [
                "" if not np.isfinite(x) else f"{float(x):.6f}" for x in row
            ])


def save_heatmap(path, matrix, camera_ids, title, colorbar_label):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping heatmap:", path)
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto")
    labels = [f"C{int(c) + 1}" for c in camera_ids]
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Camera")
    ax.set_ylabel("Camera")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_camera_matrices(features, pids, camids, kappa=20.0):
    camera_ids = np.sort(np.unique(camids))
    n_cam = len(camera_ids)
    cam_to_idx = {int(c): i for i, c in enumerate(camera_ids)}

    # pid_camera[cam][pid] = normalized mean feature for that identity-camera pair.
    pid_camera = {int(c): {} for c in camera_ids}
    within_by_cam = defaultdict(list)

    unique_pids = np.unique(pids)
    for pid in unique_pids:
        idx_pid = np.where(pids == pid)[0]
        for cam in np.unique(camids[idx_pid]):
            idx = idx_pid[camids[idx_pid] == cam]
            f = features[idx]
            centroid = F.normalize(f.mean(dim=0, keepdim=True), p=2, dim=1)[0]
            pid_camera[int(cam)][int(pid)] = centroid

            if len(idx) > 1:
                # Mean cosine distance from images to their own identity-camera centroid.
                d = 1.0 - torch.mv(f, centroid)
                within_by_cam[int(cam)].append(float(d.mean().item()))

    discrepancy = np.full((n_cam, n_cam), np.nan, dtype=np.float64)
    affinity = np.full((n_cam, n_cam), np.nan, dtype=np.float64)
    confidence = np.zeros((n_cam, n_cam), dtype=np.float64)
    shared_count = np.zeros((n_cam, n_cam), dtype=np.int64)
    within_dispersion = np.full(n_cam, np.nan, dtype=np.float64)

    for cam in camera_ids:
        values = within_by_cam[int(cam)]
        if values:
            within_dispersion[cam_to_idx[int(cam)]] = float(np.mean(values))

    pair_rows = []
    offdiag_distances = []

    for i, cam_a in enumerate(camera_ids):
        discrepancy[i, i] = 0.0
        affinity[i, i] = 1.0
        confidence[i, i] = 1.0
        shared_count[i, i] = len(pid_camera[int(cam_a)])

        for j in range(i + 1, n_cam):
            cam_b = camera_ids[j]
            common_pids = sorted(
                set(pid_camera[int(cam_a)].keys())
                & set(pid_camera[int(cam_b)].keys())
            )
            n_shared = len(common_pids)
            shared_count[i, j] = shared_count[j, i] = n_shared

            if n_shared == 0:
                continue

            dists = []
            for pid in common_pids:
                fa = pid_camera[int(cam_a)][int(pid)]
                fb = pid_camera[int(cam_b)][int(pid)]
                dists.append(float((1.0 - torch.dot(fa, fb)).item()))

            dists = np.asarray(dists, dtype=np.float64)
            mean_d = float(np.mean(dists))
            median_d = float(np.median(dists))
            std_d = float(np.std(dists))
            conf = float(n_shared / (n_shared + kappa))

            discrepancy[i, j] = discrepancy[j, i] = mean_d
            confidence[i, j] = confidence[j, i] = conf
            offdiag_distances.append(mean_d)

            wa = within_dispersion[i]
            wb = within_dispersion[j]
            within_mean = np.nanmean([wa, wb]) if np.isfinite([wa, wb]).any() else np.nan
            excess = max(0.0, mean_d - within_mean) if np.isfinite(within_mean) else np.nan

            pair_rows.append({
                "camera_a": int(cam_a) + 1,
                "camera_b": int(cam_b) + 1,
                "shared_identities": n_shared,
                "mean_cross_camera_cosine_distance": mean_d,
                "median_cross_camera_cosine_distance": median_d,
                "std_cross_camera_cosine_distance": std_d,
                "confidence": conf,
                "within_camera_dispersion_a": wa,
                "within_camera_dispersion_b": wb,
                "excess_discrepancy_diagnostic": excess,
            })

    tau = float(np.median(offdiag_distances)) if offdiag_distances else 1.0
    if not np.isfinite(tau) or tau <= 1e-12:
        tau = 1.0

    for i in range(n_cam):
        for j in range(n_cam):
            if np.isfinite(discrepancy[i, j]):
                affinity[i, j] = float(np.exp(-discrepancy[i, j] / tau))

    return (
        camera_ids, discrepancy, affinity, confidence, shared_count,
        within_dispersion, pair_rows, tau
    )


def evaluate_one_camera_pair(qf, qpid, gf, gpid, chunk_size=128):
    if len(qf) == 0 or len(gf) == 0:
        return None

    qf = F.normalize(qf.float(), p=2, dim=1)
    gf = F.normalize(gf.float(), p=2, dim=1)

    aps, inps, cmc1, cmc5 = [], [], [], []
    total_positive = 0

    for start in range(0, len(qf), chunk_size):
        q_chunk = qf[start:start + chunk_size]
        # Euclidean distance on L2-normalized features has the same ranking as cosine distance.
        dist = (1.0 - torch.matmul(q_chunk, gf.t())).cpu().numpy()
        order = np.argsort(dist, axis=1)

        for local_i, ranking in enumerate(order):
            q_index = start + local_i
            matches = (gpid[ranking] == qpid[q_index]).astype(np.int32)
            positive_positions = np.flatnonzero(matches)

            if len(positive_positions) == 0:
                continue

            total_positive += int(len(positive_positions))
            cumulative = np.cumsum(matches)
            precision = cumulative / (np.arange(len(matches)) + 1.0)
            ap = float((precision * matches).sum() / len(positive_positions))
            last_positive_rank = int(positive_positions[-1]) + 1
            inp = float(len(positive_positions) / last_positive_rank)

            aps.append(ap)
            inps.append(inp)
            cmc1.append(float(cumulative[0] > 0))
            cmc5.append(float(cumulative[min(4, len(cumulative) - 1)] > 0))

    if not aps:
        return None

    return {
        "valid_queries": len(aps),
        "positive_pairs": total_positive,
        "mAP": float(np.mean(aps)),
        "Rank1": float(np.mean(cmc1)),
        "Rank5": float(np.mean(cmc5)),
        "mINP": float(np.mean(inps)),
    }


def evaluate_camera_pairs(test_path, mode_name, output_dir):
    pack, feats, pids, camids = load_pack(test_path)
    if "num_query" not in pack:
        raise KeyError(f"{test_path} does not contain num_query")
    num_query = int(pack["num_query"])

    # Remove invalid/junk identities (PID = -1)
    qf, qpid, qcam = feats[:num_query], pids[:num_query], camids[:num_query]
    gf, gpid, gcam = feats[num_query:], pids[num_query:], camids[num_query:]

    q_valid = qpid >= 0
    qf, qpid, qcam = qf[q_valid], qpid[q_valid], qcam[q_valid]

    g_valid = gpid >= 0
    gf, gpid, gcam = gf[g_valid], gpid[g_valid], gcam[g_valid]

    camera_ids = np.sort(np.unique(np.concatenate([qcam, gcam])))
    index = {int(c): i for i, c in enumerate(camera_ids)}
    map_matrix = np.full((len(camera_ids), len(camera_ids)), np.nan)
    r1_matrix = np.full_like(map_matrix, np.nan)
    rows = []

    for cam_q in camera_ids:
        qi = np.where(qcam == cam_q)[0]
        for cam_g in camera_ids:
            if cam_q == cam_g:
                continue
            gi = np.where(gcam == cam_g)[0]
            if len(qi) == 0 or len(gi) == 0:
                continue

            # Only queries having at least one same-identity gallery image in this target camera
            # are valid for this directed camera-pair evaluation.
            has_positive = np.isin(qpid[qi], np.unique(gpid[gi]))
            valid_qi = qi[has_positive]
            result = evaluate_one_camera_pair(
                qf[valid_qi], qpid[valid_qi], gf[gi], gpid[gi]
            )

            if result is None:
                continue

            i, j = index[int(cam_q)], index[int(cam_g)]
            map_matrix[i, j] = result["mAP"]
            r1_matrix[i, j] = result["Rank1"]

            rows.append({
                "feature_mode": mode_name,
                "query_camera": int(cam_q) + 1,
                "gallery_camera": int(cam_g) + 1,
                "query_count_total": int(len(qi)),
                "query_count_valid": result["valid_queries"],
                "gallery_count": int(len(gi)),
                "positive_pairs": result["positive_pairs"],
                "mAP": result["mAP"],
                "Rank1": result["Rank1"],
                "Rank5": result["Rank5"],
                "mINP": result["mINP"],
            })

    return camera_ids, map_matrix, r1_matrix, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features", required=True)
    parser.add_argument("--test_native", required=True)
    parser.add_argument("--test_neutral", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--kappa", type=float, default=20.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_pack, train_features, train_pids, train_camids = load_pack(args.train_features)
    (
        camera_ids, discrepancy, affinity, confidence, shared_count,
        within_dispersion, pair_rows, tau
    ) = build_camera_matrices(
        train_features, train_pids, train_camids, kappa=args.kappa
    )

    np.save(out / "camera_discrepancy.npy", discrepancy)
    np.save(out / "camera_affinity.npy", affinity)
    np.save(out / "camera_confidence.npy", confidence)
    np.save(out / "camera_shared_identity_count.npy", shared_count)
    np.save(out / "camera_within_dispersion.npy", within_dispersion)

    write_matrix_csv(out / "camera_discrepancy.csv", discrepancy, camera_ids)
    write_matrix_csv(out / "camera_affinity.csv", affinity, camera_ids)
    write_matrix_csv(out / "camera_confidence.csv", confidence, camera_ids)
    write_matrix_csv(out / "camera_shared_identity_count.csv", shared_count, camera_ids)

    save_heatmap(
        out / "camera_discrepancy.png", discrepancy, camera_ids,
        "Camera-pair appearance discrepancy", "Mean cosine distance"
    )
    save_heatmap(
        out / "camera_affinity.png", affinity, camera_ids,
        "Camera affinity", "exp(-distance / tau)"
    )
    save_heatmap(
        out / "camera_confidence.png", confidence, camera_ids,
        "Camera-pair confidence", "n_shared / (n_shared + kappa)"
    )

    if pair_rows:
        fields = list(pair_rows[0].keys())
        with open(out / "camera_pair_relation_details.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(pair_rows)

    print("\nCamera relationship analysis")
    print("Camera IDs (internal, zero-based):", camera_ids.tolist())
    print("Affinity temperature tau:", tau)
    print("Shared-identity count matrix:\n", shared_count)
    print("Discrepancy matrix:\n", np.round(discrepancy, 4))
    print("Affinity matrix:\n", np.round(affinity, 4))
    print("Saved relationship matrices to:", out)

    all_metric_rows = []
    modes = [("native", args.test_native)]
    if args.test_neutral:
        modes.append(("neutral", args.test_neutral))

    for mode, path in modes:
        cam_ids_eval, map_matrix, r1_matrix, rows = evaluate_camera_pairs(
            path, mode, out
        )
        all_metric_rows.extend(rows)
        np.save(out / f"camera_pair_mAP_{mode}.npy", map_matrix)
        np.save(out / f"camera_pair_Rank1_{mode}.npy", r1_matrix)
        write_matrix_csv(out / f"camera_pair_mAP_{mode}.csv", map_matrix, cam_ids_eval)
        write_matrix_csv(out / f"camera_pair_Rank1_{mode}.csv", r1_matrix, cam_ids_eval)
        save_heatmap(
            out / f"camera_pair_mAP_{mode}.png", map_matrix, cam_ids_eval,
            f"Directed camera-pair mAP ({mode})", "mAP"
        )
        save_heatmap(
            out / f"camera_pair_Rank1_{mode}.png", r1_matrix, cam_ids_eval,
            f"Directed camera-pair Rank-1 ({mode})", "Rank-1"
        )

    if all_metric_rows:
        fields = list(all_metric_rows[0].keys())
        with open(out / "camera_pair_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_metric_rows)

        # Join symmetric train-set camera relations with directed test-set retrieval results.
        metric_lookup = defaultdict(lambda: defaultdict(list))
        for row in all_metric_rows:
            a = int(row["query_camera"])
            b = int(row["gallery_camera"])
            key = (min(a, b), max(a, b))
            metric_lookup[key][row["feature_mode"]].append(row)

        relation_performance_rows = []
        for rel in pair_rows:
            a, b = int(rel["camera_a"]), int(rel["camera_b"])
            key = (a, b)
            out_row = dict(rel)
            for mode in ["native", "neutral"]:
                directions = metric_lookup[key].get(mode, [])
                for metric in ["mAP", "Rank1", "Rank5", "mINP"]:
                    vals = [float(x[metric]) for x in directions]
                    out_row[f"{mode}_{metric}_mean_over_directions"] = (
                        float(np.mean(vals)) if vals else ""
                    )
                out_row[f"{mode}_valid_directions"] = len(directions)
            relation_performance_rows.append(out_row)

        if relation_performance_rows:
            fields = list(relation_performance_rows[0].keys())
            with open(out / "camera_relation_vs_retrieval.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(relation_performance_rows)

            # Optional Spearman analysis: expected negative association between discrepancy and mAP.
            try:
                from scipy.stats import spearmanr
                summary_rows = []
                for mode in ["native", "neutral"]:
                    xs_map, ys_map, xs_r1, ys_r1 = [], [], [], []
                    for row in relation_performance_rows:
                        x = float(row["mean_cross_camera_cosine_distance"])
                        y_map = row.get(f"{mode}_mAP_mean_over_directions", "")
                        y_r1 = row.get(f"{mode}_Rank1_mean_over_directions", "")
                        if y_map != "":
                            xs_map.append(x)
                            ys_map.append(float(y_map))
                        if y_r1 != "":
                            xs_r1.append(x)
                            ys_r1.append(float(y_r1))
                    rho_map = np.nan
                    if len(ys_map) >= 3 and np.std(xs_map) > 0 and np.std(ys_map) > 0:
                        rho_map = float(spearmanr(xs_map, ys_map).statistic)
                    rho_r1 = np.nan
                    if len(ys_r1) >= 3 and np.std(xs_r1) > 0 and np.std(ys_r1) > 0:
                        rho_r1 = float(spearmanr(xs_r1, ys_r1).statistic)
                    summary_rows.append({
                        "feature_mode": mode,
                        "num_camera_pairs_for_mAP": len(ys_map),
                        "spearman_discrepancy_vs_mAP": rho_map,
                        "num_camera_pairs_for_Rank1": len(ys_r1),
                        "spearman_discrepancy_vs_Rank1": rho_r1,
                    })
                with open(out / "camera_relation_correlation.csv", "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(summary_rows)
            except ImportError:
                print("scipy not installed; skipped Spearman correlation summary.")

    print("Saved directed camera-pair retrieval metrics to:", out / "camera_pair_metrics.csv")


if __name__ == "__main__":
    main()
