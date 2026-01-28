import argparse
import csv
import math
import os
import random
import shutil
import time


SUMMARY_REQUIRED_BASE_COLUMNS = {
    "tag",
    "scheme",
    "hotspot_p95_mean_ms",
    "hotspot_mean_ms",
}
SUMMARY_OVERLOAD_COLUMNS = ("overload_ratio_q1000", "overload_ratio")
SUMMARY_MIGRATED_COLUMNS = ("migrated_weight_total", "migrated_robots_total")
DEFAULT_KNEE_WEIGHT = 0.5
DEFAULT_LAMBDA_SWEEP = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
DEFAULT_SEEDS = "0,1,2"

WINDOW_REQUIRED_COLUMNS = {
    "t_start",
    "t_end",
    "p95_ms",
    "overload_ratio_q1000",
    "generated",
    "hotspot_overload_ratio_q1000",
}
SCALABILITY_REQUIRED_COLUMNS = {
    "N",
    "load_scale",
    "scheme",
    "hotspot_p95_mean_ms",
    "hotspot_mean_ms",
    "overload_ratio_q1000",
    "migrated_weight_total",
    "seed",
}


def _parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_seeds(text):
    if text is None:
        return []
    parts = []
    for chunk in str(text).replace(";", ",").split(","):
        value = chunk.strip()
        if not value:
            continue
        parts.append(int(value))
    return parts


def _bootstrap_ci(values, alpha=0.05, iters=1000, rng=None):
    values = [value for value in values if value is not None and not _is_nan(value)]
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = rng or random.Random(0)
    means = []
    n = len(values)
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low_idx = int((alpha / 2.0) * (len(means) - 1))
    high_idx = int((1.0 - alpha / 2.0) * (len(means) - 1))
    return means[low_idx], means[high_idx]


def _is_nan(value):
    return isinstance(value, float) and math.isnan(value)


def _cost_value(row):
    value = row.get("cost_per_request")
    if value is None or _is_nan(value):
        value = row.get("migrated_weight_total")
    return value


def _parse_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_load_multiplier(row):
    for key in ("load_multiplier", "load_scale", "load"):
        if key in row:
            return _parse_float(row.get(key))
    return None


def _select_single_row(rows, filters, prefer_latest=True):
    matches = []
    for row in rows:
        ok = True
        for key, expected in filters.items():
            if key not in row:
                ok = False
                break
            value = row.get(key)
            if isinstance(expected, float):
                if value is None or _is_nan(value) or abs(value - expected) > 1.0e-6:
                    ok = False
                    break
            else:
                if value != expected:
                    ok = False
                    break
        if ok:
            matches.append(row)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    if prefer_latest:
        for key in ("timestamp", "run_id", "run_ts"):
            if key in matches[0]:
                scored = []
                for row in matches:
                    score = _parse_float(row.get(key))
                    scored.append((score if score is not None else -1.0, row))
                scored.sort(key=lambda item: item[0], reverse=True)
                print(
                    f"Warning: {len(matches)} rows match {filters}; selecting latest by {key}."
                )
                return scored[0][1]
    print(f"Warning: {len(matches)} rows match {filters}; selecting first deterministically.")
    return matches[0]


def _is_within_interval(value, low, high):
    return value is not None and not _is_nan(value) and low <= value <= high


def _distance_to_interval(value, low, high):
    if value is None or _is_nan(value):
        return float("inf")
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _format_float(value, decimals):
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def _format_ratio_label(value):
    if value is None or _is_nan(value):
        return "N/A"
    if value < 1.0e-3:
        return "<1e-3"
    return f"{value:.3f}"


def _resolve_summary_column(headers, candidates, label, path):
    for name in candidates:
        if name in headers:
            return name
    expected = ", ".join(f"'{name}'" for name in candidates)
    raise RuntimeError(f"Missing required {label} column in {path}. Expected one of: {expected}.")


def _load_summary_file(summary_path):
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"summary CSV not found: {summary_path}")

    with open(summary_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        missing = [col for col in SUMMARY_REQUIRED_BASE_COLUMNS if col not in headers]
        if missing:
            raise RuntimeError(
                f"Missing required columns in {summary_path}: {', '.join(sorted(missing))}"
            )
        overload_key = _resolve_summary_column(
            headers, SUMMARY_OVERLOAD_COLUMNS, "overload ratio (queue > 1000)", summary_path
        )
        migrated_key = _resolve_summary_column(
            headers, SUMMARY_MIGRATED_COLUMNS, "migrated load", summary_path
        )

        rows = []
        for row in reader:
            item = {}
            for key, value in row.items():
                target_key = key
                if key == overload_key and overload_key != "overload_ratio_q1000":
                    target_key = "overload_ratio_q1000"
                if key == migrated_key and migrated_key != "migrated_weight_total":
                    target_key = "migrated_weight_total"
                if target_key in ("tag", "scheme", "profile"):
                    item[target_key] = value
                elif target_key == "N":
                    item[target_key] = int(float(value)) if value not in (None, "") else None
                else:
                    item[target_key] = _parse_float(value)
            rows.append(item)
    return rows


def _load_window_rows(path, extra_required=None):
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing window_kpis CSV: {path}")

    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        required = set(WINDOW_REQUIRED_COLUMNS)
        if extra_required:
            required.update(extra_required)
        missing = [col for col in required if col not in headers]
        if missing:
            raise RuntimeError(
                f"Missing required columns in {path}: {', '.join(sorted(missing))}"
            )
        return list(reader)


def _mean(values):
    if not values:
        raise RuntimeError("No values provided for mean.")
    return sum(values) / len(values)


def _hotspot_mean_metric(rows, t_hot_start, t_hot_end, key, label):
    values = []
    for row in rows:
        t_start = _parse_float(row.get("t_start"))
        t_end = _parse_float(row.get("t_end"))
        if t_start is None or t_end is None:
            continue
        if t_start >= t_hot_start and t_end <= t_hot_end:
            value = _parse_float(row.get(key))
            if value is not None and not _is_nan(value):
                values.append(value)
    if not values:
        return float("nan")
    return _mean(values)


def _hotspot_mean_p95(rows, t_hot_start, t_hot_end):
    return _hotspot_mean_metric(rows, t_hot_start, t_hot_end, "p95_ms", "p95_ms")


def _hotspot_mean_mean(rows, t_hot_start, t_hot_end):
    return _hotspot_mean_metric(rows, t_hot_start, t_hot_end, "mean_ms", "mean_ms")


def _overload_ratio(rows, key, label):
    values = []
    for row in rows:
        value = _parse_float(row.get(key))
        if value is None or _is_nan(value):
            raise RuntimeError(f"Invalid {label} value.")
        values.append(value)
    if not values:
        raise RuntimeError(f"No {label} values found.")
    return _mean(values)


def _overload_ratio_q1000(rows):
    return _overload_ratio(rows, "overload_ratio_q1000", "overload_ratio_q1000")


def _save_figure(fig, png_path):
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    pdf_path = os.path.splitext(png_path)[0] + ".pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    return png_path, pdf_path


def _find_row(rows, tag, scheme, expected_n=None, path=None):
    matches = [row for row in rows if row.get("tag") == tag and row.get("scheme") == scheme]
    if expected_n is not None:
        filtered = []
        for row in matches:
            if row.get("N") is None:
                filtered.append(row)
            elif int(row.get("N")) == int(expected_n):
                filtered.append(row)
        matches = filtered
    if not matches:
        location = f" in {path}" if path else ""
        raise RuntimeError(f"Missing row for tag={tag} scheme={scheme}{location}")
    if expected_n is not None:
        for row in matches:
            if row.get("N") is not None and int(row.get("N")) != int(expected_n):
                location = f" in {path}" if path else ""
                raise RuntimeError(
                    f"Row for tag={tag} scheme={scheme} has N={row.get('N')}, expected {expected_n}{location}"
                )
    if len(matches) > 1 and expected_n is not None:
        location = f" in {path}" if path else ""
        raise RuntimeError(f"Multiple rows for tag={tag} scheme={scheme} N={expected_n}{location}")
    return matches[0]


def _require_value(row, key, label, path):
    value = row.get(key)
    if value is None:
        raise RuntimeError(f"Missing {key} for {label} in {path}")
    return value


def _normalize_profiles(profile_rows, eps=1e-9):
    if not profile_rows:
        raise RuntimeError("No S2 profile rows available for knee selection.")
    xs = [_cost_value(row) for row in profile_rows]
    ys = [row["overload_ratio_q1000"] for row in profile_rows]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    range_x = max_x - min_x
    range_y = max_y - min_y
    for row in profile_rows:
        cost_val = _cost_value(row)
        row["cost_value"] = cost_val
        row["norm_x"] = (cost_val - min_x) / (range_x + eps)
        row["norm_y"] = (row["overload_ratio_q1000"] - min_y) / (range_y + eps)
    return min_x, max_x, min_y, max_y


def _same_profile(row_a, row_b):
    if row_a is None or row_b is None:
        return False
    tag_a = row_a.get("tag")
    tag_b = row_b.get("tag")
    if tag_a and tag_b:
        return tag_a == tag_b
    return row_a is row_b


def _select_balanced_profile(profile_rows, lite_row, strong_row, weight=DEFAULT_KNEE_WEIGHT):
    unique_tags = {row.get("tag") for row in profile_rows if row.get("tag") is not None}
    allow_overlap = len(unique_tags) <= 1
    for row in profile_rows:
        row["score"] = math.hypot(weight * row["norm_x"], (1.0 - weight) * row["norm_y"])

    sorted_rows = sorted(
        profile_rows,
        key=lambda row: (
            row["score"],
            row["overload_ratio_q1000"],
            row["migrated_weight_total"],
        ),
    )
    if allow_overlap:
        return sorted_rows[0]
    for row in sorted_rows:
        if _same_profile(row, lite_row) or _same_profile(row, strong_row):
            continue
        return row
    return sorted_rows[0]


def _select_lite_profile(profile_rows):
    return min(
        profile_rows,
        key=lambda row: (_cost_value(row), row["overload_ratio_q1000"]),
    )


def _select_strong_profile(profile_rows):
    return min(
        profile_rows,
        key=lambda row: (row["overload_ratio_q1000"], _cost_value(row)),
    )


def _profile_by_name(profile_rows, name):
    for row in profile_rows:
        if row.get("profile") == name:
            return row
    return None


def _format_lambda_tag(lambda_value):
    text = f"{lambda_value:g}"
    return text.replace(".", "p")


def _build_role_map(lite_row, balanced_row, strong_row):
    role_map = {}
    for label, row in (
        ("Ours-Lite", lite_row),
        ("Ours-Balanced", balanced_row),
        ("Ours-Strong", strong_row),
    ):
        if row is None:
            continue
        role_map.setdefault(row.get("tag"), []).append(label)
    return role_map


def _run_lambda_sweep(
    output_dir,
    figures_dir,
    seed,
    state_rate_hz,
    zone_rate,
    profile_overrides,
    role_map,
    n_zones,
    central_rate_override,
    lambda_values=None,
):
    from arm_sim.experiments.run_hotspot import run_scheme

    lambda_values = lambda_values or DEFAULT_LAMBDA_SWEEP
    sweep_rows = []
    for lambda_value in lambda_values:
        lambda_tag = _format_lambda_tag(lambda_value)
        results_by_tag = {}
        tags_to_run = sorted(role_map.keys())
        for tag in tags_to_run:
            overrides = profile_overrides.get(tag, {})
            sweep_tag = f"{tag}_lambda{lambda_tag}"
            run_scheme(
                "S2",
                seed,
                output_dir,
                state_rate_hz=state_rate_hz,
                zone_service_rate_msgs_s=zone_rate,
                write_csv=True,
                n_zones_override=n_zones,
                central_service_rate_msgs_s_override=central_rate_override,
                dmax_ms=30.0,
                tag=sweep_tag,
                migrate_penalty_lambda=lambda_value,
                **overrides,
            )
            path = os.path.join(output_dir, f"window_kpis_{sweep_tag}_s2.csv")
            rows = _load_window_rows(path, extra_required={"mean_ms", "migrated_weight_total"})
            hotspot_p95 = _hotspot_mean_metric(rows, 30.0, 80.0, "p95_ms", "p95_ms")
            mean_ms = _hotspot_mean_metric(rows, 30.0, 80.0, "mean_ms", "mean_ms")
            overload_ratio = _overload_ratio_q1000(rows)
            migrated_weight_total = _parse_float(rows[0].get("migrated_weight_total"))
            if migrated_weight_total is None:
                raise RuntimeError(f"Missing migrated_weight_total in {path}")
            results_by_tag[tag] = {
                "overload_ratio_q1000": overload_ratio,
                "hotspot_p95_mean_ms": hotspot_p95,
                "migrated_weight_total": migrated_weight_total,
                "mean_ms": mean_ms,
            }

        rows_for_lambda = []
        for tag, roles in role_map.items():
            if tag not in results_by_tag:
                continue
            metrics = results_by_tag[tag]
            for role in roles:
                rows_for_lambda.append(
                    {
                        "lambda": lambda_value,
                        "profile_name": role,
                        "overload_ratio_q1000": metrics["overload_ratio_q1000"],
                        "hotspot_p95_mean_ms": metrics["hotspot_p95_mean_ms"],
                        "migrated_weight_total": metrics["migrated_weight_total"],
                        "mean_ms": metrics["mean_ms"],
                    }
                )
        sweep_rows.extend(rows_for_lambda)

        if rows_for_lambda:
            best = min(
                rows_for_lambda,
                key=lambda row: (row["overload_ratio_q1000"], row["migrated_weight_total"]),
            )
            print(
                f"lambda={lambda_value:g} "
                f"best_overload={best['profile_name']} "
                f"overload_ratio_q1000={best['overload_ratio_q1000']:.3f} "
                f"migrated_weight_total={best['migrated_weight_total']:.3f}"
            )

    sweep_path = os.path.join(figures_dir, "summary_s2_lambda_sweep.csv")
    with open(sweep_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "lambda",
                "profile_name",
                "overload_ratio_q1000",
                "hotspot_p95_mean_ms",
                "migrated_weight_total",
                "mean_ms",
            ]
        )
        for row in sweep_rows:
            writer.writerow(
                [
                    _format_float(row["lambda"], 3),
                    row["profile_name"],
                    _format_float(row["overload_ratio_q1000"], 6),
                    _format_float(row["hotspot_p95_mean_ms"], 3),
                    _format_float(row["migrated_weight_total"], 6),
                    _format_float(row["mean_ms"], 3),
                ]
            )
    return sweep_path


def _plot_fig1(fig_path, values_by_label, labels=None):
    import matplotlib.pyplot as plt

    labels = labels or list(values_by_label.keys())
    p95_ms = [values_by_label[label]["hotspot_p95_mean_ms"] for label in labels]
    mean_ms = [values_by_label[label]["hotspot_mean_ms"] for label in labels]
    p95_s = [value / 1000.0 for value in p95_ms]
    mean_s = [value / 1000.0 for value in mean_ms]
    p95_draw = [0.0 if _is_nan(value) else value for value in p95_s]
    mean_draw = [0.0 if _is_nan(value) else value for value in mean_s]

    fig, ax = plt.subplots()
    x = list(range(len(labels)))
    width = 0.35
    bars = ax.bar(
        [pos - width / 2 for pos in x],
        p95_draw,
        width=width,
        label="Hotspot latency (P95)",
    )
    mean_bars = ax.bar(
        [pos + width / 2 for pos in x],
        mean_draw,
        width=width * 0.7,
        alpha=0.6,
        label="Hotspot latency (Mean)",
    )
    ax.set_ylabel("Hotspot latency (s)", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(True, axis="y", which="major", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=10, frameon=False)

    safe_vals = [value for value in p95_s + mean_s if not _is_nan(value)]
    max_val = max(safe_vals) if safe_vals else 1.0
    ax.set_ylim(0, max_val * 1.25)

    for rect, value in zip(bars, p95_s):
        label = "N/A" if _is_nan(value) else f"{value:.1f}s"
        y_pos = (value if not _is_nan(value) else max_val * 0.05) * 1.08
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            y_pos,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
        )
    for rect, value in zip(mean_bars, mean_s):
        label = "N/A" if _is_nan(value) else f"{value:.1f}s"
        y_pos = (value if not _is_nan(value) else max_val * 0.05) * 1.08
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            y_pos,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    return _save_figure(fig, fig_path)


def _plot_fig2(fig_path, values_by_label, labels=None):
    import matplotlib.pyplot as plt

    labels = labels or list(values_by_label.keys())
    thresholds = [500, 1000, 1500]
    values_by_threshold = {
        500: [values_by_label[label]["overload_ratio_q500"] for label in labels],
        1000: [values_by_label[label]["overload_ratio_q1000"] for label in labels],
        1500: [values_by_label[label]["overload_ratio_q1500"] for label in labels],
    }

    fig, ax = plt.subplots()
    x = list(range(len(labels)))
    width = 0.22
    offsets = [-width, 0.0, width]
    for offset, threshold in zip(offsets, thresholds):
        values = values_by_threshold[threshold]
        bars = ax.bar(
            [pos + offset for pos in x],
            values,
            width=width,
            label=f"Overload ratio (queue > {threshold})",
        )
        for rect, value in zip(bars, values):
            label_text = _format_ratio_label(value)
            y_pos = (value if value is not None and not _is_nan(value) else 0.0) + 0.015
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                y_pos,
                label_text,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Overload ratio (queue > threshold)", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=9, frameon=False)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    return _save_figure(fig, fig_path)


def _plot_fig3(fig_path, points, offset_log=None):
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox

    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    ax.set_xlabel("Reconfiguration cost per request", fontsize=12)
    ax.set_ylabel("Overload ratio (queue > 1000)", fontsize=12)
    ax.tick_params(labelsize=11)

    label_bbox = dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5)
    offsets = {
        "Static-edge": (10, -12),
        "Ours-Lite": (10, 8),
        "Ours-Balanced": (10, 8),
        "Ours-Strong": (-18, 10),
    }

    chosen_offsets = {}
    for point in points:
        is_star = point["label"] == "Ours-Balanced"
        marker = "*" if is_star else "o"
        size = 180 if is_star else 70
        edge = "black" if is_star else "none"
        ax.scatter([point["x"]], [point["y"]], marker=marker, s=size, edgecolors=edge, linewidths=0.8)
        label = point["label"]
        base_offset = offsets.get(point["label"], (6, 6))
        candidates = [
            base_offset,
            (base_offset[0] + 8, base_offset[1] + 8),
            (base_offset[0] + 12, base_offset[1] - 6),
            (base_offset[0] - 12, base_offset[1] + 10),
            (base_offset[0] - 16, base_offset[1] - 8),
        ]
        text_obj = None
        for candidate in candidates:
            if text_obj is not None:
                text_obj.remove()
            text_obj = ax.annotate(
                label,
                (point["x"], point["y"]),
                textcoords="offset points",
                xytext=candidate,
                fontsize=10,
                bbox=label_bbox,
                clip_on=False,
            )
            fig.canvas.draw()
            text_bbox = text_obj.get_window_extent(fig.canvas.get_renderer())
            marker_xy = ax.transData.transform((point["x"], point["y"]))
            marker_bbox = Bbox.from_bounds(marker_xy[0] - 6, marker_xy[1] - 6, 12, 12)
            if not text_bbox.overlaps(marker_bbox):
                chosen_offsets[label] = candidate
                break
        if text_obj is None:
            chosen_offsets[label] = base_offset

    xs = [point["x"] for point in points]
    max_x = max(xs)
    x_min = min(-20.0, -0.05 * max_x)
    x_max = max(max_x * 1.08, max_x + 40.0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.05, 1.05)

    if offset_log is not None:
        offset_log.update(chosen_offsets)

    return _save_figure(fig, fig_path)


def _write_summary(path, rows):
    columns = [
        "N",
        "tag",
        "scheme",
        "profile",
        "hotspot_p95_mean_ms",
        "hotspot_mean_ms",
        "hotspot_latency_n",
        "overload_ratio_q500",
        "overload_ratio_q1000",
        "overload_ratio_q1500",
        "hotspot_overload_ratio_q1000",
        "migrated_weight_total",
        "generated_requests",
        "completed_requests",
        "completion_ratio",
        "hotspot_controller_id",
        "accepted_migrations",
        "rejected_no_feasible_target",
        "rejected_budget",
        "rejected_safety",
        "fallback_attempts",
        "fallback_success",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [
                    row.get("N", ""),
                    row.get("tag", ""),
                    row.get("scheme", ""),
                    row.get("profile", ""),
                    f"{row.get('hotspot_p95_mean_ms', 0.0):.3f}"
                    if row.get("hotspot_p95_mean_ms") is not None
                    else "",
                    f"{row.get('hotspot_mean_ms', 0.0):.3f}"
                    if row.get("hotspot_mean_ms") is not None
                    else "",
                    row.get("hotspot_latency_n", ""),
                    f"{row.get('overload_ratio_q500', 0.0):.3f}"
                    if row.get("overload_ratio_q500") is not None
                    else "",
                    f"{row.get('overload_ratio_q1000', 0.0):.6f}"
                    if row.get("overload_ratio_q1000") is not None
                    else "",
                    f"{row.get('overload_ratio_q1500', 0.0):.6f}"
                    if row.get("overload_ratio_q1500") is not None
                    else "",
                    f"{row.get('hotspot_overload_ratio_q1000', 0.0):.6f}"
                    if row.get("hotspot_overload_ratio_q1000") is not None
                    else "",
                    row.get("migrated_weight_total", ""),
                    row.get("generated_requests", ""),
                    row.get("completed_requests", ""),
                    f"{row.get('completion_ratio', 0.0):.3f}"
                    if row.get("completion_ratio") is not None
                    else "",
                    row.get("hotspot_controller_id", ""),
                    row.get("accepted_migrations", ""),
                    row.get("rejected_no_feasible_target", ""),
                    row.get("rejected_budget", ""),
                    row.get("rejected_safety", ""),
                    row.get("fallback_attempts", ""),
                    row.get("fallback_success", ""),
                ]
            )


def _write_selected_profiles(path, selections):
    columns = [
        "profile_name",
        "selected_role",
        "tag",
        "scheme",
        "profile",
        "x_cost",
        "y_overload",
        "x_norm",
        "y_norm",
        "score",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for selection in selections:
            row = selection["row"]
            writer.writerow(
                [
                    selection["profile_name"],
                    selection["selected_role"],
                    row.get("tag", ""),
                    row.get("scheme", ""),
                    row.get("profile", ""),
                    _format_float(row.get("migrated_weight_total"), 6),
                    _format_float(row.get("overload_ratio_q1000"), 6),
                    _format_float(row.get("norm_x"), 6),
                    _format_float(row.get("norm_y"), 6),
                    _format_float(row.get("score"), 6),
                ]
            )


def _window_has_thresholds(path):
    if not os.path.isfile(path):
        return False
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        required = {
            "overload_ratio_q500",
            "overload_ratio_q1000",
            "overload_ratio_q1500",
            "generated",
            "hotspot_overload_ratio_q1000",
        }
        return required.issubset(set(headers))


def _load_scalability_summary(path):
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing scalability summary: {path}")
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        scheme_key = "scheme"
        if scheme_key not in headers and "strategy" in headers:
            scheme_key = "strategy"
        p95_key = "hotspot_p95_mean_ms"
        if p95_key not in headers and "hotspot_p95_latency" in headers:
            p95_key = "hotspot_p95_latency"
        missing = [
            col
            for col in SCALABILITY_REQUIRED_COLUMNS
            if col not in headers and col not in ("scheme", "hotspot_p95_mean_ms")
        ]
        if scheme_key == "scheme" and "scheme" not in headers:
            missing.append("scheme")
        if p95_key == "hotspot_p95_mean_ms" and "hotspot_p95_mean_ms" not in headers:
            missing.append("hotspot_p95_mean_ms")
        if missing:
            raise RuntimeError(
                f"Missing required columns in {path}: {', '.join(sorted(set(missing)))}"
            )
        rows = []
        for row in reader:
            rows.append(
                {
                    "N": int(float(row["N"])),
                    "load_scale": float(row["load_scale"]),
                    "scheme": row[scheme_key],
                    "hotspot_p95_mean_ms": _parse_float(row[p95_key]),
                    "hotspot_mean_ms": _parse_float(row["hotspot_mean_ms"]),
                    "overload_ratio_q1000": _parse_float(row["overload_ratio_q1000"]),
                    "migrated_weight_total": _parse_float(row["migrated_weight_total"]),
                    "seed": int(float(row["seed"])),
                }
            )
    return rows


def _mean_std(values):
    values = [value for value in values if value is not None and not _is_nan(value)]
    if not values:
        raise RuntimeError("No values for mean/std.")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def _aggregate_scalability(rows, key):
    data = {}
    for row in rows:
        scheme = row["scheme"]
        load_scale = row["load_scale"]
        n_val = row["N"]
        value = row.get(key)
        if value is None or _is_nan(value):
            continue
        data.setdefault(scheme, {}).setdefault(load_scale, {}).setdefault(n_val, []).append(value)
    aggregated = {}
    for scheme, by_load in data.items():
        aggregated[scheme] = {}
        for load_scale, by_n in by_load.items():
            aggregated[scheme][load_scale] = {}
            for n_val, values in by_n.items():
                aggregated[scheme][load_scale][n_val] = _mean_std(values)
    return aggregated


def _aggregate_scalability_simple(rows, key):
    data = {}
    for row in rows:
        scheme = row["scheme"]
        n_val = row["N"]
        value = row.get(key)
        if value is None or _is_nan(value):
            continue
        data.setdefault(scheme, {}).setdefault(n_val, []).append(value)
    aggregated = {}
    for scheme, by_n in data.items():
        aggregated[scheme] = {}
        for n_val, values in by_n.items():
            aggregated[scheme][n_val] = _mean_std(values)
    return aggregated


def _plot_fig4_scaled_load(fig_path, rows, load_map=None):
    import matplotlib.pyplot as plt

    schemes = ["Static-edge", "Ours-Lite", "Ours-Balanced", "Ours-Strong"]
    markers = ["o", "s", "D", "^"]
    n_values = sorted({row["N"] for row in rows})

    p95_data = _aggregate_scalability_simple(rows, "hotspot_p95_mean_ms")
    overload_data = _aggregate_scalability_simple(rows, "overload_ratio_q1000")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5), constrained_layout=True)
    if load_map:
        _ = [f"{n}→{load_map[n][0]:.3f}" for n in sorted(load_map.keys())]
    ax_latency, ax_overload = axes

    for scheme, marker in zip(schemes, markers):
        if scheme not in p95_data or scheme not in overload_data:
            continue
        y_means = []
        y_errs = []
        for n_val in n_values:
            mean, std = p95_data[scheme].get(n_val, (None, None))
            if mean is None:
                y_means.append(None)
                y_errs.append(0.0)
                continue
            y_means.append(mean / 1000.0)
            y_errs.append(std / 1000.0)
        if any(err > 0.0 for err in y_errs):
            ax_latency.errorbar(
                n_values,
                y_means,
                yerr=y_errs,
                marker=marker,
                linewidth=1.5,
                capsize=3,
                label=scheme,
            )
        else:
            ax_latency.plot(
                n_values, y_means, marker=marker, linewidth=1.5, label=scheme
            )

        y_means = []
        y_errs = []
        for n_val in n_values:
            mean, std = overload_data[scheme].get(n_val, (None, None))
            if mean is None:
                y_means.append(None)
                y_errs.append(0.0)
                continue
            y_means.append(mean)
            y_errs.append(std)
        if any(err > 0.0 for err in y_errs):
            ax_overload.errorbar(
                n_values,
                y_means,
                yerr=y_errs,
                marker=marker,
                linewidth=1.5,
                capsize=3,
                label=scheme,
            )
        else:
            ax_overload.plot(
                n_values, y_means, marker=marker, linewidth=1.5, label=scheme
            )

    ax_latency.set_xlabel("Number of zones (N)", fontsize=12)
    ax_latency.set_ylabel("Hotspot P95 latency (s)", fontsize=12)
    ax_latency.set_xticks(n_values)
    ax_latency.tick_params(labelsize=11)
    ax_latency.grid(True, axis="y", alpha=0.3)

    ax_overload.set_xlabel("Number of zones (N)", fontsize=12)
    ax_overload.set_ylabel("Overload ratio (queue > 1000)", fontsize=12)
    ax_overload.set_xticks(n_values)
    ax_overload.tick_params(labelsize=11)
    ax_overload.grid(True, axis="y", alpha=0.3)
    ax_overload.legend(loc="lower right", frameon=False, fontsize=9)

    fig.text(
        0.5,
        0.01,
        "Fig4: per-zone constant load scaling (q>1000); base rates from N=16.",
        ha="center",
        fontsize=9,
    )

    return _save_figure(fig, fig_path)


def _plot_fig4_scalability(fig_path, rows):
    import matplotlib.pyplot as plt

    schemes = ["Static-edge", "Ours-Lite", "Ours-Balanced", "Ours-Strong"]
    markers = ["o", "s", "D", "^"]
    n_values = sorted({row["N"] for row in rows})
    load_scales = sorted({row["load_scale"] for row in rows})
    line_styles = {1.0: "-", 2.0: "--", 3.0: ":", 4.0: "-."}

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_map = {
        scheme: color_cycle[index % len(color_cycle)] if color_cycle else None
        for index, scheme in enumerate(schemes)
    }

    p95_data = _aggregate_scalability(rows, "hotspot_p95_mean_ms")
    overload_data = _aggregate_scalability(rows, "overload_ratio_q1000")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    ax_latency, ax_overload = axes

    for scheme, marker in zip(schemes, markers):
        if scheme not in p95_data or scheme not in overload_data:
            continue
        for load_scale in load_scales:
            style = line_styles.get(load_scale, "-")
            color = color_map.get(scheme)
            y_means = []
            y_errs = []
            for n_val in n_values:
                mean, std = p95_data[scheme].get(load_scale, {}).get(n_val, (None, None))
                if mean is None:
                    y_means.append(None)
                    y_errs.append(0.0)
                    continue
                y_means.append(mean / 1000.0)
                y_errs.append(std / 1000.0)
            if any(err > 0.0 for err in y_errs):
                ax_latency.errorbar(
                    n_values,
                    y_means,
                    yerr=y_errs,
                    marker=marker,
                    linewidth=1.4,
                    capsize=3,
                    linestyle=style,
                    color=color,
                )
            else:
                ax_latency.plot(
                    n_values,
                    y_means,
                    marker=marker,
                    linewidth=1.4,
                    linestyle=style,
                    color=color,
                )

            y_means = []
            y_errs = []
            for n_val in n_values:
                mean, std = overload_data[scheme].get(load_scale, {}).get(n_val, (None, None))
                if mean is None:
                    y_means.append(None)
                    y_errs.append(0.0)
                    continue
                y_means.append(mean)
                y_errs.append(std)
            if any(err > 0.0 for err in y_errs):
                ax_overload.errorbar(
                    n_values,
                    y_means,
                    yerr=y_errs,
                    marker=marker,
                    linewidth=1.4,
                    capsize=3,
                    linestyle=style,
                    color=color,
                )
            else:
                ax_overload.plot(
                    n_values,
                    y_means,
                    marker=marker,
                    linewidth=1.4,
                    linestyle=style,
                    color=color,
                )

    ax_latency.set_xlabel("Number of zones (N)", fontsize=12)
    ax_latency.set_ylabel("Hotspot P95 latency (s)", fontsize=12)
    ax_latency.set_xticks(n_values)
    ax_latency.tick_params(labelsize=11)
    ax_latency.grid(True, axis="y", alpha=0.3)

    ax_overload.set_xlabel("Number of zones (N)", fontsize=12)
    ax_overload.set_ylabel("Overload ratio (queue > 1000)", fontsize=12)
    ax_overload.set_xticks(n_values)
    ax_overload.tick_params(labelsize=11)
    ax_overload.grid(True, axis="y", alpha=0.3)
    from matplotlib.lines import Line2D
    scheme_handles = [
        Line2D(
            [0],
            [0],
            color=color_map.get(scheme),
            marker=marker,
            linestyle="-",
            label=scheme,
        )
        for scheme, marker in zip(schemes, markers)
    ]
    load_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=line_styles.get(scale, "-"),
            label=f"Load ×{int(scale)}",
        )
        for scale in load_scales
    ]
    legend_schemes = ax_overload.legend(
        handles=scheme_handles, loc="lower right", frameon=False, fontsize=9
    )
    ax_overload.add_artist(legend_schemes)
    ax_overload.legend(
        handles=load_handles, loc="upper left", frameon=False, fontsize=9
    )

    return _save_figure(fig, fig_path)


def _print_scalability_summary(rows):
    schemes = ["Static-edge", "Ours-Lite", "Ours-Balanced", "Ours-Strong"]
    n_values = sorted({row["N"] for row in rows})
    load_scales = sorted({row["load_scale"] for row in rows})
    p95_data = _aggregate_scalability(rows, "hotspot_p95_mean_ms")
    overload_data = _aggregate_scalability(rows, "overload_ratio_q1000")

    for load_scale in load_scales:
        for n_val in n_values:
            for scheme in schemes:
                p95_mean, _ = p95_data.get(scheme, {}).get(load_scale, {}).get(n_val, (None, None))
                over_mean, _ = overload_data.get(scheme, {}).get(load_scale, {}).get(n_val, (None, None))
                if p95_mean is None or over_mean is None:
                    continue
                print(
                    f"load_x{int(load_scale)} N={n_val} {scheme} "
                    f"hotspot_p95_mean_ms={p95_mean:.3f} "
                    f"overload_ratio_q1000={over_mean:.3f}"
                )


def _load_scaled_fig4_rows(path):
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing Fig4 scaled-load summary: {path}")
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            n_val = _parse_int(row.get("N"))
            load_mult = _parse_load_multiplier(row)
            scheme = row.get("scheme") or row.get("method") or row.get("strategy")
            p95_ms = _parse_float(row.get("hotspot_p95_mean_ms")) or _parse_float(
                row.get("hotspot_p95_latency")
            )
            overload = _parse_float(row.get("overload_ratio_q1000"))
            rows.append(
                {
                    "N": n_val,
                    "load_multiplier": load_mult,
                    "scheme": scheme,
                    "hotspot_p95_mean_ms": p95_ms,
                    "overload_ratio_q1000": overload,
                    "seed": _parse_int(row.get("seed")),
                    "timestamp": row.get("timestamp"),
                    "run_id": row.get("run_id"),
                }
            )
    return rows


def _write_fig4_scaled_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "N",
                "load_multiplier",
                "scheme",
                "hotspot_p95_s",
                "overload_ratio_q1000",
                "reconfig_cost_total_migrated_load",
            ]
        )
        for row in rows:
            p95_s = (
                row["hotspot_p95_mean_ms"] / 1000.0
                if row.get("hotspot_p95_mean_ms") is not None
                else float("nan")
            )
            writer.writerow(
                [
                    row.get("N"),
                    f"{row.get('load_multiplier', 0.0):.3f}",
                    row.get("scheme", ""),
                    f"{p95_s:.6f}",
                    f"{row.get('overload_ratio_q1000', 0.0):.6f}",
                    f"{row.get('reconfig_cost', 0.0):.6f}",
                ]
            )


def _write_fig4_calibration_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "N",
                "iter",
                "m",
                "overload_ratio",
                "overload_ratio_q1500",
                "hotspot_p95_s",
                "status",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.get("N"),
                    row.get("iter"),
                    f"{row.get('m', 0.0):.3f}",
                    f"{row.get('overload_ratio', 0.0):.6f}",
                    f"{row.get('overload_ratio_q1500', 0.0):.6f}",
                    f"{row.get('hotspot_p95_s', 0.0):.6f}",
                    row.get("status", ""),
                ]
            )


def _window_has_columns(path, required):
    if not os.path.isfile(path):
        return False
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        return required.issubset(headers)


def _calibration_tag(n_zones, multiplier, standard_load, prefix="Calib"):
    text = f"{multiplier:.3f}".replace(".", "p")
    load_text = f"{standard_load:.3f}".replace(".", "p")
    return f"{prefix}N{n_zones}_L{load_text}_m{text}"


def _run_static_edge_for_multiplier(
    output_dir,
    seed,
    state_rate_hz_base,
    standard_load,
    zone_rate,
    n_zones,
    n_robots,
    central_rate_override,
    load_multiplier,
):
    from arm_sim.experiments.run_hotspot import run_scheme

    tag = _calibration_tag(n_zones, load_multiplier, standard_load)
    path = os.path.join(output_dir, f"window_kpis_{tag}_s1.csv")
    required = {"t_start", "t_end", "overload_ratio_q1000", "overload_ratio_q1500"}
    if not _window_has_columns(path, required):
        run_scheme(
            "S1",
            seed,
            output_dir,
            state_rate_hz=state_rate_hz_base * standard_load * load_multiplier,
            zone_service_rate_msgs_s=zone_rate,
            write_csv=True,
            n_zones_override=n_zones,
            n_robots_override=n_robots,
            central_service_rate_msgs_s_override=central_rate_override,
            dmax_ms=30.0,
            tag=tag,
        )
    rows = _load_window_rows(
        path,
        extra_required={
            "overload_ratio_q1000",
            "overload_ratio_q1500",
            "mean_ms",
            "p95_ms",
        },
    )
    ratio = _overload_ratio(rows, "overload_ratio_q1000", "overload_ratio_q1000")
    ratio_q1500 = _overload_ratio(rows, "overload_ratio_q1500", "overload_ratio_q1500")
    hotspot_p95 = _hotspot_mean_p95(rows, 30.0, 80.0)
    return ratio, ratio_q1500, hotspot_p95


def calibrate_load_multiplier_for_N(
    output_dir,
    seed,
    state_rate_hz_base,
    standard_load,
    zone_rate,
    n_zones,
    n_robots,
    central_rate_override,
    target_low=0.2,
    target_high=0.6,
    threshold=1000,
    max_iters=12,
    m_low=0.5,
    m_high=2.5,
    calibration_log=None,
):
    lo = float(m_low)
    hi = float(m_high)
    evaluated = {}
    iter_count = 0

    def eval_multiplier(multiplier):
        multiplier = round(float(multiplier), 3)
        if multiplier in evaluated:
            return evaluated[multiplier]
        nonlocal iter_count
        ratio, ratio_q1500, hotspot_p95 = _run_static_edge_for_multiplier(
            output_dir,
            seed,
            state_rate_hz_base,
            standard_load,
            zone_rate,
            n_zones,
            n_robots,
            central_rate_override,
            multiplier,
        )
        if ratio < target_low:
            status = "low"
        elif ratio > target_high:
            status = "high"
        else:
            status = "ok"
        print(
            f"Calib N={n_zones} m={multiplier:.3f} overload_ratio_q{threshold}={ratio:.3f} {status}"
        )
        iter_count += 1
        if calibration_log is not None:
            calibration_log.append(
                {
                    "N": n_zones,
                    "iter": iter_count,
                    "m": multiplier,
                    "overload_ratio": ratio,
                    "overload_ratio_q1500": ratio_q1500,
                    "hotspot_p95_s": hotspot_p95 / 1000.0 if hotspot_p95 is not None else float("nan"),
                    "status": status,
                }
            )
        evaluated[multiplier] = ratio
        return ratio

    best_multiplier = lo
    best_ratio = eval_multiplier(lo)
    best_dist = _distance_to_interval(best_ratio, target_low, target_high)

    ratio_lo = best_ratio
    ratio_hi = eval_multiplier(hi)
    expand_iters = 0
    while ratio_lo > target_high and lo > 0.02 and expand_iters < 8:
        hi = lo
        lo = max(lo / 2.0, 0.01)
        ratio_lo = eval_multiplier(lo)
        expand_iters += 1
    while ratio_hi < target_low and hi < 10.0 and expand_iters < 12:
        lo = hi
        hi = hi * 2.0
        ratio_hi = eval_multiplier(hi)
        expand_iters += 1

    for _ in range(max_iters):
        mid = (lo + hi) / 2.0
        ratio_mid = eval_multiplier(mid)
        if _is_within_interval(ratio_mid, target_low, target_high):
            return mid, ratio_mid
        if ratio_mid < target_low:
            lo = mid
        else:
            hi = mid
        dist = _distance_to_interval(ratio_mid, target_low, target_high)
        if dist < best_dist:
            best_dist = dist
            best_multiplier = mid
            best_ratio = ratio_mid

    print(
        f"Warning: could not hit target for N={n_zones}; "
        f"using m={best_multiplier:.3f} ratio={best_ratio:.3f}"
    )
    return best_multiplier, best_ratio


def _s2_profile_overrides(tag_suffix=""):
    suffix = tag_suffix or ""
    return {
        f"S2_conservative{suffix}": {
            "q_high_override": 700,
            "q_low_override": 500,
            "cooldown_s_override": 5.0,
            "policy_period_s_override": 2.5,
            "move_k_override": 3,
            "candidate_sample_m_override": 4,
            "p2c_k_override": 2,
            "beta_capacity_override": 0.95,
            "budget_gamma_override": 0.25,
            "fixed_k": True,
            "move_k_fixed": 3,
        },
        f"S2_neutral{suffix}": {
            "q_high_override": 500,
            "q_low_override": 360,
            "cooldown_s_override": 3.0,
            "policy_period_s_override": 3.0,
            "move_k_override": 5,
            "candidate_sample_m_override": 6,
            "p2c_k_override": 2,
            "beta_capacity_override": 0.92,
            "budget_gamma_override": 0.48,
            "fixed_k": True,
            "move_k_fixed": 5,
        },
        f"S2_aggressive{suffix}": {
            "q_high_override": 420,
            "q_low_override": 300,
            "cooldown_s_override": 2.0,
            "policy_period_s_override": 2.5,
            "move_k_override": 6,
            "candidate_sample_m_override": 8,
            "p2c_k_override": 3,
            "beta_capacity_override": 0.88,
            "budget_gamma_override": 0.50,
            "fixed_k": True,
            "move_k_fixed": 6,
        },
    }


def _run_main_if_needed(
    output_dir,
    seed,
    state_rate_hz,
    zone_rate,
    n_zones,
    n_robots,
    central_rate_override,
    main_tag,
    schemes,
):
    missing = []
    for scheme in schemes:
        path = os.path.join(output_dir, f"window_kpis_{main_tag}_{scheme.lower()}.csv")
        if not _window_has_thresholds(path):
            missing.append(scheme)
    if not missing:
        return

    from arm_sim.experiments.run_hotspot import run_scheme

    for scheme in schemes:
        run_scheme(
            scheme,
            seed,
            output_dir,
            state_rate_hz=state_rate_hz,
            zone_service_rate_msgs_s=zone_rate,
            write_csv=True,
            n_zones_override=n_zones,
            n_robots_override=n_robots,
            central_service_rate_msgs_s_override=central_rate_override,
            dmax_ms=30.0,
            tag=main_tag,
        )


def _run_profiles_if_needed(
    output_dir,
    seed,
    state_rate_hz,
    zone_rate,
    n_zones,
    n_robots,
    central_rate_override,
    tag_suffix,
):
    from arm_sim.experiments.run_hotspot import run_scheme

    profiles = _s2_profile_overrides(tag_suffix)

    for tag, overrides in profiles.items():
        path = os.path.join(output_dir, f"window_kpis_{tag}_s2.csv")
        if _window_has_thresholds(path):
            continue
        run_scheme(
            "S2",
            seed,
            output_dir,
            state_rate_hz=state_rate_hz,
            zone_service_rate_msgs_s=zone_rate,
            write_csv=True,
            n_zones_override=n_zones,
            n_robots_override=n_robots,
            central_service_rate_msgs_s_override=central_rate_override,
            dmax_ms=30.0,
            tag=tag,
            **overrides,
        )

    return profiles


def _summarize_from_windows(output_dir, tags, n_zones, hotspot_zone_id):
    summary_rows = []
    for tag, scheme, profile in tags:
        path = os.path.join(output_dir, f"window_kpis_{tag}_{scheme.lower()}.csv")
        rows = _load_window_rows(
            path,
            extra_required={
                "mean_ms",
                "overload_ratio_q500",
                "overload_ratio_q1500",
                "generated",
                "hotspot_overload_ratio_q1000",
            },
        )
        hotspot_p95 = _hotspot_mean_p95(rows, 30.0, 80.0)
        hotspot_mean = _hotspot_mean_mean(rows, 30.0, 80.0)
        ratio_q500 = _overload_ratio(rows, "overload_ratio_q500", "overload_ratio_q500")
        ratio_q1000 = _overload_ratio_q1000(rows)
        ratio_q1500 = _overload_ratio(rows, "overload_ratio_q1500", "overload_ratio_q1500")
        hotspot_overload_ratio_q1000 = _hotspot_mean_metric(
            rows,
            30.0,
            80.0,
            "hotspot_overload_ratio_q1000",
            "hotspot_overload_ratio_q1000",
        )
        hotspot_latency_n = sum(
            int(float(row.get("completed", 0)))
            for row in rows
            if _parse_float(row.get("t_start")) is not None
            and _parse_float(row.get("t_end")) is not None
            and _parse_float(row.get("t_start")) >= 30.0
            and _parse_float(row.get("t_end")) <= 80.0
        )
        generated_requests = sum(int(float(row.get("generated", 0))) for row in rows)
        completed_requests = sum(int(float(row.get("completed", 0))) for row in rows)
        completion_ratio = (
            completed_requests / generated_requests if generated_requests > 0 else float("nan")
        )
        first = rows[0] if rows else {}
        summary_rows.append(
            {
                "N": n_zones,
                "tag": tag,
                "scheme": scheme,
                "profile": profile,
                "hotspot_p95_mean_ms": hotspot_p95,
                "hotspot_mean_ms": hotspot_mean,
                "hotspot_latency_n": hotspot_latency_n,
                "overload_ratio_q500": ratio_q500,
                "overload_ratio_q1000": ratio_q1000,
                "overload_ratio_q1500": ratio_q1500,
                "hotspot_overload_ratio_q1000": hotspot_overload_ratio_q1000,
                "migrated_weight_total": first.get("migrated_weight_total", ""),
                "generated_requests": generated_requests,
                "completed_requests": completed_requests,
                "completion_ratio": completion_ratio,
                "hotspot_controller_id": hotspot_zone_id,
                "accepted_migrations": first.get("policy_reassign_ops", ""),
                "rejected_no_feasible_target": first.get("rejected_no_feasible_target", ""),
                "rejected_budget": first.get("rejected_budget", ""),
                "rejected_safety": first.get("rejected_safety", ""),
                "fallback_attempts": first.get("fallback_attempts", ""),
                "fallback_success": first.get("fallback_success", ""),
            }
        )
    return summary_rows


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate paper figures and summaries.")
    parser.add_argument(
        "--lambda-sweep",
        action="store_true",
        help="Run migration penalty lambda sweep for S2 profiles.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Plot figures from existing summary CSVs without running simulations.",
    )
    return parser.parse_args()


def _select_profile_row(rows, scheme, profile, n_val, path):
    matches = []
    for row in rows:
        if row.get("scheme") != scheme:
            continue
        if row.get("profile") != profile:
            continue
        if row.get("N") is not None and int(row.get("N")) != int(n_val):
            continue
        matches.append(row)
    if not matches:
        raise RuntimeError(
            f"Missing summary row for scheme={scheme} profile={profile} N={n_val} in {path}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple summary rows for scheme={scheme} profile={profile} N={n_val} in {path}"
        )
    return matches[0]


def _resolve_mnt_dir(base_dir):
    env_dir = os.getenv("MNT_DATA_DIR")
    if env_dir:
        return env_dir
    if os.path.basename(os.path.abspath(base_dir)) == "outputs":
        base_dir = os.path.dirname(os.path.abspath(base_dir))
    if os.path.isdir("/mnt/data"):
        return "/mnt/data"
    return os.path.join(base_dir, "mnt", "data")


def _resolve_timestamp_dir(base_dir):
    env_dir = os.getenv("FIGURES_TIMESTAMP_DIR")
    if env_dir:
        return env_dir
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if os.path.basename(os.path.abspath(base_dir)) == "outputs":
        return os.path.join(base_dir, f"figures_{stamp}")
    return os.path.join(base_dir, "outputs", f"figures_{stamp}")


def _copy_figures(outputs, base_dir, extra_files=None):
    pngs = [path for path in outputs if path.lower().endswith(".png")]
    if not pngs:
        return None, None
    timestamp_dir = _resolve_timestamp_dir(base_dir)
    os.makedirs(timestamp_dir, exist_ok=True)
    mnt_dir = _resolve_mnt_dir(base_dir)
    os.makedirs(mnt_dir, exist_ok=True)

    mapping = {}
    for path in pngs:
        name = os.path.basename(path)
        if name.startswith("fig1_"):
            mapping["fig1"] = path
        elif name.startswith("fig2_"):
            mapping["fig2"] = path
        elif name.startswith("fig3_"):
            mapping["fig3"] = path
        elif name.startswith("fig4_"):
            mapping["fig4"] = path

    fixed_names = {
        "fig1": "fig1_hotspot_p95_main_constrained.png",
        "fig2": "fig2_overload_ratio_main_constrained.png",
        "fig3": "fig3_tradeoff_scatter_constrained.png",
        "fig4": "fig4_scaledload_scalability.png",
    }

    for key, src in mapping.items():
        dst = os.path.join(timestamp_dir, os.path.basename(src))
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
        fixed = fixed_names.get(key)
        if fixed:
            shutil.copy2(src, os.path.join(mnt_dir, fixed))

    for path in extra_files or []:
        if not path or not os.path.isfile(path):
            continue
        dst = os.path.join(timestamp_dir, os.path.basename(path))
        if os.path.abspath(path) != os.path.abspath(dst):
            shutil.copy2(path, dst)

    return timestamp_dir, mnt_dir


def _plot_from_summaries(output_dir, figures_dir_main, figures_dir_fig4):
    summary_main_path = os.path.join(figures_dir_main, "summary_main_and_ablations.csv")
    summary_profiles_path = os.path.join(figures_dir_main, "summary_s2_profiles.csv")
    main_rows = _load_summary_file(summary_main_path)
    profile_rows = _load_summary_file(summary_profiles_path)

    if all(row.get("N") is None for row in main_rows):
        raise RuntimeError(
            f"Summary {summary_main_path} missing N column; cannot enforce N=16."
        )
    if all(row.get("N") is None for row in profile_rows):
        raise RuntimeError(
            f"Summary {summary_profiles_path} missing N column; cannot enforce N=16."
        )

    n_zones = 16
    static_row = _select_profile_row(main_rows, "S1", "baseline", n_zones, summary_main_path)
    lite_row = _select_profile_row(profile_rows, "S2", "conservative", n_zones, summary_profiles_path)
    balanced_row = _select_profile_row(profile_rows, "S2", "neutral", n_zones, summary_profiles_path)
    strong_row = _select_profile_row(profile_rows, "S2", "aggressive", n_zones, summary_profiles_path)

    labels_order = ["Static-edge", "Ours-Lite", "Ours-Balanced", "Ours-Strong"]
    values_by_label = {
        "Static-edge": static_row,
        "Ours-Lite": lite_row,
        "Ours-Balanced": balanced_row,
        "Ours-Strong": strong_row,
    }

    fig1_path = os.path.join(figures_dir_main, "fig1_hotspot_p95_main_constrained.png")
    fig2_path = os.path.join(figures_dir_main, "fig2_overload_ratio_main_constrained.png")
    fig3_path = os.path.join(figures_dir_main, "fig3_tradeoff_scatter_constrained.png")

    outputs = []
    outputs.extend(_plot_fig1(fig1_path, values_by_label, labels=labels_order))
    outputs.extend(_plot_fig2(fig2_path, values_by_label, labels=labels_order))

    points = [
        {
            "label": "Static-edge",
            "x": _cost_value(static_row),
            "y": static_row.get("overload_ratio_q1000"),
        },
        {
            "label": "Ours-Lite",
            "x": _cost_value(lite_row),
            "y": lite_row.get("overload_ratio_q1000"),
        },
        {
            "label": "Ours-Balanced",
            "x": _cost_value(balanced_row),
            "y": balanced_row.get("overload_ratio_q1000"),
        },
        {
            "label": "Ours-Strong",
            "x": _cost_value(strong_row),
            "y": strong_row.get("overload_ratio_q1000"),
        },
    ]
    fig3_offsets = {}
    outputs.extend(_plot_fig3(fig3_path, points, offset_log=fig3_offsets))

    fig4_summary_path = os.path.join(figures_dir_fig4, "summary_fig4_scaledload.csv")
    fig4_rows_raw = _load_summary_file(fig4_summary_path)
    fig4_rows = []
    for row in fig4_rows_raw:
        if row.get("scheme") == "S1" and row.get("profile") == "baseline":
            label = "Static-edge"
        elif row.get("scheme") == "S2" and row.get("profile") == "conservative":
            label = "Ours-Lite"
        elif row.get("scheme") == "S2" and row.get("profile") == "neutral":
            label = "Ours-Balanced"
        elif row.get("scheme") == "S2" and row.get("profile") == "aggressive":
            label = "Ours-Strong"
        else:
            continue
        fig4_rows.append(
            {
                "N": row.get("N"),
                "scheme": label,
                "hotspot_p95_mean_ms": row.get("hotspot_p95_mean_ms"),
                "overload_ratio_q1000": row.get("overload_ratio_q1000"),
            }
        )
    expected = {"Static-edge", "Ours-Lite", "Ours-Balanced", "Ours-Strong"}
    present = {row["scheme"] for row in fig4_rows}
    if expected - present:
        raise RuntimeError(f"Fig4 missing schemes: {sorted(expected - present)}")
    fig4_path = os.path.join(figures_dir_fig4, "fig4_scaledload_scalability.png")
    outputs.extend(_plot_fig4_scaled_load(fig4_path, fig4_rows))
    if fig3_offsets:
        print("Fig3 label offsets:")
        for label, offset in fig3_offsets.items():
            print(f"{label}: {offset}")
    extra_files = [
        summary_main_path,
        summary_profiles_path,
        fig4_summary_path,
        os.path.join(figures_dir_main, "summary_s2_selected.csv"),
        os.path.join(figures_dir_main, "run_config.json"),
        os.path.join(figures_dir_fig4, "run_config.json"),
        os.path.join(figures_dir_fig4, "ablation_n16_strong.csv"),
    ]
    timestamp_dir, mnt_dir = _copy_figures(outputs, output_dir, extra_files=extra_files)
    if timestamp_dir:
        print(f"Copied PNGs to {timestamp_dir}")
    if mnt_dir:
        print(f"Copied PNGs to {mnt_dir}")
    return outputs


def main():
    args = _parse_args()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = os.path.join(base_dir, "outputs")
    figures_dir = os.path.join(output_dir, "figure_z16_noc")
    os.makedirs(figures_dir, exist_ok=True)
    figures_dir_fig4 = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir_fig4, exist_ok=True)

    if args.plot_only:
        outputs = _plot_from_summaries(output_dir, figures_dir, figures_dir_fig4)
        print("Generated outputs:")
        for path in outputs:
            print(path)
        return

    seed = 123
    state_rate_hz_base = 10
    standard_load = 2.0  # standard_load corresponds to the previous load×2 baseline.
    zone_rate = 200
    n_zones = 16
    base_n_zones = 8
    base_n_robots = 100
    robots_per_zone = base_n_robots / float(base_n_zones)
    n_robots = int(round(robots_per_zone * n_zones))
    central_rate_override = n_zones * zone_rate
    profile_version = "p28"
    tag_suffix_base = f"_z{n_zones}_std{standard_load:.1f}_{profile_version}"
    hotspot_zone_id = 0

    print("Using N=16 for Fig1–Fig3 main figures.")
    calib_multiplier, calib_ratio = calibrate_load_multiplier_for_N(
        output_dir,
        seed,
        state_rate_hz_base,
        standard_load,
        zone_rate,
        n_zones,
        n_robots,
        central_rate_override,
        target_low=0.55,
        target_high=0.65,
        m_low=0.5,
        m_high=2.5,
    )
    tag_suffix = f"{tag_suffix_base}_m{calib_multiplier:.3f}".replace(".", "p")
    main_tag = f"Main_constrained{tag_suffix}"
    ratio_1000, ratio_1500, _ = _run_static_edge_for_multiplier(
        output_dir,
        seed,
        state_rate_hz_base,
        standard_load,
        zone_rate,
        n_zones,
        n_robots,
        central_rate_override,
        calib_multiplier,
    )
    print(
        f"Standard load multiplier for N=16: {calib_multiplier:.3f} "
        f"(Static-edge overload_ratio_q1000={calib_ratio:.3f})"
    )
    state_rate_hz = state_rate_hz_base * standard_load * calib_multiplier
    profile_overrides_main = _s2_profile_overrides(tag_suffix)
    with open(os.path.join(figures_dir, "main_config.txt"), "w") as handle:
        handle.write(f"N={n_zones}\n")
        handle.write(f"standard_load={standard_load:.3f}\n")
        handle.write(f"load_multiplier={calib_multiplier:.3f}\n")
        for label, key in (
            ("lite", f"S2_conservative{tag_suffix}"),
            ("balanced", f"S2_neutral{tag_suffix}"),
            ("strong", f"S2_aggressive{tag_suffix}"),
        ):
            overrides = profile_overrides_main.get(key, {})
            handle.write(f"{label}_q_high={overrides.get('q_high_override')}\n")
            handle.write(f"{label}_q_low={overrides.get('q_low_override')}\n")
            handle.write(f"{label}_cooldown_s={overrides.get('cooldown_s_override')}\n")
            handle.write(
                f"{label}_policy_period_s={overrides.get('policy_period_s_override')}\n"
            )
            handle.write(f"{label}_move_k={overrides.get('move_k_override')}\n")
            handle.write(
                f"{label}_candidate_sample_m={overrides.get('candidate_sample_m_override')}\n"
            )
            handle.write(f"{label}_p2c_k={overrides.get('p2c_k_override')}\n")
            handle.write(f"{label}_beta_capacity={overrides.get('beta_capacity_override')}\n")
            handle.write(f"{label}_budget_gamma={overrides.get('budget_gamma_override')}\n")
            handle.write(f"{label}_fixed_k={overrides.get('fixed_k')}\n")
            handle.write(f"{label}_move_k_fixed={overrides.get('move_k_fixed')}\n")

    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    fig13_calib_path = os.path.join(results_dir, "fig1_3_calibration.csv")
    with open(fig13_calib_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "N",
                "chosen_multiplier",
                "achieved_overload_1000",
                "achieved_overload_1500",
                "seed",
                "timestamp",
            ]
        )
        writer.writerow(
            [
                n_zones,
                f"{calib_multiplier:.3f}",
                f"{ratio_1000:.6f}",
                f"{ratio_1500:.6f}",
                seed,
                int(time.time()),
            ]
        )

    _run_main_if_needed(
        output_dir,
        seed,
        state_rate_hz,
        zone_rate,
        n_zones,
        n_robots,
        central_rate_override,
        main_tag,
        schemes=["S1", "S2"],
    )
    _run_profiles_if_needed(
        output_dir,
        seed,
        state_rate_hz,
        zone_rate,
        n_zones,
        n_robots,
        central_rate_override,
        tag_suffix,
    )

    summary_tags = [
        (main_tag, "S1", "baseline"),
        (main_tag, "S2", "neutral"),
        (f"S2_conservative{tag_suffix}", "S2", "conservative"),
        (f"S2_neutral{tag_suffix}", "S2", "neutral"),
        (f"S2_aggressive{tag_suffix}", "S2", "aggressive"),
    ]
    summary_rows = _summarize_from_windows(output_dir, summary_tags, n_zones, hotspot_zone_id)
    summary_main_path = os.path.join(figures_dir, "summary_main_and_ablations.csv")
    summary_profiles_path = os.path.join(figures_dir, "summary_s2_profiles.csv")
    _write_summary(summary_main_path, summary_rows)
    _write_summary(
        summary_profiles_path,
        [row for row in summary_rows if row["tag"].startswith("S2_")],
    )

    main_rows_all = _load_summary_file(summary_main_path)
    profile_rows_all = _load_summary_file(summary_profiles_path)
    if all(row.get("N") is None for row in main_rows_all):
        raise RuntimeError(
            f"Summary {summary_main_path} missing N column; cannot enforce N=16."
        )
    if all(row.get("N") is None for row in profile_rows_all):
        raise RuntimeError(
            f"Summary {summary_profiles_path} missing N column; cannot enforce N=16."
        )

    main_baseline = {}
    for scheme in ["S1"]:
        row = _find_row(
            main_rows_all, main_tag, scheme, expected_n=n_zones, path=summary_main_path
        )
        main_baseline[scheme] = {
            "hotspot_p95_mean_ms": _require_value(
                row, "hotspot_p95_mean_ms", f"{main_tag} {scheme}", summary_main_path
            ),
            "hotspot_mean_ms": _require_value(
                row, "hotspot_mean_ms", f"{main_tag} {scheme}", summary_main_path
            ),
            "overload_ratio_q500": _require_value(
                row, "overload_ratio_q500", f"{main_tag} {scheme}", summary_main_path
            ),
            "overload_ratio_q1000": _require_value(
                row, "overload_ratio_q1000", f"{main_tag} {scheme}", summary_main_path
            ),
            "overload_ratio_q1500": _require_value(
                row, "overload_ratio_q1500", f"{main_tag} {scheme}", summary_main_path
            ),
            "migrated_weight_total": _require_value(
                row, "migrated_weight_total", f"{main_tag} {scheme}", summary_main_path
            ),
        }

    profile_rows = []
    for row in profile_rows_all:
        if row.get("scheme") != "S2":
            continue
        if row.get("N") is not None and int(row.get("N")) != int(n_zones):
            raise RuntimeError(
                f"Profile row {row.get('tag')} has N={row.get('N')}, expected {n_zones}"
            )
        profile_rows.append(
            {
                "tag": row.get("tag"),
                "scheme": row.get("scheme"),
                "profile": row.get("profile"),
                "hotspot_p95_mean_ms": _require_value(
                    row, "hotspot_p95_mean_ms", row.get("tag"), summary_profiles_path
                ),
                "hotspot_mean_ms": _require_value(
                    row, "hotspot_mean_ms", row.get("tag"), summary_profiles_path
                ),
                "overload_ratio_q500": _require_value(
                    row, "overload_ratio_q500", row.get("tag"), summary_profiles_path
                ),
                "migrated_weight_total": _require_value(
                    row, "migrated_weight_total", row.get("tag"), summary_profiles_path
                ),
                "overload_ratio_q1000": _require_value(
                    row, "overload_ratio_q1000", row.get("tag"), summary_profiles_path
                ),
                "overload_ratio_q1500": _require_value(
                    row, "overload_ratio_q1500", row.get("tag"), summary_profiles_path
                ),
            }
        )

    if len(profile_rows) < 3:
        raise RuntimeError("Expected at least 3 S2 profile rows for knee selection.")

    _normalize_profiles(profile_rows)
    lite_row = _profile_by_name(profile_rows, "conservative")
    balanced_row = _profile_by_name(profile_rows, "neutral")
    strong_row = _profile_by_name(profile_rows, "aggressive")

    if lite_row is None or balanced_row is None or strong_row is None:
        print("Warning: missing explicit profile labels; falling back to score-based selection.")
        lite_row = lite_row or _select_lite_profile(profile_rows)
        strong_row = strong_row or _select_strong_profile(profile_rows)
        balanced_row = balanced_row or _select_balanced_profile(profile_rows, lite_row, strong_row)

    selected_profiles = [
        {"profile_name": "Ours-Lite", "selected_role": "Lite", "row": lite_row},
        {"profile_name": "Ours-Balanced", "selected_role": "Balanced", "row": balanced_row},
        {"profile_name": "Ours-Strong", "selected_role": "Strong", "row": strong_row},
    ]
    selected_profiles_path = os.path.join(figures_dir, "summary_s2_selected.csv")
    _write_selected_profiles(selected_profiles_path, selected_profiles)

    fig1_path = os.path.join(figures_dir, "fig1_hotspot_p95_main_constrained.png")
    fig2_path = os.path.join(figures_dir, "fig2_overload_ratio_main_constrained.png")
    fig3_path = os.path.join(figures_dir, "fig3_tradeoff_scatter_constrained.png")

    outputs = []
    labels_order = ["Static-edge", "Ours-Lite", "Ours-Balanced", "Ours-Strong"]
    values_by_label = {
        "Static-edge": main_baseline["S1"],
        "Ours-Lite": lite_row,
        "Ours-Balanced": balanced_row,
        "Ours-Strong": strong_row,
    }
    print("Verification table (N=16 calibrated):")
    for label in labels_order:
        row = values_by_label[label]
        print(
            f"{label} "
            f"hotspot_p95_ms={row.get('hotspot_p95_mean_ms')} "
            f"overload_q500={row.get('overload_ratio_q500')} "
            f"overload_q1000={row.get('overload_ratio_q1000')} "
            f"overload_q1500={row.get('overload_ratio_q1500')} "
            f"cost_per_request={_cost_value(row)}"
        )

    static_ratio = values_by_label["Static-edge"].get("overload_ratio_q1000")
    lite_ratio = values_by_label["Ours-Lite"].get("overload_ratio_q1000")
    balanced_ratio = values_by_label["Ours-Balanced"].get("overload_ratio_q1000")
    strong_ratio = values_by_label["Ours-Strong"].get("overload_ratio_q1000")
    balanced_ratio_q1500 = values_by_label["Ours-Balanced"].get("overload_ratio_q1500")
    strong_ratio_q1500 = values_by_label["Ours-Strong"].get("overload_ratio_q1500")
    static_p95 = values_by_label["Static-edge"].get("hotspot_p95_mean_ms")
    lite_p95 = values_by_label["Ours-Lite"].get("hotspot_p95_mean_ms")
    lite_cost = values_by_label["Ours-Lite"].get("migrated_weight_total")
    balanced_cost = values_by_label["Ours-Balanced"].get("migrated_weight_total")
    strong_cost = values_by_label["Ours-Strong"].get("migrated_weight_total")

    if not _is_within_interval(static_ratio, 0.55, 0.65):
        raise RuntimeError(
            f"Static-edge overload_ratio_q1000 out of range: {static_ratio}"
        )
    if lite_ratio is None or static_ratio is None or lite_ratio > static_ratio - 0.10:
        raise RuntimeError(
            f"Ours-Lite overload_ratio_q1000 must be at least 0.10 below Static-edge "
            f"(lite={lite_ratio}, static={static_ratio})"
        )
    if balanced_ratio is None or balanced_ratio > 0.30:
        raise RuntimeError(
            f"Ours-Balanced overload_ratio_q1000 out of range: {balanced_ratio}"
        )
    if balanced_ratio is None or lite_ratio is None or balanced_ratio > lite_ratio - 0.10:
        raise RuntimeError(
            f"Ours-Balanced overload_ratio_q1000 must be at least 0.10 below Ours-Lite "
            f"(balanced={balanced_ratio}, lite={lite_ratio})"
        )
    if strong_ratio is None or balanced_ratio is None or strong_ratio > balanced_ratio - 0.05:
        raise RuntimeError(
            f"Ours-Strong overload_ratio_q1000 must be at least 0.05 below Ours-Balanced "
            f"(strong={strong_ratio}, balanced={balanced_ratio})"
        )
    if (
        lite_cost is None
        or balanced_cost is None
        or strong_cost is None
        or not (strong_cost > balanced_cost >= lite_cost >= 0)
    ):
        raise RuntimeError(
            "Reconfiguration cost ordering violated: expected "
            "Strong > Balanced >= Lite >= 0."
        )
    if lite_p95 is None or static_p95 is None or lite_p95 >= static_p95:
        raise RuntimeError(
            f"Ours-Lite hotspot_p95_mean_ms must improve over Static-edge "
            f"(lite={lite_p95}, static={static_p95})"
        )
    outputs.extend(_plot_fig1(fig1_path, values_by_label, labels=labels_order))
    outputs.extend(_plot_fig2(fig2_path, values_by_label, labels=labels_order))
    points = [
        {
            "label": "Static-edge",
            "x": _cost_value(main_baseline["S1"]),
            "y": main_baseline["S1"]["overload_ratio_q1000"],
        },
        {
            "label": "Ours-Lite",
            "x": _cost_value(lite_row),
            "y": lite_row["overload_ratio_q1000"],
        },
        {
            "label": "Ours-Balanced",
            "x": _cost_value(balanced_row),
            "y": balanced_row["overload_ratio_q1000"],
        },
        {
            "label": "Ours-Strong",
            "x": _cost_value(strong_row),
            "y": strong_row["overload_ratio_q1000"],
        },
    ]
    fig3_offsets = {}
    outputs.extend(_plot_fig3(fig3_path, points, offset_log=fig3_offsets))
    if fig3_offsets:
        print("Fig3 label offsets:")
        for label, offset in fig3_offsets.items():
            print(f"{label}: {offset}")

    print(
        "Selected Ours-Balanced (profile=neutral): "
        f"{balanced_row.get('tag')} "
        f"x={_cost_value(balanced_row)} "
        f"y={balanced_row['overload_ratio_q1000']:.3f} "
        f"score={balanced_row.get('score', 0.0):.6f}"
    )
    print(
        "Selected Ours-Lite (min x): "
        f"{lite_row.get('tag')} "
        f"x={_cost_value(lite_row)} "
        f"y={lite_row['overload_ratio_q1000']:.3f}"
    )
    print(
        "Selected Ours-Strong (min y): "
        f"{strong_row.get('tag')} "
        f"x={_cost_value(strong_row)} "
        f"y={strong_row['overload_ratio_q1000']:.3f}"
    )

    if args.lambda_sweep:
        role_map = _build_role_map(lite_row, balanced_row, strong_row)
        sweep_path = _run_lambda_sweep(
            output_dir,
            figures_dir,
            seed,
            state_rate_hz,
            zone_rate,
            _s2_profile_overrides(tag_suffix),
            role_map,
            n_zones,
            central_rate_override,
        )
        print(sweep_path)

    fig4_rows = []
    fig4_schemes = ["Static-edge", "Ours-Lite", "Ours-Balanced", "Ours-Strong"]
    profile_overrides_fig4 = _s2_profile_overrides()
    fig4_loads = {}
    fig4_calibration_log = []
    fig4_targets = {
        8: (0.35, 0.45),
        16: (0.45, 0.55),
        32: (0.50, 0.60),
        64: (0.55, 0.65),
    }
    for n_val in [8, 16, 32, 64]:
        n_robots_fig4 = int(round(robots_per_zone * n_val))
        central_rate_fig4 = n_val * zone_rate
        target_low, target_high = fig4_targets[n_val]
        load_multiplier, ratio = calibrate_load_multiplier_for_N(
            output_dir,
            seed,
            state_rate_hz_base,
            standard_load,
            zone_rate,
            n_val,
            n_robots_fig4,
            central_rate_fig4,
            target_low=target_low,
            target_high=target_high,
            m_low=0.5,
            m_high=2.5,
            calibration_log=fig4_calibration_log,
        )
        fig4_loads[n_val] = (load_multiplier, ratio)

        for scheme in fig4_schemes:
            if scheme == "Static-edge":
                run_tag = _calibration_tag(n_val, load_multiplier, standard_load)
                scheme_key = "S1"
            else:
                suffix = scheme.replace("Ours-", "").lower()
                run_tag = f"Fig4N{n_val}_{suffix}_m{load_multiplier:.3f}".replace(".", "p")
                scheme_key = "S2"

            path = os.path.join(output_dir, f"window_kpis_{run_tag}_{scheme_key.lower()}.csv")
            required = {"t_start", "t_end", "overload_ratio_q1000", "mean_ms", "p95_ms"}
            if not _window_has_columns(path, required):
                from arm_sim.experiments.run_hotspot import run_scheme

                overrides = {}
                if scheme == "Ours-Lite":
                    overrides = profile_overrides_fig4["S2_conservative"]
                elif scheme == "Ours-Balanced":
                    overrides = profile_overrides_fig4["S2_neutral"]
                elif scheme == "Ours-Strong":
                    overrides = profile_overrides_fig4["S2_aggressive"]

                run_scheme(
                    scheme_key,
                    seed,
                    output_dir,
                    state_rate_hz=state_rate_hz_base * standard_load * load_multiplier,
                    zone_service_rate_msgs_s=zone_rate,
                    write_csv=True,
                    n_zones_override=n_val,
                    n_robots_override=n_robots_fig4,
                    central_service_rate_msgs_s_override=central_rate_fig4,
                    dmax_ms=30.0,
                    tag=run_tag,
                    **overrides,
                )

            rows = _load_window_rows(path, extra_required={"mean_ms", "p95_ms"})
            hotspot_p95 = _hotspot_mean_p95(rows, 30.0, 80.0)
            overload_ratio = _overload_ratio(rows, "overload_ratio_q1000", "overload_ratio_q1000")
            migrated_weight = _parse_float(rows[0].get("migrated_weight_total")) if rows else 0.0

            fig4_rows.append(
                {
                    "N": n_val,
                    "load_multiplier": load_multiplier,
                    "scheme": scheme,
                    "hotspot_p95_mean_ms": hotspot_p95,
                    "overload_ratio_q1000": overload_ratio,
                    "reconfig_cost": migrated_weight or 0.0,
                }
            )

    expected_schemes = set(fig4_schemes)
    for n_val in [8, 16, 32, 64]:
        rows_n = [row for row in fig4_rows if row["N"] == n_val]
        schemes_n = {row["scheme"] for row in rows_n}
        if schemes_n != expected_schemes:
            raise RuntimeError(
                f"Fig4 missing schemes for N={n_val}: expected {expected_schemes}, got {schemes_n}"
            )
        static_row = next(row for row in rows_n if row["scheme"] == "Static-edge")
        if abs(static_row.get("reconfig_cost", 0.0)) > 1.0e-6:
            raise RuntimeError(f"Static-edge reconfig_cost is not zero for N={n_val}.")
        if _is_within_interval(
            static_row.get("overload_ratio_q1000"), 0.2, 0.6
        ):
            if not any(
                row["scheme"].startswith("Ours-") and row.get("reconfig_cost", 0.0) > 0.0
                for row in rows_n
            ):
                raise RuntimeError(
                    f"Expected at least one Ours-* with reconfig_cost > 0 for N={n_val}."
                )

    fig4_scaled_path = os.path.join(
        output_dir, "figures", "fig4_scaledload_scalability.png"
    )
    outputs.extend(_plot_fig4_scaled_load(fig4_scaled_path, fig4_rows, load_map=fig4_loads))
    summary_fig4_path = os.path.join(
        output_dir, "figures", "summary_fig4_scaledload.csv"
    )
    _write_fig4_scaled_csv(summary_fig4_path, fig4_rows)
    fig4_calibration_path = os.path.join(
        output_dir, "results", "fig4_calibration.csv"
    )
    _write_fig4_calibration_csv(fig4_calibration_path, fig4_calibration_log)
    print(summary_fig4_path)
    print(fig4_calibration_path)
    print("Fig4 calibrated multipliers:")
    for n_val, (multiplier, ratio) in fig4_loads.items():
        print(
            f"N={n_val} load_multiplier={multiplier:.3f} "
            f"Static-edge overload_ratio_q1000={ratio:.3f}"
        )
    print("Fig4 selected rows:")
    for row in fig4_rows:
        p95_s = row["hotspot_p95_mean_ms"] / 1000.0 if row.get("hotspot_p95_mean_ms") else float("nan")
        print(
            f"N={row['N']} load_multiplier={row['load_multiplier']:.3f} "
            f"{row['scheme']} hotspot_p95_s={p95_s:.3f} "
            f"overload_ratio_q1000={row['overload_ratio_q1000']:.3f}"
        )

    lite_row = next(
        (row for row in selected_profiles if row["profile_name"] == "Ours-Lite"),
        None,
    )
    balanced_row = next(
        (row for row in selected_profiles if row["profile_name"] == "Ours-Balanced"),
        None,
    )
    lite_ratio = lite_row["row"].get("overload_ratio_q1000") if lite_row else None
    balanced_ratio = balanced_row["row"].get("overload_ratio_q1000") if balanced_row else None
    if lite_ratio is not None and balanced_ratio is not None:
        diff = lite_ratio - balanced_ratio
        print(
            f"Fig3 separation check (N=16): "
            f"Ours-Lite={lite_ratio:.3f} "
            f"Ours-Balanced={balanced_ratio:.3f} "
            f"diff={diff:.3f}"
        )

    print("Generated outputs:")
    for path in outputs:
        print(path)
    print(summary_main_path)
    print(summary_profiles_path)
    print(selected_profiles_path)
    extra_files = [
        summary_main_path,
        summary_profiles_path,
        selected_profiles_path,
        summary_fig4_path,
        os.path.join(figures_dir, "run_config.json"),
        os.path.join(figures_dir_fig4, "run_config.json"),
        os.path.join(figures_dir_fig4, "ablation_n16_strong.csv"),
    ]
    timestamp_dir, mnt_dir = _copy_figures(outputs, output_dir, extra_files=extra_files)
    if timestamp_dir:
        print(f"Copied PNGs to {timestamp_dir}")
    if mnt_dir:
        print(f"Copied PNGs to {mnt_dir}")


if __name__ == "__main__":
    main()
