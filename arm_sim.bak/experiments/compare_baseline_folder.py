import argparse
import csv
import hashlib
import os


def _load_csv(path):
    with open(path, "r", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_float(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except ValueError:
        return None


def _sha256(path, block_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _key(row):
    return (row.get("scheme"), row.get("profile"), row.get("N"))


def _compare_csv(base_rows, new_rows, fields):
    base_map = {_key(r): r for r in base_rows}
    new_map = {_key(r): r for r in new_rows}
    diffs = []
    for key, base_row in base_map.items():
        if key not in new_map:
            diffs.append((key, "missing", None))
            continue
        new_row = new_map[key]
        for field in fields:
            base_val = _parse_float(base_row.get(field))
            new_val = _parse_float(new_row.get(field))
            if base_val is None or new_val is None:
                continue
            if base_val != new_val:
                diffs.append((key, field, new_val - base_val))
    return diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--new_dir", required=True)
    args = parser.parse_args()

    base_main = _load_csv(os.path.join(args.baseline_dir, "summary_main_and_ablations.csv"))
    new_main = _load_csv(os.path.join(args.new_dir, "summary_main_and_ablations.csv"))
    base_fig4 = _load_csv(os.path.join(args.baseline_dir, "summary_fig4_scaledload.csv"))
    new_fig4 = _load_csv(os.path.join(args.new_dir, "summary_fig4_scaledload.csv"))

    fields = [
        "hotspot_p95_mean_s",
        "hotspot_mean_s",
        "fixed_hotspot_overload_ratio_q1000",
        "overload_ratio_q1000",
        "migrated_weight_total",
    ]
    diffs = _compare_csv(base_main, new_main, fields)
    diffs += _compare_csv(base_fig4, new_fig4, fields)
    if diffs:
        print("CSV differences:")
        for item in diffs[:50]:
            print(item)
    else:
        print("CSVs match for selected fields.")

    for name in (
        "fig1_hotspot_p95_main_constrained.png",
        "fig2_overload_ratio_main_constrained.png",
        "fig3_tradeoff_scatter_constrained.png",
        "fig4_scaledload_scalability.png",
    ):
        base_path = os.path.join(args.baseline_dir, name)
        new_path = os.path.join(args.new_dir, name)
        if not os.path.isfile(base_path) or not os.path.isfile(new_path):
            continue
        base_hash = _sha256(base_path)
        new_hash = _sha256(new_path)
        print(f"{name}: baseline={base_hash} new={new_hash}")


if __name__ == "__main__":
    main()
