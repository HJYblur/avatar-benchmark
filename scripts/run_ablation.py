"""
Automate the ablation evaluation pipeline over repeated runs.

For every *condition* in an ablation study you typically train N repeats (e.g. 3
checkpoints with different seeds). This script, for each checkpoint:

    1. runs ``inference.py --checkpoint <ckpt>`` (writes renders to the config's
       ``inference.output_dir``, default ``output/``), then
    2. runs ``src/evaluation/compute_metrics.py`` which logs a per-run JSON.

It then aggregates the per-run mean metrics *across the repeats* of each
condition and reports mean / std / var / min / max (sample std, ddof=1), so you
get statistically stable numbers per condition (baseline, no_fusion, ...).

Inference reuses the same ``output/`` folder each repeat, so runs happen
strictly sequentially (inference -> eval) before the next checkpoint overwrites
the renders.

Usage:
    python scripts/run_ablation.py --manifest scripts/ablation_manifest.yaml

Manifest schema (YAML):

    config: configs/nlfgs_gpu.yaml     # config passed to inference.py + eval
    target_root: processed             # optional; default = data.processed_root
    preds_root: output                 # optional; default = inference.output_dir
    eval:
      no_mask: true                    # true -> --no-mask (README default)
      use_crop: false
      test_views: null                 # null, or a list like [0, 120, 240]
    conditions:
      baseline:
        checkpoints:
          - models/checkpoints/baseline_r1.ckpt
          - models/checkpoints/baseline_r2.ckpt
          - models/checkpoints/baseline_r3.ckpt
      no_fusion:
        checkpoints:
          - models/checkpoints/no_fusion_r1.ckpt
          - ...
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
INFERENCE_SCRIPT = REPO_ROOT / "inference.py"
METRICS_SCRIPT = REPO_ROOT / "src" / "evaluation" / "compute_metrics.py"

METRIC_KEYS = ["psnr", "ssim", "lpips_alex", "lpips_vgg"]
METRIC_LABELS = {
    "psnr": "PSNR",
    "ssim": "SSIM",
    "lpips_alex": "LPIPS(Alex)",
    "lpips_vgg": "LPIPS(VGG)",
}
# Higher is better (True) vs lower is better (False) — used only for display hints.
METRIC_HIGHER_BETTER = {"psnr": True, "ssim": True, "lpips_alex": False, "lpips_vgg": False}


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _load_manifest(manifest_path: Path) -> dict:
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest {manifest_path} must be a YAML mapping.")
    if not manifest.get("conditions"):
        raise ValueError("Manifest must define at least one entry under 'conditions'.")
    return manifest


def _sample_stats(values: list[float]) -> dict:
    """mean / std / var / min / max over a list of per-run means (sample, ddof=1)."""
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"mean": None, "std": None, "var": None, "min": None, "max": None, "n": 0}
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = var ** 0.5
    else:
        var = 0.0
        std = 0.0
    return {"mean": mean, "std": std, "var": var, "min": min(vals), "max": max(vals), "n": n}


def _run(cmd: list[str], log_path: Path, dry_run: bool) -> int:
    """Run a subprocess, teeing combined stdout/stderr to ``log_path``. Returns exit code."""
    printable = " ".join(cmd)
    print(f"    $ {printable}", flush=True)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        log_file.write(f"# command: {printable}\n")
        log_file.write(f"# started: {datetime.now().isoformat(timespec='seconds')}\n\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        proc.wait()
        log_file.write(f"\n# exit_code: {proc.returncode}\n")
    return proc.returncode


def _build_inference_cmd(python_exe: str, config: str, checkpoint: str) -> list[str]:
    return [python_exe, str(INFERENCE_SCRIPT), "--config", config, "--checkpoint", checkpoint]


def _build_eval_cmd(
    python_exe: str,
    config: str,
    target_root: str,
    preds_root: str,
    eval_cfg: dict,
    checkpoint: str,
    condition: str,
    log_dir: Path,
    run_name: str,
) -> list[str]:
    cmd = [
        python_exe,
        str(METRICS_SCRIPT),
        "--config-path",
        config,
        "--target-root",
        target_root,
        "--preds-root",
        preds_root,
        "--log-dir",
        str(log_dir),
        "--run-name",
        run_name,
        "--checkpoint",
        checkpoint,
        "--tag",
        condition,
    ]
    if eval_cfg.get("no_mask", True):
        cmd.append("--no-mask")
    else:
        cmd.append("--use-mask")
    if eval_cfg.get("use_crop", False):
        cmd.append("--use-crop")
    test_views = eval_cfg.get("test_views")
    if test_views:
        cmd.append("--test-views")
        cmd.extend(str(int(v)) for v in test_views)
    return cmd


def _write_summary_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_summary_csv(path: Path, condition_stats: dict) -> None:
    header = ["condition", "num_repeats"]
    for m in METRIC_KEYS:
        label = METRIC_LABELS[m]
        header += [f"{label}_mean", f"{label}_std", f"{label}_var", f"{label}_min", f"{label}_max"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for condition, stats in condition_stats.items():
            row = [condition, stats["num_repeats_succeeded"]]
            for m in METRIC_KEYS:
                s = stats["metrics"][m]
                row += [
                    _fmt(s["mean"]),
                    _fmt(s["std"]),
                    _fmt(s["var"]),
                    _fmt(s["min"]),
                    _fmt(s["max"]),
                ]
            writer.writerow(row)


def _fmt(v, digits: int = 6) -> str:
    return "" if v is None else f"{v:.{digits}f}"


def _write_summary_md(path: Path, condition_stats: dict) -> None:
    lines = ["# Ablation evaluation summary", ""]
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    header = "| Condition | Repeats | " + " | ".join(
        f"{METRIC_LABELS[m]} ({'↑' if METRIC_HIGHER_BETTER[m] else '↓'})" for m in METRIC_KEYS
    ) + " |"
    sep = "|" + "---|" * (2 + len(METRIC_KEYS))
    lines.append(header)
    lines.append(sep)
    for condition, stats in condition_stats.items():
        cells = [condition, str(stats["num_repeats_succeeded"])]
        for m in METRIC_KEYS:
            s = stats["metrics"][m]
            if s["mean"] is None:
                cells.append("—")
            else:
                cells.append(f"{s['mean']:.4f} ± {s['std']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Values are `mean ± std` across repeats (sample std, ddof=1). ↑ higher is better, ↓ lower is better.")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Automate ablation inference + evaluation with cross-repeat stats.")
    parser.add_argument("--manifest", required=True, help="Path to ablation manifest YAML")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base directory for results (default: ablation_results/<timestamp>)",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for subprocesses")
    parser.add_argument("--skip-inference", action="store_true", help="Only run evaluation (assume renders exist)")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep going if a repeat fails")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    manifest_path = _resolve(args.manifest)
    manifest = _load_manifest(manifest_path)

    config = manifest.get("config", "configs/nlfgs_gpu.yaml")
    eval_cfg = manifest.get("eval", {}) or {}
    conditions = manifest["conditions"]

    # Resolve default target/preds roots from the config if not overridden.
    cfg_dict = {}
    try:
        cfg_dict = yaml.safe_load(_resolve(config).read_text()) or {}
    except FileNotFoundError:
        print(f"Warning: config {config} not found; relying on manifest roots.", flush=True)
    target_root = manifest.get("target_root") or cfg_dict.get("data", {}).get("processed_root", "processed")
    preds_root = manifest.get("preds_root") or cfg_dict.get("inference", {}).get("output_dir", "output")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        results_dir = _resolve(args.output_dir)
    else:
        results_dir = REPO_ROOT / "ablation_results" / stamp
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Keep a copy of the manifest used for this run.
    (results_dir / "manifest_used.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    print(f"Ablation run -> {results_dir}")
    print(f"  config={config}  target_root={target_root}  preds_root={preds_root}")
    print(f"  conditions: {list(conditions.keys())}\n")

    per_run_records = []  # flat list of every repeat's outcome
    condition_stats = {}

    for condition, spec in conditions.items():
        checkpoints = (spec or {}).get("checkpoints", []) if isinstance(spec, dict) else spec
        if not checkpoints:
            print(f"[{condition}] no checkpoints listed; skipping.", flush=True)
            continue
        print(f"[{condition}] {len(checkpoints)} repeat(s)")

        run_means = {m: [] for m in METRIC_KEYS}
        succeeded = 0

        for idx, ckpt in enumerate(checkpoints, 1):
            run_name = f"{condition}_r{idx}"
            print(f"  - repeat {idx}/{len(checkpoints)}: {ckpt}")
            record = {
                "condition": condition,
                "repeat": idx,
                "checkpoint": ckpt,
                "run_name": run_name,
                "status": "ok",
            }

            # 1) Inference
            if not args.skip_inference:
                inf_log = logs_dir / f"{run_name}_inference.log"
                rc = _run(_build_inference_cmd(args.python, config, ckpt), inf_log, args.dry_run)
                if rc != 0:
                    record["status"] = f"inference_failed(rc={rc})"
                    print(f"    ! inference failed (rc={rc}); see {inf_log}", flush=True)
                    per_run_records.append(record)
                    if args.continue_on_error:
                        continue
                    raise SystemExit(f"Inference failed for {run_name}; aborting (use --continue-on-error).")

            # 2) Evaluation (writes <logs_dir>/<run_name>.json + .log)
            eval_console_log = logs_dir / f"{run_name}_eval_console.log"
            eval_cmd = _build_eval_cmd(
                args.python, config, target_root, preds_root, eval_cfg, ckpt, condition, logs_dir, run_name
            )
            rc = _run(eval_cmd, eval_console_log, args.dry_run)
            if rc != 0:
                record["status"] = f"eval_failed(rc={rc})"
                print(f"    ! evaluation failed (rc={rc}); see {eval_console_log}", flush=True)
                per_run_records.append(record)
                if args.continue_on_error:
                    continue
                raise SystemExit(f"Evaluation failed for {run_name}; aborting (use --continue-on-error).")

            if args.dry_run:
                per_run_records.append(record)
                continue

            # 3) Read the per-run JSON produced by compute_metrics.
            run_json_path = logs_dir / f"{run_name}.json"
            try:
                run_json = json.loads(run_json_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                record["status"] = f"missing_json({exc})"
                print(f"    ! could not read {run_json_path}: {exc}", flush=True)
                per_run_records.append(record)
                if args.continue_on_error:
                    continue
                raise SystemExit(f"Missing/invalid eval JSON for {run_name}; aborting.")

            run_metrics = {}
            for m in METRIC_KEYS:
                mean_val = run_json.get("metrics", {}).get(m, {}).get("mean")
                run_metrics[m] = mean_val
                if mean_val is not None:
                    run_means[m].append(mean_val)
            record["metrics"] = run_metrics
            record["num_samples"] = run_json.get("num_samples")
            succeeded += 1
            per_run_records.append(record)
            pretty = ", ".join(
                f"{METRIC_LABELS[m]}={run_metrics[m]:.4f}" for m in METRIC_KEYS if run_metrics[m] is not None
            )
            print(f"    -> {pretty}", flush=True)

        condition_stats[condition] = {
            "num_repeats_listed": len(checkpoints),
            "num_repeats_succeeded": succeeded,
            "metrics": {m: _sample_stats(run_means[m]) for m in METRIC_KEYS},
        }
        print()

    if args.dry_run:
        print("Dry run complete (no metrics aggregated).")
        return 0

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(manifest_path),
        "config": config,
        "target_root": target_root,
        "preds_root": preds_root,
        "eval": eval_cfg,
        "conditions": condition_stats,
        "runs": per_run_records,
    }
    _write_summary_json(results_dir / "summary.json", payload)
    _write_summary_csv(results_dir / "summary.csv", condition_stats)
    _write_summary_md(results_dir / "summary.md", condition_stats)

    print("###############################################")
    print("Ablation summary (mean ± std across repeats)")
    print("###############################################")
    for condition, stats in condition_stats.items():
        print(f"[{condition}]  repeats: {stats['num_repeats_succeeded']}/{stats['num_repeats_listed']}")
        for m in METRIC_KEYS:
            s = stats["metrics"][m]
            if s["mean"] is None:
                print(f"    {METRIC_LABELS[m]:<12}: n/a")
            else:
                print(
                    f"    {METRIC_LABELS[m]:<12}: {s['mean']:.4f} ± {s['std']:.4f} "
                    f"(var={s['var']:.6f}, min={s['min']:.4f}, max={s['max']:.4f}, n={s['n']})"
                )
        print()
    print(f"Wrote: {results_dir / 'summary.json'}")
    print(f"Wrote: {results_dir / 'summary.csv'}")
    print(f"Wrote: {results_dir / 'summary.md'}")
    print(f"Per-run logs under: {logs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
