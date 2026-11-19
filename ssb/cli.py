"""Command line: ssb build | estimate | run | eval | report."""

from __future__ import annotations

import argparse
import sys
import time

from .config import ConfigError, load_config


def cmd_build(args: argparse.Namespace) -> int:
    from .build import Builder, write_dataset

    cfg = load_config(args.config)
    t0 = time.time()
    samples, paras = Builder(cfg).build()
    manifest = write_dataset(cfg, samples, paras)
    print(f"wrote {cfg.dataset.path}: {manifest['samples']} samples, "
          f"{manifest['questions']} questions, {manifest['paragraphs']} distinct paragraphs "
          f"({time.time() - t0:.0f}s)")
    print(f"  by axis: {manifest['by_axis']}")
    print(f"  by task: {manifest['by_task']}")
    print(f"  prompt tokens: {manifest['prompt_tokens_total']:,} total, "
          f"max {manifest['prompt_tokens_max']:,}")
    print("  all samples passed their checks")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    from .run import estimate

    cfg = load_config(args.config)
    rows = estimate(cfg)
    total = 0.0
    print(f"{'model':<18} {'prompts':>8} {'prompt tokens':>14} {'prior tok/s':>12} {'GPU-h':>7}")
    for r, m in zip(rows, cfg.models, strict=True):
        total += r["gpu_hours"]
        print(f"{r['model']:<18} {r['prompts']:>8,} {r['prompt_tokens']:>14,} "
              f"{m.prior_tokens_per_s:>12,.0f} {r['gpu_hours']:>7.2f}")
    print(f"{'total':<18} {'':>8} {'':>14} {'':>12} {total:>7.2f}   "
          f"(budget {cfg.run.max_gpu_hours} h)")
    print("Priors are rough T4 figures; `ssb run` times real batches before the bulk of the "
          "work and switches to the fallback model or stops if the budget is exceeded.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .run import BudgetExceeded, Runner, TimeLimitReached

    cfg = load_config(args.config)
    runner = Runner(cfg, time_limit_hours=args.time_limit_hours, force=args.force,
                    probe_only=args.probe_only)
    print(f"{cfg.name}: {runner.n_questions} questions, {len(runner.samples)} samples per model")
    try:
        runner.run(only=args.model)
    except BudgetExceeded as e:
        print(f"stopped: {e}", file=sys.stderr)
        return 4
    except TimeLimitReached as e:
        print(f"paused: {e}", file=sys.stderr)
        return 3
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .report import evaluate

    cfg = load_config(args.config)
    res = evaluate(cfg)
    print(res["leaderboard"].to_string(index=False))
    for h in res["hypotheses"]:
        print(f"{h['id']} [{h['model']}] {h['verdict']}: {h['evidence']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import write_report

    cfg = load_config(args.config)
    update = args.readme if args.readme is not None else cfg.name == "main"
    write_report(cfg, update_readme=update)
    print(f"wrote {cfg.run.output_dir}/REPORT.md and figures in {cfg.figures_dir}/"
          + ("; updated README.md" if update else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ssb", description="Semantic Saturation Benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, func, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", default="configs/main.yaml")
        p.set_defaults(func=func)
        return p

    add("build", cmd_build, "generate the dataset and verify every sample")
    add("estimate", cmd_estimate, "print prompts, tokens and prior GPU-hours per model")
    p = add("run", cmd_run, "generate answers (resumable; cached)")
    p.add_argument("--model", action="append", help="only this model name (repeatable)")
    p.add_argument("--time-limit-hours", type=float, default=None,
                   help="stop cleanly after this long; rerun to resume")
    p.add_argument("--force", action="store_true", help="ignore the GPU-hour budget")
    p.add_argument("--probe-only", action="store_true",
                   help="time one batch per N for each model, print the estimate, stop")
    add("eval", cmd_eval, "score generations and compute metrics")
    p = add("report", cmd_report, "metrics, figures, REPORT.md and README results")
    p.add_argument("--readme", action=argparse.BooleanOptionalAction, default=None,
                   help="update README.md (default: only for the main config)")
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
