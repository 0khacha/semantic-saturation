"""Command line: ssb build | run | report."""

from __future__ import annotations

import argparse
import os
import sys
import time

# Plain HTTP downloads from the Hugging Face Hub: the Xet transfer backend hung
# mid-download in testing, freezing a build. Set HF_HUB_DISABLE_XET=0 to re-enable it.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from .config import ConfigError, load_config  # noqa: E402


def cmd_build(args: argparse.Namespace) -> int:
    from .build import Builder, write_dataset

    cfg = load_config(args.config)
    t0 = time.time()
    builder = Builder(cfg)
    samples, paras = builder.build()
    m = write_dataset(cfg, builder, samples, paras)
    print(f"wrote {cfg.data.path}: {m['samples']} samples from {m['bases']} base examples, "
          f"{m['paragraphs']} distinct paragraphs ({time.time() - t0:.0f}s)")
    for k, v in m["by_task_axis"].items():
        print(f"  {k:28s} {v:6d}")
    print(f"  context tokens: {m['context_tokens_total']:,} total, max {m['context_tokens_max']:,}")
    print(f"  build info: {m['build_info']}")
    print("  all samples passed their checks")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .run import Runner

    cfg = load_config(args.config)
    r = Runner(cfg)
    print(f"{cfg.name}: {len(r.samples)} samples on {r.device}")
    r.run(only=args.model, estimate_only=args.estimate)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import write_report

    cfg = load_config(args.config)
    update = args.readme if args.readme is not None else cfg.name == "main"
    res = write_report(cfg, update_readme=update)
    for h in res["hypotheses"]:
        print(f"{h['id']}: {h['verdict']}")
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

    add("build", cmd_build, "build the dataset and verify every sample")
    p = add("run", cmd_run, "score every sample with every model (resumable)")
    p.add_argument("--model", action="append", help="only this model (repeatable)")
    p.add_argument("--estimate", action="store_true",
                   help="time one batch per N for each model, print the estimate, stop")
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
