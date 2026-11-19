"""Command line: ssb build | run | report."""

from __future__ import annotations

import argparse
import sys
import time

from .config import ConfigError, load_config


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ssb", description="Semantic Saturation Benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build", help="build the dataset and verify every sample")
    p.add_argument("--config", default="configs/main.yaml")
    p.set_defaults(func=cmd_build)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
