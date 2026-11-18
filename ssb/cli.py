"""Command line: `ssb build`."""

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ssb", description="Semantic Saturation Benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build", help="generate the dataset and verify every sample")
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
