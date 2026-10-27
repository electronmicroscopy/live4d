"""Convert an acquisition flag file into shared live4d YAML without physics defaults."""
import argparse
from pathlib import Path
import shlex
import yaml

from live4d.config import PipelineConfig
from live4d.parallax.algorithm import ParallaxAlgorithm
from live4d.parallax.settings import DEFAULTS


def convert(flags, *, out_dir="out", batch_size=512):
    algorithm = ParallaxAlgorithm()
    parser = argparse.ArgumentParser()
    algorithm.add_cli_args(parser)
    parser.add_argument("--accumulate", default="scan")
    parser.add_argument("--pass-average", type=float, default=1.0)
    args = parser.parse_args(algorithm.preprocess_argv(flags))
    problems = algorithm.check_args(args)
    if problems:
        raise ValueError("\n".join(problems))
    parallax = {key: value for key, value in vars(args).items() if key in DEFAULTS
                and value is not None and value != DEFAULTS[key]}
    parallax.update(enabled=True, out_dir=str(out_dir))
    values = dict(scan=dict(height=args.scan[0], width=args.scan[1]),
                  batch_size=batch_size, parallax=parallax)
    PipelineConfig(**values)
    return values


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m live4d.parallax.tools.migrate_flags", description=__doc__)
    parser.add_argument("--flags", type=Path, required=True, help="Existing acquisition flags, e.g. site.flags.")
    parser.add_argument("--output", type=Path, required=True, help="New YAML file; existing files are preserved.")
    parser.add_argument("--out-dir", default="out", help="Reconstruction output directory in the YAML.")
    parser.add_argument("--batch-size", type=int, default=512, help="Shared decoded batch size.")
    return parser


def main():
    args = build_parser().parse_args()
    values = convert(shlex.split(args.flags.read_text()), out_dir=args.out_dir, batch_size=args.batch_size)
    with args.output.open("x") as stream:
        yaml.safe_dump(values, stream, sort_keys=False)


if __name__ == "__main__":
    main()
