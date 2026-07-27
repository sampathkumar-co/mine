from __future__ import annotations

import argparse
import json
from pathlib import Path

from .search import EvolutionConfig, evolve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mini-ORIGIN local substrate discovery")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run an evolutionary discovery experiment")
    demo.add_argument("--generations", type=int, default=12)
    demo.add_argument("--population", type=int, default=24)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--output", type=Path, default=Path("results/demo.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        config = EvolutionConfig(
            generations=args.generations,
            population_size=args.population,
            elite_count=max(2, min(6, args.population // 5)),
            seed=args.seed,
        )
        result = evolve(config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(json.dumps({"best_fitness": result.best_fitness, "task_scores": result.task_scores}, indent=2))


if __name__ == "__main__":
    main()
