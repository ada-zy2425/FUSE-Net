"""Run one or several explicitly selected seeds without changing evaluation splits."""

import json

from .config import Config, training_parser


def main(argv=None):
    parser = training_parser()
    args = vars(parser.parse_args(argv))
    seeds, output_dir = args.pop("seeds"), args.pop("output_dir")
    if len(seeds) != len(set(seeds)):
        parser.error("seeds must be unique")
    configs = [Config(**args, seed=seed) for seed in seeds]
    if output_dir.exists():
        parser.error("output_dir already exists; choose a new experiment directory")
    # Lazy imports allow --help without downloading or importing any models.
    from .data_loader import make_loaders
    from .runtime import seed_everything, write_json
    from .solver import Solver, summarize_runs
    results = []
    for config in configs:
        seed_everything(config.seed)
        loaders, tokenizer = make_loaders(config)
        result = Solver(config, loaders, tokenizer, output_dir / ("seed_" + str(config.seed))).train()
        results.append(result)
        write_json(output_dir / "summary.json", {"seeds": seeds[:len(results)], "metrics": summarize_runs(results)})
        print(json.dumps({"seed": config.seed, "test": result}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
