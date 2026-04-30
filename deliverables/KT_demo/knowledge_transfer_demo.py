"""
Knowledge transfer demo driver for the warehouse labor prediction pipeline.

Run after creating demo data with:
  python deliverables/KT_demo/data_split_for_demo.py

This script can either print the command-line workflow or execute Step 2:
  python deliverables/KT_demo/knowledge_transfer_demo.py --print_commands
  python deliverables/KT_demo/knowledge_transfer_demo.py --run

Step 2 workflow:
  1. Preprocess the historical training split
  2. Train the main model on the historical split
  3. Run primary and secondary predictions on the 50-row prediction queue
  4. Run incremental learning with the last-day split
"""

import argparse
import shlex
import subprocess
from pathlib import Path


DEFAULT_DEMO_ROOT = "deliverables/KT_demo/demo_data"
DEFAULT_MODELS_DIR = "deliverables/KT_demo/demo_models"
DEFAULT_OUTPUT_DIR = "deliverables/KT_demo/demo_outputs"
DEFAULT_BUDGET_MIN = 30
DEFAULT_TREES = 300
DEFAULT_UPDATE_TREES = 50


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print or run the knowledge transfer demo command sequence."
    )
    parser.add_argument("--warehouse", default="OE", help="Warehouse code.")
    parser.add_argument("--demo_root", default=DEFAULT_DEMO_ROOT)
    parser.add_argument("--models_dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--budget_min", type=int, default=DEFAULT_BUDGET_MIN)
    parser.add_argument(
        "--trees",
        type=int,
        default=DEFAULT_TREES,
        help="Training trees for demo model. Increase for production-quality retraining.",
    )
    parser.add_argument(
        "--update_trees",
        type=int,
        default=DEFAULT_UPDATE_TREES,
        help="Trees to add during incremental update.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        help="Optional explicit prediction CSV. Defaults to WH_predict_50.csv.",
    )
    parser.add_argument("--run", action="store_true", help="Execute Step 2 commands.")
    parser.add_argument(
        "--print_commands",
        action="store_true",
        help="Print Step 2 commands without running them.",
    )
    return parser.parse_args()


def build_paths(args):
    warehouse = args.warehouse.upper()
    demo_root = Path(args.demo_root)
    predict_file = (
        Path(args.predict_file)
        if args.predict_file
        else demo_root / "predict" / warehouse / f"{warehouse}_predict_50.csv"
    )

    paths = {
        "warehouse": warehouse,
        "training_data": demo_root / "training_data",
        "incremental_csv": demo_root / "incremental" / warehouse / f"{warehouse}_activity_lastday.csv",
        "predict_csv": predict_file,
        "models_dir": Path(args.models_dir),
        "output_dir": Path(args.output_dir),
    }
    return paths


def command_list(args, paths):
    wh = paths["warehouse"]
    output_dir = paths["output_dir"]

    return [
        [
            "python",
            "deliverables/preprocess.py",
            wh,
            "--data_dir",
            str(paths["training_data"]),
        ],
        [
            "python",
            "deliverables/model_training.py",
            wh,
            "--data_dir",
            str(paths["training_data"]),
            "--models_dir",
            str(paths["models_dir"]),
            "--trees",
            str(args.trees),
        ],
        [
            "python",
            "deliverables/predict_primary.py",
            wh,
            str(paths["predict_csv"]),
            "--data_dir",
            str(paths["training_data"]),
            "--models_dir",
            str(paths["models_dir"]),
            "--out",
            str(output_dir / f"{wh}_primary_before_update.csv"),
        ],
        [
            "python",
            "deliverables/predict_secondary.py",
            wh,
            str(paths["predict_csv"]),
            str(args.budget_min),
            "--data_dir",
            str(paths["training_data"]),
            "--models_dir",
            str(paths["models_dir"]),
            "--out",
            str(output_dir / f"{wh}_secondary_before_update.csv"),
        ],
        [
            "python",
            "deliverables/update_model_incremental.py",
            wh,
            "--new_data",
            str(paths["incremental_csv"]),
            "--data_dir",
            str(paths["training_data"]),
            "--models_dir",
            str(paths["models_dir"]),
            "--trees",
            str(args.update_trees),
        ],
        [
            "python",
            "deliverables/predict_primary.py",
            wh,
            str(paths["predict_csv"]),
            "--data_dir",
            str(paths["training_data"]),
            "--models_dir",
            str(paths["models_dir"]),
            "--out",
            str(output_dir / f"{wh}_primary_after_update.csv"),
        ],
        [
            "python",
            "deliverables/predict_secondary.py",
            wh,
            str(paths["predict_csv"]),
            str(args.budget_min),
            "--data_dir",
            str(paths["training_data"]),
            "--models_dir",
            str(paths["models_dir"]),
            "--out",
            str(output_dir / f"{wh}_secondary_after_update.csv"),
        ],
    ]


def print_commands(commands):
    print("Step 1: create demo splits")
    print("python deliverables/KT_demo/data_split_for_demo.py")
    print()
    print("Step 2: run the pipeline demo")
    for command in commands:
        print(" ".join(shlex.quote(part) for part in command))


def validate_inputs(paths):
    required = [
        paths["training_data"] / paths["warehouse"] / f"{paths['warehouse']}_Activity.csv",
        paths["training_data"] / paths["warehouse"] / f"{paths['warehouse']}_Locations.csv",
        paths["training_data"] / paths["warehouse"] / f"{paths['warehouse']}_Products.csv",
        paths["incremental_csv"],
        paths["predict_csv"],
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "Demo input files are missing. Run Step 1 first:\n"
            "  python deliverables/KT_demo/data_split_for_demo.py\n\n"
            f"Missing:\n{missing_text}"
        )


def run_commands(commands):
    for command in commands:
        print("", flush=True)
        print("$ " + " ".join(shlex.quote(part) for part in command), flush=True)
        subprocess.run(command, check=True)


def main():
    args = parse_args()
    paths = build_paths(args)
    commands = command_list(args, paths)

    if args.print_commands or not args.run:
        print_commands(commands)

    if args.run:
        paths["output_dir"].mkdir(parents=True, exist_ok=True)
        validate_inputs(paths)
        run_commands(commands)


if __name__ == "__main__":
    main()
