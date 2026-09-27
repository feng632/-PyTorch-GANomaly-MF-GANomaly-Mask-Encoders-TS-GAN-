"""生成 AITEX 域内 TS-GAN 实验的固定原图级划分清单。"""

import argparse
from pathlib import Path

import yaml

from anomaly_reproduction.aitex_domain import create_aitex_manifest, save_aitex_manifest
from anomaly_reproduction.train import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "data" / "aitex_split_seed42.yaml",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normal-train-per-fabric", type=int, default=12)
    parser.add_argument("--normal-validation-per-fabric", type=int, default=4)
    parser.add_argument("--anomaly-train-per-type", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = create_aitex_manifest(
        args.data_root,
        seed=args.seed,
        normal_train_per_fabric=args.normal_train_per_fabric,
        normal_validation_per_fabric=args.normal_validation_per_fabric,
        anomaly_train_per_type=args.anomaly_train_per_type,
    )
    save_aitex_manifest(manifest, args.output)
    print(yaml.safe_dump(manifest["summary"], allow_unicode=True, sort_keys=False))
    print(f"划分清单已保存：{args.output.resolve()}")


if __name__ == "__main__":
    main()
