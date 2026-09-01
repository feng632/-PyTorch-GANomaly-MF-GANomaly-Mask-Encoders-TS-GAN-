"""PyCharm-friendly launcher for the four-crop GANomaly experiment."""

import sys

from anomaly_reproduction.train import PROJECT_ROOT, main


if __name__ == "__main__":
    config_path = PROJECT_ROOT / "configs" / "ganomaly_grid_four_crops.yaml"
    sys.argv = [sys.argv[0], "--config", str(config_path)]
    main()
