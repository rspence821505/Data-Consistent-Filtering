from pathlib import Path
import yaml


def load_experiment_config(config_path: Path) -> dict:
    """Load experiment configuration from a YAML file.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    config : dict
        Dictionary containing experiment configuration parameters.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
