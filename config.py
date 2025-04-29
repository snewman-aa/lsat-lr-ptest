from pathlib import Path
import yaml

def load_config(config_path: Path = None) -> dict:
    """
    Load YAML configuration from the project root.

    :param config_path: Optional path to config.yaml. Defaults to <project_root>/config.yaml.
    :return: Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
