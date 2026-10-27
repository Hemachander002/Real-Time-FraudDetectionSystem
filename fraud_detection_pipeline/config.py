from pathlib import Path
import json

PROJECT_ROOT = Path.cwd()
CONFIG_PATH = PROJECT_ROOT / "configs" / "training_config.json"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing {CONFIG_PATH}; run from the project root.")
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SEED = int(CONFIG.get("random_state", 42))
TARGET = CONFIG.get("target_column", "isfraud")
DROP_COLUMNS = CONFIG.get("drop_columns", [])
REQUIRED_COLUMNS = CONFIG.get("required_columns", [])
OUTPUT_PATH = Path(CONFIG["model_output_path"])
if not OUTPUT_PATH.is_absolute():
    OUTPUT_PATH = PROJECT_ROOT / OUTPUT_PATH
