import pandas as pd
import logging

from pathlib import Path
from src.config import RAW_FILES


# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

#   Load a CSV file into a pandas DataFrame.
# implementing daat provenances
def load_csv(path):
    df = pd.read_csv(
        path,
        low_memory=False,
    )

    df["source_dataset"] = path.stem

    logger.info("Loaded %s rows", len(df))

    return df


# Load every raw dataset.
def load_all() -> dict[str, pd.DataFrame]:
    datasets = {}

    for file in RAW_FILES:
        datasets[file.stem] = load_csv(file)

    logger.info("Loaded %s datasets.", len(datasets))

    return datasets
