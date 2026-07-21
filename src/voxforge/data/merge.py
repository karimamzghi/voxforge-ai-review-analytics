import pandas as pd

#     Merge all datasets.
def merge_datasets(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:

    return pd.concat(
        datasets.values(),
        ignore_index=True,
    )

