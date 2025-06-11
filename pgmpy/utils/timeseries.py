import pandas as pd


def from_sktime_to_dbn(
    df: pd.DataFrame, instance_col: str | None = None
) -> pd.DataFrame:
    """Convert sktime style time series data to DBN data format.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe where the index can be a ``DatetimeIndex``/``RangeIndex``
        representing the time dimension or a ``MultiIndex`` with ``(instance,
        time)`` levels.
    instance_col : str | None, default=None
        If the instance id is stored as a column instead of the index, specify
        the column name here.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns of the form ``(variable, time_slice)`` suitable
        for :meth:`pgmpy.models.DynamicBayesianNetwork.fit`.
    """
    df = df.copy()

    if instance_col is not None:
        df = df.set_index(instance_col, append=True)
        df.index = df.index.reorder_levels([instance_col, 0])

    if not isinstance(df.index, pd.MultiIndex):
        df["__instance"] = 0
        df = df.set_index("__instance", append=True)
        df.index = df.index.reorder_levels(["__instance", 0])

    df.index.names = ["instance", "time"]

    grouped = df.sort_index().groupby(level=0)
    rows = []
    idx = []
    for inst, gdf in grouped:
        gdf = gdf.droplevel(0)
        gdf = gdf.reset_index(drop=True)
        row = {}
        for t, (_, values) in enumerate(gdf.iterrows()):
            for col, val in values.items():
                row[(col, t)] = val
        rows.append(row)
        idx.append(inst)

    result = pd.DataFrame(rows, index=idx)
    if result.columns:
        result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result


def from_dbn_to_sktime(dbn_data: pd.DataFrame) -> pd.DataFrame:
    """Convert data from DBN format back to sktime style format.

    Parameters
    ----------
    dbn_data : pandas.DataFrame
        DataFrame with columns of the form ``(variable, time_slice)``.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by ``(instance, time)`` with variables as columns.
    """
    df = dbn_data.copy()
    if not isinstance(df.columns, pd.MultiIndex):
        if all(isinstance(c, tuple) and len(c) == 2 for c in df.columns):
            df.columns = pd.MultiIndex.from_tuples(df.columns)
        else:
            raise ValueError("Columns must be a MultiIndex or tuple names.")

    variables = sorted({c[0] for c in df.columns})
    times = sorted({c[1] for c in df.columns})

    rows = []
    index = []
    for inst, row in df.iterrows():
        for t in times:
            rows.append([row[(var, t)] for var in variables])
            index.append((inst, t))

    result = pd.DataFrame(
        rows,
        columns=variables,
        index=pd.MultiIndex.from_tuples(index, names=["instance", "time"]),
    )
    return result
