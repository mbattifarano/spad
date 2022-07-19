import pandas as pd
import seaborn as sns


def get_data(filename):
    return pd.read_parquet(filename)


def split_year_month(df: pd.DataFrame):
    idx_cols = df.index.names
    df.reset_index(inplace=True)
    df["year"] = df.year_month.apply(lambda t: t.year)
    df["month"] = df.year_month.apply(lambda t: t.month)
    df.set_index(
        ["year", "month", "activity_type", "tag_name", "tag_value"],
        inplace=True,
    )


def tag_zscore(df):
    grouped_tag_count = df.groupby(
        ["activity_type", "tag_name", "tag_value"]
    ).tag_count
    return (
        df.tag_count - grouped_tag_count.transform("mean")
    ) / grouped_tag_count.transform("std")


def month_compare(df: pd.DataFrame):
    year_groups = df.groupby("year")
    df_2019 = year_groups.get_group(2019)
    df_2019.reset_index("year", inplace=True)
    df_2020 = year_groups.get_group(2020)
    df_2020.reset_index("year", inplace=True)
    return df_2019.join(df_2020, how="outer", lsuffix="_2019", rsuffix="_2020")
