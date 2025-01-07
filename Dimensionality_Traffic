# %% IMPORTS
from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig, tight_layout
from dslabs_functions import plot_line_chart, HEIGHT
import matplotlib.pyplot as plt

# %%
file_tag = "TRAFFIC"
target = "Total"
data: DataFrame = read_csv(
    "data/forecast_traffic_single.csv",
    index_col="Timestamp",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)
series: Series = data[target]
print("Nr. Records = ", series.shape[0])
print("First timestamp", series.index[0])
print("Last timestamp", series.index[-1])

figure(figsize=(3 * HEIGHT, HEIGHT / 1))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} every 15mins",
)

tight_layout()
savefig("images/{}_{}_weekly.png".format(file_tag, target), dpi=300)
show()

# %%
from pandas import Index, Period


def ts_aggregation_by(
    data: Series | DataFrame,
    gran_level: str = "D",
    agg_func: str = "mean",
) -> Series | DataFrame:
    df: Series | DataFrame = data.copy()
    index: Index[Period] = df.index.to_period(gran_level)
    df = df.groupby(by=index, dropna=True, sort=True).agg(agg_func)
    df.index.drop_duplicates()
    df.index = df.index.to_timestamp()

    return df


ss_days: Series = ts_aggregation_by(series, "D")
figure(figsize=(3 * HEIGHT, HEIGHT / 1))
plot_line_chart(
    ss_days.index.to_list(),
    ss_days.to_list(),
    xlabel="days",
    ylabel=target,
    title=f"{file_tag} daily mean {target}",
)
tight_layout()
savefig("images/{}_{}_daily.png".format(file_tag, target), dpi=300)

show()

# %%
