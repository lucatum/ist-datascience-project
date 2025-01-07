# %% IMPORTS
from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig, tight_layout
from dslabs_functions import plot_line_chart, HEIGHT
import matplotlib.pyplot as plt

# %%
file_tag = "COVID"
target = "deaths"
data: DataFrame = read_csv(
    "data/forecast_covid_single.csv",
    index_col="date",
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
    title=f"{file_tag} weekly {target} (accumulated)",
)

tight_layout()
savefig("images/{}_{}_weekly.png".format(file_tag, target), dpi=300)
show()

# %%
# Calculate daily unique deaths
covid_data = data.copy()
covid_data["daily_deaths"] = data["deaths"].diff().fillna(0).astype(int)

# Display the first few rows of the modified dataframe
covid_data["deaths"]
covid_data.to_csv("data/forecast_covid_single_change.csv", index=True)

# %%
target = "daily_deaths"
series: Series = covid_data[target]
print("Nr. Records = ", series.shape[0])
print("First timestamp", series.index[0])
print("Last timestamp", series.index[-1])

figure(figsize=(3 * HEIGHT, HEIGHT / 1))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} weekly (unique changes)",
)

tight_layout()
savefig("images/{}_{}_weekly_change.png".format(file_tag, target), dpi=300)
show()

# %% QUARTERLY

from pandas import Index, Period


def ts_aggregation_by(
    data: Series | DataFrame,
    gran_level: str = "Q",
    agg_func: str = "mean",
) -> Series | DataFrame:
    df: Series | DataFrame = data.copy()
    index: Index[Period] = df.index.to_period(gran_level)
    df = df.groupby(by=index, dropna=True, sort=True).agg(agg_func)
    df.index.drop_duplicates()
    df.index = df.index.to_timestamp()

    return df


ss_days: Series = ts_aggregation_by(series, "Q")
figure(figsize=(3 * HEIGHT, HEIGHT / 1))
plot_line_chart(
    ss_days.index.to_list(),
    ss_days.to_list(),
    xlabel="days",
    ylabel=target,
    title=f"{file_tag} quarterly mean (unique changes) {target}",
)
tight_layout()
savefig("images/{}_{}_quarterly_change.png".format(file_tag, target), dpi=300)
show()

# %% STATIONARY
from pandas import Series
from matplotlib.pyplot import subplots, show, gca
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from dslabs_functions import HEIGHT, set_chart_labels


def plot_components(
    series: Series,
    title: str = "",
    x_label: str = "time",
    y_label: str = "",
) -> list[Axes]:
    decomposition: DecomposeResult = seasonal_decompose(series, model="add")
    components: dict = {
        "observed": series,
        "trend": decomposition.trend,
        "seasonal": decomposition.seasonal,
        "residual": decomposition.resid,
    }
    rows: int = len(components)
    fig: Figure
    axs: list[Axes]
    fig, axs = subplots(rows, 1, figsize=(3 * HEIGHT, rows * HEIGHT))
    fig.suptitle(f"{title}")
    i: int = 0
    for key in components:
        set_chart_labels(axs[i], title=key, xlabel=x_label, ylabel=y_label)
        axs[i].plot(components[key])
        i += 1
    return axs


from pandas import DataFrame, Series, read_csv
from matplotlib.pyplot import figure, show
from dslabs_functions import plot_line_chart

file_tag = "COVID"
filename = "data/forecast_covid_single.csv"
index = "date"
target = "deaths"
series: Series = covid_data["daily_deaths"]

plot_components(
    series,
    title=f"{file_tag} weekly {target}",
    x_label=series.index.name,
    y_label=target,
)
tight_layout()
savefig("images/Stationarity/{}_{}_components2.png".format(file_tag, target), dpi=300)
show()

# %%
from matplotlib.pyplot import plot, legend

figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} stationary study",
    name="original",
)
n: int = len(series)
plot(series.index, [series.mean()] * n, "r-", label="mean")
legend()
tight_layout()
savefig(
    "images/Stationarity/{}_{}_stationarity_study1.png".format(file_tag, target),
    dpi=300,
)
show()


# %%
BINS = 10
mean_line: list[float] = []

for i in range(BINS):
    segment: Series = series[i * n // BINS : (i + 1) * n // BINS]
    mean_value: list[float] = [segment.mean()] * (n // BINS)
    mean_line += mean_value
mean_line += [mean_line[-1]] * (n - len(mean_line))

figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} stationary study",
    name="original",
    show_stdev=True,
)
n: int = len(series)
plot(series.index, mean_line, "r-", label="mean")
legend()
tight_layout()
savefig(
    "images/Stationarity/{}_{}_stationarity_study2.png".format(file_tag, target),
    dpi=300,
)

show()


# %%
from statsmodels.tsa.stattools import adfuller


def eval_stationarity(series: Series) -> bool:
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.3f}")
    print(f"p-value: {result[1]:.3f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.3f}")
    return result[1] <= 0.05


print(f"The series {('is' if eval_stationarity(series) else 'is not')} stationary")

# %%
