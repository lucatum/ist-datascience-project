# %%
from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show
from dslabs_functions import plot_line_chart, HEIGHT
from sklearn.preprocessing import StandardScaler

file_tag = "TRAFFIC"
filename = "data/forecast_traffic_single.csv"
index = "Timestamp"
target = "Total"
data: DataFrame = read_csv(
    filename,
    index_col=index,
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)
series: Series = data[target]

figure(figsize=(3 * HEIGHT, HEIGHT / 1))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} hourly {target}",
)
show()

# %%
from dslabs_functions import plot_line_chart

ss_diff: Series = series.diff()
figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    ss_diff.index.to_list(),
    ss_diff.to_list(),
    title="Differentiation",
    xlabel=series.index.name,
    ylabel=target,
)
show()

# %%
ss_diff: Series = series.diff().diff()
figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    ss_diff.index.to_list(),
    ss_diff.to_list(),
    title="Differentiation",
    xlabel=series.index.name,
    ylabel=target,
)
show()

# %%
ss_diff_csv_path = "data/diff2_traffic.csv"
ss_diff.to_csv(ss_diff_csv_path, header=True)

# %% EVAL

# %%
from numpy import arange
from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import savefig, tight_layout
from sklearn.linear_model import LinearRegression
from dslabs_functions import series_train_test_split, plot_forecasting_eval

filename: str = "data/diff2_traffic.csv"
file_tag: str = "TRAFFIC"
target: str = "Total"
timecol: str = "Timestamp"
# %%
data: DataFrame = read_csv(
    filename, index_col=timecol, sep=",", decimal=".", parse_dates=True
)
series: Series = data[target]
series.fillna(0, inplace=True)
data.fillna(0, inplace=True)
train, test = series_train_test_split(data, trn_pct=0.90)

scaler = StandardScaler()

trnX = scaler.fit_transform(arange(len(train)).reshape(-1, 1))
trnY = train.to_numpy()
tstX = scaler.transform(arange(len(train), len(data)).reshape(-1, 1))
tstY = test.to_numpy()

model = LinearRegression()
model.fit(trnX, trnY)

prd_trn: Series = Series(model.predict(trnX), index=train.index)
prd_tst: Series = Series(model.predict(tstX), index=test.index)

plot_forecasting_eval(
    train, test, prd_trn, prd_tst, title=f"{file_tag} (1 Diff) - Linear Regression"
)
tight_layout()
savefig(f"images/{file_tag}_linear_regression_eval_diff.png", dpi=300)

# %%
from dslabs_functions import plot_forecasting_series

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} (1 Diff) - Linear Regression",
    xlabel=timecol,
    ylabel=target,
)
tight_layout()
savefig(f"images/{file_tag}_linear_regression_forecast_diff.png", dpi=300)

# %%
