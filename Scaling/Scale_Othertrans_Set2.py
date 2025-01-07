# %%
from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, tight_layout
from dslabs_functions import plot_line_chart, HEIGHT
from sklearn.preprocessing import StandardScaler

file_tag = "TRAFFIC"
filename = "data/new/hourly_traffic.csv"
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


ss_diff: Series = series.diff()


# %%
ss_diff_csv_path = "data/scale_traffic.csv"
ss_diff.to_csv(ss_diff_csv_path, header=True)

# %% EVAL

# %%
from numpy import arange
from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import savefig
from sklearn.linear_model import LinearRegression
from dslabs_functions import series_train_test_split, plot_forecasting_eval

file_tag = "TRAFFIC"
filename = "data/new/hourly_traffic.csv"
index = "Timestamp"
target = "Total"
timecol = "Timestamp"
# %%
datapre: DataFrame = read_csv(
    filename, index_col=timecol, sep=",", decimal=".", parse_dates=True
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
datapre_standardized = scaler.fit_transform(datapre)
data = DataFrame(datapre_standardized, index=datapre.index, columns=datapre.columns)
data = data.fillna(0)
series: Series = data[target]
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
    train, test, prd_trn, prd_tst, title=f"{file_tag} (Z-scaled) - Linear Regression"
)
tight_layout()
savefig(f"images/{file_tag}_linear_regression_eval_zscale.png", dpi=300)

# %%
from dslabs_functions import plot_forecasting_series

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} (Z-scaled) - Linear Regression",
    xlabel=timecol,
    ylabel=target,
)
tight_layout()
savefig(f"images/{file_tag}_linear_regression_forecast_zscale.png", dpi=300)

# %%
