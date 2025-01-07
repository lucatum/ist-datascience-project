# %%
from numpy import arange
from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import savefig
from sklearn.linear_model import LinearRegression
from dslabs_functions import series_train_test_split, plot_forecasting_eval
from sklearn.preprocessing import StandardScaler

filename: str = "data/forecast_traffic_single.csv"
file_tag: str = "TRAFFIC"
target: str = "Total"
timecol: str = "Timestamp"
# %%
data: DataFrame = read_csv(
    filename, index_col=timecol, sep=",", decimal=".", parse_dates=True
)
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
    train, test, prd_trn, prd_tst, title=f"{file_tag} - Linear Regression"
)
savefig(f"images/{file_tag}_linear_regression_eval.png")

# %%
from dslabs_functions import plot_forecasting_series

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - Linear Regression",
    xlabel=timecol,
    ylabel=target,
)
savefig(f"images/{file_tag}_linear_regression_forecast.png")

# %%
