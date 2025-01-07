# %%
from pandas import Series
from sklearn.base import RegressorMixin


class PersistenceOptimistRegressor(RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last: float = 0.0
        return

    def fit(self, X: Series):
        self.last = X.iloc[-1]
        # print(self.last)
        return

    def predict(self, X: Series):
        prd: list = X.shift().values.ravel()
        prd[0] = self.last
        prd_series: Series = Series(prd)
        prd_series.index = X.index
        return prd_series


# %%
from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import savefig, tight_layout
from dslabs_functions import (
    series_train_test_split,
    plot_forecasting_eval,
    plot_forecasting_series,
)

filename: str = "data/new/hourly_traffic.csv"
file_tag: str = "Traffic (Hourly)"
target: str = "Total"
timecol: str = "Timestamp"

data: DataFrame = read_csv(
    filename, index_col=timecol, sep=",", decimal=".", parse_dates=True
)
series: Series = data[target]

train, test = series_train_test_split(data, trn_pct=0.90)

fr_mod = PersistenceOptimistRegressor()
fr_mod.fit(train)
prd_trn: Series = fr_mod.predict(train)
prd_tst: Series = fr_mod.predict(test)

plot_forecasting_eval(
    train, test, prd_trn, prd_tst, title=f"{file_tag} - Persistence Optimist"
)
tight_layout()
savefig(f"images/{file_tag}_persistence_optim_eval.png", dpi=300)

# %%
plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - Persistence Optimist",
    xlabel=timecol,
    ylabel=target,
)
tight_layout()
savefig(f"images/{file_tag}_persistence_optim_forecast.png", dpi=300)


# %%
class PersistenceRealistRegressor(RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last = 0
        self.estimations = [0]
        self.obs_len = 0

    def fit(self, X: Series):
        for i in range(1, len(X)):
            self.estimations.append(X.iloc[i - 1])
        self.obs_len = len(self.estimations)
        self.last = X.iloc[len(X) - 1]
        prd_series: Series = Series(self.estimations)
        prd_series.index = X.index
        return prd_series

    def predict(self, X: Series):
        prd: list = len(X) * [self.last]
        prd_series: Series = Series(prd)
        prd_series.index = X.index
        return prd_series


# %%
fr_mod = PersistenceRealistRegressor()
fr_mod.fit(train)
prd_trn: Series = fr_mod.predict(train)
prd_tst: Series = fr_mod.predict(test)

plot_forecasting_eval(
    train, test, prd_trn, prd_tst, title=f"{file_tag} - Persistence Realist"
)
tight_layout()
savefig(f"images/{file_tag}_persistence_real_eval.png", dpi=300)

# %%
plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - Persistence Realist",
    xlabel=timecol,
    ylabel=target,
)
tight_layout()
savefig(f"images/{file_tag}_persistence_real_forecast.png", dpi=300)

# %%
