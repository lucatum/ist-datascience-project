# %%
from pandas import read_csv, DataFrame, Series
from statsmodels.tsa.arima.model import ARIMA
from dslabs_functions import series_train_test_split, HEIGHT
import numpy as np

filename: str = "data/new/weekly_covid.csv"
file_tag: str = "COVID (Weekly)"
target: str = "weekly_deaths"
timecol: str = "date"
measure: str = "R2"

data: DataFrame = read_csv(
    filename, index_col=timecol, sep=",", decimal=".", parse_dates=True
)
data = np.log(data + 0.000001).fillna(0)
series: Series = data[target]
train, test = series_train_test_split(data, trn_pct=0.90)

predictor = ARIMA(train, order=(2, 1, 0))
model = predictor.fit()
print(model.summary())

# %%
import matplotlib.pyplot as plt

residuals = model.resid[1:]
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(title="Density", kind="kde", ax=ax[1])
plt.show()
# %%
model.plot_diagnostics(figsize=(2 * HEIGHT, 1.5 * HEIGHT))
# %%
from matplotlib.pyplot import figure, savefig, subplots
from dslabs_functions import FORECAST_MEASURES, DELTA_IMPROVE, plot_multiline_chart


def arima_study(train: Series, test: Series, measure: str = "R2"):
    d_values = (0, 1, 2)
    p_params = (1, 2, 3, 5, 7, 10)
    q_params = (1, 3, 5, 7)

    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "ARIMA", "metric": measure, "params": ()}
    best_performance: float = -100000

    fig, axs = subplots(1, len(d_values), figsize=(len(d_values) * HEIGHT, HEIGHT))
    for i in range(len(d_values)):
        d: int = d_values[i]
        values = {}
        for q in q_params:
            yvalues = []
            for p in p_params:
                arima = ARIMA(train, order=(p, d, q))
                model = arima.fit()
                prd_tst = model.forecast(steps=len(test), signal_only=False)
                eval: float = FORECAST_MEASURES[measure](test, prd_tst)
                # print(f"ARIMA ({p}, {d}, {q})", eval)
                if (
                    eval > best_performance
                    and abs(eval - best_performance) > DELTA_IMPROVE
                ):
                    best_performance: float = eval
                    best_params["params"] = (p, d, q)
                    best_model = model
                yvalues.append(eval)
            values[q] = yvalues
        plot_multiline_chart(
            p_params,
            values,
            ax=axs[i],
            title=f"ARIMA d={d} ({measure})",
            xlabel="p",
            ylabel=measure,
            percentage=flag,
        )
    print(
        f"ARIMA best results achieved with (p,d,q)=({best_params['params'][0]:.0f}, {best_params['params'][1]:.0f}, {best_params['params'][2]:.0f}) ==> measure={best_performance:.2f}"
    )

    return best_model, best_params


# %%
from matplotlib.pyplot import savefig, tight_layout

best_model, best_params = arima_study(train, test, measure=measure)
tight_layout()
savefig(f"images/{file_tag}_arima_{measure}_study.png", dpi=300)


# %%
from dslabs_functions import plot_forecasting_eval

params = best_params["params"]
prd_trn = best_model.predict(start=0, end=len(train) - 1)
prd_tst = best_model.forecast(steps=len(test))

plot_forecasting_eval(
    train,
    test,
    prd_trn,
    prd_tst,
    title=f"{file_tag} - ARIMA (p={params[0]}, d={params[1]}, q={params[2]})",
)
tight_layout()
savefig(f"images/{file_tag}_arima_{measure}_eval.png", dpi=300)

# %%
from dslabs_functions import plot_forecasting_series

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - ARIMA ",
    xlabel=timecol,
    ylabel=target,
)
tight_layout()
savefig(f"images/{file_tag}_arima_{measure}_forecast.png", dpi=300)

# %%
