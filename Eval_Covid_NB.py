# %% IMPORTS
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, subplots, savefig, show, tight_layout
from numpy import ndarray
from pandas import read_csv, DataFrame
from seaborn import heatmap
from pandas import read_csv, DataFrame
from dslabs_functions import HEIGHT, plot_multi_scatters_chart
from dslabs_functions import get_variable_types
from typing import Literal
from numpy import array, ndarray
from matplotlib.pyplot import figure, savefig, show
from sklearn.tree import DecisionTreeClassifier
from dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    read_train_test_from_files,
)
from dslabs_functions import plot_evaluation_results, plot_multiline_chart
from dslabs_functions import read_train_test_from_files


# %% NB ALTERTATIVES STUDY
def trees_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    d_max: int = 10,
    lag: int = 2,
    metric="accuracy",
) -> tuple:
    criteria: list[Literal["entropy", "gini"]] = ["entropy", "gini"]
    depths: list[int] = [i for i in range(2, d_max + 1, lag)]

    best_model: DecisionTreeClassifier | None = None
    best_params: dict = {"name": "DT", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    for c in criteria:
        y_tst_values: list[float] = []
        for d in depths:
            clf = DecisionTreeClassifier(
                max_depth=d, criterion=c, min_impurity_decrease=0
            )
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance = eval
                best_params["params"] = (c, d)
                best_model = clf
            # print(f'DT {c} and d={d}')
        values[c] = y_tst_values
    print(f'DT best with {best_params["params"][0]} and d={best_params["params"][1]}')
    plot_multiline_chart(
        depths,
        values,
        title=f"DT Models ({metric})",
        xlabel="d",
        ylabel=metric,
        percentage=True,
    )

    return best_model, best_params


file_tag = "class_pos_covid"
train_filename = "data/class_pos_covid_train_selected.csv"
test_filename = "data/class_pos_covid_test_selected.csv"
target = "CovidPos"
eval_metric = "accuracy"

trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
    train_filename, test_filename, target
)
print(f"Train#={len(trnX)} Test#={len(tstX)}")
print(f"Labels={labels}")

# %% ANALYSIS

from numpy import array, ndarray
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_bar_chart


def naive_Bayes_study(
    trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, metric: str = "accuracy"
) -> tuple:
    estimators: dict = {
        "GaussianNB": GaussianNB(),
        "BernoulliNB": BernoulliNB(),
    }

    xvalues: list = []
    yvalues: list = []
    best_model = None
    best_params: dict = {"name": "", "metric": metric, "params": ()}
    best_performance = 0
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY: array = estimators[clf].predict(tstX)
        eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
        if eval - best_performance > DELTA_IMPROVE:
            best_performance: float = eval
            best_params["name"] = clf
            best_params[metric] = eval
            best_model = estimators[clf]
        yvalues.append(eval)
        # print(f'NB {clf}')
    plot_bar_chart(
        xvalues,
        yvalues,
        title=f"Naive Bayes Models ({metric})",
        ylabel=metric,
        percentage=True,
    )

    return best_model, best_params


figure()
best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, eval_metric)
tight_layout()
savefig(f"images/Eval/NB/{file_tag}_nb_{eval_metric}_study.png")
show()

# %% ANALYSIS RECALL

figure()
best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, "recall")
tight_layout()
savefig(f"images/Eval/NB/{file_tag}_nb_recall_study.png")
show()

# %% PERFORMANCE ANALYSIS

from dslabs_functions import plot_evaluation_results

prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)
figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f'images/Eval/NB/{file_tag}_{params["name"]}_best_{params["metric"]}_eval.png')
show()

# %%
