# %% IMPORTS
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, subplots, savefig, show
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

# %%
file_tag = "class_credit_score"
train_filename = "data/new/class_credit_score_over.csv"
test_filename = "data/new/class_credit_score_test_zscore_scaled_switched.csv"
target = "Credit_Score"
eval_metric = "precision"

# train
train_dropped: DataFrame = read_csv(
    f"data/new/class_credit_score_over.csv", sep=",", decimal="."
)
train_dropped = train_dropped.drop(
    [
        "Age",
        "Num_Credit_Card",
        "ChangedCreditLimit",
        "CreditUtilizationRatio",
        "TotalEMIpermonth",
        "Amountinvestedmonthly",
        "MonthlyBalance",
    ],
    axis=1,
)

# %%

train_dropped.to_csv("data/new/class_credit_score_train_selected.csv", index=False)


# test
test_dropped: DataFrame = read_csv(
    f"data/new/class_credit_score_test_zscore_scaled_switched.csv", sep=",", decimal="."
)
test_dropped = test_dropped.drop(
    [
        "Age",
        "Num_Credit_Card",
        "ChangedCreditLimit",
        "CreditUtilizationRatio",
        "TotalEMIpermonth",
        "Amountinvestedmonthly",
        "MonthlyBalance",
    ],
    axis=1,
)
test_dropped.to_csv("data/new/class_credit_score_test_selected.csv", index=False)

# %%
