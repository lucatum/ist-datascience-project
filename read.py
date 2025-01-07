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

# %%
original: DataFrame = read_csv(f"data/class_credit_score.csv", sep=",", decimal=".")

original.Credit_Score.describe()
# %%
