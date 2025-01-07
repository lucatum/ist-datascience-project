# %% IMPORTS
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, subplots, savefig, show
from numpy import ndarray
from pandas import read_csv, DataFrame, Series
from seaborn import heatmap
from dslabs_functions import HEIGHT, plot_multi_scatters_chart
from dslabs_functions import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# EXERCISE 1
# %% DATASET 1: Combine Datasets
train_first_approach = pd.read_csv(
    "data/datasetCovid_Outliers_first_approach_train.csv"
)
test_first_approach = pd.read_csv("data/datasetCovid_Outliers_first_approach_test.csv")
combined_sets = pd.concat([train_first_approach, test_first_approach])
combined_sets.to_csv("data/class_pos_covid_mvi_encoded.csv", index=False)
# %%
file = "class_pos_covid"
data = pd.read_csv(f"data/{file}_mvi_encoded.csv", na_values="")
target = "CovidPos"
target_data = data.pop(target)

# All variables are now numeric, so we can apply the scaler directly to the dataframe
scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
data_z_scaled = scaler.fit_transform(data)

# Convert the scaled data back to a DataFrame
df_z_scaled = pd.DataFrame(data_z_scaled, index=data.index, columns=data.columns)

# Add the target variable back into the DataFrame
df_z_scaled[target] = target_data

# Save the scaled data to a new CSV file
df_z_scaled.to_csv(f"data/{file}_scaled.csv", index=False)
# %% MIN MAX SCALER
from sklearn.preprocessing import MinMaxScaler

# Specify the file name and target variable
file = "class_pos_covid"
target = "CovidPos"

# Read the CSV file
data = pd.read_csv(f"data/{file}_mvi_encoded.csv", na_values="")

# Store the target variable's data and remove it from the dataset
target_data = data.pop(target)

# Apply MinMax Scaler to the dataframe
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
data_minmax_scaled = scaler.fit_transform(data)

# Convert the scaled data back to a DataFrame
data_minmax_scaled = pd.DataFrame(
    data_minmax_scaled, index=data.index, columns=data.columns
)

# Add the target variable back into the DataFrame
data_minmax_scaled[target] = target_data

# Save the scaled data to a new CSV file
data_minmax_scaled.to_csv(f"data/{file}_scaled_minmax.csv", index=False)


# %% GRAPHS
fig, axs = subplots(1, 3, figsize=(23, 10), squeeze=False)
# Original data
axs[0, 0].set_title("Original data")
data.boxplot(ax=axs[0, 0])
axs[0, 0].tick_params(axis="x", rotation=90)  # Rotate x-axis labels
# Z-score normalization
axs[0, 1].set_title("Z-score normalization")
df_z_scaled.boxplot(ax=axs[0, 1])
axs[0, 1].tick_params(axis="x", rotation=90)  # Rotate x-axis labels
# MinMax normalization
axs[0, 2].set_title("MinMax normalization")
data_minmax_scaled.boxplot(ax=axs[0, 2])
axs[0, 2].tick_params(axis="x", rotation=90)  # Rotate x-axis labels
# Save the figure
savefig(f"images/Scaling/{file}_scaling.png", dpi=300)
show()
# %% ------------------------------------------------- KNN Study
from typing import Literal
from numpy import array, ndarray
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_multiline_chart
from dslabs_functions import read_train_test_from_files, plot_evaluation_results


def knn_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    k_max: int = 19,
    lag: int = 2,
    metric="recall",
) -> tuple[KNeighborsClassifier | None, dict]:
    dist: list[Literal["manhattan", "euclidean", "chebyshev"]] = [
        "manhattan",
        "euclidean",
        "chebyshev",
    ]

    kvalues: list[int] = [i for i in range(1, k_max + 1, lag)]
    best_model: KNeighborsClassifier | None = None
    best_params: dict = {"name": "KNN", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict[str, list] = {}
    for d in dist:
        y_tst_values: list = []
        for k in kvalues:
            clf = KNeighborsClassifier(n_neighbors=k, metric=d)
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance: float = eval
                best_params["params"] = (k, d)
                best_model = clf
            # print(f'KNN {d} k={k}')
        values[d] = y_tst_values
    print(f'KNN best with k={best_params["params"][0]} and {best_params["params"][1]}')
    plot_multiline_chart(
        kvalues,
        values,
        title=f"KNN Models ({metric})",
        xlabel="k",
        ylabel=metric,
        percentage=True,
    )

    return best_model, best_params


##############
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv(f"data/{file}_mvi_encoded.csv", na_values="")
target = "CovidPos"
X = data.drop(columns=[target])
y = data[target]

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
# MINMAX SCALER
# Apply standard scaling to both training and test feature sets
# scaler = StandardScaler()
scaler = MinMaxScaler(feature_range=(0, 1))
trnX_scaled = scaler.fit_transform(trnX)
tstX_scaled = scaler.transform(tstX)

# Convert scaled arrays back to DataFrames
trnX_scaled_df = pd.DataFrame(trnX_scaled, columns=X.columns)
tstX_scaled_df = pd.DataFrame(tstX_scaled, columns=X.columns)

# Concatenate the scaled features with the target variable
train_scaled = pd.concat([trnX_scaled_df, trnY.reset_index(drop=True)], axis=1)
test_scaled = pd.concat([tstX_scaled_df, tstY.reset_index(drop=True)], axis=1)

train_scaled.to_csv("data/class_pos_covid_train_zcore_scaled.csv", index=False)
test_scaled.to_csv("data/class_pos_covid_test_zscore_scaled.csv", index=False)

# Z-SCORE SCALER
# Apply standard scaling to both training and test feature sets
scalerZ = StandardScaler()
trnX_scaledZ = scalerZ.fit_transform(trnX)
tstX_scaledZ = scalerZ.transform(tstX)

# Convert scaled arrays back to DataFrames
trnX_scaled_dfZ = pd.DataFrame(trnX_scaledZ, columns=X.columns)
tstX_scaled_dfZ = pd.DataFrame(tstX_scaledZ, columns=X.columns)

# Concatenate the scaled features with the target variable
train_scaledZ = pd.concat([trnX_scaled_dfZ, trnY.reset_index(drop=True)], axis=1)
test_scaledZ = pd.concat([tstX_scaled_dfZ, tstY.reset_index(drop=True)], axis=1)

train_scaledZ.to_csv("data/class_pos_covid_train_zscore_scaled.csv", index=False)
test_scaledZ.to_csv("data/class_pos_covid_test_zscore_scaled.csv", index=False)
# %%
##############
train_filename = "data/class_pos_covid_train_minmax_scaled.csv"
test_filename = "data/class_pos_covid_test_minmax_scaled.csv"
target = "CovidPos"
eval_metric = "recall"

trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
    train_filename, test_filename, target
)
print(f"Train#={len(trnX)} Test#={len(tstX)}")
print(f"Labels={labels}")

figure()
best_model, params = knn_study(trnX, trnY, tstX, tstY, k_max=25, metric=eval_metric)
savefig(f"images/Scaling/{file}_knn_{eval_metric}_study.png", dpi=300)
show()

# %%
prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)
figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(
    f'images/Scaling/{file}_knn_{params["name"]}_best_{params["metric"]}_eval.png',
    dpi=300,
)
show()

# %%
from matplotlib.pyplot import figure, savefig

distance: Literal["manhattan", "euclidean", "chebyshev"] = params["params"][1]
K_MAX = 25
kvalues: list[int] = [i for i in range(1, K_MAX, 2)]
y_tst_values: list = []
y_trn_values: list = []
acc_metric: str = "accuracy"
for k in kvalues:
    clf = KNeighborsClassifier(n_neighbors=k, metric=distance)
    clf.fit(trnX, trnY)
    prd_tst_Y: array = clf.predict(tstX)
    prd_trn_Y: array = clf.predict(trnX)
    y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
    y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

figure()
plot_multiline_chart(
    kvalues,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"KNN overfitting study for {distance}",
    xlabel="K",
    ylabel=str(eval_metric),
    percentage=True,
)
savefig(f"images/Scaling/{file}_knn_overfitting.png", dpi=300)
show()

# %%
