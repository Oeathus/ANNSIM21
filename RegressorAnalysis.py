import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

plt.close("all")

log_folder = "logs" + os.path.sep + "classifier" + os.path.sep
analysis_folder = "analysis" + os.path.sep
knnData = pd.read_csv(log_folder + 'knnr.log.csv')
svrData = pd.read_csv(log_folder + 'svr.log.csv')
linearData = pd.read_csv(log_folder + 'linear.log.csv')
dnnData = pd.read_csv(log_folder + 'dnnr.log.csv')
wnnData = pd.read_csv(log_folder + 'wnnr.log.csv')
dwnnData = pd.read_csv(log_folder + 'dwnnr.log.csv')
lstmData = pd.read_csv(log_folder + 'lstmr.log.csv')

desiredAgg = {
    'R^2': 'max',
    'MSE': 'min',
    'RMSE': 'min',
    'Pearson Corr': 'max'
}

knnFinalRanking = knnData[["Normalization", "R^2", "MSE", "RMSE", "Pearson Corr"]].copy()
knnFinalRanking = knnFinalRanking.groupby("Normalization", as_index=False).agg(desiredAgg)
knnFinalRanking["Model"] = "KNN"

svrFinalRanking = svrData[["Normalization", "R^2", "MSE", "RMSE", "Pearson Corr"]].copy()
svrFinalRanking = svrFinalRanking.groupby("Normalization", as_index=False).agg(desiredAgg)
svrFinalRanking["Model"] = "SVR"

linearFinalRanking = linearData[["Normalization", "R^2", "MSE", "RMSE", "Pearson Corr"]].copy()
linearFinalRanking = linearFinalRanking.groupby("Normalization", as_index=False).agg(desiredAgg)
linearFinalRanking["Model"] = "Linear"

lstmFinalRanking = lstmData[["Model", "Normalization", "R^2", "MSE", "RMSE", "Pearson Corr"]].copy()
lstmFinalRanking = lstmFinalRanking.groupby(["Model", "Normalization"], as_index=False).agg(desiredAgg)

dnnFinalRanking = dnnData[["Normalization", "R^2", "MSE", "RMSE", "Pearson Corr"]].copy()
dnnFinalRanking = dnnFinalRanking.groupby("Normalization", as_index=False).agg(desiredAgg)
dnnFinalRanking["Model"] = "DNN"

wnnFinalRanking = wnnData[["Normalization", "R^2", "MSE", "RMSE", "Pearson Corr"]].copy()
wnnFinalRanking = wnnFinalRanking.groupby("Normalization", as_index=False).agg(desiredAgg)
wnnFinalRanking["Model"] = "WNN"

dwnnFinalRanking = dwnnData[["Normalization", "R^2", "MSE", "RMSE", "Pearson Corr"]].copy()
dwnnFinalRanking = dwnnFinalRanking.groupby("Normalization", as_index=False).agg(desiredAgg)
dwnnFinalRanking["Model"] = "DWNN"

finalRanking = pd.concat([
    knnFinalRanking,
    svrFinalRanking,
    linearFinalRanking,
    lstmFinalRanking,
    dnnFinalRanking,
    wnnFinalRanking,
    dwnnFinalRanking
], ignore_index=True)
finalRanking = finalRanking[['Model'] + [c for c in finalRanking if c not in ['Model']]]

# finalRanking = finalRanking[finalRanking["Model"]]
print(finalRanking)

metrics = ["R^2", "RMSE", "Pearson Corr"]
# finalRanking["R^2"] = np.log(finalRanking["R^2"])
# finalRanking["RMSE"] = np.log(finalRanking["RMSE"])

for metric in metrics:
    values = np.array(finalRanking[metric])
    if metric == "RMSE":
        finalRanking.sort_values(by=metric, ascending=True, inplace=True)
    else:
        finalRanking.sort_values(by=metric, ascending=False, inplace=True)

    clrs = ['red']
    for i in range(0, len(values) - 1):
        clrs.append('grey')

    metricName = metric
    # if metric != "Pearson Corr":
    #     metricName = "log(" + metric + ")"

    sns.catplot(x="Model",
                y=metric,
                kind="swarm",
                hue="Normalization",
                data=finalRanking).set_xticklabels(rotation=45)
    plt.title("Final Ranking " + metricName + " X Model, Hue: Normalization")
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.15)
    plt.savefig(analysis_folder + "FinalRegressorRanking." + metric + "XModelHNormalization.png")
    plt.show()

    sns.catplot(x="Model",
                y=metric,
                kind="bar",
                data=finalRanking,
                palette=clrs).set_xticklabels(rotation=45)
    plt.title("Final Ranking " + metricName + " X Model")
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.18)
    plt.savefig(analysis_folder + "FinalRegressorRanking." + metric + "XModel.png")
    plt.show()

plt.close("all")

finalRanking.to_csv(analysis_folder + "FinalRanking.csv")
print("Best R^2")
print(finalRanking.iloc[finalRanking['R^2'].argmax()])
print("Best RMSE")
print(finalRanking.iloc[finalRanking['RMSE'].argmin()])
print("Best Pearson Correlation")
print(finalRanking.iloc[finalRanking['Pearson Corr'].argmax()])
