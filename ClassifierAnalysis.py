import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

plt.close("all")

log_folder = "logs" + os.path.sep + "classifier" + os.path.sep
analysis_folder = "analysis" + os.path.sep
knnData = pd.read_csv(log_folder + 'knn.new_data_path.log.csv')
svmData = pd.read_csv(log_folder + 'svm.new_data_path.log.csv')
rccData = pd.read_csv(log_folder + 'rcc.new_data_path.log.csv')
lstmData = pd.read_csv(log_folder + 'lstm.new_data_path.log.csv')
dnnData = pd.read_csv(log_folder + 'dnn.new_data_path.log.csv')
wnnData = pd.read_csv(log_folder + 'wnn.new_data_path.log.csv')
dwnnData = pd.read_csv(log_folder + 'dwnn.new_data_path.log.csv')

knnNeighborsData = pd.read_csv(log_folder + "inner_knn_log" + os.path.sep +
                               "ROC+BUF+SYR+ALB.ZScore.None.csv")

knnCatPlots = ["Source",
               "Normalization",
               "Random State"]
svmCatPlots = ["Source",
               "Normalization",
               "Kernel",
               "Random State"]
lstmCatPlots = ["Source",
                "Normalization",
                "Sequence Length"]
dnnCatPlots = ["Source",
               "Normalization",
               "Layers",
               "Random State"]
wnnCatPlots = ["Source",
               "Normalization",
               "Random State"]
rccCatPlots = ["Source",
               "Normalization",
               "Reservoir Size",
               "Random State"]

for i in knnCatPlots:
    for j in knnCatPlots:
        np.random.seed(42)
        if i == j:
            continue
        sns.catplot(x=i,
                    y="Accuracy",
                    kind="swarm",
                    hue=j,
                    data=knnData)
        plt.title("KNN Accuracy X " + i + ", Hue: " + j)
        plt.subplots_adjust(bottom=0.1, top=0.9)
        plt.savefig(analysis_folder + "KNN.AccuracyX" + i + "H" + j + ".png")
        # plt.show()
        plt.close("all")

for i in knnCatPlots:
    np.random.seed(42)
    sns.scatterplot(x="Best N_Neighbors",
                    y="Accuracy",
                    hue=i,
                    data=knnData)
    plt.title("KNN Accuracy X Best N_Neighbors, Hue: " + i)
    plt.subplots_adjust(bottom=0.1, top=0.9)
    plt.savefig(analysis_folder + "KNN.AccuracyXNNeighborsH" + i + ".png")
    # plt.show()
    plt.close("all")

var_labels = ["Source", "Normalization"]
knnData = knnData.groupby(var_labels, as_index=False).agg({"Accuracy": "max"})

sns.catplot(x="Source",
            y="Accuracy",
            kind="bar",
            hue="Normalization",
            data=knnData).set(ylim=(0.93, 0.97))  # .set_xticklabels(rotation=45)
plt.title("KNN Accuracy X Source, Hue: Normalization")
plt.subplots_adjust(bottom=0.2, top=0.9)
plt.savefig("KNNRanking.AccuracyXSourceHNormalization.png")
# plt.show()

knnData.sort_values(by="Accuracy", ascending=False, inplace=True)

for vl in var_labels:
    labels = np.array(knnData[vl])
    values = np.array(knnData.Accuracy)
    clrs = ['grey' if (x < max(values)) else 'red' for x in values]

    sns.catplot(x=vl,
                y="Accuracy",
                kind="bar",
                data=knnData,
                palette=clrs).set(ylim=(0.93, 1.00))  # .set_xticklabels(rotation=45)
    plt.title("KNN Ranking Accuracy X " + str(vl))
    plt.subplots_adjust(bottom=0.2, top=0.9)
    plt.savefig("KNNRanking.AccuracyX"+str(vl)+".png")
    # plt.show()

sns.lineplot(x="NNeighbors",
             y="Accuracy",
             data=knnNeighborsData)
plt.title("Best NNeighbors Search for ROC+BUF+SYR+ALB.ZScore.None")
plt.savefig("BestKNN.AccuracyXNNeighbors.png")
plt.show()

sns.lineplot(x="NNeighbors",
             y="Process Runtime",
             data=knnNeighborsData)
plt.title("Best NNeighbors Search for ROC+BUF+SYR+ALB.ZScore.None")
plt.savefig("BestKNN.ProcessRuntimeXNNeighbors.png")
plt.show()

for i in svmCatPlots:
    for j in svmCatPlots:
        np.random.seed(42)
        if i == j:
            continue
        sns.catplot(x=i,
                    y="Accuracy",
                    kind="strip",
                    hue=j,
                    data=svmData,
                    legend_out=True)
        plt.title("SVM Accuracy X " + i + ", Hue: " + j)
        plt.subplots_adjust(bottom=0.1, top=0.9)
        plt.savefig(analysis_folder + "SVM.AccuracyX" + i + "H" + j + ".png")
        # plt.show()
        plt.close('all')

for i in rccCatPlots:
    for j in rccCatPlots:
        np.random.seed(42)
        if i == j:
            continue
        sns.catplot(x=i,
                    y="Accuracy",
                    kind="strip",
                    hue=j,
                    data=rccData,
                    legend_out=True)
        plt.title("RCC Accuracy X " + i + ", Hue: " + j)
        plt.subplots_adjust(bottom=0.1, top=0.9)
        plt.savefig(analysis_folder + "RCC.AccuracyX" + i + "H" + j + ".png")
        # plt.show()
        plt.close('all')

for i in lstmCatPlots:
    for j in lstmCatPlots:
        np.random.seed(42)
        if i == j:
            continue
        sns.catplot(x=i,
                    y="Accuracy",
                    kind="strip",
                    hue=j,
                    data=lstmData)
        plt.title("LSTM Accuracy X " + i + ", Hue: " + j)
        plt.subplots_adjust(bottom=0.1, top=0.9)
        plt.savefig(analysis_folder + "LSTM.AccuracyX" + i + "H" + j + ".png")
        # plt.show()
        plt.close('all')

for i in dnnCatPlots:
    for j in dnnCatPlots:
        np.random.seed(42)
        if i == j:
            continue
        sns.catplot(x=i,
                    y="Accuracy",
                    kind="strip",
                    hue=j,
                    data=dnnData,
                    legend_out=True)
        plt.title("DNN Accuracy X " + i + ", Hue: " + j)
        plt.subplots_adjust(bottom=0.1, top=0.9)
        plt.savefig(analysis_folder + "DNN.AccuracyX" + i + "H" + j + ".png")
        # plt.show()
        plt.close('all')

for i in wnnCatPlots:
    for j in wnnCatPlots:
        np.random.seed(42)
        if i == j:
            continue
        sns.catplot(x=i,
                    y="Accuracy",
                    kind="strip",
                    hue=j,
                    data=wnnData,
                    legend_out=True)
        plt.title("WNN Accuracy X " + i + ", Hue: " + j)
        plt.subplots_adjust(bottom=0.1, top=0.9)
        plt.savefig(analysis_folder + "WNN.AccuracyX" + i + "H" + j + ".png")
        # plt.show()
        plt.close('all')

for i in dnnCatPlots:
    for j in dnnCatPlots:
        np.random.seed(42)
        if i == j:
            continue
        sns.catplot(x=i,
                    y="Accuracy",
                    kind="strip",
                    hue=j,
                    data=dwnnData,
                    legend_out=True)
        plt.title("DWNN Accuracy X " + i + ", Hue: " + j)
        plt.subplots_adjust(bottom=0.1, top=0.9)
        plt.savefig(analysis_folder + "DWNN.AccuracyX" + i + "H" + j + ".png")
        # plt.show()
        plt.close('all')

knnFinalRanking = knnData[["Source", "Normalization", "Accuracy"]].copy()
knnFinalRanking = knnFinalRanking.groupby(["Source", "Normalization"], as_index=False).agg({'Accuracy': 'max'})
knnFinalRanking["AI"] = "KNN"

svmFinalRanking = svmData[["Source", "Normalization", "Accuracy"]].copy()
svmFinalRanking = svmFinalRanking.groupby(["Source", "Normalization"], as_index=False).agg({'Accuracy': 'max'})
svmFinalRanking["AI"] = "SVM"

rccFinalRanking = rccData[["Source", "Normalization", "Accuracy"]].copy()
rccFinalRanking = rccFinalRanking.groupby(["Source", "Normalization"], as_index=False).agg({'Accuracy': 'max'})
rccFinalRanking["AI"] = "RCC"

lstmFinalRanking = lstmData[["Source", "Normalization", "Accuracy"]].copy()
lstmFinalRanking = lstmFinalRanking.groupby(["Source", "Normalization"], as_index=False).agg({'Accuracy': 'max'})
lstmFinalRanking["AI"] = "LSTM"

dnnFinalRanking = dnnData[["Source", "Normalization", "Accuracy"]].copy()
dnnFinalRanking = dnnFinalRanking.groupby(["Source", "Normalization"], as_index=False).agg({'Accuracy': 'max'})
dnnFinalRanking["AI"] = "DNN"

wnnFinalRanking = wnnData[["Source", "Normalization", "Accuracy"]].copy()
wnnFinalRanking = wnnFinalRanking.groupby(["Source", "Normalization"], as_index=False).agg({'Accuracy': 'max'})
wnnFinalRanking["AI"] = "WNN"

dwnnFinalRanking = dwnnData[["Source", "Normalization", "Accuracy"]].copy()
dwnnFinalRanking = dwnnFinalRanking.groupby(["Source", "Normalization"], as_index=False).agg({'Accuracy': 'max'})
dwnnFinalRanking["AI"] = "DWNN"

finalRanking = pd.concat([
    knnFinalRanking,
    svmFinalRanking,
    rccFinalRanking,
    lstmFinalRanking,
    dnnFinalRanking,
    wnnFinalRanking,
    dwnnFinalRanking
], ignore_index=True)

finalRankingMixed = finalRanking[finalRanking["Source"] == "ROC+BUF+SYR+ALB"].copy()
finalRankingMixed["Source"] = "Mixed"

finalRankingROC = finalRanking[finalRanking["Source"] == "ROC"].copy()

rankings = {
    "Mixed": finalRankingMixed,
    "ROC": finalRankingROC
}

for source, ranking in rankings.items():
    sns.catplot(x="AI",
                y="Accuracy",
                kind="bar",
                hue="Normalization",
                data=ranking).set(ylim=(0.935, 0.965))
    plt.title(source + " Final Ranking Accuracy X AI, Hue: Normalization")
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.15)
    plt.savefig(analysis_folder + source + "FinalRanking.AccuracyXAIHNormalization.png")
    # plt.show()

    sns.catplot(x="Normalization",
                y="Accuracy",
                kind="swarm",
                hue="AI",
                data=ranking)
    plt.title(source + " Final Ranking Accuracy X Normalization, Hue: AI")
    plt.subplots_adjust(bottom=0.1, top=0.9)
    plt.savefig(analysis_folder + source + "FinalRanking.AccuracyXNormalizationHAI.png")
    # plt.show()

    ranking = ranking.groupby("AI", as_index=False).agg({"Accuracy": "max"})

    ranking.sort_values(by="Accuracy", ascending=False, inplace=True)

    labels = np.array(ranking.AI)
    values = np.array(ranking.Accuracy)
    clrs = ['grey' if (x < max(values)) else 'red' for x in values]

    sns.catplot(x="AI",
                y="Accuracy",
                kind="bar",
                data=ranking,
                palette=clrs).set(ylim=(0.945, 0.965))
    plt.title(source + " Final Ranking Accuracy X AI")
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.18)
    plt.savefig(analysis_folder + source + "FinalRanking.AccuracyXAI.png")
    # plt.show()


plt.close("all")
