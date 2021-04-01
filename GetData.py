from urllib.request import urlopen
import re
import os
import pandas as pd
import numpy as np
from io import StringIO
import datetime

pd.set_option("display.max_columns", None, "display.min_rows", 20)
raw_data_folder = "data" + os.path.sep + "raw_data" + os.path.sep
regressor_data_folder = "data" + os.path.sep + "regressor_data" + os.path.sep
classifier_data_folder = "data" + os.path.sep + "classifier_data" + os.path.sep

baseURL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
settings = "latlon=no&elev=no&trace=empty&direct=no&report_type=1&report_type=2"
dataDesired = "&data=p01i&data=tmpf&data=dwpf&data=relh&data=drct&data=sknt&data=alti&data=mslp&data=vsby"
timezone = "&tz=America%2FNew_York"
format = "&format=onlycomma"
missing = "&missing=empty"
startts = datetime.datetime(2010, 1, 1)
endts = datetime.datetime(2020, 12, 31)
start = startts.strftime("&year1=%Y&month1=%m&day1=%d&")
end = endts.strftime("&year2=%Y&month2=%m&day2=%d&")

station = "&station=ALB&station=SCH&station=BUF&station=IAG&station=SYR&station=FZY&station=ROC"
station = "&station=ROC"
station = "&station=BUF&station=IAG"
station = "&station=ALB&station=SCH"
station = "&station=SYR&station=FZY"

stations = {
    'Albany2Station': '&station=ALB&station=SCH',
    'Buffalo2Station': '&station=BUF&station=IAG',
    'Syracuse2Station': '&station=SYR&station=FZY',
    'Rochester1Station': '&station=ROC'
}

for filePrefix, station in stations.items():
    builtURL = baseURL + settings + station + dataDesired + timezone + format + missing + start + end

    rawData = urlopen(builtURL, timeout=300).read().decode("utf-8")
    data = pd.read_csv(StringIO(rawData))
    data.to_csv(raw_data_folder + filePrefix + "HourlyRawData.csv", index=False)

    data.loc[data.vsby < 0, 'vsby'] = np.NaN
    data['interval_id'] = (pd.to_datetime(data['valid'], format='%Y-%m-%d %H:%M') - pd.Timestamp(
        "2010-01-01")) // pd.Timedelta('1h')
    data.drop(columns=['valid'], inplace=True)
    data = data.groupby(['interval_id', 'station']).agg({'p01i': 'max', 'tmpf': 'mean',
                                                         'dwpf': 'mean', 'relh': 'mean',
                                                         'drct': 'mean', 'sknt': 'mean',
                                                         'alti': 'mean', 'mslp': 'mean',
                                                         'vsby': 'mean'})
    data = data.groupby('interval_id').agg({'p01i': 'mean', 'tmpf': 'mean',
                                            'dwpf': 'mean', 'relh': 'mean',
                                            'drct': 'mean', 'sknt': 'mean',
                                            'alti': 'mean', 'mslp': 'mean',
                                            'vsby': 'mean'})
    data = data.reset_index()
    data.loc[data.p01i.isna(), 'p01i'] = 0.0
    data.interpolate(axis=1, inplace=True)
    data.to_csv(raw_data_folder + filePrefix + "HourlyAveraged.csv", index=False)

rocData = pd.read_csv(raw_data_folder + "Rochester1StationHourlyAveraged.csv")
rocData["interval_id"] = rocData["interval_id"].astype(np.int64)
rocData["rains_next_interval"] = np.nan
rocData["next_p01i"] = np.nan
for i in range(rocData.shape[0] - 1):
    currentIntervalID = int(rocData.loc[i, "interval_id"])
    nextIntervalID = int(rocData.loc[i + 1, "interval_id"])
    if currentIntervalID + 1 == nextIntervalID:
        rocData.loc[i, "next_p01i"] = rocData.loc[i + 1, "p01i"]
        rainNextInterval = float(rocData.loc[i + 1, "p01i"])
        if rainNextInterval > 0.01:
            rocData.loc[i, "rains_next_interval"] = 1
        else:
            rocData.loc[i, "rains_next_interval"] = 0

print(rocData.info())
rocData.dropna(inplace=True)
rocData["rains_next_interval"] = rocData["rains_next_interval"].astype(np.int64)

roc_classifier_data = rocData.drop(columns=['next_p01i']).copy()
roc_regressor_data = rocData.drop(columns=['rains_next_interval']).copy()

roc_classifier_data.to_csv(classifier_data_folder + "ROC.None.csv", index=False)
roc_regressor_data.to_csv(regressor_data_folder + "ROC.None.csv", index=False)

bufData = pd.read_csv(raw_data_folder + "Buffalo2StationHourlyAveraged.csv")
bufData["interval_id"] = bufData["interval_id"].astype(np.int64)
syrData = pd.read_csv(raw_data_folder + "Syracuse2StationHourlyAveraged.csv")
syrData["interval_id"] = syrData["interval_id"].astype(np.int64)
albData = pd.read_csv(raw_data_folder + "Albany2StationHourlyAveraged.csv")
albData["interval_id"] = albData["interval_id"].astype(np.int64)
albData.columns = albData.columns.map(lambda x: x if x == 'interval_id' else str(x) + '_alb')

bufSyrData = pd.merge(syrData, bufData, on="interval_id", suffixes=("_syr", "_buf"), how='outer')
albBufSyrData = pd.merge(bufSyrData, albData, on="interval_id", suffixes=(None, None), how='outer')

all_classifier_data = pd.merge(roc_classifier_data, albBufSyrData, on="interval_id", suffixes=(None, None), how='left')
all_regressor_data = pd.merge(roc_regressor_data, albBufSyrData, on="interval_id", suffixes=(None, None), how='left')

all_classifier_data.interpolate(inplace=True)
all_regressor_data.interpolate(inplace=True)
all_classifier_data.dropna(inplace=True)
all_regressor_data.dropna(inplace=True)

all_classifier_data = all_classifier_data[[c for c in all_classifier_data if c not in ['rains_next_interval']] + ['rains_next_interval']]
all_regressor_data = all_regressor_data[[c for c in all_regressor_data if c not in ['next_p01i']] + ['next_p01i']]

all_classifier_data.to_csv(classifier_data_folder + "ROC+BUF+SYR+ALB.None.csv", index=False)
all_regressor_data.to_csv(regressor_data_folder + "ROC+BUF+SYR+ALB.None.csv", index=False)

roc_mm_classifier_data = roc_classifier_data.copy()
roc_zs_classifier_data = roc_classifier_data.copy()
roc_mm_regressor_data = roc_regressor_data.copy()
roc_zs_regressor_data = roc_regressor_data.copy()

for col in rocData.columns:
    if col != 'interval_id' and col != 'next_p01i' and col != 'rains_next_interval':
        roc_mm_classifier_data[col] = (roc_mm_classifier_data[col] - roc_mm_classifier_data[col].min()) / (roc_mm_classifier_data[col].max() - roc_mm_classifier_data[col].min())
        roc_zs_classifier_data[col] = ((roc_zs_classifier_data[col] - roc_zs_classifier_data[col].mean()) / roc_zs_classifier_data[col].std())
        roc_mm_regressor_data[col] = (roc_mm_regressor_data[col] - roc_mm_regressor_data[col].min()) / (roc_mm_regressor_data[col].max() - roc_mm_regressor_data[col].min())
        roc_zs_regressor_data[col] = ((roc_zs_regressor_data[col] - roc_zs_regressor_data[col].mean()) / roc_zs_regressor_data[col].std())

roc_mm_classifier_data.to_csv(classifier_data_folder + "ROC.MinMax.csv", index=False)
roc_zs_classifier_data.to_csv(classifier_data_folder + "ROC.ZScore.csv", index=False)
roc_mm_regressor_data.to_csv(regressor_data_folder + "ROC.MinMax.csv", index=False)
roc_zs_regressor_data.to_csv(regressor_data_folder + "ROC.ZScore.csv", index=False)

all_mm_regressor_data = all_regressor_data.copy()
all_zs_regressor_data = all_regressor_data.copy()
all_mm_classifier_data = all_classifier_data.copy()
all_zs_classifier_data = all_classifier_data.copy()

for col in rocData.columns:
    if col != 'interval_id' and col != 'next_p01i' and col != 'rains_next_interval':
        all_mm_regressor_data[col] = (all_mm_regressor_data[col] - all_mm_regressor_data[col].min()) / (all_mm_regressor_data[col].max() - all_mm_regressor_data[col].min())
        all_zs_regressor_data[col] = ((all_zs_regressor_data[col] - all_zs_regressor_data[col].mean()) / all_zs_regressor_data[col].std())
        all_mm_classifier_data[col] = (all_mm_classifier_data[col] - all_mm_classifier_data[col].min()) / (all_mm_classifier_data[col].max() - all_mm_classifier_data[col].min())
        all_zs_classifier_data[col] = ((all_zs_classifier_data[col] - all_zs_classifier_data[col].mean()) / all_zs_classifier_data[col].std())

all_mm_regressor_data.to_csv(regressor_data_folder + "ROC+BUF+SYR+ALB.MinMax.csv", index=False)
all_zs_regressor_data.to_csv(regressor_data_folder + "ROC+BUF+SYR+ALB.ZScore.csv", index=False)
all_mm_classifier_data.to_csv(classifier_data_folder + "ROC+BUF+SYR+ALB.MinMax.csv", index=False)
all_zs_classifier_data.to_csv(classifier_data_folder + "ROC+BUF+SYR+ALB.ZScore.csv", index=False)
