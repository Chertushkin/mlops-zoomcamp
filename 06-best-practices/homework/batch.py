#!/usr/bin/env python
# coding: utf-8

import pickle
import sys
import typing as tp

import pandas as pd
import numpy as np


def get_model():
    with open("model.bin", "rb") as f_in:
        dv, lr = pickle.load(f_in)
    return dv, lr

def prepare_data(df, categorical):
    df["duration"] = df.dropOff_datetime - df.pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    return prepare_data(df, categorical)



def main(year: int, month: int, categorical: tp.List):
    # input_file = f"https://raw.githubusercontent.com/alexeygrigorev/datasets/master/nyc-tlc/fhv/fhv_tripdata_{year:04d}-{month:02d}.parquet"
    
    input_file = f"s3://nyc-duration-prediction-misha/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet"
    output_file = f"s3://nyc-duration-prediction-misha/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions_new.parquet"

    df = read_data(input_file, categorical)
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    dv, lr = get_model()
    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print("predicted mean duration:", y_pred.mean())
    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["predicted_duration"] = y_pred
    df_result.to_parquet(output_file, engine="pyarrow", index=False)
    print(np.sum(df_result['predicted_duration']))


def get_input_params():
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    categorical = ["PUlocationID", "DOlocationID"]

    return year, month, categorical


if __name__ == "__main__":
    year, month, categorical = get_input_params()
    main(year, month, categorical)
