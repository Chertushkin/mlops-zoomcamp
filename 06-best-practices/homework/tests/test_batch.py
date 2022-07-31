from datetime import datetime
import pandas as pd
import numpy as np

# from ..batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


def prepare_data(df, categorical):
    df["duration"] = df.dropOff_datetime - df.pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


def test_prepare_data():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    columns = ["PUlocationID", "DOlocationID", "pickup_datetime", "dropOff_datetime"]
    categorical = ["PUlocationID", "DOlocationID"]
    df = pd.DataFrame(data, columns=columns)

    expected_data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
    ]
    columns = ["PUlocationID", "DOlocationID", "pickup_datetime", "dropOff_datetime"]
    categorical = ["PUlocationID", "DOlocationID"]
    expected_df = pd.DataFrame(expected_data, columns=columns)

    new_df = prepare_data(df, categorical)
    new_expected_df = prepare_data(expected_df, categorical)
    assert np.all(new_df == new_expected_df)

    year = 2021
    month = 1
    output_file = f"s3://nyc-duration-prediction-misha/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet"
    new_df.to_parquet(
        output_file,
        engine="pyarrow",
        compression=None,
        index=False
    )
    print('here')

    


test_prepare_data()