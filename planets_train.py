import os
import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


OBJID = "objid"
RA = "ra"
DEC = "dec"
CLEAN = "clean"
ROWC = "rowc"
colc = "colc"
CLASS = "class"

rfc_params = {
    "n_estimators": 10,
    "criterion": "gini",
    "min_samples_split": 10,
    "max_features": "auto",
    "bootstrap": True,
    "n_jobs": -1
}


def nan_filler(df: pd.DataFrame, old_nan):
    df.replace(old_nan, np.nan, inplace=True)


def fill_numeric_knn(df: pd.DataFrame, scaler, imputer_params: dict = {"n_neighbors": 5, "metric": "nan_euclidean", "weights": "uniform"})->pd.DataFrame:
    """
    """
    # Scaling
    columns = df.columns
    scl = scaler()
    df = scl.fit_transform(df)
    knn_imputer = KNNImputer(**imputer_params)
    transformed_df = knn_imputer.fit_transform(df)
    out_df = pd.DataFrame(transformed_df)
    out_df.columns = columns
    return out_df


def fill_median(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    """
    return df[columns].fillna(df[columns].median())


def preprocess_train_dataset(file, target_column, drop_columns=None):
    df = pd.read_csv(file)
    y = df[target_column]
    df.drop(target_column, axis=1, inplace=True)
    if drop_columns:
        df.drop(drop_columns, inplace=True, axis=1)
    nan_filler(df, old_nan="na")
    df = fill_median(df, columns=df.columns)
    return df, y


def preprocess_unlabeled_dataset(file, drop_columns=None):
    df = pd.read_csv(file)
    if drop_columns:
        df.drop(drop_columns, inplace=True, axis=1)
    nan_filler(df, old_nan="na")
    df = fill_median(df, columns=df.columns)
    return df


def train_model(model, train_set, y_train):
    model.fit(train_set, y_train)


def evaluate_model(model, test_set):
    return model.predict(test_set)


def main():
    # reading arguments
    args = sys.argv[1:]
    train_file = args[0]
    unlabeled_file = args[1]
    test_file = args[2]
    results_file = args[3]
    x_train, y_train = preprocess_train_dataset(file=train_file, target_column=CLASS, drop_columns=OBJID)
    pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(**rfc_params))
    train_model(model=pipeline, train_set=x_train, y_train=y_train)
    x_test = preprocess_unlabeled_dataset(file=test_file)
    objid = x_test[OBJID]
    x_test.drop(OBJID, axis=1, inplace=True)
    predictions = evaluate_model(pipeline, x_test)
    out_df = pd.DataFrame({"objid": objid, "predictions": predictions})
    out_df.to_csv(results_file, index=False)


if __name__ == "__main__":
    main()

