import os
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

# Constants

TRAIN_FILE = "train.csv"
VAL_FILE = "val.csv"
UNLABELED = "unlabeled_v2.csv"
TRAIN_REPORT_PATH = "reports/train_planets_report.html"
VAL_REPORT_PATH = "reports/val_planets_report.html"
UNLABELED_REPORT_PATH = "reports/unlabeled_planets_report.html"

# Features
OBJID = "objid"
RA = "ra"
DEC = "dec"
CLEAN = "clean"
ROWC = "rowc"
colc = "colc"
CLASS = "class"

train = pd.read_csv(TRAIN_FILE)
val = pd.read_csv(VAL_FILE)


def nan_filler(df: pd.DataFrame, old_nan):
    df.replace(old_nan, np.nan, inplace=True)



nan_filler(train, "na")
nan_filler(val, "na")
unlabeled = pd.read_csv(UNLABELED)
nan_filler(unlabeled, "na")
train = train.astype(np.float32)
val = val.astype(np.float32)
unlabeled = unlabeled.astype(np.float32)

if not os.path.exists(TRAIN_REPORT_PATH):
    report = ProfileReport(train, title='Train Planet Report', explorative=True)
    report.to_file(TRAIN_REPORT_PATH)

if not os.path.exists(VAL_REPORT_PATH):
    report = ProfileReport(val, title='Validation Planet Report', explorative=True)
    report.to_file(VAL_REPORT_PATH)

if not os.path.exists(UNLABELED_REPORT_PATH):
    report = ProfileReport(unlabeled, title='Unlabeled Planet Report', explorative=True)
    report.to_file(UNLABELED_REPORT_PATH)
