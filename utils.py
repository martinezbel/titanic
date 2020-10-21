from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]


def load_data():
    df = pd.read_csv("train.csv")
    df.drop(["PassengerId"], axis=1, inplace=True)
    df.drop(["Name"], axis=1, inplace=True)
    df.drop(["Ticket"], axis=1, inplace=True)
    df.drop(["Cabin"], axis=1, inplace=True)
    df.fillna(df.mean(), inplace=True)
#dummies, Convert categorical variable into dummy/indicator variables.
    df = pd.concat(
        [df, pd.get_dummies(df["Sex"].astype("category"), prefix="sex")], axis=1
    )
    df = pd.concat(
        [df, pd.get_dummies(df["Embarked"].astype("category"), prefix="embarked")],
        axis=1,
    )
    df.drop(["Sex"], axis=1, inplace=True)
    df.drop(["Embarked"], axis=1, inplace=True)

    return df


def read_markdown_file(path):
    return Path(path).read_text()


def train_rf(df, n_estimators=100, max_depth=3):
    target = "Survived"
    features = [c for c in df.columns.values if c != target]
    X = df[features]
    y = df[target].astype("category")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(X_train, y_train)

    fig, ax = plt.subplots()
    plot_confusion_matrix(clf, X_test, y_test, ax=ax)

    return clf, fig