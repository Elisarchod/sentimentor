import numpy as np
import pandas as pd
from typing import List, Dict, Union
import plotly.express as px
import plotly.graph_objects as go
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score

stopwords.words('english')


def load_data(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(f'{file_name}.csv', index_col="ID")
    print(df.shape)
    df = df[df.sum().sort_values(ascending=False).index]
    df = df.loc[df.sum(axis=1).sort_values(ascending=False).index]
    return df


def drop_stop_words(df: pd.DataFrame) -> pd.DataFrame:
    words_to_drop = [col for col in df.columns if col in stopwords.words('english')]
    df.drop(words_to_drop, axis=1, inplace=True)
    print(df.shape)
    return df


def get_correlation_pairs(tfidf: pd.DataFrame, corr_type: str) -> pd.DataFrame:
    correlations = tfidf.corr(corr_type)
    correlations_stacked = correlations.abs().unstack().sort_values(ascending=False)
    correlations_stacked = correlations_stacked[correlations_stacked < 1]
    return correlations_stacked


def combine_pairs(X: pd.DataFrame, strong_corr_pairs: List[str]) -> pd.DataFrame:
    droped = []
    for pair in strong_corr_pairs:
        pair_splited = pair.split("_")
        if all([False if word in droped else True for word in pair_splited]):
            X[pair] = X[pair_splited].mean(axis=1)
            X.drop(pair_splited, axis=1, inplace=True)
            droped.extend(pair_splited)
    print(X.shape)
    return X


def calc_strong_pairs(tfidf_sk: pd.DataFrame, threshold: float, plot: bool = False):
    correlations_stacked = get_correlation_pairs(
        tfidf_sk, "pearson").to_frame().rename(columns={0: "pearson"}).join(
        get_correlation_pairs(tfidf_sk, "spearman").to_frame().rename(columns={0: "spearman"})
    )
    correlations_stacked = correlations_stacked.iloc[range(0, len(correlations_stacked), 2)]
    correlations_stacked.index = ["_".join(pair) for pair in correlations_stacked.index.to_list()]

    # strong_corr = []
    # for cor in ["pearson", "spearman"]:
    #     strong_corr.extend(
    #         correlations_stacked[cor].sort_values(ascending=False)
    #     .head(threshold)
    #     .index.to_list()
    #     )

    strong_corr_pairs = correlations_stacked.query("pearson > @threshold").index.to_list()
    strong_corr = strong_corr_pairs

    if plot:
        correlations_stacked_plot = correlations_stacked.loc[strong_corr].reset_index().sort_values("pearson")
        fig = go.Figure()
        for cor in ["pearson", "spearman"]:
            fig.add_trace(
                go.Scatter(
                    y=correlations_stacked_plot["index"],
                    x=correlations_stacked_plot[cor],
                    mode='markers',
                    text=correlations_stacked_plot[["pearson", "spearman"]],
                    name=cor)
            )
        fig.show()
    return strong_corr_pairs, correlations_stacked


def plot_low_tfidf_values(tfidf_sk: pd.DataFrame, X: pd.DataFrame, y: pd.Series, plot: bool=False) -> pd.DataFrame:
    tfidf_sk_plot = (tfidf_sk
        .mean(axis=0).to_frame().rename(columns={0: "tfidf"})
        .join(
        X.replace({0: None})
            .notnull().sum(axis=0)
            .to_frame()
            .rename(columns={0: "count_review"}))
        .sort_values("tfidf", ascending=False)
        .join(
        X
            .join(y.to_frame())
            .groupby("rating", as_index=False)
            .mean().T[1]
            .to_frame()
            .rename(
            columns={1: "rating_freq"}
        )
    )).reset_index().sort_values("tfidf", ascending=False)

    tfidf_sk_plot["rating_freq_log"] = np.log(tfidf_sk_plot["rating_freq"])
    tfidf_sk_plot["rating_freq_log_abs"] = tfidf_sk_plot["rating_freq_log"].abs()
    tfidf_sk_plot.loc[tfidf_sk_plot["rating_freq_log"] == -np.inf, "rating_freq_log_abs"] = 0
    if plot:
        fig = px.scatter(
            tfidf_sk_plot, x="tfidf", y="index", color="rating_freq_log_abs",
            hover_data=["count_review", "rating_freq_log_abs"]
        )
        fig.show()

    return tfidf_sk_plot


def calc_low_label_diff(X: pd.DataFrame, y: pd.Series, threshold: int, plot: bool=False) -> pd.DataFrame:
    data = X.join(y)
    label_freq = data.groupby(["rating"]).mean().T
    label_freq["diff"] = label_freq[1].sub(label_freq[0]).abs()
    label_freq = label_freq.sort_values("diff", ascending=False).reset_index()
    if plot:
        fig = px.scatter(
            label_freq,
            x="diff",
            y="index"
        )
        fig.show()
    return label_freq[label_freq["diff"] > threshold]["index"], label_freq


def run_transformer(X: pd.DataFrame, transformer = None) -> Union[TfidfTransformer, pd.DataFrame]:
    if transformer is None:
        transformer = TfidfTransformer()
        transformer = transformer.fit(X)
    tfidf_sk = pd.DataFrame(
        transformer.transform(X).toarray(),
        columns=X.columns,
        index=X.index
    )
    return transformer, tfidf_sk


def plot_prediction_distribution(clf, X, y, threshold: float):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=clf.predict_proba(X[y==1])[:, 1], name="positive"))
    fig.add_trace(go.Histogram(x=clf.predict_proba(X[y==0])[:, 1], name="negetive"))

    fig.add_trace(
        go.Scatter(
            x=[threshold, threshold],
            y= [0, 50],
            mode='lines', name='lines')
    )

    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    print(f"Acuuracy score {accuracy_score(y, clf.predict_proba(X)[:, 1] > threshold)}")
    fig.show()
