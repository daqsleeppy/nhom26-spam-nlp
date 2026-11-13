"""Core experiment from Nhom26_NLP_CuoiKy_NangCap.ipynb.

All model fitting and tuning occur on train only. Test is held out until
hyperparameters/model choice are decided through training CV.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
DATA_URL = (
    "https://raw.githubusercontent.com/trannguyenthaituan251209/"
    "vietnamese_sms_dataset/main/full_dataset.csv"
)
MODEL_NAMES = [
    "TF-IDF + Naive Bayes (baseline)",
    "TF-IDF + Logistic Regression",
    "TF-IDF + Linear SVM",
    "Character n-gram + Linear SVM",
]


def prepare_dataset(raw: pd.DataFrame):
    """Mirror the notebook's original cleaning and return stats."""
    if not {"message", "label"}.issubset(raw.columns):
        raise ValueError("CSV phải có hai cột tên chính xác là 'message' và 'label'.")
    orig = len(raw)
    df = raw[["message", "label"]].copy()
    df = df.dropna(subset=["message", "label"]).copy()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"].ne("")].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    after_invalid = len(df)

    conflicts = df.groupby("message")["label"].nunique()
    conflict_msgs = set(conflicts[conflicts > 1].index)
    df = df[~df["message"].isin(conflict_msgs)].copy()
    after_conflict = len(df)
    df = df.drop_duplicates(subset=["message"], keep="first").reset_index(drop=True)

    count_by_class = df["label"].value_counts().to_dict()
    if df["label"].nunique() != 2 or min(count_by_class.values()) < 8 or len(df) < 30:
        raise ValueError("Cần >=30 tin nhắn và mỗi lớp 0/1 có ít nhất 8 mẫu sau khi làm sạch.")
    stats = {
        "raw": orig,
        "invalid_removed": orig - after_invalid,
        "conflict_messages": len(conflict_msgs),
        "conflict_rows_removed": after_invalid - after_conflict,
        "duplicates_removed": after_conflict - len(df),
        "cleaned": len(df),
        "ham": int(count_by_class.get(0, 0)),
        "spam": int(count_by_class.get(1, 0)),
    }
    return df, stats


def build_models() -> Dict[str, Pipeline]:
    def word_vectorizer():
        return TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), max_features=10000,
            sublinear_tf=True
        )

    def char_vectorizer():
        return TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=2,
            max_features=50000, sublinear_tf=True
        )

    return {
        MODEL_NAMES[0]: Pipeline([
            ("tfidf", word_vectorizer()),
            ("clf", MultinomialNB(alpha=0.5)),
        ]),
        MODEL_NAMES[1]: Pipeline([
            ("tfidf", word_vectorizer()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
        MODEL_NAMES[2]: Pipeline([
            ("tfidf", word_vectorizer()),
            ("clf", LinearSVC(
                C=1.0, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
        MODEL_NAMES[3]: Pipeline([
            ("tfidf", char_vectorizer()),
            ("clf", LinearSVC(
                C=1.0, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
    }


def model_metrics(y_true, y_pred):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision Spam": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall Spam": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1 Spam": float(f1_score(y_true, y_pred, zero_division=0)),
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_experiment(
    cleaned: pd.DataFrame,
    tune_best: bool = True,
    progress: Optional[Callable[[float, str], None]] = None,
):
    """Runs 5-fold train CV, selects model, optionally tunes and evaluates holdout."""
    if progress is None:
        progress = lambda fraction, label: None
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned["message"], cleaned["label"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=cleaned["label"]
    )
    if y_train.value_counts().min() < 5:
        raise ValueError("Tập train phải có ít nhất 5 mẫu mỗi lớp để chạy CV 5-fold.")
    if set(X_train).intersection(set(X_test)):
        raise ValueError("Rò rỉ dữ liệu: nội dung trùng giữa train và test.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "precision_spam": "precision",
               "recall_spam": "recall", "f1_spam": "f1"}
    models = build_models()
    cv_rows = []
    for i, (name, estimator) in enumerate(models.items()):
        progress(0.04 + 0.46*i/len(models), f"Cross-validation 5-fold: {name}")
        score = cross_validate(
            estimator, X_train, y_train, cv=cv, scoring=scoring,
            n_jobs=1, error_score="raise"
        )
        row = {"Mô hình": name}
        for metric in scoring:
            arr = score[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(arr))
            row[f"{metric}_std"] = float(np.std(arr))
        cv_rows.append(row)
    cv_table = pd.DataFrame(cv_rows).sort_values("f1_spam_mean", ascending=False).reset_index(drop=True)
    best_name = str(cv_table.iloc[0]["Mô hình"])
    progress(0.52, f"Mô hình được chọn bằng CV: {best_name}")

    fitted_models = {}
    test_rows = []
    preds = {}
    for i, (name, estimator) in enumerate(models.items()):
        progress(0.54 + 0.24*i/len(models), f"Huấn luyện mô hình đầy đủ: {name}")
        fitted = clone(estimator).fit(X_train, y_train)
        fitted_models[name] = fitted
        pred = np.asarray(fitted.predict(X_test), dtype=int)
        preds[name] = pred
        test_rows.append({"Mô hình": name, **model_metrics(y_test, pred)})

    final_model = fitted_models[best_name]
    best_params = {"ghi_chu": "Giữ siêu tham số mặc định trong notebook"}
    tuned_cv_f1 = None
    final_name = best_name
    if tune_best:
        progress(0.80, "GridSearchCV 5-fold cho mô hình đã chọn")
        clf = models[best_name].named_steps["clf"]
        grid_params = {"clf__alpha": [0.1, 0.5, 1.0]} if isinstance(clf, MultinomialNB) \
            else {"clf__C": [0.1, 1.0, 3.0]}
        grid = GridSearchCV(
            clone(models[best_name]), param_grid=grid_params, cv=cv,
            scoring="f1", n_jobs=1, refit=True, error_score="raise"
        )
        grid.fit(X_train, y_train)
        final_model = grid.best_estimator_
        best_params = grid.best_params_
        tuned_cv_f1 = float(grid.best_score_)
        final_name = "FINAL (CV + tuning)"
        pred = np.asarray(final_model.predict(X_test), dtype=int)
        preds[final_name] = pred
        test_rows.append({"Mô hình": final_name, **model_metrics(y_test, pred)})

    final_pred = np.asarray(final_model.predict(X_test), dtype=int)
    confusion = confusion_matrix(y_test, final_pred, labels=[0, 1])
    errors = pd.DataFrame({
        "message": X_test.reset_index(drop=True),
        "actual": y_test.reset_index(drop=True).astype(int),
        "predicted": pd.Series(final_pred).astype(int),
    })
    errors = errors.loc[errors["actual"] != errors["predicted"]].copy()
    errors["error_type"] = errors["actual"].map({1: "FN · Bỏ sót Spam", 0: "FP · Báo nhầm"})
    progress(1.0, "Đã đánh giá xong trên tập test cố định")
    return {
        "models": fitted_models,
        "final_model": final_model,
        "final_name": final_name,
        "best_name": best_name,
        "cv_table": cv_table,
        "test_table": pd.DataFrame(test_rows),
        "errors": errors,
        "confusion": confusion,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_label_0": int((y_test == 0).sum()),
        "test_label_1": int((y_test == 1).sum()),
        "best_params": best_params,
        "tuned_cv_f1": tuned_cv_f1,
        "tune_best": bool(tune_best),
    }


def get_all_models(exp):
    result = dict(exp["models"])
    if exp["tune_best"]:
        result["final_name"] = exp["final_model"]
    return result


def classify_one(estimator, message: str):
    if not message or not message.strip():
        raise ValueError("Tin nhắn không được để trống.")
    pred = int(estimator.predict([message])[0])
    clf = estimator.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        class_names = list(clf.classes_)
        prob = float(estimator.predict_proba([message])[0][class_names.index(1)])
        return {"label": pred, "score": prob, "score_type": "Xác suất Spam (chưa hiệu chuẩn)"}
    if hasattr(clf, "decision_function"):
        margin = float(np.ravel(estimator.decision_function([message]))[0])
        return {"label": pred, "score": margin, "score_type": "Margin SVM (không phải xác suất)"}
    return {"label": pred, "score": None, "score_type": "Không có điểm số"}


def matching_features(estimator, message: str, max_items: int = 12):
    """Token contributions for linear estimators; descriptive, not causal."""
    vec = estimator.named_steps["tfidf"]
    clf = estimator.named_steps["clf"]
    x = vec.transform([message]).tocsr()
    if isinstance(clf, MultinomialNB):
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    elif hasattr(clf, "coef_"):
        weights = clf.coef_[0]
    else:
        return pd.DataFrame(columns=["Đặc trưng", "Đóng góp vào điểm Spam"])
    active_idx = x.indices
    values = x.data * weights[active_idx]
    if len(values) == 0:
        return pd.DataFrame(columns=["Đặc trưng", "Đóng góp vào điểm Spam"])
    top_idx = np.argsort(-np.abs(values))[:max_items]
    tokens = vec.get_feature_names_out()
    return pd.DataFrame({
        "Đặc trưng": tokens[active_idx[top_idx]],
        "Đóng góp vào điểm Spam": values[top_idx].astype(float),
    })


def pack_results(exp, stats):
    """Downloads results and the group's own sklearn pickle models. No raw SMS included."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("nhom26_cv_results.csv", exp["cv_table"].to_csv(index=False).encode("utf-8-sig"))
        z.writestr("nhom26_test_results.csv", exp["test_table"].to_csv(index=False).encode("utf-8-sig"))
        z.writestr("nhom26_confusion_matrix.csv", pd.DataFrame(
            exp["confusion"], index=["Thật: bình thường", "Thật: Spam"],
            columns=["Dự đoán: bình thường", "Dự đoán: Spam"]
        ).to_csv().encode("utf-8-sig"))
        metadata = {
            "dataset_stats": stats,
            "train": exp["train_size"], "test": exp["test_size"],
            "random_state": RANDOM_STATE, "test_size": 0.2, "cv_folds": 5,
            "best_cv_model": exp["best_name"], "final_name": exp["final_name"],
            "best_params": exp["best_params"], "tuned_cv_f1": exp["tuned_cv_f1"],
        }
        z.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"))
        for path, model in [
            ("spam_model_baseline.joblib", exp["models"][MODEL_NAMES[0]]),
            ("spam_model_final.joblib", exp["final_model"]),
        ]:
            payload = io.BytesIO()
            joblib.dump(model, payload)
            z.writestr(path, payload.getvalue())
    return buffer.getvalue()
