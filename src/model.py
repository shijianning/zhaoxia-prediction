"""机器学习模型模块。

把「历史气象特征 + 观测标签」喂给 scikit-learn 分类器，
输出当天窗口出现漂亮朝霞/晚霞的概率。

特征向量（与特征工程 + 规则评分一致）：
[cloud_cover, cloud_low, cloud_mid, cloud_high, humidity, wind,
 precip, temp, day_of_year_sin, day_of_year_cos, rule_score]
"""
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def feature_vector(f):
    """把特征字典转换为模型输入向量。"""
    doy = float(f.get("day_of_year", 1))
    return [
        float(f.get("cloud_cover", 0.0)),
        float(f.get("cloud_low", 0.0)),
        float(f.get("cloud_mid", 0.0)),
        float(f.get("cloud_high", 0.0)),
        float(f.get("humidity", 0.0)),
        float(f.get("wind", 0.0)),
        float(f.get("precip", 0.0)),
        float(f.get("temp", 0.0)),
        float(np.sin(2 * np.pi * doy / 365.0)),
        float(np.cos(2 * np.pi * doy / 365.0)),
        float(f.get("rule_score", 0.0)),
    ]


class GlowModel:
    """朝霞晚霞分类器封装。"""

    def __init__(self, model_type="logistic"):
        self.model_type = model_type
        self.model = None
        self.n_samples = 0
        self.metrics = {}

    def train(self, rows):
        """rows: 特征字典列表，每个含 'glow' 标签(0/1)。返回评估指标。"""
        X = np.array([feature_vector(r) for r in rows], dtype=float)
        y = np.array([int(r["glow"]) for r in rows], dtype=int)

        if self.model_type == "rf":
            est = RandomForestClassifier(
                n_estimators=200, max_depth=6,
                random_state=42, class_weight="balanced",
            )
        else:
            est = LogisticRegression(max_iter=1000, class_weight="balanced")

        self.model = make_pipeline(StandardScaler(), est)
        self.model.fit(X, y)

        self.n_samples = len(rows)
        pred = self.model.predict(X)
        acc = accuracy_score(y, pred)
        metrics = {"accuracy": round(acc, 4), "n_samples": self.n_samples}
        if len(set(y)) == 2:
            proba = self.model.predict_proba(X)[:, 1]
            metrics["auc"] = round(roc_auc_score(y, proba), 4)
        self.metrics = metrics
        return metrics

    def predict_proba(self, f):
        """返回窗口出现好霞的概率(0-1)。"""
        X = np.array([feature_vector(f)], dtype=float)
        return float(self.model.predict_proba(X)[0, 1])

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            return None
        return joblib.load(path)
