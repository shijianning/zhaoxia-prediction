"""机器学习模型模块。

把「历史气象特征 + 观测标签」喂给 scikit-learn 分类器，
输出当天窗口出现漂亮朝霞/晚霞的概率。

特征向量（与特征工程一致）：
[cloud_cover, cloud_low, cloud_mid, cloud_high, humidity, wind,
 precip, temp, day_of_year_sin, day_of_year_cos, visibility, aod]

**为什么特征里没有 rule_score？**
早期版本把规则分 `rule_score` 作为一维特征喂进模型，而弱监督标签恰恰又是
`rule_score >= 65 → 1` 生成的 —— 标签成了输入的确定性函数，任何一个容量足够
的分类器都能把它几乎完美地拟合出来。实测（合成 1500 样本复现该流程）：
全量自评 AUC=1.000、留出 30% 仍 AUC=1.000，而单靠 `rule_score` 一维就有 AUC=0.920。
也就是说该指标衡量的是"能否复刻规则引擎"，对"能否预测真实出霞"没有任何证据力，
同时规则分还会与「云结构/通透度」等原始因子双重计数。

注意一个反直觉之处：**只把 `rule_score` 从特征里删掉，AUC 并不会回落**。因为
`rule_score` 本身是原始气象量的确定性函数，弱标签 = `threshold(rule_score(原始量))`
仍然是原始量的确定性函数（实测移除后留出 30% AUC 仍为 1.000）。删特征只消除了
「双重计数」，并不能破除这层确定性关系——这是「模型有没有资格报指标」的问题。

因此：
1. 特征向量中移除 `rule_score`，消除与其他因子的共线双重计数；
2. 评估改为**按日期排序的时序留出**（用最近一段做测试），不再用训练集自评；
3. 由 `train.py` 在「真实观测标注不足」时干脆不启用模型（弱监督标签只能用来
   预训练，不能用来宣称模型的预测能力）——**这一步才是真正堵住泄漏的关键**。
"""
import datetime as dt
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 特征名（顺序即向量顺序），便于排查与文档
FEATURE_NAMES = [
    "cloud_cover", "cloud_low", "cloud_mid", "cloud_high",
    "humidity", "wind", "precip", "temp",
    "doy_sin", "doy_cos", "visibility", "aod",
]

# 缺失值填充用的中性值（仅作用于模型输入，不参与规则评分）
NEUTRAL_VISIBILITY_M = 10000.0
NEUTRAL_AOD = 0.3


def _num(v, default=0.0):
    """None 安全取数（分层云量等字段可能缺失，直接用 float(None) 会崩）。"""
    return float(v) if v is not None else float(default)


def feature_vector(f):
    """把特征字典转换为模型输入向量（12 维）。

    缺失时用中性值填充，避免被当成"极差通透"而误判：
    - 能见度 10km、AOD 0.3 约等于规则引擎的中性分（不奖励也不惩罚）。
    """
    doy = float(f.get("day_of_year") or 1)
    return [
        _num(f.get("cloud_cover")),
        _num(f.get("cloud_low")),
        _num(f.get("cloud_mid")),
        _num(f.get("cloud_high")),
        _num(f.get("humidity")),
        _num(f.get("wind")),
        _num(f.get("precip")),
        _num(f.get("temp")),
        float(np.sin(2 * np.pi * doy / 365.0)),
        float(np.cos(2 * np.pi * doy / 365.0)),
        _num(f.get("visibility"), NEUTRAL_VISIBILITY_M),
        _num(f.get("aod"), NEUTRAL_AOD),
    ]


class GlowModel:
    """朝霞晚霞分类器封装。"""

    def __init__(self, model_type="logistic"):
        self.model_type = model_type
        self.model = None
        self.n_samples = 0
        # 真实观测标注条数：决定该模型是否有资格参与打分（见 predict.py）
        self.real_labels = 0
        self.trained_at = None
        self.metrics = {}

    # ---- 内部：构造估计器 ----
    def _make_estimator(self):
        if self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=200, max_depth=6,
                random_state=42, class_weight="balanced",
            )
        return LogisticRegression(max_iter=1000, class_weight="balanced")

    def train(self, rows, holdout_ratio=0.3, real_labels=None, real_source="post"):
        """rows: 特征字典列表，每个含 'glow' 标签(0/1)、'date'、'source'。

        评估采用**按日期排序的时序留出**，且留出集**只取真实观测样本**
        （source == real_source）。这一点很关键：若留出集里混入弱监督样本，
        由于弱标签本就是规则分的函数，AUC 会重新虚高，指标又失去意义。

        样本不足以稳定切分时只报训练集指标并置 holdout=None。

        返回评估指标字典；训练集类别不足 2 类时返回 {"ok": False, ...}。
        """
        if not rows:
            return {"ok": False, "reason": "无样本"}

        X = np.array([feature_vector(r) for r in rows], dtype=float)
        y = np.array([int(r["glow"]) for r in rows], dtype=int)

        order = sorted(range(len(rows)), key=lambda i: str(rows[i].get("date", "")))

        # 留出集：只从真实观测样本里，按日期取最近的 holdout_ratio 一段
        real_idx = [i for i in order if rows[i].get("source") == real_source]
        n_real = len(real_idx)
        n_test = int(round(n_real * holdout_ratio))
        if n_real >= 20 and n_test >= 5:
            test_idx = real_idx[n_real - n_test:]
            test_set = set(test_idx)
            train_idx = [i for i in order if i not in test_set]
        else:
            train_idx, test_idx = order, []

        can_train = len(train_idx) >= 10
        if not can_train:
            return {"ok": False, "reason": "训练样本不足 10 条"}

        y_tr = y[train_idx]
        if len(set(y_tr.tolist())) < 2:
            return {"ok": False, "reason": "训练集只有一个类别，无法训练"}

        self.model = make_pipeline(StandardScaler(), self._make_estimator())
        self.model.fit(X[train_idx], y_tr)

        self.n_samples = len(rows)
        if real_labels is not None:
            self.real_labels = int(real_labels)
        self.trained_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

        metrics = {
            "ok": True,
            "n_samples": len(rows),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "real_labels": self.real_labels,
            "real_rows": n_real,
            "accuracy": round(float(accuracy_score(y_tr, self.model.predict(X[train_idx]))), 4),
        }
        if len(set(y_tr.tolist())) == 2:
            p_tr = self.model.predict_proba(X[train_idx])[:, 1]
            metrics["auc"] = round(float(roc_auc_score(y_tr, p_tr)), 4)

        # ---- 留出集：唯一有预测意义的指标（纯真实观测） ----
        if test_idx:
            y_te = y[test_idx]
            metrics["holdout_accuracy"] = round(
                float(accuracy_score(y_te, self.model.predict(X[test_idx]))), 4)
            if len(set(y_te.tolist())) == 2:
                p_te = self.model.predict_proba(X[test_idx])[:, 1]
                metrics["holdout_auc"] = round(float(roc_auc_score(y_te, p_te)), 4)
            else:
                metrics["holdout_note"] = "留出集只有一个类别，AUC 不可计算"
        else:
            metrics["holdout_note"] = (
                f"真实观测样本仅 {n_real} 条，不足以切出留出集，"
                f"训练集指标不具预测意义"
            )
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
        """加载模型。旧版本（特征维度不同，如含 rule_score 的 13 维）一律弃用。

        返回 None 表示"无可用模型"，调用方据此退回纯规则评分。
        """
        if not os.path.exists(path):
            return None
        try:
            m = joblib.load(path)
        except Exception:
            return None
        if not isinstance(m, cls):
            return None
        # 兼容旧 pickle：补齐后来新增的属性
        if not hasattr(m, "real_labels"):
            m.real_labels = 0
        if not hasattr(m, "trained_at"):
            m.trained_at = None
        # 特征维度必须与当前 FEATURE_NAMES 一致，否则打分时形状不匹配
        try:
            if int(m.model.n_features_in_) != len(FEATURE_NAMES):
                return None
        except Exception:
            return None
        return m
