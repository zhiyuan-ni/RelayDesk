"""classification_report 的规格测试。期望值都是手算出来的，见每条测试里的注释。"""
from app.evals.metrics import classification_report, confusion_pairs


def test_1_all_correct():
    r = classification_report(["a", "b", "a"], ["a", "b", "a"])
    assert r["accuracy"] == 1.0 and r["macro_f1"] == 1.0
    assert r["per_class"]["a"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2}


def test_1_empty_input():
    assert classification_report([], []) == {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}}


def test_3_hand_computed_example():
    # 真实:  refund refund invoice invoice invoice login
    # 预测:  refund invoice invoice invoice refund  login
    y_true = ["refund", "refund", "invoice", "invoice", "invoice", "login"]
    y_pred = ["refund", "invoice", "invoice", "invoice", "refund", "login"]
    r = classification_report(y_true, y_pred)

    assert r["accuracy"] == 0.6667                                   # 6 条对了 4 条
    # refund:  TP=1 FP=1 FN=1 -> P=0.5    R=0.5    F1=0.5
    assert r["per_class"]["refund"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    # invoice: TP=2 FP=1 FN=1 -> P=0.6667 R=0.6667 F1=0.6667
    assert r["per_class"]["invoice"] == {"precision": 0.6667, "recall": 0.6667, "f1": 0.6667, "support": 3}
    # login:   TP=1 FP=0 FN=0 -> 全是 1
    assert r["per_class"]["login"]["f1"] == 1.0
    assert r["macro_f1"] == 0.7222                                   # (0.5 + 0.6667 + 1.0) / 3
    assert list(r["per_class"]) == ["invoice", "login", "refund"]    # 按类别名排序


def test_3_class_never_predicted_gets_zero_not_a_crash():
    r = classification_report(["a", "b"], ["a", "a"])
    assert r["per_class"]["b"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    assert r["per_class"]["a"] == {"precision": 0.5, "recall": 1.0, "f1": 0.6667, "support": 1}


def test_2_label_only_in_predictions_is_not_listed():
    r = classification_report(["a", "a"], ["a", "other"])
    assert list(r["per_class"]) == ["a"] and r["per_class"]["a"]["recall"] == 0.5


def test_5_macro_f1_exposes_a_lazy_majority_classifier():
    # 10 条里 8 条是 a。全部预测成 a：准确率 0.8 看着不错，宏平均 F1 揭穿了它
    y_true = ["a"] * 8 + ["b", "c"]
    r = classification_report(y_true, ["a"] * 10)
    assert r["accuracy"] == 0.8 and r["macro_f1"] == 0.2963          # (0.8889 + 0 + 0) / 3


def test_confusion_pairs():
    pairs = confusion_pairs(["refund", "refund", "invoice", "login"], ["payment", "payment", "refund", "login"])
    assert pairs[0] == ("refund", "payment", 2) and ("invoice", "refund", 1) in pairs
