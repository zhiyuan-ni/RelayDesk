"""分类指标：准确率、精确率、召回率、F1、宏平均 F1。纯 Python 实现，不依赖任何库。

为什么只看准确率不够：
  	假设 100 条评测里有 60 条是物流咨询。一个把所有消息都判成"物流咨询"的分类器，准确率有 60%，
  	但它对退款、发票、登录故障的识别能力是零。准确率被样本多的类别主导了。

  	宏平均 F1 先给每个类别单独算 F1，再取平均，每个类别不论样本多少，权重都一样。
  	上面那个分类器的宏平均 F1 会非常低，如实反映出它其实什么都不会。

对某一个类别 C 来说：
	TP  真实是 C，也预测成了 C          预测对了
  	FP  真实不是 C，却预测成了 C        误报，别人的被我抢来了
  	FN  真实是 C，却预测成了别的        漏报，我的被别人抢走了

  	精确率 precision = TP / (TP + FP)   我说是 C 的那些里，有多少真的是 C
  	召回率 recall    = TP / (TP + FN)   真正的 C 里面，我找出来了多少
  	F1              = 2 * P * R / (P + R)   两者的调和平均，任何一个低，F1 就低
"""
from typing import Any


def safe_div(a: float, b: float) -> float:
    """分母为 0 时返回 0.0。某个类别一次都没被预测到时，精确率的分母就是 0。"""
    return a / b if b else 0.0


def classification_report(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """根据真实标签和预测标签计算各项指标。

    ───────────── 练习：请你实现 ─────────────
    y_true 和 y_pred 等长，第 i 个元素分别是第 i 条样本的真实标签和预测标签。

    返回的字典形如：
        {
            "accuracy":	0.8333,
            "macro_f1": 0.7778,
            "per_class":	{
            "refund":		{"precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2},
            "invoice": 		{...},
          	},
        }

    规格，对应 tests/test_metrics.py：
      	1. accuracy = 预测正确的条数 / 总条数。总条数为 0 时是 0.0
      	2. 只统计在 y_true 里出现过的类别，按类别名排序。只在预测里出现、真实标签里没有的类别不单独列出，
        	它造成的错误已经体现在别的类别的召回率里了
      	3. 对每个类别算出 TP、FP、FN，再算 precision、recall、f1。除法一律用 safe_div
      	4. support 是这个类别在 y_true 里出现的次数，也就是 TP + FN
      	5. macro_f1 是各类别 f1 的算术平均。没有任何类别时是 0.0
      	6. 所有小数用 round(x, 4) 保留 4 位

    提示：
      	- zip(y_true, y_pred) 可以同时遍历两个列表
      	- 统计 TP 的一种写法： sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
      	- 类别列表： sorted(set(y_true))
    大约 15 行。这是目前最长的一个练习，但每一步都是照着公式写。
	"""
    per_class = {}
    for label in sorted(set(y_true)):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        support = tp + fn
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return {
        "accuracy": round(safe_div(sum(1 for t, p in zip(y_true, y_pred) if t == p), len(y_true)), 4),
        "macro_f1": round(safe_div(sum(v["f1"] for v in per_class.values()), len(per_class)), 4),
        "per_class": per_class,
    }


def confusion_pairs(y_true: list[str], y_pred: list[str], top: int = 8) -> list[tuple[str, str, int]]:
    """最常见的错误：(真实类别, 被误判成的类别, 次数)。改进系统时，先看这张表最上面的几行。"""
    counts: dict[tuple[str, str], int] = {}
    for t, p in zip(y_true, y_pred):
        if t != p:
            counts[(t, p)] = counts.get((t, p), 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [(t, p, n) for (t, p), n in ranked[:top]]
