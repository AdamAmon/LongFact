"""NLI 判定封装：对（证据, 假设句）做支持/矛盾/中立判断的简单接口。

此实现直接加载 `AutoTokenizer` + `AutoModelForSequenceClassification`，对句对进行编码并返回
模型预测标签与概率。默认模型为 `facebook/bart-large-mnli`。提供 `is_supported` 便捷方法
用于判定是否被证据支持（基于阈值）。
"""
from typing import Tuple
import torch

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


class NLIChecker:
    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: int = -1):
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise ImportError("transformers is required for NLIChecker")
        self.device = torch.device('cpu') if device == -1 else torch.device(f'cuda:{device}')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        # label mapping provided by model config, e.g. {0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'}
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def check(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """对 (premise, hypothesis) 返回预测标签与概率 (label, score)。"""
        inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            score, idx = torch.max(probs, dim=-1)
            label = self.id2label.get(int(idx.item()), str(idx.item()))
            return label, float(score.item())

    def is_supported(self, premise: str, hypothesis: str, threshold: float = 0.6) -> bool:
        """基于阈值判断 hypothesis 是否被 premise 支持（ENTAILMENT 且概率 >= threshold）。"""
        label, score = self.check(premise, hypothesis)
        return (label.upper() in ('ENTAILMENT', 'ENTAILS')) and (score >= threshold)


def simple_demo():
    checker = NLIChecker()
    p = "Alice went to the store to buy apples."
    h = "Alice visited a supermarket to purchase fruit."
    print(checker.check(p, h))


if __name__ == "__main__":
    simple_demo()
