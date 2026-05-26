"""NLI 判定封装：对（证据, 假设句）做支持/矛盾/中立判断的简单接口。

此实现直接加载 `AutoTokenizer` + `AutoModelForSequenceClassification`，对句对进行编码并返回
模型预测标签与概率。默认模型为 `facebook/bart-large-mnli`。提供 `is_supported` 便捷方法
用于判定是否被证据支持（基于阈值）。
"""
from typing import Tuple
import torch
import traceback

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import BitsAndBytesConfig
except Exception as e:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    BitsAndBytesConfig = None
    print('nli_check: transformers import failed:', e)
    traceback.print_exc()

from config import DEFAULT_NLI_MODEL, PREFERRED_PRECISION, DEFAULT_TORCH_COMPILE


class NLIChecker:
    def __init__(self, model_name: str = DEFAULT_NLI_MODEL, device: int = -1, load_in_8bit: bool = False, precision: str = 'auto', torch_compile: bool = False):
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise ImportError("transformers is required for NLIChecker")
        self.device = torch.device('cpu') if device == -1 else torch.device(f'cuda:{device}')
        self.precision = precision or PREFERRED_PRECISION or 'auto'
        self.torch_compile = torch_compile or DEFAULT_TORCH_COMPILE
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Support 8-bit loading via bitsandbytes when requested
        if load_in_8bit and BitsAndBytesConfig is not None:
            try:
                # device_map='auto' will place parameters on GPU where possible
                bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    device_map='auto',
                    quantization_config=bnb_cfg,
                    trust_remote_code=True,
                )
            except Exception as e:
                print(f'nli_check: 8bit load failed for {model_name}:', e)
                traceback.print_exc()
                # fallback to normal loading
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        else:
            # If fp16 requested and GPU device available, try loading with float16 dtype
            try:
                if (self.precision in ('fp16', 'half')) and device >= 0:
                    try:
                        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
                        if self.torch_compile and hasattr(torch, 'compile'):
                            try:
                                self.model = torch.compile(self.model)
                            except Exception:
                                pass
                    except Exception:
                        # fallback to normal
                        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
                else:
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            except Exception:
                # fallback
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

        # label mapping provided by model config, e.g. {0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'}
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.last_error: str = None

    def check(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """对 (premise, hypothesis) 返回预测标签与概率 (label, score)。"""
        try:
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, max_length=1024).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                score, idx = torch.max(probs, dim=-1)
                label = self.id2label.get(int(idx.item()), str(idx.item()))
                # clear last_error on success
                self.last_error = None
                return label, float(score.item())
        except Exception as e:
            # Do not raise here; return a safe error token so pipeline continues.
            err = traceback.format_exc()
            print("Exception in NLIChecker.check:", e)
            print(err)
            self.last_error = err
            return 'ERROR', 0.0

    def is_supported(self, premise: str, hypothesis: str, threshold: float = 0.6) -> bool:
        """基于阈值判断 hypothesis 是否被 premise 支持（ENTAILMENT 且概率 >= threshold）。"""
        label, score = self.check(premise, hypothesis)
        return (label.upper() in ('ENTAILMENT', 'ENTAILS')) and (score >= threshold)

    def check_batch(self, premises: list, hypotheses: list) -> list:
        """对多个 (premise, hypothesis) 批量判定，返回 [(label, score), ...].

        This uses a single forward pass for efficiency when available.
        """
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have the same length")
        try:
            inputs = self.tokenizer(premises, hypotheses, return_tensors='pt', truncation=True, padding=True, max_length=1024).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                max_scores, idxs = torch.max(probs, dim=-1)
                labels = [self.id2label.get(int(i.item()), str(i.item())) for i in idxs]
                self.last_error = None
                return list(zip(labels, [float(s.item()) for s in max_scores]))
        except Exception as e:
            err = traceback.format_exc()
            print('nli_check: batch inference failed:', e)
            print(err)
            self.last_error = err
            # return error markers for each input so callers can continue
            return [('ERROR', 0.0) for _ in premises]

    def check_with_evidence(self, evidences: list, hypothesis: str, strategy: str = 'max') -> tuple:
        """对同一 hypothesis 使用多条 evidence 分别做 NLI，并根据 strategy 聚合结果。

        strategies:
          - 'max': 取最大置信度对应的 (label, score)
          - 'majority': 投票选择标签（并返回该标签的平均分）
          - 'any_entailment': 如果任意 evidence 的 entailment score >= threshold 返回该对

        返回 (agg_label, agg_score, per_evidence_list) 其中 per_evidence_list 为 [(label, score, evidence), ...]
        """
        if not evidences:
            return 'NO_EVIDENCE', 0.0, []

        # Use batch checking to improve throughput when possible
        try:
            pairs = self.check_batch(evidences, [hypothesis] * len(evidences))
            # if batch returned error markers, force fallback to serial
            if any((isinstance(p, tuple) and p[0] == 'ERROR') for p in pairs):
                raise RuntimeError('batch returned ERROR markers')
            per = [(lab, sc, e) for (lab, sc), e in zip(pairs, evidences)]
        except Exception as e:
            # fallback to serial checks preserving error handling
            print('nli_check: batch check failed, falling back to serial checks:', e)
            traceback.print_exc()
            per = []
            for ev in evidences:
                try:
                    lab, sc = self.check(ev, hypothesis)
                except Exception as e2:
                    print("Exception in check_with_evidence for evidence:", (ev[:200] if isinstance(ev, str) else str(ev)))
                    print(traceback.format_exc())
                    lab, sc = 'ERROR', 0.0
                per.append((lab, sc, ev))

        if not per:
            self.last_error = None
            return 'NO_EVIDENCE', 0.0, []

        if strategy == 'max':
            best = max(per, key=lambda x: x[1])
            return best[0], best[1], per

        if strategy == 'majority':
            from collections import Counter

            cnt = Counter([p[0].upper() for p in per])
            lab = cnt.most_common(1)[0][0]
            # average score for that label
            scores = [p[1] for p in per if p[0].upper() == lab]
            avg = float(sum(scores) / len(scores)) if scores else 0.0
            return lab, avg, per

        if strategy == 'any_entailment':
            for lab, sc, e in per:
                if lab.upper() in ('ENTAILMENT', 'ENTAILS'):
                    return lab, sc, per
            # fallback to max
            best = max(per, key=lambda x: x[1])
            return best[0], best[1], per

        # unknown strategy -> fallback to max
        best = max(per, key=lambda x: x[1])
        return best[0], best[1], per


def simple_demo():
    checker = NLIChecker()
    p = "Alice went to the store to buy apples."
    h = "Alice visited a supermarket to purchase fruit."
    print(checker.check(p, h))


if __name__ == "__main__":
    simple_demo()
