"""Scoring engine: LettuceDetect's token-classification method without depending on the package.

Two interchangeable backends run the same ModernBERT model:
- "torch": transformers + PyTorch (local use; also supports the large model)
- "onnx":  ONNX Runtime, no PyTorch (serverless hosts such as Vercel, where PyTorch is too big)

Tokenisation, prompt format and context chunking are shared, and mirror
lettucedetect.detectors.transformer.TransformerDetector and PromptUtils (lettucedetect 0.2.3).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

# lettucedetect/prompts/qa_prompt_en.txt and summary_prompt_en.txt, verbatim.
QA_PROMPT = (
    "Briefly answer the following question:\n{question}\n"
    "Bear in mind that your response should be strictly based on the following {n} passages:\n{context}\n"
    'In case the passages do not contain the necessary information to answer the question, please reply with: '
    '"Unable to answer based on given passages."\noutput:'
)
SUMMARY_PROMPT = "Summarize the following text:\n{context}\noutput:"


def format_context(context: list[str], question: str | None) -> str:
    block = "\n".join(f"passage {i + 1}: {p}" for i, p in enumerate(context))
    if question is None:
        return SUMMARY_PROMPT.format(context=block)
    return QA_PROMPT.format(question=question, n=len(context), context=block)


class Engine:
    def __init__(self, tokenizer_file: str | Path, run_logits, max_length: int = 4096) -> None:
        """run_logits(input_ids[int64 1xN], attention_mask[int64 1xN]) -> logits[1xNx2]."""
        self.max_length = max_length
        self._run = run_logits
        self.tok = Tokenizer.from_file(str(tokenizer_file))
        self.tok.no_padding()
        self.tok.enable_truncation(max_length=max_length, strategy="only_first")

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    def group_passages(self, context: list[str], question: str | None, answer: str) -> list[list[str]]:
        """Group passages so each group, wrapped in the prompt, fits max_length with the answer."""
        budget = self.max_length - self.count(answer) - 3
        if self.count(format_context(context, question)) <= budget:
            return [context]
        passage_budget = budget - self.count(format_context([""], question))
        if passage_budget <= 0:
            return [context]
        groups: list[list[str]] = []
        current: list[str] = []
        used = 0
        for i, passage in enumerate(context):
            n = self.count(f"passage {i + 1}: {passage}")
            cost = n + (1 if current else 0)
            if current and used + cost > passage_budget:
                groups.append(current)
                current, used, cost = [], 0, n
            current.append(passage)
            used += cost
        if current:
            groups.append(current)
        return groups or [context]

    def token_probs(self, context: list[str], question: str | None, text: str):
        """P(hallucinated) per token of text, with each token's (start, end) offsets in text.

        A long context is split into groups that fit the model. A token counts as supported
        if ANY group supports it (min across groups). LettuceDetect uses max(), which flags
        facts that appear in only one group.
        """
        groups = self.group_passages(context, question, text)
        best, offsets = None, None
        for group in groups:
            enc = self.tok.encode(format_context(group, question), text)
            idx = [i for i, s in enumerate(enc.sequence_ids) if s == 1]
            ids = np.array([enc.ids], dtype=np.int64)
            logits = self._run(ids, np.ones_like(ids))[0][idx].astype(np.float64)
            e = np.exp(logits - logits.max(-1, keepdims=True))
            p = e[:, 1] / e.sum(-1)
            best = p if best is None else np.minimum(best, p)
            offsets = [enc.offsets[i] for i in idx]
        return best.tolist(), offsets, len(groups)


def onnx_engine(model_dir: str | Path) -> Engine:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 1
    session = ort.InferenceSession(str(Path(model_dir) / "model.onnx"), opts, providers=["CPUExecutionProvider"])
    return Engine(Path(model_dir) / "tokenizer.json",
                  lambda ids, mask: session.run(None, {"input_ids": ids, "attention_mask": mask})[0])


def torch_engine(model_id: str) -> Engine:
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForTokenClassification

    model = AutoModelForTokenClassification.from_pretrained(model_id).eval()

    def run(ids, mask):
        with torch.no_grad():
            return model(input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(mask)).logits.numpy()

    return Engine(hf_hub_download(model_id, "tokenizer.json"), run)
