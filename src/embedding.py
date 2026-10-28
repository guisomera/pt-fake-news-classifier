"""Shared code to turn a text into a BERTimbau vector.

The settings here must match src/bert_features.py (same model, same truncation,
same [CLS] pooling). Otherwise the classifier trained on the saved vectors would
receive vectors made in a different way.
"""

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LENGTH = 512


def load_bert():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    bert = AutoModel.from_pretrained(MODEL_NAME)
    bert.eval()
    return tokenizer, bert


def embed(texts, tokenizer, bert):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = bert(**inputs)
    return output.last_hidden_state[:, 0, :].numpy()
