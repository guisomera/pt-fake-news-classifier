import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from data_split import get_split

LIMIT = None  # quick test; change to None to process everything

name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(name, do_lower_case=False)
bert = AutoModel.from_pretrained(name)
bert.eval()


def embed(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = bert(**inputs)
    return output.last_hidden_state[:, 0, :].numpy()


def embed_all(texts, batch_size=4):
    start_time = time.time()
    chunks = []
    for start in range(0, len(texts), batch_size):
        chunks.append(embed(texts[start:start + batch_size]))
        print(f"{start + batch_size}/{len(texts)} texts, {time.time() - start_time:.0f}s", flush=True)
    return np.vstack(chunks)


train, test = get_split()
if LIMIT:
    train = train.sample(LIMIT, random_state=0)
    test = test.sample(LIMIT, random_state=0)

t0 = time.time()
X_train = embed_all(train["text"].tolist())
X_test = embed_all(test["text"].tolist())
print(f"total: {time.time() - t0:.0f}s")

out = "data/bert_vectors_small.npz" if LIMIT else "data/bert_vectors.npz"
np.savez(
    out,
    X_train=X_train, y_train=train["label"].to_numpy().astype(str), ids_train=train["id"].to_numpy(),
    X_test=X_test, y_test=test["label"].to_numpy().astype(str), ids_test=test["id"].to_numpy(),
)
print("saved", out, X_train.shape, X_test.shape)