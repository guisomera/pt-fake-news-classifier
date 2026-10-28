# Portuguese Fake News Classifier

Classifies Brazilian Portuguese news as **fake** or **true**. The project compares a simple baseline (TF-IDF + logistic regression) with a pretrained Hugging Face model (BERTimbau) on the same test set, and serves the better one through a FastAPI API.

## Problem

Given the text of a news article in Portuguese, predict whether it is fake or true.

## Data

[Fake.br Corpus](https://github.com/roneysco/Fake.br-Corpus): 7,200 news articles, 3,600 fake and 3,600 true. Each fake article is paired with a true one on the same topic.

- I use the `size_normalized_texts` version, where the fake text is cut to the length of its true pair. In the full texts, fake news are much shorter than true ones, and a rule based only on text length already separates them well (about 94% accuracy on held-out data, in a quick check). Using the normalized version removes that shortcut.
- The data is split 80/20 by article id, so a fake article and its true pair always stay on the same side of the split. Result: 5,760 articles for training and 1,440 for testing (720 fake and 720 true), with `random_state=42`.
- Only public data is used.

## Approach

1. **Baseline** (`src/baseline.py`): TF-IDF turns each text into word-weight numbers; logistic regression learns to separate fake from true.
2. **BERTimbau** (`src/bert_features.py`, `src/bert_eval.py`): the pretrained model [`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased) turns each text into a 768-number vector (the `[CLS]` vector, texts truncated to 512 tokens). A logistic regression is trained on top of these vectors. BERTimbau itself is **not fine-tuned**; it is used as a frozen feature extractor.
3. **API** (`src/api.py`): FastAPI service that runs BERTimbau + the trained classifier.

## Results

Same test set for both models: 1,440 articles, one split.

| Model | Accuracy | Errors (of 1,440) | Fake predicted as true | True predicted as fake |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.90 | 137 | 75 | 62 |
| BERTimbau (frozen) + logistic regression | 0.97 | 49 | 29 | 20 |

Per class (precision / recall / F1):

| Model | fake | true |
|---|---|---|
| TF-IDF + logistic regression | 0.91 / 0.90 / 0.90 | 0.90 / 0.91 / 0.91 |
| BERTimbau + logistic regression | 0.97 / 0.96 / 0.97 | 0.96 / 0.97 / 0.97 |

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Get the data (not tracked by git)
mkdir -p data/raw
git clone https://github.com/roneysco/Fake.br-Corpus.git data/raw/Fake.br-Corpus

# 2. Baseline
python src/baseline.py

# 3. BERTimbau vectors (slow: it processes all 7,200 articles on CPU)
python src/bert_features.py
python src/bert_eval.py

# 4. Train and save the classifier used by the API
python src/train_final.py

# 5. Start the API
uvicorn --app-dir src api:app
```

Try it (the first start downloads the BERTimbau model from Hugging Face):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Paste a news article in Portuguese here."}'
```

The response has the predicted `label` (`fake` or `true`), its `probability`, and the `probabilities` of both classes. Interactive docs are at `http://127.0.0.1:8000/docs`.

## Limitations

- **One split, one run.** The numbers come from a single train/test split. I did not repeat the experiment with other splits, so I have no estimate of how much the results vary.
- **Possible shortcuts.** Length was neutralized, but other differences between fake and true sources (writing style, vocabulary of each outlet) may still make the task easier than real-world fake news detection. I did not run an error analysis to check this.
- **Paired data.** Fake and true articles are paired by topic, which is not how news arrives in real life. Performance on other sources or other periods is unknown.
- **Truncation.** Texts longer than 512 tokens are cut. I did not measure how many articles are affected.
- **Frozen model.** BERTimbau is not fine-tuned. Fine-tuning could improve results, but it was not tested.
- **Not a fact-checker.** The model detects patterns of text seen in this corpus. It does not check facts.

## Credits

- Dataset: Fake.br Corpus (see the repository above for the original paper).
- Model: BERTimbau, by Souza, Nogueira and Lotufo (BRACIS 2020).
