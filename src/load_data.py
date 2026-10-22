from pathlib import Path
import pandas as pd

base = Path("data/raw/Fake.br-Corpus/size_normalized_texts")

lista = []
for label in ["fake", "true"]:
    for arquivo in (base / label).glob("*.txt"):
        texto = arquivo.read_text(encoding="utf-8")
        lista.append({"text": texto, "label": label, "id": int(arquivo.stem)})

tabela = pd.DataFrame(lista)
print(tabela.shape)