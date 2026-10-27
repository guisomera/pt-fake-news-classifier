from sklearn.model_selection import train_test_split
from load_data import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import  classification_report, confusion_matrix

tabela = load_data() 

ids_treino, ids_teste = train_test_split(tabela["id"].unique(), test_size=0.2, random_state=42)   #Separa treino e prova
treino = tabela[tabela["id"].isin(ids_treino)]   #Seleciona os dados de treino
prova = tabela[tabela["id"].isin(ids_teste)]   #Seleciona os dados de prova

vetorizador = TfidfVectorizer()   #Transforma os textos em vetores
X_treino = vetorizador.fit_transform(treino["text"])   #Transforma os textos de treino em vetores e da peso
X_prova = vetorizador.transform(prova["text"])   #Transforma os textos de prova em vetores 

modelo = LogisticRegression(max_iter=1000)   #Cria o modelo
modelo.fit(X_treino, treino["label"])  #Treina o modelo

palpites = modelo.predict(X_prova) #Faz previsões

print(classification_report(prova["label"], palpites))  #Compara com  o gabarito
print(confusion_matrix(prova["label"], palpites))  #Mostra acertos e erros
print(modelo.classes_)