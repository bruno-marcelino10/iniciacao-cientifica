# Manipulação de Dados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Pré-processamento dos Dados
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Modelos Utilizados
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Otimização
from sklearn.model_selection import train_test_split, GridSearchCV

# Métricas de Avaliação: Classificação
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def treat(n, dfs):
    
    # Iterator
    df = dfs[n]
    print("\nDataFrame Escolhido:", n, "anos pré-falência")
    
    # Leading with NA's
    df.dropna(inplace = True)
    print("\nQuantidade de Amostras:\n", df["Alvo"].value_counts())

    # Train and Test
    X = df.drop("Alvo", axis = 1)
    y = df["Alvo"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

    # Standardizing
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    # Principal Component Analysis
    pca = PCA()
    X_train = pca.fit_transform(X_train_norm)
    X_test = pca.transform(X_test_norm)
    print("\nExplicação de Cada Componente:", pca.explained_variance_ratio_.cumsum().round(2))
    print("--------------------------------------------------")
    return X_train, X_test, y_train, y_test

def models(df, obj):
    '''
    Entradas:
    df: Tupla com X_train, X_test, y_train, y_test
    obj: String definindo o tipo de modelo (log, rf, mlp)
    
    Saídas:
    Tupla contendo f1-score, auc, acurácia, 
    '''
    
    # Desempacotando Features 
    X_train, X_test, y_train, y_test = df
    
    # Regressão Logística
    if obj == "log":
        modelo = LogisticRegression(max_iter = 1000, random_state = 42)
        modelo.fit(X_train, y_train)
        print("Modelo: Regressão Logística")
        
    # Random Forest
    if obj == "rf":
        modelo = RandomForestClassifier(random_state = 42)
        parametros = {'max_depth': [1, 5, None],
                      'min_samples_leaf': [1, 3, 5],
                      'min_samples_split': [2, 4],
                      'n_estimators': [25, 50, 100],
                      'criterion': ["gini", 'entropy']}
        
        modelo = GridSearchCV(modelo, parametros, n_jobs = -1, cv = 3, scoring = "roc_auc")
        modelo.fit(X_train, y_train)
        print("Modelo: Random Forest")
    
    # Multi-Layer Perceptron
    if obj == "mlp":
        modelo = MLPClassifier(max_iter = 1000, random_state = 42)
        parametros = {'hidden_layer_sizes': [(50,50), (50,100,50), (100,)],
                      'activation': ['tanh', 'relu'],
                      'solver': ['sgd', 'adam', "lbfgs"],
                      'alpha': [0.0001, 0.05],
                      'learning_rate': ['constant','adaptive']}

        modelo = GridSearchCV(modelo, parametros, n_jobs = -1, cv = 3, scoring = "roc_auc")
        modelo.fit(X_train, y_train)
        print("Modelo: Multi-Layer Perceptron")

    # Predictions
    y_pred = modelo.predict(X_test)
    
    # Classification Metrics
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print("F1-Score:", round(f1,4))
    print("AUC:", round(auc,4))
    print("Acurácia:", round(acc,4))
    print("--------------------------------------------------")
    return [f1, auc, acc, obj]