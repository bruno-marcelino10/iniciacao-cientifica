# Manipulação de Dados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Pré-processamento dos Dados
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Modelos Utilizados
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Otimização
from sklearn.model_selection import train_test_split, GridSearchCV

# Métricas de Avaliação: Classificação
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score

def treat(n, dfs):
    
    # Iterator
    df = dfs[n]
    print("\nDataFrame Escolhido:", n+1, "anos pré-falência")
    
    # Leading with NA's
    df.dropna(inplace = True)
    print("\nQuantidade de Amostras:\n", df["Alvo"].value_counts())

    # Train and Test
    X = df.drop("Alvo", axis = 1)
    y = df["Alvo"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

    # Standardizing
    scaler = StandardScaler()
    X_train_pad = scaler.fit_transform(X_train)
    X_test_pad = scaler.transform(X_test)
    
    return X_train_pad, X_test_pad, y_train, y_test

def selectors(df, obj):    
    '''
    Entradas:
    df: Tupla com X_train padronizado, X_test padronizado, y_train, y_test
    obj: String definindo o tipo de modelo
    
    Saídas:
    Tupla com X_train decomposto, X_test decomposto, y_train, y_test   
    '''
    
    X_train_pad, X_test_pad, y_train, y_test = df

    # Principal Component Analysis
    if obj == "pca":
        pca = PCA()
        X_train = pca.fit_transform(X_train_pad)
        X_test = pca.transform(X_test_pad)
        print("Decomposição: PCA")

    if obj == "kpca":
        kpca = KernelPCA()
        X_train = kpca.fit_transform(X_train_pad)
        X_test = kpca.transform(X_test_pad)
        print("Decomposição: Kernel PCA")

    if obj == "lda":
        lda = LinearDiscriminantAnalysis()
        X_train = lda.fit_transform(X_train_pad, y_train)
        X_test = lda.transform(X_test_pad)
        print("Decomposição: LDA")

    print("--------------------------------------------------")

    return X_train, X_test, y_train, y_test

def models(df, obj, selector):
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
        parametros = {
            #'max_depth': [1, 5, None], # Retirar
            'min_samples_leaf': [1, 3, 5],
            'min_samples_split': [2, 4],
            #'n_estimators': [25, 50, 100], # Retirar
            'criterion': ["gini", 'entropy']
            }
        
        modelo = GridSearchCV(modelo, parametros, n_jobs = -1, cv = 2, scoring = "roc_auc")
        modelo.fit(X_train, y_train)
        print("Modelo: Random Forest")
    
    # Multi-Layer Perceptron
    if obj == "mlp":
        modelo = MLPClassifier(max_iter = 1000, random_state = 42)
        parametros = {
            #'hidden_layer_sizes': [(50,50), (50,100,50), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam', "lbfgs"],
            #'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive']
            }

        modelo = GridSearchCV(modelo, parametros, n_jobs = -1, cv = 2, scoring = "roc_auc")
        modelo.fit(X_train, y_train)
        print("Modelo: Multi-Layer Perceptron")

    if obj == "svm":
        modelo = SVC(random_state = 42)
        parametros = {'kernel' : ["poly", 'linear', 'rbf', 'sigmoid']}

        modelo = GridSearchCV(modelo, parametros, n_jobs = -1, cv = 2, scoring = "roc_auc")
        modelo.fit(X_train, y_train)
        print("Modelo: Support Vector Classifier")

    if obj == "xgb":
        modelo = XGBClassifier(random_state = 42)
        parametros = {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5]
        }

        modelo = GridSearchCV(modelo, parametros, n_jobs = -1, cv = 2, scoring = "roc_auc")
        modelo.fit(X_train, y_train)
        print("Modelo: XGBoost")

    # Predictions
    y_pred = modelo.predict(X_test)
    
    # Classification Metrics
    prec = precision_score(y_test, y_pred, zero_division=1) # Maximizar precision
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    print("Precisão:", round(prec,4))
    print("AUC:", round(auc,4))
    print("Acurácia:", round(acc,4))
    print("--------------------------------------------------")
    return [prec, auc, acc, obj, selector]