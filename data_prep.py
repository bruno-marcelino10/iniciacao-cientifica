from data_import import Data
import numpy as np
import pandas as pd

def prepare(data):
    target = data.target
    # Alterando índices
    target.index = target["Empresa"]
    target.drop(["Empresa"], axis = 1, inplace = True)

    # Montando o alvo
    print("Classificações: ",list(set(target["Classificação"])))

    target["Recuperação Judicial"] = np.where(target["Classificação"] == "Recuperação Judicial", 1, 0)
    target["Amostra Falida"] = np.where(target["Classificação"] == "Amostra Falida", 1, 0)
    target["Alvo"] = target["Amostra Falida"] + target["Recuperação Judicial"]
    target.drop(["Recuperação Judicial", "Amostra Falida"], axis = 1, inplace = True)
    return data

def wrangle(ano_pre_fal, data):
    # monta df para 1 a 5 anos pré-falência

    df = pd.DataFrame(index = data.features.keys())

    for indicador in indicadores:
        coluna = []
        for empresa in features.keys():
            data = features[empresa]
            inds = data.columns[0]
            val_ind = data[data[inds] == indicador].iloc[0, 6-ano_pre_fal]
            coluna.append(val_ind)

        df[indicador] = coluna
    
    df.replace('#DIV/0!', np.nan, inplace = True)
    df.replace(0, np.nan, inplace = True)
    df = df.join(target["Alvo"]) # Acrescenta os alvos

    return df

