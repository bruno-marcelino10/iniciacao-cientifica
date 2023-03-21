from data_import import get_data
from data_prep import prepare, wrangle
from data_modelling import treat, models 
import warnings
import pandas as pd

if __name__ == '__main__':
    warnings.filterwarnings("ignore") # ignora warnings caso existam
    
    # Importando dados de planilhas do Google Planilhas
    dfs = get_data() 
    
    # Prepara alvo binário (0: não faliu, 1: faliu)
    dfs["target"] = prepare(dfs["target"])

    # Separa coluna com os indicadores
    indicadores = list(dfs["features"]["BOMBRIL"].iloc[35:, 0])
    
    # Cria uma lista com 5 dataframes contendo (empresas x indicadores) para cada ano pré-falência
    dfs = [wrangle(i, dfs, indicadores) for i in range(1,6)]    

    # Modelagem
    tabela = []
    for i in range(5):
        df = treat(i, dfs) # tratamento prévio da base de i anos pré-falência
        for j in ["log", "rf", "mlp"]: 
            linha = models(df, j) # estimação dos três modelos em cima desta base
            linha.append(i)
            tabela.append(linha)
    
    df = pd.DataFrame(tabela) # criação de tabela com as métricas de avaliação para cada modelo em cada base
    df.rename(columns = {0: "F1-Score", 1: "AUC", 2: "Acurácia", 3: "Modelo", 4: "Anos"}, inplace = True)
    df = round(df, 4)
    print(df)
