from data_import import get_data
from data_prep import prepare, wrangle
from data_modelling import treat, selectors, models
from data_viz import create_plot

from dynaconf import Dynaconf
import warnings
import time
import pandas as pd

settings = Dynaconf(core_loaders=["JSON"], settings_files="project/settings.json")

if __name__ == '__main__':

    warnings.filterwarnings("ignore") # ignora warnings caso existam
    start_time = time.time() # contagem do tempo de execução

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
        create_plot(i, df[0], df[2]) # cria plots do t-SNE
        for j in settings["SELECTORS"]:
            df_decomp = selectors(df, j)
            for k in settings["MODELS"]: 
                linha = models(df_decomp, k, j) # estimação dos modelos em cima desta base
                linha.append(i+1)
                tabela.append(linha)
    
    df = pd.DataFrame(tabela) # criação de tabela com as métricas de avaliação para cada modelo em cada base
    df.rename(columns = {0: "Precisão", 1: "AUC", 2: "Acurácia", 3: "Modelo", 4: "Seletor", 5: "Anos"}, inplace = True)
    df = round(df, 4)

    print("\nResultados Consolidados:")
    print(df)

    end_time = time.time() # contagem do tempo de execução
    print("\nTempo decorrido: {:.2f} segundos".format(end_time - start_time))