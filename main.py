from data_import import Data
from data_prep import prepare, wrangle
from data_modelling import treat, models 

if __name__ == '__main__':
    
    # Importing and preparing...
    df = Data()
    df.get_data()
    df = prepare(df)

    sheet_names = ["df_" + str(i) + "y" for i in range(1,6)]
    dfs = [wrangle(i) for i in range(1,6)]    

    # Modelling...
    tabela = []
    for i in range(1,6):
        df = treat(i)
        for j in ["log", "rf", "mlp"]:
            linha = models(df, j)
            linha.append(i)
            tabela.append(linha)
    
    print(tabela)