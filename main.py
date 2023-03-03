from data_import import get_data
from data_prep import prepare, wrangle
from data_modelling import treat, models 
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    
    # Importing...
    dfs = get_data() 
    
    # Prepare binary target
    dfs["target"] = prepare(dfs["target"])

    sheet_names = ["df_" + str(i) + "y" for i in range(1,6)]
    indicadores = list(dfs["features"]["BOMBRIL"].iloc[35:, 0]) # Separa coluna com os indicadores
    dfs = [wrangle(i, dfs, indicadores) for i in range(1,6)]    

    # Modelling...
    tabela = []
    for i in range(5):
        df = treat(i, dfs)
        for j in ["log", "rf", "mlp"]:
            linha = models(df, j)
            linha.append(i)
            tabela.append(linha)
    
    df = pd.DataFrame(tabela)
    df.rename(columns = {0: "F1-Score", 1: "AUC", 2: "Acurácia", 3: "Modelo", 4: "Anos"}, inplace = True)
    df = round(df, 4)
    print(df)
