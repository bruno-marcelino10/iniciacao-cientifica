import pandas as pd

def get_data():

    print("Iniciando Importação...")
    data = dict()
    # Features
    sheet_id = "1xOPRcRkJrGzxb65dieEHrWHnqXYrrSpx7Ack3pr6Pdw"
    features = pd.read_excel(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx", sheet_name = None)
    data["features"] = features

    # Targets
    sheet_id = "116cM2eSTve3UHHYESOdWXRiYxx8gGAPPLlonNDNIn6M"
    target = pd.read_excel(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx")
    data["target"] = target

    res = 0
    miss = []
    for i in list(features.keys()):
        if i in list(target["Empresa"].values):
            res += 1
        else:
            miss.append(i)
    
    # Validation
    validation = res == len(features.keys())
    data["validation"] = validation
    
    print("Número de Empresas (Target):", res)
    print("Número de Empresas (Features):", len(features.keys())) # Numero de empresas
    print("Empresas Faltando:", miss)
    
    print("Finalizando Importação...\n")
    return data