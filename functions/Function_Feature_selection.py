# Importo librerie utili
import pandas as pd
import streamlit as st

# Funzione per scegliere le feature categoriche, numeriche e target da usare
def Feature_selection(dataframe, task) :
    """
    La funzione accetta come argomenti il dataframe da controllare e il tipo di analisi. La funzione converte anche in modo corretto le colonne numeriche
    La funzione restituisce il dataframe con valori numerici corretti, la colonna target, e le liste delle colonne categoriche e numeriche
    """
    # Check per controllare che il dataframe non sia vuoto + selezione della colonna target
    if dataframe.empty == False :
        target = st.multiselect('Select the target:', dataframe.columns, key = 100)

    # Lista colonne
    Lista_col = dataframe.drop(target, axis = 1).columns.tolist()
    Lista_col.insert(0, 'All')
    Lista_col.insert(0, 'None')

    # Menu dropdown per multiselezione delle colonne categoriche
    categorical = st.multiselect("Select categorical features:", Lista_col, key = 1)
    if "All" in categorical:
        categorical = dataframe.drop(target, axis = 1).columns.tolist()
    elif "None" in categorical :
        pass

    # Menu dropdown per multiselezione delle colonne numeriche
    numerical = st.multiselect("Select numerical features:", Lista_col, key = 2)
    if "All" in numerical:
        numerical = dataframe.drop(target, axis = 1).columns.tolist()
    elif "None" in numerical :
        pass

    # Conversione delle colonne numeriche e Target
    # Può capitare che dopo la lettura di un file CSV, i dati numerici (con decimali) non siano convertiti in modo corretto
    if task == "Regression" :
        for i in numerical + target :
            dataframe[i] = dataframe[i].astype(str).str.replace(',', '.').astype(float)
            dataframe[i].apply(pd.to_numeric)
    else :
        for i in numerical :
            dataframe[i] = dataframe[i].astype(str).str.replace(',', '.').astype(float)
            dataframe[i].apply(pd.to_numeric)
            
    return dataframe, target, categorical, numerical
