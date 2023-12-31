# Importo librerie utili
import pandas as pd
import streamlit as st


# Funzione per scegliere le feature categoriche, numeriche e target da usare
def Feature_selection(dataframe) :
    """
    La funzione accetta come unico argomento il dataframe da controllare. La funzione converte anche in modo corretto le colonne numeriche
    La funzione restituisce il dataframe con valori numerici corretti, la colonna target, e le liste delle colonne categoriche e numeriche
    """
    # Check per controllare se il dataframe non sia vuoto
    if dataframe.empty == False :
        Tar = st.multiselect('Select the target:', dataframe.columns, key = 100)

    # Lista colonne
    Lista_col = dataframe.drop(Tar,axis = 1).columns.tolist()
    Lista_col.insert(0,'All')
    Lista_col.insert(0,'None')

    # Menu dropdown per multiselezione delle colonne categoriche
    Categ = st.multiselect("Select categorical features:", Lista_col, key = 1)
    if "All" in Categ:
        Categ = dataframe.drop(Tar, axis = 1).columns.tolist()
    elif "None" in Categ :
        pass

    # Menu dropdown per multiselezione delle colonne numeriche
    Numer = st.multiselect("Select numerical features:", Lista_col, key = 2)
    if "All" in Numer:
        Numer = dataframe.drop(Tar,axis = 1).columns.tolist()
    elif "None" in Numer :
        pass

    # Conversione delle colonne numeriche
    # Può capitare che dopo la lettura di un file CSV, i dati numerici (con decimali) non siano convertiti in modo corretto
    for i in Numer :
        dataframe[i] = dataframe[i].astype(str).str.replace(',', '.').astype(float)
        dataframe[i].apply(pd.to_numeric)

    return dataframe, Tar, Categ, Numer
