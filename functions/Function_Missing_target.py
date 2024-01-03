# Importo librerie utili
import pandas as pd
import numpy as np
import streamlit as st

# La funzione sostituisce i valori mancanti nelle colonne categoriche in base al metodo scelto dall'utente
def Missing_target(dataframe, categorical, numerical, target, a, b, step_further) :
    """
    La funzione accetta come argomenti:
    - dataframe: il dataframe da controllare e correggere
    - categorical: la lista di colonne categoriche scelte dall'utente
    - numerical: la lista di colonne numeriche scelte dall'utente
    - target: colonna target scelta dall'utente
    - a: array valori eliminati dalle colonne categoriche (se tale metodo di imputation è stato scelto)
    - b: array valori eliminati dalle colonne numeriche (se tale metodo di imputation è stato scelto)
    - step_further: l'ultimo flag aggiornato
    La funzione restituisce:
    - Dataframe modificato
    - Flag step_further
    """
    # Check dello step e delle scelte dell'utente
    if step_further == 2 and (categorical or numerical):
        step_further = 3
        miss = np.unique( np.concatenate( (a, b, dataframe[target].isnull().to_numpy().nonzero()[0]), axis = 0) )
        dataframe = dataframe.drop(index = miss, axis = 0)
        dataframe = dataframe.reset_index()
        
    return dataframe, step_further
