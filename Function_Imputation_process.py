# Importo librerie utili
import pandas as pd
import numpy as np
import streamlit as st

# La funzione sostituisce i valori mancanti nelle colonne categoriche in base al metodo scelto dall'utente
def Imputation_process(dataframe, Categ, Numer) :
    """
    La funzione accetta come argomenti:
    - dataframe: il dataframe da controllare e correggere
    - Categ: la lista di colonne categoriche scelte dall'utente
    - Numer: la lista di colonne numeriche scelte dall'utente
    La funzione restituisce:
    - Dataframe modificato
    - Numeri relativi ai metodi utilizzati
    - Liste che contiengono il metodo e il valore per la sostituzione (utili nella Pipeline finale)
    - Flag step_further
    - Array con i valori eliminati dal dataframe (se si è scelto tale metodo)
    """

    # CATEGORICAL
    # Inizializzazione dello step_further e del flag relativo al metodo di sostituzione
    step_further = 0
    categ_impute = 0
    numer_impute = 0

    # Controllo che ci siano valori mancanti nel dataframe e che ci siano colonne numeriche
    if Categ and dataframe.isna().sum().sum() != 0:
        # Scelta del metodo di imputation
        Sub_categ = st.selectbox(
            'Which imputation method (for NA values) do you like best for categorical features?',
            ['','Substitute null values with string NAN','Substitute null values with the mode','Drop rows (careful!)'])

        if Sub_categ == '' :                                                            # Nessuna imputation (lo script non va avanti se non si sceglie un metodo - grazie al flag step_further)
            Value_imputation_cat = None
            a = np.array([])
        elif Sub_categ == 'Substitute null values with string NAN':                     # Sostituzione con stringa "NAN"
            dataframe[Categ] = dataframe[Categ].fillna("NAN")
            Value_imputation_cat = "NAN"
            a = np.array([])
            step_further = 1
            categ_impute = 1
        elif Sub_categ == 'Substitute null values with the mode':                       # Sostituzione con la moda
            Value_imputation_cat = dataframe[Categ].mode().iloc[0]
            dataframe[Categ] = dataframe[Categ].fillna(Value_imputation_cat)
            a = np.array([])
            step_further = 1
            categ_impute = 2
        elif Sub_categ == 'Drop rows (careful!)' :                                      # Rimozione delle righe con valori mancanti
            Value_imputation_cat = None
            a = dataframe[Categ].isnull().to_numpy().nonzero()[0]
            step_further = 1
            categ_impute = 3

    # Se non ci sono colonne categoriche
    else :
        step_further = 1
        Sub_categ = ""
        Value_imputation_cat = None
        a = np.array([])
        pass       

    #--------------------------------------------------------------------------------------------------------------------------------------------
    # NUMERICAL
    # Controllo che ci siano valori mancanti nel dataframe e che ci siano colonne numeriche
    if Numer and dataframe.isna().sum().sum() != 0:
        Sub_num = st.selectbox(
            'Which imputation method (for NA values) do you like best for numerical features?',
            ['','Substitute null values with the mean','Substitute null values with the median','Drop rows (careful!)'])

        if Sub_num == '' :                                                              # Nessuna imputation
            Value_imputation = None
            b = np.array([])
            step_further = 2
        elif Sub_num == 'Substitute null values with the mean':                         # Sostituzione con media
            dataframe[Numer] = dataframe[Numer].fillna(dataframe[Numer].mean())
            Value_imputation = dataframe[Numer].mean()
            b = np.array([])
            step_further = 2
            numer_impute = 1
        elif Sub_num == 'Substitute null values with the median':                       # Sostituzione con mediana
            dataframe[Numer] = dataframe[Numer].fillna(dataframe[Numer].median())
            Value_imputation = dataframe[Numer].median()
            b = np.array([])
            step_further = 2
            numer_impute = 2
        elif Sub_num == 'Drop rows (careful!)' :                                        # Rimozione delle righe con valori mancanti
            Value_imputation = None
            b = dataframe[Numer].isnull().to_numpy().nonzero()[0]
            step_further = 2
            numer_impute = 3

    # Se non ci sono colonne numeriche
    else :
        step_further = 2
        Sub_num = ""
        Value_imputation = None
        b = np.array([])
        pass        

    # Creo liste dei parametri che mi serviranno nell'applicazione della pipeline finale
    Sub_num_list = [Sub_num, Value_imputation]
    Sub_categ_list = [Sub_categ, Value_imputation_cat]

    return dataframe, categ_impute, numer_impute, Sub_categ_list, Sub_num_list, step_further, a, b