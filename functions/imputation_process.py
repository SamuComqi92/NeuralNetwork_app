# Importo librerie utili
import pandas as pd
import numpy as np
import streamlit as st

# La funzione sostituisce i valori mancanti nelle colonne categoriche in base al metodo scelto dall'utente
def imputation(dataframe, categorical, numerical) :
    """
    La funzione accetta come argomenti:
    - dataframe: il dataframe da controllare e correggere
    - categorical: la lista di colonne categoriche scelte dall'utente
    - numerical: la lista di colonne numeriche scelte dall'utente
    La funzione restituisce:
    - Dataframe modificato
    - Numeri relativi ai metodi utilizzati
    - Liste che contiengono il metodo e il valore per la sostituzione (utili nella Pipeline finale)
    - Flag step_further
    - Array con i valori eliminati dal dataframe (se si è scelto tale metodo)
    """

    # Categorical features
    step_further, categ_impute, numer_impute = 0, 0, 0                                                # Inizializzazione dello step_further e del flag relativo al metodo di sostituzione                            
    if categorical and dataframe.isna().sum().sum() != 0:                                             # Controllo che ci siano valori mancanti nel dataframe e che ci siano colonne numeriche
        categorical_method = st.selectbox(                                                            # Scelta del metodo di imputation
            'Which imputation method (for NA values) do you like best for categorical features?',
            ['','Substitute null values with string NAN','Substitute null values with the mode','Drop rows (careful!)'])

        if categorical_method == 'Drop rows (careful!)' :                                             # Rimozione delle righe con valori mancanti
            a = dataframe[categorical].isnull().to_numpy().nonzero()[0]                               # Righe eliminate     
            value_imputation_cat, step_further, categ_impute = None, 1, 3
        else :
            a = np.array([])
            if categorical_method == '' :                                                             # Nessuna imputation (lo script non va avanti se non si sceglie un metodo)
                value_imputation_cat = None                                                           # La sospensione è dovuta allo step_further non aggiornato
            elif categorical_method == 'Substitute null values with string NAN':                      # Sostituzione con stringa "NAN"
                value_imputation_cat = "NAN"
                dataframe[categorical] = dataframe[categorical].fillna(value_imputation_cat)
                step_further, categ_impute = 1, 1
            elif categorical_method == 'Substitute null values with the mode':                        # Sostituzione con la moda
                value_imputation_cat = dataframe[categorical].mode().iloc[0]
                dataframe[categorical] = dataframe[categorical].fillna(value_imputation_cat)
                step_further, categ_impute = 1, 2
    else :                                                                                            # Se non ci sono colonne categoriche
        step_further, categorical_method, value_imputation_cat, a = 1, "", None, np.array([])

    # Numerical features
    if numerical and dataframe.isna().sum().sum() != 0:                                              # Controllo che ci siano valori mancanti nel dataframe e che ci siano colonne numeriche
        numerical_method = st.selectbox(                                                             # Scelta del metodo di imputation
            'Which imputation method (for NA values) do you like best for numerical features?',
            ['','Substitute null values with the mean','Substitute null values with the median','Drop rows (careful!)'])

        if numerical_method == 'Drop rows (careful!)' :                                              # Rimozione delle righe con valori mancanti
            b = dataframe[Numer].isnull().to_numpy().nonzero()[0]                                    # Righe eliminate
            value_imputation_num, step_further, numer_impute = None, 2, 3
        else :
            b = np.array([])
            if numerical_method == '' :                                                              # Nessuna imputation
                value_imputation_num = None
            elif numerical_method == 'Substitute null values with the mean':                         # Sostituzione con media
                value_imputation_num = dataframe[numerical].mean()
                dataframe[numerical] = dataframe[numerical].fillna(value_imputation_num)
                step_further, numer_impute = 2, 1
            elif numerical_method == 'Substitute null values with the median':                       # Sostituzione con mediana
                value_imputation_num = dataframe[numerical].median()
                dataframe[numerical] = dataframe[numerical].fillna(value_imputation_num)
                step_further, numer_impute = 2, 2
    else :                                                                                           # Se non ci sono colonne numeriche
        step_further, numerical_method, value_imputation_num, b = 2, "", None, np.array([])  

    # Creo liste dei parametri che mi serviranno nell'applicazione della pipeline finale
    list_numerical = [numerical_method, value_imputation_num]
    list_categorical = [categorical_method, value_imputation_cat]

    return dataframe, categ_impute, numer_impute, list_categorical, list_numerical, step_further, a, b
