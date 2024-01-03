# Importo librerie utili
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# La funzione trasforma il set di training in base alla scelta dell'utente, standardizzando tutti i valori numerici
def standardize_X_train(dataframe, X_train, X_test, step_further) :
    """
    La funzione accetta i seguenti argomenti:
    - dataframe: dataframe per controllare la presenza di valori mancanti
    - X_train: set attributi per il training
    - X_test: set attributi per la validation
    - step_further: flag di avanzamento
    La funzione restituisce:
    - X_train
    - X_test
    - Tra_num: trasformazione applicata
    - step_further
    - flag_stand: tipologia standardizzazione
    """

    st.text("")
    st.text("")
    st.text("")
    st.write("### Standardization")
    st.write("The standardization defined from the Training set will be applied to the Test set")

    # Inizializzazione flag relative al metodo utilizzato
    flag_stand = 0

    # Menu dropdown per la scelta del motodo da utilizzare
    Tra_num = st.selectbox( 'How would you like to normalize data?', ['','MinMaxScaler','StandardScaler','Do not normalize'] )

    # Gestione dei vari casi
    if Tra_num == '' :                                        # L'app non va avanti se non viene selezionato un valore (grazie a step_further non aggiornaot)
        pass
    elif Tra_num == 'MinMaxScaler':
        Set_scaler = MinMaxScaler()
        Set_scaler.fit(X_train)
        X_train = pd.DataFrame(Set_scaler.transform(X_train), columns = X_train.columns)
        X_test = pd.DataFrame(Set_scaler.transform(X_test), columns = X_test.columns)
        flag_stand, step_further = 1, 6
    elif Tra_num == 'StandardScaler':
        Set_scaler = StandardScaler()
        Set_scaler.fit(X_train)
        X_train = pd.DataFrame(Set_scaler.transform(X_train), columns = X_train.columns)
        X_test = pd.DataFrame(Set_scaler.transform(X_test), columns = X_test.columns)
        flag_stand, step_further = 2, 6
    elif Tra_num == 'Do not normalize':
        flag_stand, step_further = 3, 6

    return X_train, X_test, Tra_num, step_further, flag_stand
