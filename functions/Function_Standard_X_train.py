# Importo librerie utili
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# La funzione trasforma il set di training in base alla scelta dell'utente, standardizzando tutti i valori numerici
def Standard_X_train(dataframe, X_train, X_test, step_further) :
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
    # Inizializzazione flag relative al metodo utilizzato
    flag_stand = 0

    # Check avanzamento e valori nulli
    #if step_further == 5 and dataframe.isna().sum().sum() == 0:
    st.text("")
    st.text("")
    st.text("")
    st.write("### Standardization")
    st.write("The standardization defined from the Training set will be applied to the Test set")

    # Menu dropdown per la scelta del motodo da utilizzare
    Tra_num = st.selectbox( 'How would you like to normalize data?', ['','MinMaxScaler','StandardScaler','Do not normalize'] )

    # Gestione dei vari casi
    if Tra_num == '' :
        pass
    elif Tra_num == 'MinMaxScaler':
        flag_stand = 1
        step_further = 6
        Set_scaler = MinMaxScaler()
        Set_scaler.fit(X_train)
        X_train = pd.DataFrame(Set_scaler.transform(X_train), columns = X_train.columns)
        X_test = pd.DataFrame(Set_scaler.transform(X_test), columns = X_test.columns)
    elif Tra_num == 'StandardScaler':
        flag_stand = 2
        step_further = 6
        Set_scaler = StandardScaler()
        Set_scaler.fit(X_train)
        X_train = pd.DataFrame(Set_scaler.transform(X_train), columns = X_train.columns)
        X_test = pd.DataFrame(Set_scaler.transform(X_test), columns = X_test.columns)
    elif Tra_num == 'Do not normalize':
        flag_stand = 3
        step_further = 6
        pass

    return X_train, X_test, Tra_num, step_further, flag_stand
