# Importo librerie utili
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer, OneHotEncoder

# La funzione trasforma, se necessario, la colonna target, soprattutto in casi con modelli di Regressione
def Target_transformation(dataframe, Tar, y_train, y_test, step_further) :
    """
    La funzione accetta i seguenti argomenti:
    - dataframe: da controllare per capire come trasformare la colonna target
    - Tar: colonna target
    - y_train: target del training
    - y_test: target della validation
    - step_further: flag di avanzamento
    La funzione restituisce:
    - y_train
    - y_test
    - Norm_tar_list: lista che conterrà il metodo scelto per la Pipeline finale
    - step_further
    """
    # Inizializzazione dei flag relativi al tipo di normalizzazione usata
    flag_norm = 0
    #if step_further == 6 :

    # Menu drop down per scegliere se trasformare o meno la colonna target
    Norm_tar = st.selectbox(
            'Transform target feature with logarithms or MinMax (negative values)? (Especially for Regression problems with large numbers)',
            ['','No','Yes'])

    # Inizializzazione flag del tipo di trasformazione (10: log, 20: minmax)
    flag_transf = 0

    # Gestione delle diverse casistiche
    if Norm_tar == '':
        target_minmax = None
        pass
    elif Norm_tar == 'No' :
        target_minmax = None
        step_further = 7
        pass
    else :
        if float(dataframe[Tar].min()) >= 0. :
            target_minmax = None
            step_further = 7
            if float(dataframe[Tar].max()) > 1. :
                y_train = np.log10(y_train+1)
                y_test = np.log10(y_test+1)
                st.write("Target values have been transformed with Log(x+1)")
                flag_norm = 1
                flag_transf = 10
            else :
                step_further = 7
                st.write("Target values are already in a range between 0 and 1")
                flag_norm = 2
        elif float(dataframe[Tar].min()) < 0. :
            st.write("MinMaxScaler will be used (reducing the target values in a range between 0 and 1)")
            target_minmax = MinMaxScaler()
            flag_transf = 20
            target_minmax.fit(y_train)
            y_train = np.array(pd.DataFrame(target_minmax.transform(y_train)))
            y_test = np.array(pd.DataFrame(target_minmax.transform(y_test)))
            flag_norm = 3
            step_further = 7
    
    # Creo lista per l'applicazione finale della pipeline
    Norm_tar_list = [Norm_tar, target_minmax, flag_norm, flag_transf]

    return y_train, y_test, Norm_tar_list, step_further
