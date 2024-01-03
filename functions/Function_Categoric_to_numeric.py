# Importo librerie utili
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelBinarizer

# La funzione converte i valori categorici in valori numerici (per l'applicazione del modello di Machine Learning) in base al metodo scelto dall'utente
def Categoric_to_numeric(dataframe, categorical_features, step_further) :
    """
    La funzione accetta i seguenti argomenti:
    - dataframe: da controllare
    - categorical_features: lista di colonne categoriche scelte dall'utente
    - step_further: flag di avanzamento
    La funzione restituisce i seguenti risultati:
    - dataframe modificato
    - list_transformation: lista con informazioni relative al metodo utilizzato
    - step_further: step di avanzamento
    """

    # Menu dropdown per la scelta del metodo di conversione
    menu_transformation = st.selectbox( 'How would you like to transform categorical features?', ['','OneHotEncoder','String to numbers'] )

    # Gestione delle varie scelte
    if menu_transformation == 'OneHotEncoder':
        jobs_encoder = LabelBinarizer()                        # Definizione oggetto encoder
        for i in categorical_features :                        # Procedure di rinominazione colonne
            columns = np.unique(dataframe[i].astype(str))
            m = 0
            for j in columns :
                columns[m] = i + "_" + columns[m]
                m = m + 1
            # Trasformazione del dataframe
            transformed = pd.DataFrame(jobs_encoder.fit_transform(dataframe[i].astype(str)), columns = columns)
            dataframe = pd.concat([transformed, dataframe], axis = 1).drop([i], axis = 1)
    else :
        jobs_encoder = None
        if menu_transformation == '' :
            pass
        elif menu_transformation == 'String to numbers':
            for i in categorical_features :
                dataframe[i].replace(np.unique(dataframe[i]), np.arange(0,len(np.unique(dataframe[i]))), inplace = True)

    # Aggiornamento dello step
    step_further = 4

    # Possibilità di visualizzare il dataframe
    if menu_transformation != '':
        if st.checkbox('Show dataframe', key = 530):                
            st.write(dataframe)

    # Creazione lista per l'applicazione della pipeline finale
    list_transformation = [menu_transformation, jobs_encoder]

    return dataframe, list_transformation, step_further
