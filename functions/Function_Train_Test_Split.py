# Importo librerie utili
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split

# La funzione splitta il dataframe in Training e Test set, in base alla scelta dell'utente
def Train_Test_Split(dataframe, Categ, Numer, Tar, step_further) :
    """
    La funzione accetta i seguenti argomenti
    - dataframe: dataframe da usare per lo split
    - Categ: lista delle colonne categoriche
    - Numer: lista delle colonne numeriche
    - Tar: colonne target
    - step_further: step di avanzamento
    La funzione restitusce:
    - X: array degli attributi
    - y: array del target
    - X_train: set attributi per il training
    - X_test: set attributi per la validation
    - y_train: target per il training
    - y_test: target per la validation
    - step_further
    """
    
    st.text("")
    st.text("")
    st.text("")
    st.write("### Creation of Training and Test set")
    
    # Creazione di X (attributi) e y (target) come array
    X = np.array( dataframe.drop(Tar,axis=1).drop(["index"], axis=1) )
    y = np.array( dataframe[Tar] )

    # Slider per scegliere le dimensioni del Validation set
    # Eventuali errori dovuti a valori categorici non distribuiti in modo corretto si possono risolvere aumentando le dimensioni del training set
    Size_test = st.slider('Select the size of the Test set (% with respecto the the total number of data)',  100*1/len(X), 100*(len(X)-1)/len(X), 70.0)

    st.write('If the training process raises an error due to categorical values not found in the validation set, try to increase the training set size.')
    
    # Creazione dei due set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 - Size_test/100, random_state = 42)
    st.write("Number of data in the Training set: %d" % len(X_train))
    st.write("Number of data in the Test set: %d" % len(X_test))

    # Conversione dei due array in dataframe Pandas
    X_train = pd.DataFrame(X_train, columns = dataframe.drop(Tar,axis=1).drop(["index"], axis=1).columns)
    X_test = pd.DataFrame(X_test, columns = dataframe.drop(Tar,axis=1).drop(["index"], axis=1).columns)
    step_further = 5

    return X, y, X_train, X_test, y_train, y_test, step_further
