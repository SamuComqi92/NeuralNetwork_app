# Librerie utili
import pandas as pd
import numpy as np
import streamlit as st

# Funzione per eseguire una pipeline
def Pipeline_NN(uploaded_file_test, Selected_columns_start, Numer, Categ, Tar, Sub_num_list, Sub_categ_list, 
                Tra_categ_list, dataframe_orig, Tra_num_list, Norm_tar_list) :
    """ La funzione applica tutte le trasformazioni scelte dall'utente sul file CSV di test caricato tramite pulsante.
    La funzione restituisce il dataframe con gli attributi e la colonna target, entrambi trasformati, e il dataframe_test completo finale (trasformato).
    Gli argomenti della funzione sono i seguenti:
     - uploaded_file_test: file CSV caricato tramite pulsante con Streamlit
     - Selected_columns_start: colonne iniziali (sono escluse le colonne con troppi valori mancanti)
     - Numer: lista delle colonne numeriche
     - Categ: lista delle colonne categoriche
     - Tar: colonna target
     - Sub_num_list: lista con metodo di imputation dei valori numerici mancanti e valore (nel caso di sostituzione con media o mediana)
     - Sub_categ_list: lista con metodo di imputation dei valori categorici mancanti e valore (nel caso di sostituzione con moda oppure "NAN")
     - Tra_categ_list: lista con metodo di trasformazione delle colonne categoriche e possibile encoder
     - dataframe_orig: dataframe originale da cui prendere i valori unici
     - Tra_num_list: lista con metodo di trasformazione delle colonne numeriche e possibile encoder
     - Norm_tar_list: lista con flag di trasformazione della colonna target e possibile min_max_scaler
    """
 
    # Salvo il file caricato in un dataframe
    # Quest'ultimo verrà modificato a seconda delle scelte dell'utente
    try:
        dataframe_test = pd.read_csv(uploaded_file_test, delimiter = ';')
        if len(dataframe_test.columns) == 1 :
            dataframe_test = pd.read_csv(uploaded_file_test, delimiter = ',')
    except Exception as e:
        st.error(f"Error: {e}")
    
    # Drop rows and columns with more than 70% of missing data
    dataframe_test = dataframe_test[Selected_columns_start]

    #Convert to numeric (conversion of CSV file does not always work)
    for i in Numer :
        dataframe_test[i] = dataframe_test[i].astype(str).str.replace(',', '.').astype(float)
        dataframe_test[i].apply(pd.to_numeric)

    # Trasformazione della colonna Target
    for i in Tar :
        dataframe_test[i] = dataframe_test[i].astype(str).str.replace(',', '.').astype(float)
        dataframe_test[i].apply(pd.to_numeric)
        
    # Missing numerical features
    if Sub_num_list[0] == '' :
        pass
    elif Sub_num_list[0] == 'Substitute null values with the mean':
        dataframe_test[Numer] = dataframe_test[Numer].fillna(Sub_num_list[1])
    elif Sub_num_list[0] == 'Substitute null values with the median':
        dataframe_test[Numer] = dataframe_test[Numer].fillna(Sub_num_list[1])
    elif Sub_num_list[0] == 'Drop rows (careful!)' :
        dataframe_test[Numer] = dataframe_test[Numer].isnull().to_numpy().nonzero()[0]

    # Missing categorical features
    if Sub_categ_list[0] == '' :
        pass
    elif Sub_categ_list[0] == 'Substitute null values with string NAN':
        dataframe_test[Categ] = dataframe_test[Categ].fillna(Sub_categ_list[1])
    elif Sub_categ_list[0] == 'Substitute null values with the mode':
        dataframe_test[Categ] = dataframe_test[Categ].fillna(Sub_categ_list[1])
    elif Sub_categ_list[0] == 'Drop rows (careful!)' :
        dataframe_test[Categ] = dataframe_test[Categ].isnull().to_numpy().nonzero()[0]

    # Missing values in target column
    dataframe_test = dataframe_test.dropna(subset = Tar)

    # Categorical to numerical column transformation
    if Tra_categ_list[0] == '' :
        pass
    elif Tra_categ_list[0] == 'OneHotEncoder':
        for i in Categ :
            columns_test = np.unique(dataframe_test[i].astype(str))
            m=0
            for j in columns_test :
                columns_test[m] = i + "_" + columns_test[m]
                m=m+1
            transformed_test = pd.DataFrame(Tra_categ_list[1].transform(dataframe_test[i].astype(str)), columns = columns_test)
            dataframe_test = pd.concat([transformed_test, dataframe_test], axis=1).drop([i], axis=1)
    elif Tra_categ_list[0] == 'String to numbers':
        for i in Categ :
            st.write(Tra_categ_list[2][i])
            dataframe_test[i].replace(np.unique(dataframe_test[i]),np.arange(0,len(np.unique(dataframe_orig[i]))),inplace=True)
            
    # Creation of X (attributes) and y (target)
    X_test_final = pd.DataFrame(dataframe_test.drop(Tar,axis=1), columns = dataframe_test.drop(Tar,axis=1).columns)
    y_test_final = np.array( dataframe_test[Tar] )

    # Standardize the file
    if Tra_num_list[0] == '' :
        pass
    elif Tra_num_list[0] == 'MinMaxScaler' or Tra_num_list[0] == 'StandardScaler':
        X_test_final = pd.DataFrame(Tra_num_list[1].transform(X_test_final), columns = X_test_final.columns)
    elif Tra_num_list[0] == 'Do not normalize':
        pass

    # Transform target
    if Norm_tar_list[0] == '' or Norm_tar_list[0] == 'No' :
        pass
    else :
        if float(dataframe_test[Tar].min()) >= 0. :
            if float(dataframe_test[Tar].max()) > 1. :
                y_test_final = np.log10(y_test_final+1)
            else :
                pass
        elif float(dataframe_test[Tar].min()) < 0. :
            y_test_final = np.array(pd.DataFrame(Norm_tar_list[1].transform(y_test_final)))

    # Dopo tutte le trasformazioni, restituisco il dataframe con gli attributi e la colonna target, e il dataframe_test completo
    return X_test_final, y_test_final, dataframe_test
