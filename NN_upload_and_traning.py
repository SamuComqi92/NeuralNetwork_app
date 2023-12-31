# Importo librerie utili
import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_score,recall_score,f1_score, accuracy_score, confusion_matrix, r2_score, precision_recall_curve, roc_auc_score, roc_curve

# Importo i moduli custom presenti nella cartella "Functions" e il modello di rete neurale nella cartella "Model"
# In alternativa, lasciarli nella stessa cartella di questo file .py e chiamarli con from name_module import *
# Modello Neural Network
from model import NeuralNetwork
# Pulizia e trasformazione dataset
from functions import Function_Remove_70_missing, Function_Feature_selection, Function_Imputation_process, Function_Missing_target, Function_Categoric_to_numeric
# Training, Finalizzazione, e Pipeline
from functions import Function_Train_Test_Split, Function_Standard_X_train, Function_Target_transformation, Function_NN_Builder, Function_Metrics_plot
# Finalizzazione e Pipeline
from functions import Function_Model_Finalization, Function_Pipeline

##################################################################################################################################################################################################################
##################################################################################################################################################################################################################

# Titolo
st.write("# Predictions with Neural Networks")

# Pulsante per upload dati (file CSV)
st.write("### Upload Data")
uploaded_file = st.file_uploader("Choose a CSV file for the analysis")

# Check se il file è stato caricato o meno - Tutto lo script si basa sul caricamento o meno di un file
if uploaded_file is not None:   

    # Messaggio di caricamento            
    st.write("File successfully uploaded!")
    
    # Salvo il file caricato in un dataframe
    # Quest'ultimo verrà modificato a seconda delle scelte dell'utente
    # Leggo il dataframe con separatore ;
    # Se il separatore è diverso, uso un if per ri-leggerlo.
    dataframe = pd.read_csv(uploaded_file, delimiter=",")
    #if len( dataframe.columns ) == 1 :
    #    dataframe = pd.read_csv(uploaded_file, delimiter=",")
        
    if st.checkbox('Show dataframe', key=50):                
        st.write(dataframe)

    # Primo check sui valori mancanti
    st.write("Number of missing values")
    st.write(  pd.DataFrame( {'# Missing values':np.array(dataframe.isna().sum()),'% Missing values':np.array(100*dataframe.isna().sum()/dataframe.shape[0])},index=dataframe.columns))
    st.write("**Note**: Columns with more than 70\% of missing data will be removed from the dataset")
    st.write("**Note**: Rows with more than 70\% of missing data will be removed from the dataset")

    # Rimozione righe e colonne con più del 70% di valori mancanti    
    dataframe, Selected_columns_start = Function_Remove_70_missing.Remove_70_missing(dataframe)

    ##############################################################################################################################################################################################################
    # Selezione delle varie features (numeriche e categoriche) e della colonna target
    # Correzzione colonne numeriche del dataframe
    st.text("")
    st.text("")
    st.text("")
    st.write("### Select Target, Categorical, and Numerical features")
    dataframe, Tar, Categ, Numer = Function_Feature_selection.Feature_selection(dataframe)
   

    ##############################################################################################################################################################################################################
    # Gestione dei valori mancanti nelle colonne categoriche e categoriche, e infine i valori mancanti nella colonna Target (che vengono eliminati)
    # Sia per le colonne numeriche che categoriche, viene data la possibilità di applicare 3 metodi diversi per gestire i valori mancanti.
    st.text("")
    st.text("")
    st.text("")
    st.write("### Deal with missing data")   
    st.write("If the target features has missing values, they will be dropped from the dataset")
    dataframe, categ_impute, numer_impute, Sub_categ_list, Sub_num_list, step_further, a, b = Function_Imputation_process.Imputation_process(dataframe, Categ, Numer)
    dataframe, step_further = Function_Missing_target.Missing_target(dataframe, Categ, Numer, Tar, a, b, step_further)
    st.write("Number of missing values: %d" % dataframe.isna().sum().sum())

    # Checkbox per mostrare il dataframe pulito
    if st.checkbox('Show dataframe', key = 53):                
        st.write(dataframe)

    ##############################################################################################################################################################################################################
    # Conversione dei valori categorici in valori numerici
    # Viene data la possibilità di convertire i valori in due modi diversi
    # Check dello step, della presenza di colonne categoriche e dell'assenza di valori mancanti (altrimenti, lo script si interrompe)
    if step_further == 3 and Categ.count("None") == 0 and dataframe.isna().sum().sum() == 0 :
        dataframe, Tra_categ_list, step_further = Function_Categoric_to_numeric.Categoric_to_numeric(dataframe, Categ, step_further)
    else :
        Tra_categ = []
        jobs_encoder = None
        step_further = 4
        Tra_categ_list = [Tra_categ, jobs_encoder]

    ##############################################################################################################################################################################################################
    # Creazione dei set di Training e Validation per la valutazione del modello
    # Check iniziale per mostrare la sezione
    if step_further == 4 and (Categ or Numer) and dataframe.isna().sum().sum() == 0:
        X, y, X_train, X_test, y_train, y_test, step_further = Function_Train_Test_Split.Train_Test_Split(dataframe, Categ, Numer, Tar, step_further)
        
    ##############################################################################################################################################################################################################
    # Standardizzazione del dataframe X per il Training
    # Check avanzamento e valori nulli
    if step_further == 5 and dataframe.isna().sum().sum() == 0:
        X_train, X_test, Tra_num, step_further, flag_stand = Function_Standard_X_train.Standard_X_train(dataframe, X_train, X_test, step_further)

    ##############################################################################################################################################################################################################
    # Trasformazione della colonna Target
    if step_further == 6 :
        y_train, y_test, Norm_tar_list, step_further = Function_Target_transformation.Target_transformation(dataframe, Tar, y_train, y_test, step_further)

        # Checkbox per mostrare i meno il target
        if st.checkbox('Show Target (training set)', key = 61):                
            st.write(y_train)

    ##############################################################################################################################################################################################################
    # Costruzione della Rete Neurale
    # Dopo il primo training, è possibile finalizzare il modello ed applicare un nuovo file di test
    if step_further == 7 :

        # Scelta dei diversi parametri da parte dell'utente e costruzione dell'oggetto "Model"
        Task, Hidden_layers, Algo, Alpha, Regularization, Momentum, Early_stopping, Verbose, Max_iter, Function_, Batch, Decay, Lambda, Random_state, Patient, Final_metric = Function_NN_Builder.NN_Builder(dataframe)
        Model = NeuralNetwork.NeuralNet(task=Task, function = Function_, Hidden_layers = Hidden_layers, algo = Algo, batch = None, alpha = float(Alpha), decay = float(Decay), 
                        regularization = Regularization, Lambda = float(Lambda), Max_iter = int(Max_iter), momentum = float(Momentum), random_state = int(Random_state), verbose = Verbose,
                        early_stopping = Early_stopping, patient = int(Patient), flag_plot = False, metric = Final_metric)
    
        # Sessione di training del modello
        st.text("")
        st.text("")
        st.text("")
        if st.button('Start the training!'):
            # Training NN
            Model.Training(X_train, y_train, X_test, y_test)
            st.write('Training complete!')

            # Calcolo metriche finali (per Regressione) e plot
            Function_Metrics_plot.Metrics_plot(Model, X_train, X_test, y_train, y_test, Task, Norm_tar_list)


        ########################################################################################################################################################################################################
        # Model Finalization
        st.text("")
        st.text("")
        st.text("")
        st.write("Now, if you want, you can finalize the model!")
        st.write("The Training and the Test set will be used together to create the final model.")
        st.write("The dataset will be standardized and the Target normalized using the methods defined above.") 
        st.write(r"After the model finalization, you can download the JSON file with the model parameters and weights.")

        # Flag finalizzazione
        flag_finalization = 0

        # Pulsante finalizzazione
        if "ButBut" not in st.session_state :
            st.session_state["ButBut"] = False
        else :
            st.session_state["ButBut"] = st.button("Finalization of the model")

        # Inizio processo di finalizzazione del modello
        # Eseguo il training finale e salvo i parametri in un file JSON
        if st.session_state["ButBut"] :
            # Finalizzazione del modello
            Final_model, Tra_num_list_final, flag_finalization, Norm_tar_list_final = Function_Model_Finalization.Model_Finalization(X, y, Model, Task, Final_metric, Tra_num, Norm_tar_list, flag_stand)

            # Salvo tutto nella sessione (per far in modo che, una volta caricato il file di test, tutti parametri rimangano salvati)
            st.session_state["Final_model"] = Final_model
            st.session_state["Tra_num_list_final"] = Tra_num_list_final
            st.session_state["flag_finalization"] = flag_finalization
            st.session_state["Norm_tar_list_final"] = Norm_tar_list_final

            # Salvataggio del modello finale nel file "Best_model_parameters.json"
            file_path = "Best_model_parameters.json"
            Final_model.Save_model(file_path)

        ########################################################################################################################################################################################################
        # Model application on a new file
        st.write("")
        st.write("")
        st.write("### Upload Data for testing the model")
        st.write("If you finalized your model, you can upload a new file to apply the best model")
        st.write("The file must have the same number of columns of the original file with the same names.")

        # Upload a new file
        uploaded_file_test = st.file_uploader("Choose a CSV file to apply the best model")

        # Check se è stato caricato un file tramite pulsante
        if uploaded_file_test is not None and st.session_state["flag_finalization"] == 1:         
            
            # Trasformazione del dataset con pipeline
            X_test_final, y_test_final, dataframe_test = Function_Pipeline.Pipeline_NN(uploaded_file_test, Selected_columns_start, Numer, Categ, Tar, Sub_num_list, Sub_categ_list, 
                                                                                       Tra_categ_list, dataframe, st.session_state["Tra_num_list_final"], st.session_state["Norm_tar_list_final"])

            # Applicazione del modello finale
            Predictions_test = st.session_state["Final_model"].Predict( X_test_final )
            
            # Calcolo predizioni e probabilità (solo per classificazione)
            if Task == "Classification" :
                Predictions_prob_test = st.session_state["Final_model"].Predict_proba( X_test_final )
                dataframe_test["Predictions"] = Predictions_test
                dataframe_test["Probability"] = Predictions_prob_test
            if Task == "Regression" and st.session_state["Norm_tar_list_final"][0] == 3:
                Predictions_test = st.session_state["Norm_tar_list_final"][1].inverse_transform(Predictions_test)
                dataframe_test["Predictions"] = Predictions_test
            elif Task == "Regression" and st.session_state["Norm_tar_list_final"][0] == 1:
                Predictions_test = 10**Predictions_test + 1
                dataframe_test["Predictions"] = Predictions_test

            # Stampo il dataframe finale con le predizioni (e probabilità solo per classificazione)
            st.write("")
            st.write("Uploaded table with predictions:")
            st.write( dataframe_test )
            
            # Converto il dataframe e lo salvo in un file csv (se l'utente clicca un pulsante)
            dataframe_test = dataframe_test.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download the dataframe",
                dataframe_test,
                "Predictions.csv",
                "text/csv",
                key='download-csv'
                )

            st.write("")
            st.write("If you want to make changes to the original model:")
            st.write(" 1. Modify the model parameters accordigly")
            st.write(" 2. Re-train and re-finalize the model")
            st.write("After that, you will see the new predictions in the uploaded dataframe.")
            st.write("")
            st.write("-----------------------------")
            st.write("")
            st.write("")
            st.write("Everything is done!")
