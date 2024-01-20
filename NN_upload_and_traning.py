# Importo librerie utili
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
from itertools import product
from datetime import datetime
import matplotlib.pyplot as plt
from htbuilder.units import percent, px
from sklearn.model_selection import train_test_split
from htbuilder import HtmlElement, div, hr, a, p, img, styles
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer, OneHotEncoder
from sklearn.metrics import precision_score,recall_score,f1_score, accuracy_score, confusion_matrix, r2_score, precision_recall_curve, roc_auc_score, roc_curve

# Importo i moduli custom presenti nella cartella "Functions" e il modello di rete neurale nella cartella "Model"
# In alternativa, lasciarli nella stessa cartella di questo file .py e chiamarli con from name_module import * 
from model import NeuralNetwork                                                                                        # Modello Neural Network
from functions import remove_missing, feature_selection, imputation_process, missing_target, categoric_to_numeric      # Pulizia e trasformazione dataset
from functions import custom_split, standardize_x_train, target_transformation, nn_builder, metrics_plot               # Training, Finalizzazione, e Pipeline
from functions import model_finalization, test_pipeline                                                                # Finalizzazione e Pipeline
from functions import footer                                                                                           # Footer

##################################################################################################################################################################################################

# Imposto la pagina
st.set_page_config(
    page_title = "Prediction with Neural Networks",
    page_icon = "🧊",
    layout = "wide",
    menu_items = { 'About': "This is simple app to guide users and build a Neural Network model to make predictions." }
)
st.markdown('''<style> section.main > div {max-width:75rem} </style>''', unsafe_allow_html = True)

# Titolo
st.write("# Predictions with Neural Networks")
st.write("")
st.write("This is a simple app to guide you in the process of applying a custom Neural Network model to a dataset. Currently")
st.write("- The app supports binary classification, multivariate classification, and regression analyses.")
st.write("- The app cannot process dates in your dataset.")
st.write("")
st.write("### Upload Data")
uploaded_file = st.file_uploader("Choose a CSV file for the analysis")        # Pulsante per upload dati (file CSV)
st.write("Important: the delimiter in the csv file must be a semicolon!")

if uploaded_file is not None:                                                 # Check se il file è stato caricato o meno - Tutto lo script si basa sul caricamento o meno di un file
    st.write("File successfully uploaded!")                                   # Messaggio di caricamento     
    dataframe = pd.read_csv(uploaded_file, delimiter = ';')
    if st.checkbox('Show dataframe', key = 50):                
        st.write(dataframe)

    # Primo check sui valori mancanti
    st.write("Number of missing values")
    st.write(  pd.DataFrame( {'# Missing values': np.array(dataframe.isna().sum()), '% Missing values': np.array(100*dataframe.isna().sum()/dataframe.shape[0])}, index = dataframe.columns))
    st.write("**Note**: Columns with more than 70\% of missing data will be removed from the dataset")
    st.write("**Note**: Rows with more than 70\% of missing data will be removed from the dataset")

    # Rimozione righe e colonne con più del 70% di valori mancanti    
    dataframe, Selected_columns_start = remove_missing.remove_missing(dataframe)

    # Type of analysis
    Task1 = st.selectbox( 'What is the task of this analysis?', ['','Classification','Regression'] )
    st.write("If the task is 'Classification', the final model will predict classes; if 'Regression', it will predict numbers.")

    # L'app si avvia solo se è stata scelta la tipologia di analisi
    if Task1 :
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Selezione delle varie features (numeriche e categoriche) e della colonna target e correzione colonne numeriche del dataframe
        st.text("")
        st.text("")
        st.text("")
        st.write("### Select Target, Categorical, and Numerical features")
        dataframe, Tar, Categ, Numer = feature_selection.feature_selection(dataframe, Task1)
        st.write("Unselected columns will be excluded from the dataset.")
        
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Gestione dei valori mancanti nelle colonne categoriche e categoriche, e infine i valori mancanti nella colonna Target (che vengono eliminati)
        st.text("")
        st.text("")
        st.text("")
        st.write("### Deal with missing data")   
        st.write("If the target features has missing values, they will be dropped from the dataset")
        dataframe, categ_impute, numer_impute, Sub_categ_list, Sub_num_list, step_further, a, b = imputation_process.imputation(dataframe, Categ, Numer)
        dataframe, step_further = missing_target.missing_target(dataframe, Categ, Numer, Tar, a, b, step_further)
        st.write("Number of missing values: %d" % dataframe.isna().sum().sum())
        if st.checkbox('Show dataframe', key = 53):                
            st.write(dataframe)
    
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Conversione dei valori categorici in valori numerici    
        # Check dello step, della presenza di colonne categoriche e dell'assenza di valori mancanti (altrimenti, lo script si interrompe)
        if (step_further == 3) and (len(Categ) != 0 and Categ != ["None"]) and (dataframe.isna().sum().sum() == 0) :
            dataframe, Tra_categ_list, step_further = categoric_to_numeric.categoric_to_numeric(dataframe, Categ, step_further)
        elif (dataframe.isna().sum().sum() != 0) :
            st.write("Something's wrong with the data. Missing values are still there...")
        else :
            step_further, Tra_categ_list = 4, [[], None, []]
        
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Creazione dei set di Training e Validation per la valutazione del modello
        # Check iniziale per mostrare la sezione
        if step_further == 4 and (Categ or Numer) and dataframe.isna().sum().sum() == 0:
            X, y, X_train, X_test, y_train, y_test, step_further, final_columns = custom_split.train_test_customsplit(dataframe, Tar, step_further)
            st.session_state["Final_columns"] = final_columns
            
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Standardizzazione del dataframe X per il Training
        # Check avanzamento e valori nulli
        if step_further == 5 and dataframe.isna().sum().sum() == 0:
            X_train, X_test, Tra_num, step_further, flag_stand = standardize_x_train.standardize_x_train(dataframe, X_train, X_test, step_further)
                
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Trasformazione della colonna Target
        if step_further == 6 :
            y_train, y_test, Norm_tar_list, step_further = target_transformation.target_transformation(dataframe, Tar, y_train, y_test, step_further, Task1)
            if st.checkbox('Show Target (training set)', key = 61):                
                st.write(y_train)
    
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Costruzione della Rete Neurale
        # Dopo il primo training, è possibile finalizzare il modello ed applicare un nuovo file di test
        if step_further == 7 and Task1 :
            # Scelta dei diversi parametri da parte dell'utente e costruzione dell'oggetto "Model"
            Hidden_layers, Algo, Alpha, Regularization, Momentum, Early_stopping, Verbose, Max_iter, Function_, Batch, Decay, Lambda, Random_state, Patient, Final_metric = nn_builder.nn_builder(dataframe, Task1)
            Model = NeuralNetwork.NeuralNet(task = Task1, function = Function_, Hidden_layers = Hidden_layers, algo = Algo, batch = Batch, alpha = float(Alpha), decay = float(Decay), 
                            regularization = Regularization, Lambda = float(Lambda), Max_iter = int(Max_iter), momentum = float(Momentum), random_state = int(Random_state), verbose = Verbose,
                            early_stopping = Early_stopping, patient = int(Patient), flag_plot = False, metric = Final_metric)
            st.text("")
            st.text("")
            st.text("")
            if st.button('Start the training!'):                                                                              # Pulsante per avviare il training
                Model.Training(X_train, y_train, X_test, y_test)                                                              # Training NN
                st.write('Training complete!')
                metrics_plot.metrics_plot(Model, X_train, X_test, y_train, y_test, Task1, Norm_tar_list, Final_metric)        # Calcolo metriche finali (per Regressione) e plot


                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.tree import DecisionTreeClassifier
                # Apply models
                if Task1 == 'Classification':
                    model_log = LogisticRegression()
                    model_ran = RandomForestClassifier(n_estimators = 100, random_state=42)
                    model_tree = DecisionTreeClassifier()

                    accuracy_models = []
                    for model in [model_log, model_ran, model_tree] :
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        accuracy_models.append( accuracy_score(y_test, predictions) )

                    st.write(pd.DataFrame(accuracy_models, columns = ["Logistic regression", "Random Forest", "Decision Tree"] )#, index = ["Accuracy"]))
                             
                elif Task1 == 'Regression':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    mse = mean_squared_error(y_test, predictions)
                    st.write(f'Mean Squared Error: {mse}')
                else:
                    st.write('Invalid task. Supported tasks are "classification" and "regression".')

    
            #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # Model Finalization
            st.text("")
            st.text("")
            st.text("")
            st.write("Now, if you want, you can finalize the model!")
            st.write("The Training and the Test set will be used together to create the final model.")
            st.write("The dataset will be standardized and the Target normalized using the methods defined above.") 
            st.write(r"After the model finalization, you can download the JSON file with the model parameters and weights.")
            
            flag_finalization = 0                            # Flag finalizzazione
            if "ButBut" not in st.session_state :            # Pulsante finalizzazione
                st.session_state["ButBut"] = False
            else :
                st.session_state["ButBut"] = st.button("Finalization of the model")
    
            # Inizio processo di finalizzazione del modello: eseguo il training finale e salvo i parametri in un file JSON
            if st.session_state["ButBut"] :
                Final_model, Tra_num_list_final, flag_finalization, Norm_tar_list_final = model_finalization.finalization(X, y, Model, Task1, Final_metric, Tra_num, Norm_tar_list, flag_stand)
    
                # Salvo tutto nella sessione (per far in modo che, una volta caricato il file di test, tutti parametri rimangano salvati)
                st.session_state["Final_model"] = Final_model
                st.session_state["Tra_num_list_final"] = Tra_num_list_final
                st.session_state["Norm_tar_list_final"] = Norm_tar_list_final
                st.session_state["flag_finalization"] = flag_finalization
    
                # Salvataggio del modello finale nel file "Best_model_parameters.json"
                file_name = "Best_model_parameters.json"
                Final_model.Save_model(file_name)
    
            #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # Model application on a new file
            st.write("")
            st.write("")
            st.write("### Upload Data for testing the model")
            st.write("If you finalized your model, you can upload a new file to apply the best model")
            st.write("The file must have the same number of columns of the original file with the same names.")
            uploaded_file_test = st.file_uploader("Choose a CSV file to apply the best model")                    # Upload a new file
            if uploaded_file_test is not None and st.session_state["flag_finalization"] == 1:                     # Check se è stato caricato un file tramite pulsante   
                # Trasformazione del dataset con pipeline
                X_test_final, y_test_final, dataframe_test = test_pipeline.pipeline_nn(uploaded_file_test, Selected_columns_start, Numer, Categ, Tar, Sub_num_list, Sub_categ_list, 
                                                Tra_categ_list, st.session_state["Final_columns"], st.session_state["Tra_num_list_final"], st.session_state["Norm_tar_list_final"], Task1)
    
                Predictions_test = st.session_state["Final_model"].Predict( X_test_final )                        # Applicazione del modello finale
                if Task1 == "Classification" :                                                                    # Calcolo predizioni e probabilità (solo per classificazione)
                    Predictions_prob_test = st.session_state["Final_model"].Predict_proba( X_test_final )
                    dataframe_test["Probability"] = Predictions_prob_test
                if Task1 == "Regression" :
                    if st.session_state["Norm_tar_list_final"][0] == 3:
                        Predictions_test = st.session_state["Norm_tar_list_final"][1].inverse_transform(Predictions_test)
                    elif st.session_state["Norm_tar_list_final"][0] == 1:
                        Predictions_test = 10**Predictions_test + 1
                
                dataframe_test["Predictions"] = Predictions_test
                st.write("")
                st.write("Uploaded table with predictions:")
                st.write( dataframe_test )
                
                # Converto il dataframe e lo salvo in un file csv (se l'utente clicca un pulsante)
                dataframe_test = dataframe_test.to_csv(index = False).encode('utf-8')
                st.download_button( "Download the dataframe", dataframe_test, "Predictions.csv", "text/csv", key = 'download-csv' )
    
                st.write("")
                st.write("If you want to make changes to the original model:")
                st.write(" 1. Modify the model parameters accordingly")
                st.write(" 2. Re-train and re-finalize the model")
                st.write("After that, you will see the new predictions in the uploaded dataframe.")
                st.write("")
                st.write("-----------------------------")
                st.write("")
                st.write("")
                html_str = f"""<style>p.a {{font: bold 23.5px Sans;}}</style><p class="a">Everything is done!</p>"""                
                st.markdown(html_str, unsafe_allow_html = True)

    # If a task has not been chosen
    else :
        st.write("")
        st.write("")
        st.write("Choose a task for this analysis!")

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Footer (le funzioni utilizzate sono in functions.py)
if __name__ == "__main__":
    myargs = [ "Made in ", footer.image_render('https://avatars3.githubusercontent.com/u/45109972?s=400&v=4', width = px(25), height = px(25)), 
        " by ", footer.link_render("https://www.linkedin.com/in/samuele-campitiello-ph-d-913b90104/", "Samuele Campitiello") ]
    footer.footer(*myargs)
