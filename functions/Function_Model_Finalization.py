# Importo librerie utili
import streamlit as st
import pandas as pd
import numpy as np
from itertools import product
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_score,recall_score,f1_score, accuracy_score, confusion_matrix, r2_score, precision_recall_curve, roc_auc_score, roc_curve

# La funzione utilizza l'intero dataset a disposizione per finalizzare il modello con i parametri scelti dall'utente
def Model_Finalization(X, y, Model, Task, Final_metric, Tra_num, Norm_tar_list, flag_stand) :
    """
    La funzione accetta i seguenti argomenti:
    - X: set attributi completo (senza split)
    - y: colonna target completa (senza split)
    - Model: modello finale con i parametri scelti dall'utente
    - Task: tipo di analisi
    - Final_metric: metrica utilizzata per valutare il modello (scelta dall'utente)
    - Tra_num: metodo standardizzazione utilizzato (per salvarlo nella lista)
    - Norm_tar_list: lista con metodo di trasformazione utilizzato per la colonna target
    - flag_stand: flag relativo al metodo di standardizzazione di X utilizzato in precedenza
    La funzione restituisce:
    - st.session_state.Final_model: modello finale dopo il training completo (salvato in sessione)
    - st.session_state["Tra_num_list_final"]: lista con informazioni su standardizzazione di X (salvata in sessione)
    - st.session_state["flag_finalization"]: flag di finalizzazione (salvato in sessione)
    - Norm_tar_list_final: lista con metodo di trasformazione per la colonna target
    """
    flag_finalization = 1
    st.session_state["flag_finalization"] = flag_finalization

    # Standardizzazzione di X
    if flag_stand == 0 or flag_stand == 3 :
        XX_train = X
        pass
    elif flag_stand == 1:
        Set_scaler_x = MinMaxScaler()
        Set_scaler_x.fit(X)
        XX_train = pd.DataFrame(Set_scaler_x.transform(X))
    elif flag_stand == 2:
        Set_scaler_x = StandardScaler()
        Set_scaler_x.fit(X)
        XX_train = pd.DataFrame(Set_scaler_x.transform(X))

    # Salvo il nuovo scaler nella sessione
    if flag_stand > 0 and flag_stand < 3 and ("Set_scaler_x" not in st.session_state) :          
        st.session_state.Set_scaler_x  = Set_scaler_x

    # Creo lista per l'applicazione della pipeline finale
    Tra_num_list_final = [Tra_num, Set_scaler_x]
    st.session_state["Tra_num_list_final"] = Tra_num_list_final

    # Normalizzazione di y
    if Norm_tar_list[2] == 0 or Norm_tar_list[2] == 2 :
        target_minmax_y = None
        yy_train = y
        pass
    elif Norm_tar_list[2] == 1:
        target_minmax_y = None
        yy_train = np.log10(y+1)
    elif Norm_tar_list[2] == 3:
        target_minmax_y = MinMaxScaler()
        target_minmax_y.fit(y)
        yy_train = np.array(pd.DataFrame(target_minmax_y.transform(y)))

    # Lista per l'applicazione finale
    Norm_tar_list_final = [Norm_tar_list[2], target_minmax_y]

    # Applicazione del modello a tutto il dataset
    Model.Training(XX_train, yy_train, XX_train, yy_train)

    # Salvo il modello finale nella sessione (questo verrà utilizzato per l'applicazione finale)
    if "Final_model" not in st.session_state :          
        st.session_state.Final_model = Model
    else :
        st.session_state.Final_model = Model

    # Stampo i risultati finali rispetto a quelli calcolati con il primo training (salvati in sessione)
    # MinMax
    if Task == "Regression" and Norm_tar_list[3] == 20:
        res_tr_final = np.sqrt( ((target_minmax_y.inverse_transform(Model.Predict(XX_train)) -  target_minmax_y.inverse_transform(yy_train))**2).sum()/len(yy_train) )
        st.write('Real RMSE: {:.5f}'.format(Final_metric, res_tr_final))
        st.write("Previous RMSE: {:.5f}".format(Final_metric, st.session_state.res_tr))
    # Log(x+1)
    elif Task == "Regression" and Norm_tar_list[3] == 10:
        res_tr_final = np.sqrt( (( 10**Model.Predict(XX_train) -  10**(yy_train) )**2).sum()/len(yy_train) )
        st.write('Real RMSE: {:.5f}'.format(Final_metric, res_tr_final))
        st.write("Previous RMSE: {:.5f}".format(Final_metric, st.session_state.res_tr))
    # Nessuna trasformazione
    else :
        st.write("Final {}: {:.5f}".format(Final_metric, Model.metric_te))
        st.write("Previous {}: {:.5f}".format(Final_metric, st.session_state.res_te))

    return st.session_state.Final_model, Tra_num_list_final, st.session_state["flag_finalization"], Norm_tar_list_final
