# Importo librerie utili
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer, OneHotEncoder

# La funzione calcola le metriche finali del modello dopo il training e crea dei plot
def Metrics_plot(Model, X_train, X_test, y_train, y_test, Task, Norm_tar_list, Final_metric) :
    """
    La funzione accetta i seguenti argomenti:
    - Model: modello di rete neurale (dopo il traning)
    - X_train, X_test: set di attributi per training e validation
    - y_train, y_test: colonna target per training e validation
    - Task: task dell'analisi
    - Norm_tar_list: metodo di normalizzazione della colonna target
    - Final_metric: metrica di valutazione scelta dall'utente
    La funzione restituisce:
    - Plot Learning curve, sia per regressione che per classificazione
    - Plot y_real vs. y_predicted (solo per regressione)
    """
    # Calcolo delle metriche finali
    if Task == "Classification" :
        res_tr = Model.metric_tr
        res_te = Model.metric_te
    elif Task == "Regression" and Norm_tar_list[3] == 20:
        # E' necessario trasformare la colonna target per il calcolo delle metriche reali
        res_tr = np.sqrt( ((Norm_tar_list[1].inverse_transform(Model.Predict(X_train)) -  Norm_tar_list[1].inverse_transform(y_train))**2).sum()/len(y_train)  )
        res_te = np.sqrt( ((Norm_tar_list[1].inverse_transform(Model.Predict(X_test)) -  Norm_tar_list[1].inverse_transform(y_test))**2).sum()/len(y_test) )
        st.write('Training real RMSE: {:.5f} -- Validation real RMSE: {:.5f}'.format(res_tr, res_te))
    elif Task == "Regression" and Norm_tar_list[3] == 10:
        res_tr = np.sqrt( (( 10**Model.Predict(X_train) -  10**(y_train))**2).sum()/len(y_train)  )
        res_te = np.sqrt( (( 10**Model.Predict(X_test) -  10**(y_test))**2).sum()/len(y_test) )
        st.write('Training real RMSE: {:.5f} -- Validation real RMSE: {:.5f}'.format(res_tr, res_te))
    
    # Salvo risultati in session
    st.session_state["res_tr"]  = res_tr
    st.session_state["res_te"]  = res_te

    # Plot delle Learning curves
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize = (15, 4))
    ax1.plot(Model.cost_function_tr, '-b')
    ax1.plot(Model.cost_function_te,'-r')
    ax1.legend(["Training set","Test set"])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cost function")
    
    # Confronto risultati Test (solo per regressione)
    #st.pyplot(fig)
    if Task == "Regression" :
        if Norm_tar_list[3] == 20 :
            ax2.plot(Norm_tar_list[1].inverse_transform(Model.Predict(X_test)), Norm_tar_list[1].inverse_transform(y_test), 'ob')
            rangg = np.arange(Norm_tar_list[1].inverse_transform(Model.Predict(X_test)).min(), Norm_tar_list[1].inverse_transform(Model.Predict(X_test)).max())
            ax3.hist(Norm_tar_list[1].inverse_transform(Model.Predict(X_test)) - Norm_tar_list[1].inverse_transform(y_test), bins = 20, color = 'blue', alpha = 0.7)
        elif Norm_tar_list[3] == 10 :
            ax2.plot( 10**Model.Predict(X_test)+1, 10**(y_test)+1, 'ob')
            rangg = np.arange( (10**Model.Predict(X_test) +1 ).min(), (10**Model.Predict(X_test)).max())
            ax3.hist(10**Model.Predict(X_test) - 10**(y_test), bins = 20, color = 'blue', alpha = 0.7)
        else :
            ax2.plot(Model.Predict(X_test),y_test, 'ob')
            rangg = np.arange(Model.Predict(X_test).min(), Model.Predict(X_test).max())
            ax3.hist(Model.Predict(X_test) - y_test, bins = 20, color = 'blue', alpha = 0.7)
        ax2.plot(rangg,rangg, '-r')
        ax2.set_xlabel("Predictions")
        ax2.set_ylabel("Actual values")
        ax3.set_xlabel("Predictions - Actual")

    st.pyplot(fig)

    return True
