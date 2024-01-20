# Importo librerie utili
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer, OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, r2_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# La funzione calcola le metriche finali del modello dopo il training e crea dei plot
def metrics_plot(Model, X_train, X_test, y_train, y_test, Task, Norm_tar_list, Final_metric) :
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

    if Task == "Classification" or (Task == "Regressione" and Norm_tar_list[3] == 0) :        # Per "Classificazione", le metriche sono calcolate direttamente dal modello
        res_tr, res_te = Model.last_metric_tr, Model.last_metric_te
    elif Task == "Regression" and Norm_tar_list[3] == 20:                                     # Per "Regressione" con MinMax
        y_real_tr = Norm_tar_list[1].inverse_transform(y_train)
        y_real_te = Norm_tar_list[1].inverse_transform(y_test)
        y_predicted_tr = Norm_tar_list[1].inverse_transform(Model.Predict(X_train))
        y_predicted_te = Norm_tar_list[1].inverse_transform(Model.Predict(X_test))
        if Final_metric == "RMSE" :
            res_tr = mean_squared_error(y_real_tr, y_predicted_tr, squared = False)
            res_te = mean_squared_error(y_real_te, y_predicted_te, squared = False)
        elif Final_metric == "MAE" :
            res_tr = mean_absolute_error(y_real_tr, y_predicted_tr)
            res_te = mean_absolute_error(y_real_te, y_predicted_te)
        elif Final_metric == "R2" :
            res_tr = r2_score(y_real_tr, y_predicted_tr)
            res_te = r2_score(y_real_te, y_predicted_te)
        
    elif Task == "Regression" and Norm_tar_list[3] == 10:                                     # Per "Regressione" con Log(x+1)
        y_real_tr = 10**(y_train) + 1
        y_real_te = 10**(y_test) + 1
        y_predicted_tr = 10**Model.Predict(X_train) + 1
        y_predicted_te = 10**Model.Predict(X_test) + 1
        if Final_metric == "RMSE" :
            res_tr = mean_squared_error(y_real_tr, y_predicted_tr, squared = False)
            res_te = mean_squared_error(y_real_te, y_predicted_te, squared = False)
        elif Final_metric == "MAE" :
            res_tr = mean_absolute_error(y_real_tr, y_predicted_tr)
            res_te = mean_absolute_error(y_real_te, y_predicted_te)
        elif Final_metric == "R2" :
            res_tr = r2_score(y_real_tr, y_predicted_tr)
            res_te = r2_score(y_real_te, y_predicted_te)

    # Apply other models
    if Task == 'Classification':
        Col_final = ["Logistic regression", "Decision Tree", "Random Forest"]
        ID_final = ["Accuracy", "Precision", "Recall", "F1-score"]
        other_models = [ LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators = 100, random_state = 42) ]
        acc_models, pre_models, rec_models, f1_models = [], [], [], []
        for model in other_models :
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            acc_models.append( accuracy_score(y_test, predictions) )
            pre_models.append( precision_score(y_test, predictions, average = "weighted") )
            rec_models.append( recall_score(y_test, predictions, average = "weighted") )
            f1_models.append( f1_score(y_test, predictions, average = "weighted") )
        other_results = pd.DataFrame( [acc_models, pre_models, rec_models, f1_models],  columns = Col_final, index = ID_final ) 
        st.write(other_results)
        
    elif Task == 'Regression':
        model = RandomForestRegressor(n_estimators = 100, random_state = 42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        st.write(f'Mean Squared Error: {mse}')
    else:
        st.write('Invalid task. Supported tasks are "classification" and "regression".')
        
    st.write('Training real {}: {:.5f} -- Validation real {}: {:.5f}'.format(Final_metric, res_tr, Final_metric, res_te))
    st.session_state["res_tr"], st.session_state["res_te"]  = res_tr, res_te                 # Salvo risultati in session

    # Creazione del plot finale
    if Task == "Regression" :
        fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize = (15, 4))
    else :
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize = (10, 4))
    
    # Plot delle learning curves
    ax1.plot(Model.cost_function_tr, '-b')
    ax1.plot(Model.cost_function_te,'-r')
    ax1.legend(["Training set","Test set"])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cost function")
    
    # Confronto risultati Test (tre plot solo per Regressione)
    if Task == "Regression" :
        if Norm_tar_list[3] == 20 :
            ax2.plot(Norm_tar_list[1].inverse_transform(Model.Predict(X_test)), Norm_tar_list[1].inverse_transform(y_test), 'ob')
            ax3.hist(Norm_tar_list[1].inverse_transform(Model.Predict(X_test)) - Norm_tar_list[1].inverse_transform(y_test), bins = 20, color = 'blue', alpha = 0.7)
            rangg = np.arange(Norm_tar_list[1].inverse_transform(Model.Predict(X_test)).min(), Norm_tar_list[1].inverse_transform(Model.Predict(X_test)).max())
        elif Norm_tar_list[3] == 10 :
            ax2.plot( 10**Model.Predict(X_test)+1, 10**(y_test)+1, 'ob')
            ax3.hist(10**Model.Predict(X_test) - 10**(y_test), bins = 20, color = 'blue', alpha = 0.7)
            rangg = np.arange( (10**Model.Predict(X_test) +1 ).min(), (10**Model.Predict(X_test)).max())
        else :
            ax2.plot(Model.Predict(X_test),y_test, 'ob')
            ax3.hist(Model.Predict(X_test) - y_test, bins = 20, color = 'blue', alpha = 0.7)
            rangg = np.arange(Model.Predict(X_test).min(), Model.Predict(X_test).max())
        ax2.plot(rangg,rangg, '-r')
        ax2.set_xlabel("Predictions")
        ax2.set_ylabel("Actual values")
        ax3.set_xlabel("Predictions - Actual")
    
    # Due plot per le analisi di Classificazione
    else :
        ax2.plot(Model.metric_tr, '-b')
        ax2.plot(Model.metric_te,'-r')
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("{}".format(Final_metric))

    return st.pyplot(fig)
