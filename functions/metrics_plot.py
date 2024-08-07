# Importo librerie utili
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer, OneHotEncoder, label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, r2_score, roc_auc_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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

    if Task == "Classification" or (Task == "Regression" and Norm_tar_list[3] == 0) :        # Per "Classificazione", le metriche sono calcolate direttamente dal modello
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

    left_column, right_column = st.columns(2)             # Nella parte principale, crea tre colonne dove posso sistemare testi e bottoni
    with left_column:    
        nn_results = pd.DataFrame( [res_tr, res_te],  index = ["Training", "Validation"], columns = [Final_metric] ).T
        st.write("")
        st.write("Neural Network results:")
        st.write(nn_results)
        st.session_state["res_tr"], st.session_state["res_te"]  = res_tr, res_te                 # Salvo risultati in session
    with right_column :
        # Apply other models
        if Task == 'Classification':
            Col_final = ["Logistic regression", "Decision Tree", "Random Forest"]
            Models = [ LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators = 100, random_state = 42) ]
            results_metric = []
            for model in Models :
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                if Final_metric == "Accuracy" :
                    results_metric.append( accuracy_score(y_test, predictions) )
                elif Final_metric == "Precision" :
                    if len(np.unique(y_test)) == 2 :
                        results_metric.append( precision_score(y_test, predictions) )
                    else :
                        results_metric.append( precision_score(y_test, predictions, average = "weighted") )
                elif Final_metric == "Recall" :
                    if len(np.unique(y_test)) == 2 :
                        results_metric.append( recall_score(y_test, predictions) )
                    else :
                        results_metric.append( recall_score(y_test, predictions, average = "weighted") )
                elif Final_metric == "F1 score" :
                    if len(np.unique(y_test)) == 2 :
                        results_metric.append( f1_score(y_test, predictions) )
                    else :
                        results_metric.append( f1_score(y_test, predictions, average = "weighted") )
                elif Final_metric == "AUC" :
                    if len(np.unique(y_test)) == 2 :
                        results_metric.append( roc_auc_score(y_encoded, predictions) )
                    else :
                        y_prob = model.predict_proba(X_test)
                        y_test_bin = label_binarize(y_test, classes = list(np.unique(y_test)))
                        results_metric.append( roc_auc_score(y_test_bin, y_prob, average = "weighted", multi_class = "ovr") )
                        
        elif Task == 'Regression':
            Col_final = ["Linear regression", "Decision Tree", "Random Forest"]
            Models = [ LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(n_estimators = 100, random_state = 42) ]
            results_metric = []
            for model in Models :
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                if Norm_tar_list[3] == 20:                                     # Per "Regressione" con MinMax
                    y_real = Norm_tar_list[1].inverse_transform(y_test.reshape(1, -1))
                    y_predicted = Norm_tar_list[1].inverse_transform(predictions.reshape(1, -1))
                elif Norm_tar_list[3] == 10:                                   # Per "Regressione" con Log(x+1)
                    y_real = 10**(y_test) + 1
                    y_predicted = 10*predictions + 1

                # Final metrics
                if Final_metric == "RMSE" :
                    results_metric.append( mean_squared_error(y_real, y_predicted, squared = False) )
                elif Final_metric == "MAE" :
                    results_metric.append( mean_absolute_error(y_real, y_predicted) )
                elif Final_metric == "R2" :
                    results_metric.append( r2_score(y_real, y_predicted) )

        other_results = pd.DataFrame( results_metric,  index = Col_final, columns = [Final_metric] ).T
        st.write("")
        st.write("Other models (validation set):")
        st.write(other_results)
           

    # Creazione del plot finale
    if Task == "Regression" :
        fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize = (15, 4))
    else :
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize = (10, 4))
    
    # Plot delle learning curves
    ax1.plot(Model.cost_function_tr, '-b')
    ax1.plot(Model.cost_function_te,'-r')
    ax1.legend(["Training set","Validation set"])
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
        ax3.set_xlabel("Predictions - Actual (validation set)")
    
    # Due plot per le analisi di Classificazione
    else :
        ax2.plot(Model.metric_tr, '-b')
        ax2.plot(Model.metric_te,'-r')
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("{}".format(Final_metric))

    return st.pyplot(fig)
