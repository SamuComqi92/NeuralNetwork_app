# Importo librerie utili
import streamlit as st
import numpy as np


# La funzione permette all'utente di scegliere i vari parametri per costuire la rete neurale
def nn_builder(dataframe, Task) :
    """
    La funzione accetta i seguenti argomenti:
    - dataframe: dataframe originale
    La funzione restituisce tutti i parametri settati dall'utente
    """
    
    step_further = 8            # Aggiornamento step di avanzamento
    
    st.text("")
    st.text("")
    st.text("")
    st.write("### Build the Neural Network")
    st.write("Set all the hyper-parameters of the Neural Network")

    # Task of the analysis
    html_str = f"""<style>p.a {{font: bold 23.5px Sans;}}</style><p class="a">The task is: {Task}</p>"""
    st.markdown(html_str, unsafe_allow_html=True)

    # Main parameters
    html_str2 = f"""<style>p.a {{font: bold 15px Sans;}}</style><p class="a">These are the main parameters of your Neural Network (you may leave their default values).</p>"""
    st.markdown(html_str2, unsafe_allow_html=True)

    # Defining three columns
    left_column, center_column, right_column = st.columns(3)             # Nella parte principale, crea tre colonne dove posso sistemare testi e bottoni
    with left_column:                                                    # Qui scelgo di scrivere cose solo nela parte destra
        st.write('Hidden layers and units')
        Hidden_layers = st.text_input("Write the units for each hidden layer separated by a comma (e.g., '5,6' means that there are two layers with 5 and 6 units respectively).", "5")
        Hidden_layers = tuple(map(int, Hidden_layers.split(',')))       # Convertion of the list into a tuple

        st.text("")
        st.text("")
        st.write('Learning Rate')
        Alpha = st.text_input('Learning Rate (0.3 by default - write a value). A very small value corresponds to a slow-learning algorithm.', '0.3')

    with center_column :
        st.write('Activation function')
        Function_ = st.text_input("Write the activation function (more than one if you want to use a different function for each hidden layer) - 'Sigmoid', 'Tanh', 'Relu', 'Leaky_relu', 'Elu', 'Swish'","Sigmoid")
        Function_ = tuple(map(str, Function_.split(', ')))             # Convertion of the list into a tuple

        st.text("")
        st.text("")
        st.write("Evaluation Metric")
        if Task == 'Classification' :
            Metrics_final = ["Accuracy", "Precision", "Recall", "F1 score", "AUC"]
        else :
            Metrics_final = ["RMSE", "MAE", "R2"]
        Final_metric = st.selectbox( 'Choose the Evaluation metric (for the evaluation of the training and the validation sets).', Metrics_final )
        
    with right_column :
        st.write('Number of Iterations')
        Max_iter = st.text_input('Maximum number of iterations. The defaul value is 100. The larger the number of iterations, the larger the runtime.', '100')
        
        st.text("")
        st.text("")
        st.write('Random State')
        Random_state = st.text_input('A random state value for reproducible results (it is a value that you can choose randomly).', '0')

    # Other parameters
    st.write("")
    st.write("")
    st.text("")
    st.text("")
    html_str3 = f"""<style>p.a {{font: bold 15px Sans;}}</style><p class="a">These are other advanced parameters (you may leave their default values).</p>"""
    st.markdown(html_str3, unsafe_allow_html=True)

    left_column2, center_column2, right_column2 = st.columns(3)            # Nella parte principale, crea tre colonne dove posso sistemare testi e bottoni
    with left_column2:   
        Algo = st.selectbox( 'Optimization algorithm', ["Batch","Adam"] )

        st.text("")
        st.text("")
        Regularization = st.selectbox( 'Type of Regularization', ["None","Ridge","Lasso"] )

        st.text("")
        st.text("")
        Early_stopping = st.selectbox( 'Flag to apply Early-stopping to the algorithm (for Adam, it is better to set it to "False")', ["False","True"])

    with center_column2:   
        Batch = st.selectbox( 'Size of mini-batches (0: all data will be used)', np.arange(0,len(dataframe)) )

        st.text("")
        st.text("")
        Lambda = st.text_input('Regularization factor Lambda (0 by default)', '0')

        st.text("")
        st.text("")
        Patient = st.text_input('The number of epochs to check for early stopping (for Adam and/or with Momentum)', '5')
        
    with right_column2 :
        Verbose = st.selectbox( 'Flag to display results (0: do not deplay, 1: deplay)', [1, 0])

        st.text("")
        st.text("")
        Decay = st.text_input('Decay parameter for the Learning rate (0 by default)','0')

        st.text("")
        st.text("")
        Momentum = st.text_input('The momentum factor in the weights optimization (the value must be equal or larger than 0).', '0')

    # Check del numero di activation functions e del numero di Hidden layers
    # Se scelgo più di una funzione di attivazione, il numero complessivo deve essere uguale al numero di layer della Rete
    if len(Function_) < len(Hidden_layers) :
        Copy_func = Function_
        for i in range(len(Hidden_layers)-1) :        # Same activation function for each layer
            Copy_func = Copy_func + Function_
        Function_ = Copy_func
    elif len(Function_) > len(Hidden_layers) :
        st.write("The number of hidden layers must be equal to the number of activation functions!")

    return Hidden_layers, Algo, Alpha, Regularization, Momentum, Early_stopping, Verbose, Max_iter, Function_, Batch, Decay, Lambda, Random_state, Patient, Final_metric
