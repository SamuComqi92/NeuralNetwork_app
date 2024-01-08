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
    # Check dello step di avanzamento
    step_further = 8
    st.text("")
    st.text("")
    st.text("")
    st.write("### Build the Neural Network")
    st.write("Set all the hyper-parameters of the Neural Network")

    # Task of the analysis
    st.write("The task is:")
    html_str = f"""
        <style>
        p.a {{
          font: bold 23.5px Sans;
        }}
        </style>
        <p class="a">{Task}</p>
        """
        
    st.markdown(html_str, unsafe_allow_html=True)

    # Main parameters
    st.write("These are the main parameters of your Neural Network (you can leave their default values).")
    left_column, right_column = st.columns(2)            # Nella parte principale, crea due colonne dove posso sistemare testi e bottoni
    with left_column:                                    # Qui scelgo di scrivere cose solo nela parte destra
        st.write('Hidden layers and units')
        Hidden_layers = st.text_input("Write the units for each hidden layer separated by a comma (e.g., '5,6' means that there are two layers with 5 and 6 units respectively).", "5")
        Hidden_layers = tuple(map(int, Hidden_layers.split(',')))      # Convertion of the list into a tuple

        st.text("")
        st.text("")
        st.write('Learning Rate')
        Alpha = st.text_input('Learning Rate (0.3 by default - write a value). A very small value corresponds to a slow-learning algorithm.', '0.3')

        st.text("")
        st.text("")
        if Task == 'Classification' :
            Metrics_final = ["Accuracy", "Precision", "Recall", "F1 score", "AUC"]
        else :
            Metrics_final = ["RMSE", "MAE", "R2"]
        Final_metric = st.selectbox(
            'Choose the Evaluation metric (the metric will be used to evaluate the training and the validation sets)',
            Metrics_final)
        
        # st.text("")
        # st.text("")
        # Algo = st.selectbox(
        #     'Optimization algorithm',
        #     ["Batch","Adam"]
        #     )

        

        # st.text("")
        # st.text("")
        # Regularization = st.selectbox(
        #     'Type of Regularization',
        #     ["None","Ridge","Lasso"])

        # st.text("")
        # st.text("")
        # Momentum = st.text_input('The momentum factor in the weights optimization', '0')

        # st.text("")
        # st.text("")
        # Early_stopping = st.selectbox(
        #     'A flag to apply Early-stopping to the algorithm (for Adam, it is better to set it to "False")',
        #     ["False","True"])

        # st.text("")
        # st.text("")
        # Verbose = st.selectbox(
        #     'A flag to display results while processing (0: do not deplay, 1: deplay)',
        #     [0, 1])

    with right_column :
        st.write('Number of Iterations')
        Max_iter = st.text_input('Maximum number of iterations. The defaul value is 100. The larger the number of iterations, the larger the runtime.', '100')
        
        st.text("")
        st.text("")
        st.write('Activation function')
        Function_ = st.text_input("Write the activation function (more than one if you want to use a different function for each hidden layer) - 'Sigmoid', 'Tanh', 'Relu', 'Leaky_relu', 'Elu', 'Swish'","Sigmoid")
        Function_ = tuple(map(str, Function_.split(', ')))             # Convertion of the list into a tuple

        st.text("")
        st.text("")
        st.write('Random State')
        Random_state = st.text_input('A random state value for reproducible results', '0')
        
        # st.text("")
        # st.text("")
        # Batch = st.selectbox(
        #     'Size of mini-batches (0: all data will be used)',
        #     np.arange(0,len(dataframe))
        #     )

        # st.text("")
        # st.text("")
        # Decay = st.text_input('Decay parameter for the Learning rate (0 by default)','0')

        # st.text("")
        # st.text("")
        # Lambda = st.text_input('Regularization factor Lambda (0 by default)', '0')

        

        # st.text("")
        # st.text("")
        # Patient = st.text_input('The number of epochs to check for early stopping (it occurs in case when Adam, and/or Momentum are applied)', '5')

        

    # Check del numero di activation functions e del numero di Hidden layers
    # Se scelgo più di una funzione di attivazione, il numero complessivo deve essere uguale al numero di layer della Rete
    if len(Function_) < len(Hidden_layers) :
        Copy_func = Function_
        for i in range(len(Hidden_layers)-1) :        # Same activation function for each layer
            Copy_func = Copy_func + Function_
        Function_ = Copy_func
    elif len(Function_) > len(Hidden_layers) :
        st.write("!!! The number of hidden layers must be equal to the number of activation functions !!!")

    return Hidden_layers, Algo, Alpha, Regularization, Momentum, Early_stopping, Verbose, Max_iter, Function_, Batch, Decay, Lambda, Random_state, Patient, Final_metric
