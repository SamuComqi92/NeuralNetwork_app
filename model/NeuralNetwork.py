#Importing the library
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import json
import matplotlib.pyplot as plt
from itertools import product
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer, OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

##################################################################################################################################################################################################################

#Class NeuralNet
class NeuralNet :
    """
    The class builds a Neural Network for binary classification and regression
    It is possible to define:
    - The task of the problem ("classification", "regression")
    - The activation function ("sigmoid","tanh","relu","leaky_relu","elu","swish").
        It can be a tuple if different activation functions are specified for each layer
    - The hidden layers (a tuple where each element represents the number of nodes in each layer)
    - The Gradient Descent algorithm ("batch","adam")
    - The size of the mini-batches
    - The learning rate
    - The decay parameter for the learning rate (lr = lr/(1+decay*epoch))
    - The type of regularization ("ridge","lasso")
    - The regularization factor lambda
    - The maximum number of iterations
    - The momentum factor in the weights optimization
    - A random state for reproducible results
    - A flag to display results while processing
    - A flag to apply early stopping to the algorithm
    - The number of epochs to check for early stopping
         The check occurs in case when Adam, and/or Momentum
    - A flag to show the learning curves (i.e. training and test cost functions) updated at each epoch
    - The metric for the final model evaluation
    
    Modules:
    - Training: training the Neural Network
    - Predict: class prediction (for classification) and value (for regression)
    - Predict_proba: probability prediction (only for classification)
    - Score: performance metric (different metrics for classification and regression)
    
    Attributes:
    - last_iter: last iteration of training
    - best_weights: best weights found with training
    - cost_function_tr: cost function values for the training set up to the last iteration
    - cost_function_te: cost function values for the validation set up to the last iteration
    - metric_tr: chosen metric with self.metric computed on the training set to be saved at each iteration
    - metric_te: chosen metric with self.metric computed on the test set to be saved at each iteration
    """
    
    def __init__(self, task, function = "Sigmoid", Hidden_layers = (5,), algo = "Batch", batch = None,
                 alpha = 0.3, decay = 0.0, regularization = None, Lambda = 0.0, 
                 Max_iter = 1000, momentum = 0.8, random_state = None, verbose = 0,
                 early_stopping = True, patient = 5, flag_plot = False, metric = None) :
        self.task = task
        self.function = function
        self.Hidden_layers = Hidden_layers     
        self.algo = algo
        self.batch = batch
        self.alpha = alpha
        self.decay = decay
        self.regularization = regularization
        self.Lambda = Lambda
        self.Max_iter = Max_iter
        self.momentum = momentum
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.patient = patient
        self.flag_plot = flag_plot
        self.metric = metric

        #Change metric if None
        if self.metric is None :
            if self.task == "Classification" :
                self.metric = "Accuracy"
            else :
                self.metric = "RMSE"

        #Check if the activation function is unique or a tuple
        self.flag_tuple = 0
        if type(self.function) == tuple :
            self.flag_tuple = 1
    
        #Raise error if the tuple has not the right number of functions
        if (self.flag_tuple == 1) and (len(self.function) != len(self.Hidden_layers)) :
            raise ValueError("The number of activation functions is not correct")

    #Activation functions
    def act(function,X) :
        """
        The function returns the transformed values according to the specified activation function
        - function    activation function 
        - X           input matrix (data with no target column)
        """
        
        if function == "Sigmoid" :
            return 1. / (1. + np.exp(-X))
        elif function == "Tanh" :
            return np.tanh(X)
        elif function == "Relu" :
            return np.where(X>=0,X,0)
        elif function == "Leaky_relu" :
            return np.where(X>=0,X,0.1*X)
        elif function == "Elu" :
            return np.where(X>=0,X, 0.1*(np.exp(X) - 1))           #alpha = 0.1
        elif function == "Swish" :
            return X * (1./(1.+np.exp(-X)))
        else :
            raise ValueError("The activation function is not valid")


    #Derivative of the activation functions
    def derivative(function,X) :
        """
        The function returns the transformed values according to the derivative of the specified activation function
        - function    activation function 
        - X           input matrix (data with no target column)
        """
        
        if function == "Sigmoid" :
            return (NeuralNet.act(function,X)) * (1 - NeuralNet.act(function,X))
        elif function == "Tanh" :
            return (1 - NeuralNet.act(function,X)**2)
        elif function == "Relu" :
            return np.where(X>=0,1,0)
        elif function == "Leaky_relu" :
            return np.where(X>=0,1,0.1)
        elif function == "Elu" :
            return np.where(X>=0,1, 0.1*np.exp(X)) 
        elif function == "Swish" :
            return NeuralNet.act("Sigmoid",X) * (1 + X - NeuralNet.act(function,X))
        else :
            raise ValueError("The activation function is not valid")
            
    
    #Random initialization of weights (based on the activation function)
    def randw(self, Lin,Lout, function, X) :
        """
        Source: https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
        The function returns a matrix of random weights with dimensions Lout x (Lin + 1)
        - Lin         dimension of the input layer
        - Lout        dimension of the output layer
        - function    activation function
        - X           input matrix
        """
        
        if self.random_state is not None :
            np.random.seed(self.random_state)
        if (function == "Relu") or (function == "Leaky_relu") or (function == "Elu") :
            return np.random.normal(0.0, np.sqrt(2/X.shape[0]),(Lin+1)*Lout).reshape(Lout,Lin+1)
        elif (function == "Sigmoid") or (function == "Tanh") or (function == "Swish") :
            epsilon = (6 / (Lin+Lout))**0.5
            return np.random.rand(Lout,Lin+1)*2*epsilon - epsilon
        else :
            raise ValueError("The activation function is not valid")


    #Adding bias term to matrices    
    def Bias(m,X) :
        """
        The function returns the matrix with a bias row made of 1s
        - m           number of elements to add (on one row)
        - X           input matrix (data with no target column)
        """
        
        if m != X.shape[1] :
            raise ValueError("The value of m must match the number of columns of X")
        else :    
            return np.vstack([np.ones((1,m)),X]) 

    
    #Forward propagation
    def Forward_propagation(self, X, T) :        
        """
        The function returns the values obtained in the last layer through the forward propagation algorithm
        - X           input matrix (data with no target column)
        - T           tuple containing n matrices (i.e. weights)
        """
        
        m = X.shape[0]                                 #Number of observations
        n = len(T)                                     #Number of matrices (i.e. weights)
        X = np.hstack([np.ones((m,1)),X])              #Add bias column

        #Perform the first step of the Forward propagation (where a1 = X.T)
        if self.flag_tuple == 0 :
            a = NeuralNet.act(self.function,np.dot(T[0],X.T))            
        else :
            a = NeuralNet.act(self.function[0],np.dot(T[0],X.T))            

        #Loop to apply the forward propagation
        for i in np.arange(3,n+2) :
            if i==(n+1) :                              #Final layer
                if self.task == "Classification" :
                    a = softmax(np.dot(T[i-2],NeuralNet.Bias(m,a)), axis=0)
                else :
                    a = np.dot(T[i-2],NeuralNet.Bias(m,a))
                break
            else :                                     #Other layers
                if self.flag_tuple == 1 :
                    a = NeuralNet.act(self.function[i-2],np.dot(T[i-2],NeuralNet.Bias(m,a)))
                else :
                    a = NeuralNet.act(self.function,np.dot(T[i-2],NeuralNet.Bias(m,a)))

        return a.T
    
    
    #Forward and Back propagation
    def ForwardBack_propagation(self, X, y, T) :
        """
        The function returns the values obtained in each layer (through forward propagation) and the errors (through back propagaton)
        - X           input matrix (data with no target column)
        - y           target variable transformed with OneHotEncoder (number of columns = number of classes)
        - T           tuple containing n matrices (i.e. weights)
        """
        
        m = X.shape[1]                                   #Number of observations
        n = len(T)                                       #Number of matrices (i.e. weights)
        
        #First step of the Forward propagation
        DNODES = ()
        ANODES_n = (np.dot(T[0],X),)
        if self.flag_tuple == 0 :
            ANODES = (NeuralNet.act(self.function,np.dot(T[0],X)) ,)
        else :
            ANODES = (NeuralNet.act(self.function[0],np.dot(T[0],X)) ,)

        #Forward propagation loop
        for i in np.arange(3,n+3) :
            if i==(n+1) :                              #Final layer
                if self.task == "Classification" :
                    a = softmax(np.dot(T[i-2],NeuralNet.Bias(m,ANODES[i-3])), axis=0)
                else :
                    a = np.dot(T[i-2],NeuralNet.Bias(m,ANODES[i-3]))
                ANODES_n = ANODES_n + (np.dot(T[i-2],NeuralNet.Bias(m,ANODES[i-3])),)
                ANODES = ANODES + (a,)
                DNODES = (a - y.T,) + DNODES
                break
            else :                                     #Other layers
                if self.flag_tuple == 0 :
                    ANODES_n = ANODES_n + (np.dot(T[i-2],NeuralNet.Bias(m,ANODES[i-3])),)
                    ANODES = ANODES + (NeuralNet.act(self.function,np.dot(T[i-2],NeuralNet.Bias(m,ANODES[i-3]))),)
                else :
                    ANODES_n = ANODES_n + (np.dot(T[i-2],NeuralNet.Bias(m,ANODES[i-3])),)
                    ANODES = ANODES + (NeuralNet.act(self.function[i-2],np.dot(T[i-2],NeuralNet.Bias(m,ANODES[i-3]))),)

        #Back propagation loop
        for i in np.arange(2,n+1)[::-1] :
            if self.flag_tuple == 0 :
                DNODES = (( np.dot(T[i-1].T,DNODES[0]) * NeuralNet.derivative(self.function,NeuralNet.Bias(m,ANODES_n[i-2])) )[1:,:],) + DNODES
            else :
                DNODES = (( np.dot(T[i-1].T,DNODES[0]) * NeuralNet.derivative(self.function[i-2],NeuralNet.Bias(m,ANODES_n[i-2])) )[1:,:],) + DNODES          

        return ANODES, DNODES
    
    
    #Cost function and Gradient
    def J_Grad(self, X, yy, T) :
        """
        The function returns the Cost function (with the regularization term), the Gradient, and the performance metric
        - X           input matrix (data with no target column)
        - yy          target variable
        - labels      number of labels
        - Lambda      regularization factor
        - metric      chosen metric for extra output
        - T           tuple containing n matrices (i.e. weights)
        """
        
        m = X.shape[1]                          #Number of observations
        n = len(T)                              #Number of matrices (i.e. weights)

        #Forward and Back propagation
        anodes, dnodes = NeuralNet.ForwardBack_propagation(self, X, yy, T)

        #Accuracy/RMSE
        if self.task == "Classification" :
            res = anodes[-1].T.argmax(axis=1)
            Performance = (res == yy.argmax(axis=1)).sum()/m
        else :
            Performance = np.sqrt(((anodes[-1] - yy.T)**2).sum()/m)  

        #Metric  
        if self.task == "Classification" :
            if self.metric == "Accuracy" :
                Metricc = Performance
            elif self.metric == "Precision" :
                if yy.shape[1] == 2 :
                    Metricc = precision_score(yy.argmax(axis=1), anodes[-1].T.argmax(axis=1))
                else :
                    Metricc = precision_score(yy.argmax(axis=1), anodes[-1].T.argmax(axis=1), average=None)
            elif self.metric == "Recall" :
                if yy.shape[1] == 2 :
                    Metricc = recall_score(yy.argmax(axis=1), anodes[-1].T.argmax(axis=1))
                else :
                    Metricc = recall_score(yy.argmax(axis=1), anodes[-1].T.argmax(axis=1), average=None)
            elif self.metric == "F1 score" :
                if yy.shape[1] == 2 :
                    Metricc = f1_score(yy.argmax(axis=1), anodes[-1].T.argmax(axis=1))
                else :
                    Metricc = f1_score(yy.argmax(axis=1), anodes[-1].T.argmax(axis=1), average=None)
            elif self.metric == "AUC" :
                if yy.shape[1] == 2 :
                    Metricc = roc_auc_score(yy.argmax(axis=1), anodes[-1].T.argmax(axis=1))
                else :
                    Metricc = roc_auc_score(yy.argmax(axis=1), anodes[-1].T.argmax(axis=1), average=None, multi_class = "ovr")
            else :
                raise ValueError("Misspelled or inappropriate metric for %s" % self.task)
        else :
            if self.metric == "RMSE" :
                Metricc = mean_squared_error(yy, anodes[-1].T, squared = False)
            elif self.metric == "MAE" :
                Metricc = mean_absolute_error(yy, anodes[-1].T)
            elif self.metric == "R2" :
                Metricc = r2_score(yy, anodes[-1].T )
            else :
                raise ValueError("Misspelled or inappropriate metric for %s" % self.task)

        #Gradient of the Cost function
        Delta1 = np.dot(dnodes[0],X.T)
        Delta2 = np.dot(dnodes[1],NeuralNet.Bias(m,anodes[0]).T)
        if (self.Lambda != 0) & (self.regularization == "Ridge") :
            t1grad = (Delta1+(np.hstack([np.zeros((T[0].shape[0],1)), T[0][:,1:]]) )*self.Lambda)/m
            t2grad = (Delta2+(np.hstack([np.zeros((T[1].shape[0],1)), T[1][:,1:]]) )*self.Lambda)/m  
        elif (self.Lambda != 0) & (self.regularization == "Lasso") :
            t1grad = (Delta1+(np.hstack([np.zeros((T[0].shape[0],1)), np.abs(T[0][:,1:])]))*self.Lambda)/m
            t2grad = (Delta2+(np.hstack([np.zeros((T[1].shape[0],1)), np.abs(T[1][:,1:])]))*self.Lambda)/m  
        else :
            t1grad = Delta1/m
            t2grad = Delta2/m 

        TGRAD = (t1grad,) + (t2grad,)
        for i in np.arange(3,n+1) :
            Delta = np.dot(dnodes[i-1],NeuralNet.Bias(m,anodes[i-2]).T)
            if (self.Lambda != 0) & (self.regularization == "Ridge") :
                tgrad = (Delta+(np.hstack([ np.zeros((T[i-1].shape[0],1)), T[i-1][:,1:]]))*self.Lambda)/m 
            elif (self.Lambda != 0) & (self.regularization == "Lasso") :
                tgrad = (Delta+(np.hstack([ np.zeros((T[i-1].shape[0],1)), np.abs(T[i-1][:,1:])]))*self.Lambda)/m 
            else :
                tgrad = Delta/m
            TGRAD = TGRAD + (tgrad,)

        #Regularization term
        if (self.Lambda != 0) & (self.regularization == "Ridge") :
            REG = sum([np.sum(i[:,1:]**2) for i in T])*self.Lambda/(2*m)  
        elif (self.Lambda != 0) & (self.regularization == "Lasso") :
            REG = sum([np.sum(abs(i[:,1:])) for i in T])*self.Lambda/(2*m)  
        else :
            REG = 0

        #Cost function
        if self.task == "Classification" :
            if self.labels == 2 :        #Binary
                J = sum(sum(-np.log10(anodes[-1])*yy.T - np.log10(1-anodes[-1])*(1-yy).T))/m + REG
            else :                  #Multiclass (cross-entropy)
                J = -(yy*np.log(anodes[-1].T)).sum()/m + REG
        else :
            J = (dnodes[-1]**2).sum()/(2*m) + REG

        return J, TGRAD, Performance, Metricc

    
    #Function to predict the probability (for Classification problems)
    def Prediction_proba(self, X, T) :
        """
        The function returns the final probabilities of getting the positive class and the corresponding predicted class
        This is specific of a binary classification problem. 
        - X           input matrix (data with no target column)
        - T           tuple containing n matrices (i.e. weights)
        """
        
        Final = NeuralNet.Forward_propagation(self, X, T)
        if self.task == "Classification" :
            Predictions = Final.max(axis=1)
            #for i in np.arange(0,len(Final)) :
            #    Predictions.append(np.max(Final[i,:]))
            return Predictions
        else :
            raise ValueError("No probabilities for %s" % self.task)


    #Module to compute the model performance (with different metrics)
    def Score(self, X, y, metric) :
        """
        The function returns the accuracy (for classification) and RMSE (for regression)
        - X           input matrix (data with no target column)
        - y           target variable
        - metric      metric to evaluate the performance
        """
        
        if self.task == "Classification" :
            if metric == "Accuracy" :
                return (self.Predict(X) == y).sum()/len(y)
            elif metric == "Precision" :
                if len(np.unique(y))==2 :
                    return precision_score(y, self.Predict(X))
                else :
                    return precision_score(y, self.Predict(X), average=None)
            elif metric == "Recall" :
                if len(np.unique(y))==2 :
                    return recall_score(y, self.Predict(X))
                else :
                    return recall_score(y, self.Predict(X), average=None)
            elif metric == "F1 score" :
                if len(np.unique(y))==2 :
                    return f1_score(y, self.Predict(X))
                else :
                    return f1_score(y, self.Predict(X), average=None)
            elif metric == "AUC" :
                if len(np.unique(y))==2 :
                    return roc_auc_score(y, self.Predict_proba(X))
                else :
                    return roc_auc_score(y, NeuralNet.Forward_propagation(self, X, self.best_weights), average=None, multi_class = "ovr")
            else :
                raise ValueError("Misspelled or inappropriate metric for %s" % self.task)
        else :
            if metric == "RMSE" :
                return mean_squared_error(np.array(y).reshape(len(y),1), self.Predict(X))
            elif metric == "MAE" :
                return mean_absolute_error(np.array(y).reshape(len(y),1), self.Predict(X))
            elif metric == "R2" :
                return r2_score(y, self.Predict(X) )
            else :
                raise ValueError("Misspelled or inappropriate metric for %s" % self.task)
                

    #Training of the model        
    def Training(self, X_train, y_train, X_test, y_test) :
        """
        The function returns the last iteration, a tuple containing the weights, the cost function value at each iteration for the training and validation sets
        - X_train         input matrix (data with no target column)
        - y_train         target variable
        - X_test          matrix for cross-validation
        - y_test          target variable for cross-validation
        """
        
        #Check dimensions
        if (len(X_train) != len(y_train)) or (len(X_test) != len(y_test)):
            raise ValueError("Data and target have different dimensions")
        else :
            #Initialization values
            inputs = X_train.shape[1]               #Number of features
            Cost_value = 1e09                       #Value to start early stopping
            tolerance = 1e-06                       #Tolerance value for early stopping
            
            #Default hyperparameters of the Adam algorithm
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-07
            
            #Number of labels (in the output)
            if self.task == "Classification" :
                self.labels = len(np.unique(y_train))
            else :
                self.labels = 1

            #Random initialization of weights
            n = len(self.Hidden_layers)
            if self.flag_tuple == 0 :
                THETA = (NeuralNet.randw(self, inputs, self.Hidden_layers[0], self.function, X_train),)
            else :
                THETA = (NeuralNet.randw(self, inputs, self.Hidden_layers[0], self.function[0], X_train),)

            for i in np.arange(0,n) :
                if i==n-1 :
                    if self.flag_tuple == 0 :
                        THETA = THETA + (NeuralNet.randw(self, self.Hidden_layers[i], self.labels, self.function, X_train),)
                    else :
                        THETA = THETA + (NeuralNet.randw(self, self.Hidden_layers[i], self.labels, self.function[i], X_train),)
                    break
                else :
                    if self.flag_tuple == 0 :
                        THETA = THETA + (NeuralNet.randw(self, self.Hidden_layers[i], self.Hidden_layers[i+1], self.function, X_train),)
                    else :
                        THETA = THETA + (NeuralNet.randw(self, self.Hidden_layers[i], self.Hidden_layers[i+1], self.function[i], X_train),)
        
            #Initialization to zero of the change in the weights (to compute the momentum)
            Change = tuple([0.0*x for x in THETA])

            #Initialization of the momenta for the Adam algorithm
            M_beta = tuple([0.0*x for x in THETA])
            V_beta = tuple([0.0*x for x in THETA])
            
            #Function to update the weights
            def update_weights(G_tr_, Change_, THETA_, M_beta_, V_beta_, i) :
                #Decay of the learning rate
                if self.decay != 0 :
                    self.alpha = self.alpha / (1 + self.decay*i)

                #Update weights
                if self.algo == "Batch" :
                    #Compute changes in the weights (with momentum)
                    NewChange = ()
                    for k in range(n+1) :
                        NewChange = (NewChange + (self.alpha*G_tr_[k] + self.momentum*Change_[k],))

                    #Update weights
                    for k in range(n+1) :
                        THETA_ = (THETA_ + (THETA_[0] - NewChange[k],))[1:]

                    return THETA_, NewChange, M_beta_, V_beta_
                elif self.algo == "Adam":
                    #Compute first and second momenta
                    NewM_beta, NewV_beta = (), ()
                    for k in range(n+1) :
                        NewM_beta = (NewM_beta + (beta1*M_beta_[k] + (1-beta1)*G_tr_[k],))
                        NewV_beta = (NewV_beta + (beta2*V_beta_[k] + (1-beta2)*(G_tr_[k]*G_tr_[k]),))
                    
                    #Compute changes in the weights (with momentum)
                    NewChange = ()
                    for k in range(n+1) :
                        NewChange = (NewChange + (self.alpha*((np.sqrt(1 - beta2**(i+1)))/(1-beta1**(i+1)))*NewM_beta[k]/(np.sqrt(NewV_beta[k]) + epsilon) + self.momentum*Change_[k],))

                    #Update weights
                    for k in range(n+1) :
                        THETA_ = (THETA_ + (THETA_[0] - NewChange[k],))[1:]
                    
                    return THETA_, NewChange, NewM_beta, NewV_beta
                else :
                    raise ValueError("The algorithm used for optimizing the weights is not valid")


            #Create target array (OneHotEncoder - number of columns = number of classes)
            if self.task == "Classification" :
                Encoder = OneHotEncoder().fit(y_train.reshape((len(y_train)),1))
                yy = Encoder.transform(y_train.reshape((len(y_train)),1)).toarray()
                yy_test = Encoder.transform(y_test.reshape((len(y_test)),1)).toarray()
                # Dizionario della trasformazione
                unique_classes = Encoder.categories_[0]
                Class_convertion = { label: number for label, number in zip( unique_classes, range(len(unique_classes)) ) }
            else :
                yy = y_train
                yy_test = y_test

            #Add Bias unit
            X_train_1 = (np.hstack([np.ones((X_train.shape[0],1)),X_train])).T
            X_test_1 = (np.hstack([np.ones((X_test.shape[0],1)),X_test])).T

            #Iterative training (i: epoch)
            Cost_tr, Cost_te, Metr_tr, Metr_te = [], [], [], []
            latest_iteration = st.empty()
            output = st.empty()
            plotto_ = st.empty()
            bar = st.progress(0) 
            for i in range(self.Max_iter) :
                #Stochastic or Mini-batch algorithm
                flag_batch = 0
                if self.batch is not None :
                    flag_batch = 1
                    ratio_ = int(X_train_1.shape[1]/self.batch)
                    jj=0
                    while jj<=ratio_ :
                        if (X_train_1[:,0+self.batch*jj:self.batch*(1+jj)]).shape[1] == 0 :
                            pass
                        else :
                            #Cost function and gradient
                            J_tr, G_tr, Perf_tr, Metric_tr = NeuralNet.J_Grad(self, X_train_1[:,0+self.batch*jj:self.batch*(1+jj)], yy[0+self.batch*jj:self.batch*(1+jj),:],THETA)
                            J_te, G_te, Perf_te, Metric_te = NeuralNet.J_Grad(self, X_test_1, yy_test,THETA)

                            #Update cost function lists
                            Cost_tr.append(J_tr)
                            Cost_te.append(J_te)
                            Metr_tr.append(Metric_tr)
                            Metr_te.append(Metric_te)

                            if self.verbose != 0 :
                                if self.task == "Classification" :
                                    print('\rIteration: {}/{} -- Batch: {}/{} ----- Training cost: {:.5f} - Validation cost: {:.5f} --- Training {}: {:.5f} - Validation {}: {:.5f}'.format(i,
                                                                                                                                                        self.Max_iter,jj,ratio_,
                                                                                                                                                        J_tr,J_te,
                                                                                                                                                        self.metric, Metric_tr,
                                                                                                                                                        self.metric, Metric_te), end='')
                                else :
                                    print('\rIteration: {}/{} -- Batch: {}/{} ----- Training cost: {:.5f} - Validation cost: {:.5f} --- Training {}: {:.5f} - Validation {}: {:.5f}'.format(i,
                                                                                                                                                        self.Max_iter,jj,ratio_,
                                                                                                                                                        J_tr,J_te,
                                                                                                                                                        self.metric, Metric_tr,
                                                                                                                                                        self.metric, Metric_te), end='')
                            
                            #Update weights
                            THETA, Change, M_beta, V_beta = update_weights(G_tr, Change, THETA, M_beta, V_beta, i)
                        jj=jj+1
                
                #Classic Batch algorithm
                else :
                    #Cost function and gradient
                    J_tr, G_tr, Perf_tr, Metric_tr = NeuralNet.J_Grad(self, X_train_1, yy, THETA)
                    J_te, G_te, Perf_te, Metric_te = NeuralNet.J_Grad(self, X_test_1, yy_test, THETA)

                    #Update cost function lists
                    Cost_tr.append(J_tr)
                    Cost_te.append(J_te)
                    Metr_tr.append(Metric_tr)
                    Metr_te.append(Metric_te)

                #Show cost function curves
                if self.flag_plot == "True" :
                    self.verbose = 0                    #Results are shown on the plot
                    with output.container():
                        if self.task == "Classification" :
                            st.write('Training cost: {:.5f} -- Validation cost: {:.5f}'.format(J_tr,J_te))
                            st.write('Training {}: {:.5f} -- Validation {}: {:.5f}'.format(self.metric,Metric_tr,self.metric,Metric_te))
                        else :
                            st.write('Training cost: {:.5f} -- Validation cost: {:.5f}'.format(J_tr,J_te))
                            st.write('Training {}: {:.5f} -- Validation {}: {:.5f}'.format(self.metric,Metric_tr,self.metric, Metric_te))
      
                    latest_iteration.text(f'Iteration: {i+1} / {self.Max_iter}')
                    bar.progress((i + 1)/(self.Max_iter))                  #progress on bar updated

                    fig, ax = plt.subplots()
                    with plotto_.container() :
                        if self.task == "Classification" :
                            ax.plot(Cost_tr[0],'-b')
                            ax.plot(Cost_te[0],'-r')
                            ax.legend(["Training","Test"])
                            ax.set_xlabel("Epoch")
                            ax.set_ylabel("Cost function") 
                            st.pyplot(fig)

                #Print results (if verbose != 0)
                if self.verbose != 0 and flag_batch == 0:
                    #Progress bar
                    # Update the progress bar with each iteration.
                    with output.container():
                        if self.task == "Classification" :
                            st.write('Training cost: {:.5f} -- Validation cost: {:.5f}'.format(J_tr,J_te))
                            st.write('Training {}: {:.5f} -- Validation {}: {:.5f}'.format(self.metric,Metric_tr,self.metric,Metric_te))
                        else :
                            st.write('Training cost: {:.5f} -- Validation cost: {:.5f}'.format(J_tr,J_te))
                            st.write('Training {}: {:.5f} -- Validation {}: {:.5f}'.format(self.metric,Metric_tr, self.metric,Metric_te))
      
                    latest_iteration.text(f'Iteration: {i+1} / {self.Max_iter}')
                    bar.progress((i + 1)/(self.Max_iter))                  #progress on bar updated
                    
                
                #Early stopping
                #The condition i>Iter is added to avoid initial flactuations
                Iter = 20
                broken = 0
                if (self.early_stopping == "True") & (self.task == "Classification") :
                    if (i>Iter) and ((self.algo == "Adam") or (self.momentum != 0.0)) :
                        if (Cost_te[Iter-self.patient] - Cost_te[-1] < tolerance) :
                            broken = 1
                        else :
                            if (Cost_te[i-1] < Cost_tr[i-1]) and ((J_tr - J_te) < 0) :
                                broken = 1
                    elif (i>Iter) :
                        if (Cost_te[i-1] < Cost_tr[i-1]) and ((J_tr - J_te) < 0) :
                            broken = 1
                        else :
                            if J_te < Cost_value :
                                Cost_value = J_te
                            else :
                                broken = 1
                elif (self.early_stopping == "True") & (self.task == "Regression") :
                    if (i>Iter) and ((self.algo == "Adam") or (self.momentum != 0.0)) :
                        if (Cost_te[Iter-self.patient] - J_te < tolerance) :
                            broken = 1
                    elif (i>Iter) :
                        if abs(J_te - Cost_value) > tolerance :
                            Cost_value = J_te
                        else :
                            broken = 1

                #Stop the training with early stopping
                if broken==1 :
                    self.last_iter = i
                    self.best_weights = THETA
                    self.cost_function_tr = Cost_tr
                    self.cost_function_te = Cost_te
                    self.metric_tr = Metric_tr
                    self.metric_te = Metric_te
                    break
                
                #Update weights
                THETA, Change, M_beta, V_beta = update_weights(G_tr, Change, THETA, M_beta, V_beta, i)
                
            #Final results     
            self.last_iter = i
            self.best_weights = THETA
            self.cost_function_tr = Cost_tr
            self.cost_function_te = Cost_te
            #self.metric_tr = Perf_tr
            #self.metric_te = Perf_te
            self.metric_tr = Metric_tr
            self.metric_te = Metric_te
            self.class_conv = Class_convertion

    
    #Module to make predictions
    def Predict(self, X) :
        """
        The function returns the corresponding predicted class using the best weights found with training. 
        This is specific of a binary classification problem. In the case of regression, it return only the predicted values
        - X           input matrix (data with no target column)
        - T           tuple containing n matrices (i.e. weights)
        """
        
        Final = NeuralNet.Forward_propagation(self, X, self.best_weights)
        if self.task == "Classification" :
            Predictions = Final.argmax(axis=1)
            Predictions = [self.class_conv.get(value, 'Unknown') for value in Predictions]        # Conversione da numeri a stringhe (se necessario)
            #for i in np.arange(0,len(Final)) :
            #    Predictions.append(np.argmax(Final[i,:]))
            return Predictions
        else :
            return Final
    
    
    #Module to predict the probability (for Classification problems)
    def Predict_proba(self, X) :
        """
        The function returns the final probabilities of getting the positive class and the corresponding predicted class
        This is specific of a binary classification problem. 
        - X           input matrix (data with no target column)
        - T           tuple containing n matrices (i.e. weights)
        """
        
        Final = NeuralNet.Forward_propagation(self, X, self.best_weights)
        if self.task == "Classification" :
            Predictions = Final.max(axis=1)
            #for i in np.arange(0,len(Final)) :
            #    Predictions.append(np.max(Final[i,:]))
            return Predictions
        else :
            raise ValueError("No probabilities for %s" % self.task)

    # Module to save the best weights
    def Best_Weights(self) :
        return self.best_weights
    
    # Module to save the best model in a given path as a JSON file
    def Save_model(self, file_name) :
        # Set di variabili
        data = {
            "Task": self.task,
            "Functions": self.function,
            "Hidden_layers": self.Hidden_layers ,
            "Weights": [arr.tolist() for arr in self.best_weights],
            "Weights_shape": [arr.shape for arr in self.best_weights],
            "Algorithm": self.algo,
            "Batch": self.batch,
            "Alpha": self.alpha,
            "Decay": self.decay,
            "Regularization": self.regularization,
            "Lambda": self.Lambda,
            "Momentum": self.momentum
        }

        # Salvo i dati in un file JSON nel path indicato
        json_string = json.dumps(data)
        st.download_button(
            label = "Download best parameters",
            file_name = "Best_parameters.json",
            mime = "application/json",
            data = json_string,
        )
        return True
