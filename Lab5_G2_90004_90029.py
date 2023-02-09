#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:29:37 2020

@author: Alexandre Silva 90004 e Beatriz Pereira 90029

Lab 5 - Evaluation and Generalization

"""

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import keras
import keras.callbacks
from keras.utils import to_categorical
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import max_error as max_err
from sklearn.metrics import explained_variance_score as exp_var_scr
from sklearn.metrics import mean_absolute_error as mean_abs_err
from sklearn.metrics import r2_score 

import keras 
from tensorflow.keras.layers import Dense
from sklearn import preprocessing

#%% --- Classification ---

#load of the train and test data 
data_xtrain = np.load('Cancer_Xtrain.npy')
data_ytrain = np.load('Cancer_Ytrain.npy')
data_xtest = np.load('Cancer_Xtest.npy')
data_ytest = np.load('Cancer_Ytest.npy')

#standardize data so norms dont get to much influenced, and since some of the
#methods use considerate y \in {0,1} change the values 2 to 0.

scaler = StandardScaler()
scaler.fit(data_xtrain)
data_xtrain = scaler.transform(data_xtrain)
data_xtest = scaler.transform(data_xtest)
data_ytrain[data_ytrain == 2] = 0
data_ytest[data_ytest == 2] = 0

#plot the frequency of the training set
font = {'weight': 'normal','size': 13}
plt.figure()
plt.bar([0,1], [len(data_ytrain) - int(sum(data_ytrain)), int(sum(data_ytrain))], width=0.5, tick_label = [0,1], color = ['aqua','orangered'])
plt.xlabel('Training Data (y)', **font)
plt.ylabel('Frequency', **font)
plt.ylim(0, 55)
colors = {'output 0':'aqua', 'output 1':'orangered'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels, loc = 'upper right')
plt.savefig('Frequency_datatrain.eps', format ='eps')
plt.show()


#This function executes the classification methods. It receives a string to
#execute each and returns the module, best parameters and best score
#Most common Kernel Functions in SVM. 'linear' represents the 'Linear' kernel, 
#'rbf' is the Gaussian Kernel, 'sigmoid' represents the Sigmoidal Kernel and
#the poly refers to polynomial kernels (degree p must be given)


def classification_methods(method):
    
    cv = 5
    if method == 'svm':
        kernel_functions = ['linear', 'rbf', 'sigmoid', 'poly']
        max_degree = len(data_xtrain.transpose())
        C_range = np.logspace(-5, 10, 16)
        gamma_range = np.logspace(-10, 5, 16)
        param_grid = dict(kernel = kernel_functions, gamma=gamma_range, C=C_range, degree = range(1,max_degree+1), max_iter = [10000])
        method_module = SVC()
        
    if method == 'naive_bayes':
        param_grid = {}
        method_module = GaussianNB()
        
    if method == 'logistic_regression':
        C_range = np.logspace(-5, 10, 16)
        solver = ['liblinear', 'sag', 'saga', 'lbfgs']
        param_grid = dict(solver=solver, C = C_range, penalty = ['l2'], random_state=[0], max_iter=[10000])
        method_module = LogisticRegression()
    
    if method == 'k_nearest_neighbors':
        num_nei = [*range(1,50+1)]
        param_grid = dict(n_neighbors = num_nei)
        method_module = KNeighborsClassifier()
        
    if method == 'random_forest':
        ntrees = [*range(64,128)]
        param_grid = dict(n_estimators = ntrees, max_depth = [2], random_state = [0])
        method_module = RandomForestClassifier()
        
    grid = GridSearchCV(method_module, param_grid=param_grid, cv= cv)
    grid.fit(data_xtrain, data_ytrain.ravel())
    print("Os melhores parâmetros são %s com um score de %0.2f" 
          % (grid.best_params_, grid.best_score_))
    
    return [method_module, grid.best_params_, grid.best_score_]

#Executes de MLP classification with 3 hidden layers with sizes 32, 64 and 32
#again. It uses Early Stopping and its defined with patience = 10, batch size
# = 10 and epochs = 200
def classification_MLP():
    MLP_model = keras.Sequential()
    # Input layer
    MLP_model.add(keras.layers.Dense(9, activation='relu', input_shape=data_xtrain[0].shape, name = 'DenseInput'))
    # 1a layer, 64 neurons
    MLP_model.add(keras.layers.Dense(32, activation='relu', name = 'Dense_64_1'))
    # 2a layer, 128 neurons
    MLP_model.add(keras.layers.Dense(64, activation='relu', name = 'Dense128'))
    # 3a layer, 64 neurons
    MLP_model.add(keras.layers.Dense(32, activation='relu', name = 'Dense_64_2'))
    # Output layer
    MLP_model.add(keras.layers.Dense(2, activation='softmax', name = 'Softmax'))
    #summary
    MLP_model.summary()
    #EarlyStopping condition
    MLP_stop_my_model = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=10, restore_best_weights=True)
    #Compile the model
    MLP_model.compile('Adam', loss='categorical_crossentropy')
    #generate datasplit to create 
    datax_train_new, datax_validation_new, datay_train_new, datay_validation_new = train_test_split(data_xtrain,data_ytrain, test_size=0.2, random_state = 7)
    datay_train_new = to_categorical(datay_train_new)
    datay_validation_new = to_categorical(datay_validation_new)
    #fit the model to the data
    _history = MLP_model.fit(datax_train_new,  datay_train_new, batch_size = 10, epochs = 200, validation_data = (datax_validation_new, datay_validation_new), callbacks = [MLP_stop_my_model])
    #plot loss
    epoch_runs = np.arange(1, len(_history.history['loss']) + 1).tolist()
    plt.figure()
    plt.plot(epoch_runs, _history.history['loss'])
    plt.plot(epoch_runs, _history.history['val_loss'], color = 'red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('MLP loss with Early Stopping')
    plt.legend(['Validation Loss', 'Training Loss'])
    plt.savefig('MLP_earlystopping.eps', format ='eps')
    plt.show()
    
    MLP_predict_test = MLP_model.predict(data_xtest, batch_size=10, callbacks=[MLP_stop_my_model]) 

    return MLP_predict_test



#This function evaluates a set of metrics in order to compare some of the
#classification results we obtain
def evaluation_metrics(datay_test, datay_predictions):
    accuracy_score = metrics.accuracy_score(datay_test, datay_predictions) #accuracy score
    print('Accuracy Score: %0.2f' % (accuracy_score))
    confusion_matrix = metrics.confusion_matrix(datay_test, datay_predictions) # confusion matrix
    print('A Confusion Matrix é: \n', confusion_matrix)
    precision_score = metrics.precision_score(datay_test, datay_predictions) #precision score
    print('Precision Score: %0.2f' % (precision_score))
    recall_score = metrics.recall_score(datay_test, datay_predictions) #recall score
    print('Recall Score: %0.2f' % (recall_score))
    f_measure = metrics.f1_score(datay_test, datay_predictions) # F-measure
    print('F-measure Score: %0.2f' % (f_measure))
    jaccard_test = metrics.jaccard_score(datay_test, datay_predictions) #teste de jaccard
    print('Jaccard Score: %0.2f' % (jaccard_test)) 
    average_precision = metrics.average_precision_score(datay_test, datay_predictions) #average precision
    print('Average Precision: %0.2f' % (average_precision))
    balanced_accuracy = metrics.balanced_accuracy_score(datay_test, datay_predictions) #balanced accuracy
    print('Balanced Accuracy: %0.2f' % (balanced_accuracy))
    return confusion_matrix

#plot of the confusion metrics
def plot_confusion_matrix(confusion_matrix):
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in confusion_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in confusion_matrix.flatten()/np.sum(confusion_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Greens')
    return

#executes de code

print('\n \n ---------Logistic Regression ------------ \n \n')

lr_best = classification_methods('logistic_regression')
lr = LogisticRegression(solver = lr_best[1]['solver'], penalty = lr_best[1]['penalty'], C = lr_best[1]['C'], max_iter = lr_best[1]['max_iter'], random_state = lr_best[1]['random_state'])
lr.fit(data_xtrain, data_ytrain.ravel())
lr_predictions = lr.predict(data_xtest)
lr_metrics = evaluation_metrics(data_ytest, lr_predictions)
lr_cfm = plot_confusion_matrix(lr_metrics)

print('\n \n ---------Naive Bayes ------------ \n \n')

nbayes_best = classification_methods('naive_bayes')
nbayes = GaussianNB()
nbayes.fit(data_xtrain, data_ytrain.ravel())
nbayes_predictions = nbayes.predict(data_xtest)
nbayes_metrics = evaluation_metrics(data_ytest, nbayes_predictions)
nbayes_cfm = plot_confusion_matrix(nbayes_metrics)

print('\n \n ---------K-Nearest Neighbors ------------ \n \n')

knn_best = classification_methods('k_nearest_neighbors')
knn = KNeighborsClassifier(knn_best[1]['n_neighbors'])
knn.fit(data_xtrain, data_ytrain.ravel())
knn_predictions = knn.predict(data_xtest)
knn_metrics = evaluation_metrics(data_ytest, knn_predictions)
knn_cfm = plot_confusion_matrix(knn_metrics)

print('\n \n ---------Random Forest ------------ \n \n')

rf_best = classification_methods('random_forest')
rf = RandomForestClassifier(n_estimators=rf_best[1]['n_estimators'])
rf.fit(data_xtrain, data_ytrain.ravel())
rf_predictions = rf.predict(data_xtest)
rf_metrics = evaluation_metrics(data_ytest, rf_predictions)
rf_cfm = plot_confusion_matrix(rf_metrics)

print('\n \n ---------SVM ------------ \n \n')

svm_best = classification_methods('svm')
svm = SVC(kernel=svm_best[1]['kernel'], gamma = svm_best[1]['gamma'], C = svm_best[1]['C'], degree = svm_best[1]['degree'], max_iter= 10000)
svm.fit(data_xtrain, data_ytrain.ravel())
svm_predictions = svm.predict(data_xtest)
svm_metrics = evaluation_metrics(data_ytest, svm_predictions)
svm_cfm = plot_confusion_matrix(svm_metrics)

print('\n \n ---------MLP------------ \n \n')

mlp_pre = classification_MLP()
mlp_predictions = np.argmax(mlp_pre,axis=1)
mlp_metrics = evaluation_metrics(data_ytest, mlp_predictions)
mlp_cfm = plot_confusion_matrix(mlp_metrics)



#%%
def performance_regression(y_true, y_predict):
    print('\n Metrics scores:')
    mean_squared_error = mse(y_true, y_predict) # mean square error
    print('\n Mean Squared Error:', mean_squared_error)
    max_error = max_err(y_true, y_predict)
    print('\n Max Error:', max_error) # max error
    variance_score = exp_var_scr(y_true, y_predict) # explained variance
    print('\n Explained Variance Regression Score:', variance_score)
    mean_absolute_error = mean_abs_err(y_true, y_predict) # mean absolute error
    print('\n Mean Absolute Error:', mean_absolute_error)
    
    r2Score = r2_score(y_true, y_predict)
    print('\n Regression score function:',r2Score)
   

#%% --- Regression ---

## Load Data
x_train = np.load('Real_Estate_Xtrain.npy')
y_train = np.load('Real_Estate_ytrain.npy')
x_test = np.load('Real_Estate_Xtest.npy')
y_test = np.load('Real_Estate_ytest.npy')

scaler = preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

scaler = preprocessing.StandardScaler()

y_train = preprocessing.scale(y_train)
y_test = preprocessing.scale(y_test)

# Splitting the train data into two subsets
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2)

#%% MLP 

n=x_train.shape[1];
mlp_reg = keras.Sequential()

#Input layer
mlp_reg.add(Dense(n, activation='relu',
            input_shape=x_train[0].shape, name='Dense_in'))
 
#Hidden Layers

mlp_reg.add(Dense(16, activation='relu', name='Dense_256'))
mlp_reg.add(Dense(8, activation='relu', name='Dense_128o'))


#Output Layer
mlp_reg.add(Dense(1, activation='linear', name='Dense_o'))


mlp_reg.summary()


mlp_reg.compile(loss='mean_absolute_error', optimizer='adam')

mlp_reg_stop=keras.callbacks.EarlyStopping(monitor = 'val_loss',patience=10, restore_best_weights=True)
mlp_reg_fit = mlp_reg.fit(x_train, y_train, epochs=200, batch_size=10, 
         verbose=1, callbacks=[mlp_reg_stop], validation_data=(x_validation, y_validation))

# Plot
plt.figure()
plt.plot(mlp_reg_fit.history['loss'], ) 
plt.plot(mlp_reg_fit.history['val_loss'], color = 'red') 
plt.title('MLP regressor')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['training loss','validation loss'])

mlp_reg_predict_test = mlp_reg.predict(x_test, batch_size=10, callbacks=None)

print('\nScore test MLP Regressor:')
performance_regression(y_test, mlp_reg_predict_test)
print('\n')

#%% Linear Regressor 

# Train
Linear_reg = LinearRegression().fit(x_train, y_train)
Linear_reg_predict_test = Linear_reg.predict(x_validation)

print('\nScore train Linear Regression:')
performance_regression(y_validation, Linear_reg_predict_test)

# Test
Linear_reg = LinearRegression().fit(x_train, y_train)
Linear_reg_predict_test = Linear_reg.predict(x_test)

print('\nScore test Linear Regression:')
performance_regression(y_test, Linear_reg_predict_test)
print('\n')

#%% Lasso and Ridge Regressor

#crossvalidation
alpha_array = np.linspace(0.01, 1, 100)
regression_cv = linear_model.RidgeCV(alpha_array)
model_cv = regression_cv.fit(x_train, y_train)
best_alpha = model_cv.alpha_

# #--Ridge Regressor--

# Train 
ridge_reg = linear_model.Ridge(alpha=best_alpha).fit(x_train, y_train)
ridge_reg_predict_test = ridge_reg.predict(x_validation)

print('\nScore train Ridge Regression:')
performance_regression(y_validation, ridge_reg_predict_test)

# Test
ridge_reg = linear_model.Ridge(alpha=best_alpha).fit(x_train, y_train)
ridge_reg_predict_test = ridge_reg.predict(x_test)

print('\nScore test Ridge Regression:')
performance_regression(y_test, ridge_reg_predict_test)
print('\n')


#--Lasso Regressor--

# Train
lasso_reg = linear_model.Lasso(alpha=0.02).fit(x_train, y_train)
lasso_reg_predict_test = lasso_reg.predict(x_validation)

print('\nScore train Lasso Regression:')
performance_regression(y_validation, lasso_reg_predict_test)

# Test
lasso_reg = linear_model.Lasso(alpha=0.02).fit(x_train, y_train)
lasso_reg_predict_test = lasso_reg.predict(x_test)

print('\nScore test Lasso Regression:')
performance_regression(y_test, lasso_reg_predict_test)





