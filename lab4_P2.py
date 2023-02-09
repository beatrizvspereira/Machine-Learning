# -*- coding: utf-8 -*-
"""
@author: Alexandre Silva 90004 e Beatriz Pereira 90029

Usage: Lab AA Bayes classifiers

"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 
import sklearn.metrics as metrics

# --------------------Pergunta 2-----------------------

#load of the train data 
data1_xtrain = np.load('data1_xtrain.npy')
data1_ytrain = np.load('data1_ytrain.npy')

#load of the test data
data1_xtest = np.load('data1_xtest.npy')
data1_ytest = np.load('data1_ytest.npy')

#number of classes in the training set
num_classes = len(set(data1_ytrain.flatten()))


class_train = list()
class_test = list()
for i in range(num_classes):
    class_train.append(np.where(data1_ytrain == i+1)[0])
    class_test.append(np.where(data1_ytest == i+1)[0])


#função que executa a questão 2
def scatter_train_test():

    #valor máximo, minimo,cores e markers dos dados para fazer os plots
    min_data = min(data1_xtrain.min(),data1_xtest.min())
    max_data = min(data1_xtrain.max(),data1_xtest.max())
    colors = ['indigo', 'orange', 'red']
    markers = ['o', 'x', 'p'] 

    plt.figure()
    for i in range(2):
        plt.subplot(1,2,i+1)
        for j in range(num_classes):
            if i == 0:
                plt.scatter(data1_xtrain[class_train[j], 0], data1_xtrain[class_train[j], 1], c = colors[j], marker=markers[j])
            else:
                plt.scatter(data1_xtest[class_test[j], 0], data1_xtest[class_test[j], 1], c = colors[j], marker=markers[j])
        plt.xlim(min_data-1, max_data+1)
        plt.ylim(min_data-1, max_data+1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend([ r'$1ª$ $classe$', r'$2ª$ $classe$', r'$3ª$ $classe$'],fontsize = 8,loc = 'lower right')
    plt.savefig('plots2ndquestion.eps', format ='eps')
    plt.show()

#calcula a probabilidade de cada classe
def calculate_probabilities():
    prob=[]
    for i in range(num_classes):
        prob.append(len(class_train[i])/len(data1_ytrain))
    return prob


#Questão 3: A função retorna o vetor de previsões das classes de cada obs.
def naive_bayes(): 
    
    #calcula as médias para serem usadas na normal
    mean1_class, mean2_class = [], []
    for i in range(num_classes):
        mean1_class.append(np.mean(data1_xtrain[class_train[i], 0]))
        mean2_class.append(np.mean(data1_xtrain[class_train[i], 1]))
    
    
    #calcula as variâncias para serem usadas na normal, o var já divide por N.
    var1_class, var2_class = [], []
    for i in range(num_classes):
        var1_class.append(np.var(data1_xtrain[class_train[i], 0]))
        var2_class.append(np.var(data1_xtrain[class_train[i], 1]))
    

    #formar as normais com o stats.norm, usando o valor médio e a variância.
    norm1_class, norm2_class = [], []
    for i in range(num_classes):
        norm1_class.append(scipy.stats.norm(loc = mean1_class[i], scale = np.sqrt(var1_class[i])))
        norm2_class.append(scipy.stats.norm(loc = mean2_class[i], scale = np.sqrt(var2_class[i])))

    #decide qual o maior valor, tendo em conta que se assume independência.
    prob_class = calculate_probabilities()
    class_prediction = np.array([])
    error_of_prediction = 0
    
    for i in range(len(data1_xtest)):
        feat = data1_xtest[i]
        values = []
        for j in range(num_classes):
            values.append(prob_class[j] * norm1_class[j].pdf(feat[0]) * norm2_class[j].pdf(feat[1]))
    
        best_class = (values.index(max(values))) + 1
        class_prediction = np.append(class_prediction,best_class)
        
        if best_class != data1_ytest[i]: error_of_prediction += 1
    
    #cálculo do erro
    erro_classe_percent = ((error_of_prediction)/(len(data1_xtest)))*100
    print('O erro no Naive Bayers Classifier é de aproximadamente', round(erro_classe_percent,2),'%')
    
    return class_prediction


#Questão 4 - Scatter plot para a classificação dos dados de teste.
def scatter_test_data(class_prediction):
    
    class_pred = []
    for i in range(num_classes):
        class_pred.append(np.where(class_prediction == i+1)[0])
        
    min_data = min(data1_xtrain.min(),data1_xtest.min())
    max_data = min(data1_xtrain.max(),data1_xtest.max())
    colors = ['indigo', 'orange', 'red']
    markers = ['o', 'x', 'p']
    
    plt.figure()
    for i in range(num_classes):
        plt.scatter(data1_xtest[class_pred[i], 0], data1_xtest[class_pred[i], 1], c = colors[i]  , marker = markers[i])
    plt.xlim(min_data-1, max_data+1)
    plt.ylim(min_data-1, max_data+1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend([ r'$Previsão$ $1ª$ $classe$', r'$Previsão$ $2ª$ $classe$', r'$Previsão$ $3ª$ $classe$'],fontsize = 8,loc = 'lower right')
    plt.savefig('plots4thquestion_1.eps', format ='eps')
    plt.show()

#Questão 6 -Bayes Classifier

def bayes_classifier():
    
    #bidimensional mean (axis = 0 means column, axis = 1 means row)
     # using multivariate normal distribution
    mean_class, cov_class, norm_class = [],[],[]
    for i in range(num_classes):
        mean_class.append(np.mean(data1_xtrain[class_train[i]], axis = 0))
        cov_class.append(np.cov(np.transpose(data1_xtrain[class_train[i]]),bias = True))
        norm_class.append(scipy.stats.multivariate_normal(mean = mean_class[i], cov = cov_class[i]))
 
    prob_class = calculate_probabilities()
    class_prediction = np.array([])
    error_of_prediction = 0
    for i in range(len(data1_xtest)):
        feat= data1_xtest[i]
        values = list()
        for j in range(num_classes):
            values.append(prob_class[j] * norm_class[j].pdf(feat))
                         
        best_class = (values.index(max(values))) + 1
        class_prediction = np.append(class_prediction, best_class)
        
        if best_class != data1_ytest[i]: error_of_prediction += 1
    
    #cálculo do erro
    erro_classe_percent = ((error_of_prediction)/(len(data1_xtest)))*100

    print('O erro no Bayers Classifier é de aproximadamente', round(erro_classe_percent,2),'%')
    
    return class_prediction


def plot_prediction_bayes(class_prediction):
    
    class_pred = []
    for i in range(num_classes):
        class_pred.append(np.where(class_prediction == i+1)[0])

    min_data = min(data1_xtrain.min(),data1_xtest.min())
    max_data = min(data1_xtrain.max(),data1_xtest.max())
    colors = ['indigo', 'orange', 'red']
    markers = ['o', 'x', 'p']

    plt.figure()
    for i in range(num_classes):
        plt.scatter(data1_xtest[class_pred[i], 0], data1_xtest[class_pred[i], 1], c = colors[i]  , marker = markers[i] )
    plt.xlim(min_data-1, max_data+1)
    plt.ylim(min_data-1, max_data+1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend([ r'$Previsão$ $1ª$ $classe$', r'$Previsão$ $2ª$ $classe$', r'$Previsão$ $3ª$ $classe$'],fontsize = 8,loc = 'lower right')
    plt.savefig('plots4thquestion_2.eps', format ='eps')
    plt.show()


                             
#execução da pergunta 2.
pergunta2=scatter_train_test()

#execução pergunta 3.
class_prediction_naive = naive_bayes()

#execução pergunta 4
pergunta4_plot = scatter_test_data(class_prediction_naive)  

#Questão 5 - Accuracy score

accuracy_score_naive_bayes = metrics.accuracy_score(data1_ytest, class_prediction_naive)    

#Pergunta 6
class_prediction_bayes = bayes_classifier() 

#Accuracy score again
accuracy_score_bayes = metrics.accuracy_score(data1_ytest, class_prediction_bayes)   

#plot na 6

pergunta_6_plot = plot_prediction_bayes(class_prediction_bayes)                           