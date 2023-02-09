# -*- coding: utf-8 -*-

"""
@author: Alexandre Silva Nº90004 ; Beatriz Pereira Nº90029

Usage: Lab AA: Neural Networks


"""

#--------------- 1.1 --------------------

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

#labels das imagens
labels=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#import do dataset e load para uma lista de arrays.
mnist = keras.datasets.fashion_mnist
[(train_images, train_labels), (test_images_init, test_labels)]  = mnist.load_data()


# train data shape só para confirmar
print(train_images.shape)
print(len(train_labels))
print(test_images_init.shape)
print(len(test_labels))

#Pergunta 2, mostrar algumas imagens.
plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.title(labels[train_labels[i]])
plt.show()

plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images_init[i], cmap=plt.cm.binary)
    plt.title(labels[test_labels[i]])
plt.show()

#Pergunta 3, dividir por 255.
train_images = train_images/255.0
test_images = test_images_init/255.0

#Pergunta 4, Transformar para one-hot encoding.
train_labels = keras.utils.to_categorical(train_labels, len(labels))
#test_labels = keras.utils.to_categorical(test_labels, len(labels))

#Pergunta 5, split dos dados para para se ir aplicando no MLP
[img_train, img_val, labels_train, labels_val] = train_test_split(train_images, train_labels,test_size = 0.2, random_state= 3)

#Pergunta 6. 

img_train = np.expand_dims(img_train, axis = 3)
img_val = np.expand_dims(img_val, axis = 3)
test_images = np.expand_dims(test_images, axis = 3)


# train data shape só para confirmar a alteração
print(img_train.shape)
print(img_val.shape)
print(test_images.shape)

#--------------- 1.2 --------------------

# Pergunta 1 
# Sequential model
MLP_model = keras.Sequential()
# 1a layer
MLP_model.add(keras.layers.Flatten(input_shape=(28,28,1), name = 'Flatten'))
# 2a layer, 32 neurons
MLP_model.add(keras.layers.Dense(32, activation='relu', name = 'Dense_32'))
# 3a layer, 64 neurons
MLP_model.add(keras.layers.Dense(64, activation='relu', name = 'Dense_64'))
# Pergunta 3
# 4a layer, softmax layer
MLP_model.add(keras.layers.Dense(10, activation='softmax', name = 'Softmax_10'))
# Pergunta 4
# Summary
MLP_model.summary()

# Pergunta 5
# Early stopping monitor
MLP_stop_my_model = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Pergunta 6
# Fit do MLP aos dados de treino e validação 
MLP_model.compile('Adam', loss='categorical_crossentropy')

# --- With early stopping

# Pergunta 7

# Fitting
MLP_model_fit = MLP_model.fit(img_train, labels_train, batch_size = 200, epochs = 200, validation_data = (img_val, labels_val), callbacks = [MLP_stop_my_model])


# plot: training e validation loss
epoch_runs = np.arange(1, len(MLP_model_fit.history['loss']) + 1).tolist()
plt.figure()
plt.plot(epoch_runs, MLP_model_fit.history['val_loss'], color = 'red')
plt.plot(epoch_runs, MLP_model_fit.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('MLP loss with Early Stopping')
plt.legend(['Validation Loss', 'Training Loss'])
plt.savefig('MLP_earlystopping.eps', format ='eps')
plt.show()

# Pergunta 8 
# avaliação de performance 
predictions_mlp = MLP_model.predict(test_images, batch_size=200, callbacks=[MLP_stop_my_model])
big_predictions_mlp = np.argmax(predictions_mlp, 1)
accuracym_mlp = metrics.accuracy_score(test_labels, big_predictions_mlp) #accuracy score
confusionm_mlp = metrics.confusion_matrix(test_labels, big_predictions_mlp) # confusion matrix
print('\nMLP with Early Stopping: ')
print('\t accuracy score = ', accuracym_mlp)
print('\t confusion matrix = \n', confusionm_mlp)


# Epochs: 27
# loss: 0.2801 - val_loss: 0.3255
# Accuracy Score: 0.8675
# [[829   0  11  29   7   2 107   0  15   0]
# [  5 954   1  30   4   0   5   0   1   0]
# [ 12   1 731  11 149   0  88   0   8   0]
# [ 25   2  12 885  43   0  29   0   4   0]
# [  0   0  74  35 827   0  58   0   6   0]
# [  0   0   0   1   0 944   0  32   7  16]
# [136   0  84  34  87   0 637   0  22   0]
# [  0   0   0   0   0  24   0 957   0  19]
# [  3   0   2   6   2   2   7   3 975   0]
# [  0   0   0   1   0   8   1  53   1 936]]


# --- Without early stopping

# Pergunta 1 
# Sequential model
MLP_model_2 = keras.Sequential()
# 1a layer
MLP_model_2.add(keras.layers.Flatten(input_shape=(28,28,1), name = 'Flatten'))
# 2a layer, 32 neurons
MLP_model_2.add(keras.layers.Dense(32, activation='relu', name = 'Dense_32'))
# 3a layer, 64 neurons
MLP_model_2.add(keras.layers.Dense(64, activation='relu', name = 'Dense_64'))
# 4a layer, softmax layer
MLP_model_2.add(keras.layers.Dense(10, activation='softmax', name = 'Softmax_10'))

MLP_model_2.compile('Adam', loss='categorical_crossentropy')

# Pergunta 9

# Fitting
MLP_model_fit2 = MLP_model_2.fit(img_train, labels_train, batch_size = 200, epochs = 200, validation_data = (img_val, labels_val), callbacks =None)

# plot: training e validation loss
epoch_runs = np.arange(1, len(MLP_model_fit2.history['loss']) + 1).tolist()
plt.figure()
plt.plot(epoch_runs, MLP_model_fit2.history['val_loss'], color = 'red')
plt.plot(epoch_runs, MLP_model_fit2.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('MLP loss without Early Stopping')
plt.legend(['Validation Loss', 'Training Loss'])
plt.savefig('MLP_woearlystopping.eps', format ='eps')
plt.show()

# avaliação de performance
predictions_mlp2 = MLP_model_2.predict(test_images, batch_size=200)
big_predictions_mlp2 = np.argmax(predictions_mlp2, 1)
accuracym_mlp2 = metrics.accuracy_score(test_labels, big_predictions_mlp2)
confusionm_mlp2 = metrics.confusion_matrix(test_labels, big_predictions_mlp2)
print('\nMLP without Early Stopping: ')
print('\t accuracy score = ', accuracym_mlp2)
print('\t confusion matrix = \n', confusionm_mlp2)

# Epochs: 200
# loss: 0.0703 - val_loss: 0.8029
# Accuracy Score: 0.8672
#confusion matrix = 
 #[[816   5  17  25   6   5 116   0  10   0]
# [  4 964   3  18   5   0   5   0   1   0]
# [ 24   1 777   8 101   0  87   0   2   0]
# [ 44   9  20 839  44   0  37   0   7   0]
# [  5   2  88  23 822   0  54   1   5   0]
# [  0   0   0   2   0 943   0  36   2  17]
# [152   2  78  23  74   0 661   1   9   0]
# [  0   0   0   0   0  16   1 955   2  26]
# [  9   1   4   4   4   3  12   6 957   0]
# [  0   0   0   1   0  13   1  46   1 938]]


#--------------- 1.3 --------------------

#Pergunta 1 para inicialização.
my_model = keras.Sequential()
my_model.add(keras.layers.Conv2D(16,(3,3), activation = 'relu',input_shape=(28, 28,1)))
my_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
my_model.add(keras.layers.Conv2D(16,(3,3), activation = 'relu'))
my_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
my_model.add(keras.layers.Flatten())
my_model.add(keras.layers.Dense(32,activation = 'relu'))
my_model.add(keras.layers.Dense(10,activation= 'softmax'))

print(my_model.summary())

#Inicialização do critério criterio de paragem. Usa a função loss para definir quando para
#o algoritmo. Caso haja 10 epochs seguidas (patience = 10) onde o valor deixou
#de decrescer (mode = min), então ele para. O restore vai buscar os valores da
#epoca em que ele parou de decrescer.

stop_my_model = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', restore_best_weights = True)

#
#Compile do modelo, aplicando o modelo as imgs de treino usando as imagens de validação para o Early Stopping
my_model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001, clipnorm = 1), loss = 'categorical_crossentropy')
fitmy_model = my_model.fit(img_train, labels_train, batch_size = 200, epochs = 200, validation_data = (img_val, labels_val), callbacks = [stop_my_model])


# Pergunta 4, plot da training loss e da validation loss.
epoch_runs = np.arange(1, len(fitmy_model.history['loss']) + 1).tolist()
plt.figure()
plt.plot(epoch_runs, fitmy_model.history['val_loss'], color = 'red')
plt.plot(epoch_runs, fitmy_model.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Convolutional Neural Network (CNN)')
plt.legend(['Validation Loss', 'Training Loss'])
plt.savefig('CNN_EarlyStopping.eps', format ='eps')
plt.show()

#Pergunta 5, Accuracy e confusion matrix
predictions_cnn = my_model.predict(test_images, batch_size = 200, callbacks = [stop_my_model])
big_predictions_cnn = np.argmax(predictions_cnn, 1)
accuracym_cnn = metrics.accuracy_score(test_labels, big_predictions_cnn)
confusionm_cnn = metrics.confusion_matrix(test_labels, big_predictions_cnn)


print('\t accuracy score = ', accuracym_cnn)
print('\t confusion matrix = \n', confusionm_cnn)

# Epochs: 59
# loss: 0.1955 - val_loss: 0.2918
# Accuracy Score: 0.8918
#confusion matrix = 
# [[836   0  28  17   3   0 105   0  11   0]
# [  1 979   1  12   2   0   4   0   1   0]
# [ 12   2 838   7  67   0  67   0   7   0]
# [ 14   9  17 884  36   0  38   0   2   0]
# [  1   1  68  29 820   0  80   0   1   0]
# [  0   0   0   0   0 960   0  30   1   9]
# [109   1  71  25  63   0 717   0  14   0]
# [  0   0   0   0   0  18   0 968   0  14]
# [  3   0   4   5   3   0   5   7 973   0]
# [  1   0   0   0   0   6   0  50   0 943]]




#fiz algumas alterações À função dadas apenas para efeitos de apresentabilidade das imagens

def visualize_activations(conv_model,layer_idx,image):
    plt.figure(0)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image,cmap=plt.cm.binary)
    outputs = [conv_model.layers[i].output for i in layer_idx]
    
    visual = keras.Model(inputs = conv_model.inputs, outputs = outputs)
    
    features = visual.predict(np.expand_dims(np.expand_dims(image,0),3))  
        
    f = 1
    for fmap in features:
            square = int(np.round(np.sqrt(fmap.shape[3])))
            plt.figure(f)
            for ix in range(fmap.shape[3]):
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.subplot(square, square, ix+1)
                plt.imshow(fmap[0,:, :, ix], cmap=plt.cm.binary)
            plt.show()
            plt.pause(2)
            f +=1
            
             
#Pergunta 6, feature maps, aplicar a função dada pelos profs
for i in range(5):
    visualize_activations(my_model, [0, 2], test_images_init[17*i])
