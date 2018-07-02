from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import pandas as pd
import random
import matplotlib
from keras.layers import BatchNormalization



from keras.layers import LeakyReLU
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train=(np.reshape(X_train,(60000,784))-127.5)/127.5
X_test=(np.reshape(X_test,(10000,784))-127.5)
input_size=100
threshold=0.1

from keras.utils import np_utils
for k in range(60000):
    if y_train[k]!=0:
        y_train[k]=0
    else:
        y_train[k]=1

for k in range(10000):
    if y_test[k]!=0:
        y_test[k]=0
    else:
        y_test[k]=1
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

def discriminator():
    model=Sequential()
    model.add(Dense(300,activation='sigmoid',kernel_initializer='normal',input_dim=(784)))
    model.add(Dense(2,kernel_initializer='normal',activation='softmax'))
    return(model)
def generator():
    generator = Sequential()


    #generator.add(BatchNormalization())
    generator.add(Dense(128,input_shape=(input_size,) ))
    generator.add(LeakyReLU(alpha=0.01))
    generator.add(Dense(784,activation='tanh'))
    return(generator)
gen=generator()
discr=discriminator()
discr.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

discr.trainable = False
gan=Sequential([gen,discr])
gan.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


X0=np.array([X_train[i] for i in range(len(X_train)) if y_train[i]==1])
for k in range(500):
    print(k)
    a=b=10
    while a>threshold or b>threshold:
        a,_=discr.train_on_batch(np.array(random.sample(list(X0),25)), np_utils.to_categorical(np.ones(25),2))
        b,_=discr.train_on_batch(gen.predict(np.random.normal(0,1,(25,input_size))),np_utils.to_categorical(np.zeros(25),2))
    c=10
    while c>threshold:
        c,_=gan.train_on_batch(np.random.normal(0,1,(500,input_size)) , np_utils.to_categorical(np.ones(500),2))
def fakes():
    fig=plt.figure(figsize=(10, 10))
    columns = 5
    rows = 5
    for i in range(1, columns*rows+1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.reshape(gen.predict(np.array([np.random.normal(0, 1, input_size)])), (28, 28)), cmap='Greys')
    plt.show()
def reals():
    fig=plt.figure(figsize=(10, 10))
    columns = 5
    rows = 5
    for i in range(1, columns*rows+1):
        fig.add_subplot(rows, columns, i)
        plt.imshow( np.reshape(random.sample(list(X0),1),(28, 28)), cmap='Greys')
    plt.show()
reals()
fakes()