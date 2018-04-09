# import packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras import regularizers

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def nucleotide_to_number(text):
    dictt = {'a':'1','c':'2','g':'3','t':'4'}
    arr = []
    new_text = text.lower()
    for i in list(new_text):
        for k, j in dictt.items():
            if k == i:
                j = float(j)
                arr.append(j)
    return arr

def load_data():
    #Load Training Data
    file_train = open("train.csv", 'r')
    data_train = file_train.read()

    #Pocess Training Data
    #Split the Training Data into Three Colomns by ','
    x_train = []
    y_train = []
    id_train = []
    rows = data_train.split('\n')
    
    # Remove title
    rows = rows[1:]
    for row in rows[0:2000]:
        split_row = row.split(',')
        id_train.append(split_row[0])
        x_train.append(split_row[1])
        y_train.append(int(split_row[2]))

    # Convert to numpy array
    id_train = np.array(id_train)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    #Load Testing Data
    file_test = open("test.csv", 'r')
    data_test = file_test.read()

    #Pocess Testing Data
    #Split the Testing Data into Three Colomns by ','
    x_test = []
    y_test = []
    id_test = []
    rows = data_test.split('\n')
    # Remove title
    rows = rows[1:]
    for row in rows[:400]:
        split_row = row.split(',')
        id_test.append(split_row[0])
        x_test.append(split_row[1])

    # Convert to numpy array
    id_test = np.array(id_test).reshape(1, 400)
    x_test = np.array(x_test)
    
    #Shuffle the Training Data
    ID, X, Y = shuffle(id_train, x_train, y_train, random_state=3)
    
    #Split the Training Set as Training Set and Development Set
    id_train = ID[0: 2000]
    x_train = X[0: 2000]
    y_train = Y[0: 2000].reshape(2000, 1)

    #id_dev = ID[1600: 2000]
    x_dev = X[1600: 2000]
    y_dev = Y[1600: 2000].reshape(400, 1)
    
    X_train = []
    X_dev = []
    X_test = []
    for nuc in x_train:
        number = nucleotide_to_number(nuc)
        X_train.append(number)
    
    for nuc in x_dev:
        number = nucleotide_to_number(nuc)
        X_dev.append(number)
    
    for nuc in x_test:
        number = nucleotide_to_number(nuc)
        X_test.append(number)
    
    X_train = np.array(X_train)
    X_dev = np.array(X_dev)
    X_test = np.array(X_test)
    return X_train, y_train, X_dev, y_dev, id_test, X_test


X_train, y_train, X_dev, y_dev, id_test, X_test = load_data()

#Normalization
x_train = preprocessing.normalize(X_train, axis = 1)
x_dev = preprocessing.normalize(X_dev, axis = 1)
x_test = preprocessing.normalize(X_test, axis = 1)

# Modeling
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=14, kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(0.8))
model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(0.8))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])     
model.fit(x_train,y_train,verbose=1,shuffle=True, epochs=2000, batch_size=100, validation_split=0.2)

#Print the accuracy of development set
score = model.evaluate(x_dev, y_dev)
print(score)

#Make a prediction on testing set
classes = model.predict_classes(x_test)
print(classes)

#Output the prediction to 'csv' file
np.savetxt("f_test.csv", classes, delimiter=",")
