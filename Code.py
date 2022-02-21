## Code Written by: Russlan Jaafreh & Yoo Seong Kang

'''
Needed packages for this code :
[Pandas (1.4.1),
Numpy (1.22.2),
Pymatgen (2022.2.10)
Pymatgen.core (2022.2.10)
Keras (2.7),
Tensorflow (v2.8),
keras_tuner,
itertools (8.12.0),
collections (3.3),
matplotlib (3.5.1),
time (3.7)
]

'''

import pandas as pd 
import numpy as np
import re
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
from keras.regularizers import l1_l2
import keras_tuner as kt
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from collections import Counter
import pymatgen.core as pg

##Importing the data and making sure no NaNs are available
import os
os.chdir(r'DATA DIRECTORY')
df1 = pd.read_csv('Training_data.csv')
df1 = df1.dropna()
Before_Features = df1.drop(['Composition','ZT',], axis =1)
Before_Features = Before_Features.dropna()

##Assigning Y
Y = df1['ZT']

#Feature Engineering (Variance & Pearson)

#VARIANCE 0.16
from sklearn.feature_selection import VarianceThreshold
var_thres = VarianceThreshold(threshold=0.16)
var_thres.fit(Before_Features)
var_thres.get_support()
constant_columns = [column for column in Before_Features.columns if column not in Before_Features.columns[var_thres.get_support()]]
After_Variance = Before_Features.drop(constant_columns,axis=1)

#Pearson 0.8
import matplotlib.pyplot as plt 
import seaborn as sns
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    af_corr = dataset.drop(col_corr,axis=1)
    return af_corr

#Shape sould be (3769,53)
af_both = correlation(After_Variance, 0.80)
af_both.shape

#Plotting the heatmap
A_cor = af_both.corr()
plt.figure(figsize=(25,20))
sns.heatmap(A_cor,cmap=plt.cm.CMRmap_r,annot=False)
plt.title("Correlation",size = 20)
plt.show()

#Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(af_both, Y, test_size=0.20, random_state=415)

#MinMaxScaling of train (to avoid data leakage)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_train_sca = pd.DataFrame(X_train_sc)
X_train_sca.columns =af_both.columns

#Scaling the X_test
X_test_sc = scaler.transform(X_test)
X_test_sca = pd.DataFrame(X_test_sc)
X_test_sca.columns =af_both.columns

#Building the basic ANN model

def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(n_inputs, input_dim=n_inputs, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu', kernel_regularizer= l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    return model

#Hyperparameter optimization of the base model using Keras Tuner:
from tensorflow.keras.optimizers import Adam as adam
import keras_tuner as kt

def build_model(hp):
    model = keras.Sequential()
    model.add(Dense(53, input_dim=53, activation = hp.Choice("activation0",["relu", "sigmoid", "tanh"])))
    model.add(Dense(units=hp.Int("units1", min_value=16, max_value=64, step=16),activation=hp.Choice("activation1", ["relu", "sigmoid", "tanh"])))
    model.add(Dense(units=hp.Int("units2", min_value=16, max_value=64, step=16),activation=hp.Choice("activation2", ["relu", "sigmoid", "tanh"])))
    model.add(Dense(units=hp.Int("units3", min_value=8, max_value=16, step=8),activation=hp.Choice("activation3", ["relu", "sigmoid", "tanh"])))
    model.add(Dense(units=hp.Int("units4", min_value=4, max_value=16, step=4),activation=hp.Choice("activation4", ["relu", "sigmoid", "tanh"]),kernel_regularizer = l1_l2(l1=1e-5, l2=1e-4)))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=adam(learning_rate=hp_learning_rate), loss='mse')
    return model


build_model(kt.HyperParameters())
tuner = kt.RandomSearch(build_model,objective='val_loss',seed=1,max_trials=1000,executions_per_trial=1)
tuner.search(X_train, y_train, epochs=500, batch_size=128 ,validation_data=(X_test, y_test))

#Show and pick the hyperparameters of best 10 models
tuner.results_summary()


#IMPORTANT: After this, retrain the model one more time with the updated hyperparameters each time (this was done manually but an efficient code can be written using tuner.get_best_models method (SEE Tuner documentation for usage))

#10-fold CV for the base and 10 potetnial models (DONE MANUALLY)
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, random_state=42, shuffle=True)


from time import time
t1 = time()
cv_score = []
r2_10split = []
for train_index, test_index in kf.split(df_scaled1):
    X_train, X_test = np.array(df_scaled1)[train_index], np.array(df_scaled1)[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    model = get_model(53, 1)
    history = model.fit(X_train,y_train,epochs=500,batch_size=64 , validation_data=(X_test,y_test))
    pred_dl = model.predict(X_test)
    cv_score.append(r2_score(y_test,pred_dl))
t2= time()

#saving the CV score and the time in r2_10split
r2_10split = [(sum(cv_score)/10,t2-t1)]



#Training the model (ONE SPLIT for Visualization)
model = get_model(53, 1)
history = model.fit(X_train_sca,y_train,epochs=500,batch_size=64 , validation_data=(X_test_sca,y_test))

#Visualizing the lossVSepoch curve

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xlim((0, 500))   
plt.ylim(0, 0.1)
plt.legend(['train', 'test'], loc='upper left')
plt.show() 

#"Virtual Doping" Code

import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from collections import Counter

global RATIO
global New_composition
global Dopant
global Max_RAT
Dopant = ['Sb', 'Nb', 'Mn', 'Bi', 'Cu', 'Ag', 'In', 'Sn', 'S', 'Ga', 'Cl', 'I', 'Te', 'Li', 'Zn', 'Ca', 'Yb', 'Ge',
              'Ba', 'La', 'Pb', 'Br', 'Mg', 'Co']
Dopant2 = ["Ge"]
New_composition = []
RATIO = 0.01
MAX_RAT = 0.99

#Function to Add elements to a compound at intervals
def Make_composition(compound):
    global RATIO
    global New_composition
    global Dopant
    global Max_RAT
    composition = compound.as_dict()
    elements = list(composition.keys())
    Total_Num = int(MAX_RAT/RATIO) + 1
    U_Dope = list(set(Dopant2)-set(elements))
    print(type(U_Dope),U_Dope)
    for num in np.arange(1, Total_Num):
        for cwr in combinations_with_replacement(elements, num):
            result = Counter(cwr)
            copy_el = composition.copy()
            for key, value in result.items():
                copy_el[key] = copy_el[key] - (RATIO * value)
            for Dope in U_Dope:
                comp1 = copy_el.copy()
                comp1[Dope] = RATIO * num
                final_comp = pg.Composition.from_dict(comp1).fractional_composition
                New_composition.append(final_comp.formula.replace(" ", ""))

#Executing the code
compound = pg.Composition("PbTe")
elements = compound.elements
Make_composition(compound)
comp1 = pd.DataFrame(New_composition)

frames = [comp1]
result = pd.concat(frames)

print(result.head())


