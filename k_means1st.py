import numpy as np
import pandas
import pandas as pd
import math

from fontTools.misc.classifyTools import classify
from numpy.f2py.symbolic import normalize
from sklearn.metrics import accuracy_score
#Kn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import minmax_scale, MinMaxScaler


#Min_max normalization/ used to get weights for k-means algorithms
def normalize(lst):
    mx = max(lst)
    mn = min(lst)
    dif = int(mx)-int(mn)
    normalizedlst = []
    for i in lst:
        normalizedlst.append((i-mn)/dif)
    return normalizedlst


df = pandas.read_csv('weightheight.csv')
#print(df)

#print(df.head())
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
X = df[['Weight','Height']].values
y = df['Gender'].values
scaler = MinMaxScaler()


Y = df['Gender']
X_train,x_test,Y_train,y_test = train_test_split(X,Y,random_state=100,test_size=0.2)


X_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(x_test)
koptimal = []
#for k in range(1,100):

#    classifier = KNeighborsClassifier(n_neighbors=k)
#    classifier.fit(X_train_scaled,Y_train)
#    koptimal.append(classifier.score(x_test_scaled,y_test))

#print(koptimal.index(max(koptimal)))
test = KNeighborsClassifier(n_neighbors=59)
test.fit(X_train_scaled,Y_train)
hghinpt = input("Enter your height")
wgtinpt = input("Enter your weight")
person = [[hghinpt,wgtinpt]]
person = scaler.transform(person)
person_gender_guess = test.predict(person)
if person_gender_guess == 0:
    print("I guess you are a male")
else:
    print("I guess you are a female")




