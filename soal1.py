import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    'fertility.csv'
)

# print(df.head(2))
# print(df.columns.values)
# ['Season', 'Age', 'Childish diseases', 'Accident or serious trauma',
# 'Surgical intervention', 'High fevers in the last year',
# 'Frequency of alcohol consumption', 'Smoking habit',
# 'Number of hours spent sitting per day', 'Diagnosis']
# print(df.shape) #(100, 10)
# print(df.isnull().sum())

df = df.drop(['Season'], axis = 1)
print(df.head(2))

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

df['Childish diseases'] = label.fit_transform(df['Childish diseases'])
# print(label.classes_)   # no, yes
df['Accident or serious trauma'] = label.fit_transform(df['Accident or serious trauma'])
# print(label.classes_)   # no, yes
df['Frequency of alcohol consumption'] = label.fit_transform(df['Frequency of alcohol consumption'])
# print(label.classes_)   # ['daily' 'never' 'occasional'
df['Surgical intervention'] = label.fit_transform(df['Surgical intervention'])
# print(label.classes_)   # no, yes
df['High fevers in the last year'] = label.fit_transform(df['High fevers in the last year'])
# print(label.classes_)   #['less than 3 months ago' 'more than 3 months ago' 'no']
#['less than 3 months ago' 'more than 3 months ago' 'no']
df['Smoking habit'] = label.fit_transform(df['Smoking habit'])
# print(label.classes_)   # ['daily' 'never' 'occasional']
df['Diagnosis'] = label.fit_transform(df['Diagnosis'])
# print(label.classes_)   # ['Altered' 'Normal']

x = df.drop(['Diagnosis'], axis = 1)
y = df['Diagnosis']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y,
    test_size = .1,
)

# print(len(xtrain))
# print(len(xtest))

from sklearn.linear_model import LogisticRegression
modelLogR = LogisticRegression(solver= 'liblinear', multi_class='auto')
modelLogR.fit(xtrain, ytrain)

print('score logres:', round(modelLogR.score(xtest, ytest) * 100, 2), '%')

# 'Age', 'Childish diseases', 'Accident or serious trauma',
#             no,yes                      no,yes
# 'Surgical intervention', 'High fevers in the last year',
#             no,yes  ['< 3 months ago' '>3 months ago' 'no']
# 'Frequency of alcohol consumption', 'Smoking habit',
    # ['daily' 'never' 'occasional'  ['daily' 'never' 'occasional']
# 'Number of hours spent sitting per day    Diagnosis >> ['Altered' 'Normal']

from sklearn import tree
modeltree = tree.DecisionTreeClassifier()
modeltree.fit(xtrain, ytrain)
print('score tree:', round(modeltree.score(xtest, ytest) * 100, 2), '%')

from sklearn.ensemble import RandomForestClassifier
modelRandomForest = RandomForestClassifier(n_estimators = 50)
modelRandomForest.fit(xtrain, ytrain)
print('score randomforest:',round(modelRandomForest.score(xtest, ytest) * 100, 2), '%')


Arin1 = modelLogR.predict([[29,0,0,0,2,0,0,5]])[0]
Bebi1 = modelLogR.predict([[31,0,1,1,2,2,1,1]])[0]
Caca1 = modelLogR.predict([[25,1,0,0,0,1,1,7]])[0]
Dini1 = modelLogR.predict([[28,0,1,1,2,1,0,16]])[0]
Enno1 = modelLogR.predict([[42,1,0,0,2,1,1,8]])[0]

Arin2 = modeltree.predict([[29,0,0,0,2,0,0,5]])[0]
Bebi2 = modeltree.predict([[31,0,1,1,2,2,1,1]])[0]
Caca2 = modeltree.predict([[25,1,0,0,0,1,1,7]])[0]
Dini2 = modeltree.predict([[28,0,1,1,2,1,0,16]])[0]
Enno2 = modeltree.predict([[42,1,0,0,2,1,1,8]])[0]

Arin3 = modelRandomForest.predict([[29,0,0,0,2,0,0,5]])[0]
Bebi3 = modelRandomForest.predict([[31,0,1,1,2,2,1,1]])[0]
Caca3 = modelRandomForest.predict([[25,1,0,0,0,1,1,7]])[0]
Dini3 = modelRandomForest.predict([[28,0,1,1,2,1,0,16]])[0]
Enno3 = modelRandomForest.predict([[42,1,0,0,2,1,1,8]])[0]

if Arin1 == 0 or Bebi1 == 0 or Caca1 == 0 or Dini1 == 0 or Enno1 == 0:
    Arin1 = 'Altered'
    Bebi1 = 'Altered'
    Caca1 = 'Altered'
    Dini1 = 'Altered'
    Enno1 = 'Altered'
else :
    Arin1 = 'Normal'
    Bebi1 = 'Normal'
    Caca1 = 'Normal'
    Dini1 = 'Normal'
    Enno1 = 'Normal'

if Arin2 == 0 or Bebi2 == 0 or Caca2 == 0 or Dini2 == 0 or Enno2 == 0:
    Arin2 = 'Altered'
    Bebi2 = 'Altered'
    Caca2 = 'Altered'
    Dini2 = 'Altered'
    Enno2 = 'Altered'
else :
    Arin2 = 'Normal'
    Bebi2 = 'Normal'
    Caca2 = 'Normal'
    Dini2 = 'Normal'
    Enno2 = 'Normal'

if Arin3 == 0 or Bebi3 == 0 or Caca3 == 0 or Dini3 == 0 or Enno3 == 0:
    Arin3 = 'Altered'
    Bebi3 = 'Altered'
    Caca3 = 'Altered'
    Dini3 = 'Altered'
    Enno3 = 'Altered'
else :
    Arin3 = 'Normal'
    Bebi3 = 'Normal'
    Caca3 = 'Normal'
    Dini3 = 'Normal'
    Enno3 = 'Normal'

print('Arin, prediksi kesuburan: ', Arin1, '(Logistic Regression)')
print('Arin, prediksi kesuburan: ', Arin2, '(Decision Tree)')
print('Arin, prediksi kesuburan: ', Arin3, '(Random Forest)')

print('Bebi, prediksi kesuburan: ', Bebi1, '(Logistic Regression)')
print('Bebi, prediksi kesuburan: ', Bebi2, '(Decision Tree)')
print('Bebi, prediksi kesuburan: ', Bebi3, '(Random Forest)')

print('Caca, prediksi kesuburan: ', Caca1, '(Logistic Regression)')
print('Caca, prediksi kesuburan: ', Caca2,  '(Decision Tree)')
print('Caca, prediksi kesuburan: ', Caca3,  'Random Forest)')

print('Dini, prediksi kesuburan: ', Dini1, '(Logistic Regression)')
print('Dini, prediksi kesuburan: ', Dini2, '(Decision Tree)')
print('Dini, prediksi kesuburan: ', Dini3, '(Random Forest)')

print('Enno, prediksi kesuburan: ', Enno1, '(Logistic Regression)')
print('Enno, prediksi kesuburan: ', Enno2, '(Decision Tree)')
print('Enno, prediksi kesuburan: ', Enno3, '(Random Forest)')
