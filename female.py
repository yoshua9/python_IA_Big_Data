# -*- coding: utf-8 -*-
"""
Created on Sun May 31 17:53:06 2020

@author: Yoshua
"""
# Librerias
import numpy as nump 
import pandas as pand 
import seaborn as seans
import re
from sys import exit
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plot
from matplotlib import style
from sklearn import linear_model
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics  import roc_auc_score 


#Configurar la visualización de los datos que se muestran en el pandas
#pand.set_option('display.max_rows', 500)
#pand.set_option('display.max_columns', 500)
#pand.set_option('display.width', 1000)

#Recogemos los ficheros
test_df = pand.read_csv("test.csv")
train_df = pand.read_csv("train.csv")

#print('Datos que Faltan:')
#print(pand.isnull(train_df).sum())
#print(pand.isnull(test_df).sum())

#print('Tipos de Datos:')
#print(train_df.info())

#Describe las variables
#print(train_df.describe())

#Describe las variables
#print(train_df.head(20))

#Filtrar por las columnas con muchos nulos
#train_df = train_df.loc[:, train_df.isnull().mean() < .05]
#print(train_df.head())

#Se realiza la suma y se ordenan los valores de forma descendente cuando son "Null"
total = train_df.isnull().sum().sort_values(ascending=False)

#Sacamos los porcentajes
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
#redondeamos el porcentaje que vamos a mostrar
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

#mostramos los valores que tienen campos con tipo Null en dos  columnas(La suma de Null por columna y % sobre el global de filas)
missing_data = pand.concat([total, percent_2], axis=1, keys=['Total', '%'])

#Muestra la suma de valores NULL y el porcentaje sobre el Total
#print(missing_data.head(500))

#Ver los valores de la columna
#print(train_df['MM11_15'].describe())

#nos muestra todas las cabeceras del fichero
#print(train_df.columns.values)
'''
train_df = train_df[['train_id','AA3','AA4','AA7','AA14','AA15','DG1','DG3','DG3A','is_female'
,'DG4','DG5_1','DG5_2','DG5_3','DG5_4','DG5_5','DG5_6','DG5_7','DG5_8'
,'DG5_9','DG5_10','DG5_11','DG5_96','DG6','DG8a','DG8b','DG8c','DG9a'
,'DL0','DL1','DL4_1','DL4_2','DL4_3','DL4_4','DL4_5','DL4_6','DL4_7'
,'DL4_8','DL4_9','DL4_10','DL4_11','DL4_12','DL4_13','DL4_14','DL4_15'
,'DL4_16','DL4_17','DL4_18','DL4_19','DL4_20','DL4_21','DL4_22','DL4_23'
,'DL4_96','DL4_99','DL6','DL11','DL14','DL15','DL16','DL17','DL18','DL19'
,'DL20','DL21','DL22','DL23','DL24','DL25_1','DL25_2','DL25_3','DL25_4'
,'DL25_5','DL25_6','DL25_7','DL25_8','DL26_1','DL26_2','DL26_3','DL26_4'
,'DL26_5','DL26_6','DL26_7','DL26_8','DL26_9','DL26_10','DL26_12'
,'DL26_99','MT1','MT1A','MT2','MT10','FF1','MM1','MM2_1','MM2_2','MM2_3'
,'MM2_4','MM2_5','MM2_6','MM2_7','MM2_8','MM2_9','MM2_10','MM2_11'
,'MM2_12','MM2_13','MM2_14','MM2_15','MM2_16','MM3_1','MM3_2','MM3_3'
,'MM3_4','MM3_5','MM3_6','MM3_7','MM3_8','MM3_9','MM3_10','MM3_11'
,'MM3_12','MM3_13','MM3_14','MMP1_1','MMP1_2','MMP1_3','MMP1_4','MMP1_5'
,'MMP1_6','MMP1_7','MMP1_8','MMP1_9','MMP1_10','MMP1_11','MMP1_96'
,'IFI1_1','IFI3_1','IFI1_2','IFI3_2','IFI1_3','IFI3_3','IFI1_4','IFI1_5'
,'IFI1_6','IFI1_7','IFI1_8','IFI1_9','IFI14_1','IFI14_2','IFI14_3'
,'IFI14_4','IFI14_5','IFI14_6','IFI14_7','IFI15_1','IFI15_2','IFI15_3'
,'IFI15_4','IFI15_5','IFI15_6','IFI15_7','IFI18','FL1','FL4','FL6_1'
,'FL6_2','FL6_3','FL6_4','FL7_1','FL7_2','FL7_3','FL7_4','FL7_5','FL7_6'
,'FL8_1','FL8_2','FL8_3','FL8_4','FL8_5','FL8_6','FL8_7','FL9A','FL10'
,'FL11','FL12','FL13','FL14','FL15','FL16','FL17','FL18','FB1_1','FB1_2'
,'FB1_3','FB2','FB3','FB13','FB16_1','FB16_2','FB16_3','FB16_4','FB16_5'
,'FB16_6','FB16_7','FB16_8','FB16_96','FB18','FB19','FB19A_1','FB19A_2'
,'FB19A_3','FB19A_4','FB19A_5','FB19A_96','FB19B_1','FB19B_2','FB19B_3'
,'FB19B_4','FB19B_5','FB19B_96','FB22_1','FB22_2','FB22_3','FB22_4'
,'FB22_5','FB22_6','FB22_7','FB22_8','FB22_9','FB22_10','FB22_11'
,'FB22_12','FB22_96','FB27_1','FB27_2','FB27_3','FB27_4','FB27_5','FB27_6'
,'FB27_7','FB27_8','FB27_9','FB27_96','FB29_1','FB29_2','FB29_3','FB29_4'
,'FB29_5','FB29_6','FB29_96','LN1A','LN1B','LN2_1','LN2_2','LN2_3','LN2_4'
,'GN2','GN3','GN4','GN5']]

test_df = test_df[['test_id','AA3','AA4','AA7','AA14','AA15','DG1','DG3','DG3A'
,'DG4','DG5_1','DG5_2','DG5_3','DG5_4','DG5_5','DG5_6','DG5_7','DG5_8'
,'DG5_9','DG5_10','DG5_11','DG5_96','DG6','DG8a','DG8b','DG8c','DG9a'
,'DL0','DL1','DL4_1','DL4_2','DL4_3','DL4_4','DL4_5','DL4_6','DL4_7'
,'DL4_8','DL4_9','DL4_10','DL4_11','DL4_12','DL4_13','DL4_14','DL4_15'
,'DL4_16','DL4_17','DL4_18','DL4_19','DL4_20','DL4_21','DL4_22','DL4_23'
,'DL4_96','DL4_99','DL6','DL11','DL14','DL15','DL16','DL17','DL18','DL19'
,'DL20','DL21','DL22','DL23','DL24','DL25_1','DL25_2','DL25_3','DL25_4'
,'DL25_5','DL25_6','DL25_7','DL25_8','DL26_1','DL26_2','DL26_3','DL26_4'
,'DL26_5','DL26_6','DL26_7','DL26_8','DL26_9','DL26_10','DL26_12'
,'DL26_99','MT1','MT1A','MT2','MT10','FF1','MM1','MM2_1','MM2_2','MM2_3'
,'MM2_4','MM2_5','MM2_6','MM2_7','MM2_8','MM2_9','MM2_10','MM2_11'
,'MM2_12','MM2_13','MM2_14','MM2_15','MM2_16','MM3_1','MM3_2','MM3_3'
,'MM3_4','MM3_5','MM3_6','MM3_7','MM3_8','MM3_9','MM3_10','MM3_11'
,'MM3_12','MM3_13','MM3_14','MMP1_1','MMP1_2','MMP1_3','MMP1_4','MMP1_5'
,'MMP1_6','MMP1_7','MMP1_8','MMP1_9','MMP1_10','MMP1_11','MMP1_96'
,'IFI1_1','IFI3_1','IFI1_2','IFI3_2','IFI1_3','IFI3_3','IFI1_4','IFI1_5'
,'IFI1_6','IFI1_7','IFI1_8','IFI1_9','IFI14_1','IFI14_2','IFI14_3'
,'IFI14_4','IFI14_5','IFI14_6','IFI14_7','IFI15_1','IFI15_2','IFI15_3'
,'IFI15_4','IFI15_5','IFI15_6','IFI15_7','IFI18','FL1','FL4','FL6_1'
,'FL6_2','FL6_3','FL6_4','FL7_1','FL7_2','FL7_3','FL7_4','FL7_5','FL7_6'
,'FL8_1','FL8_2','FL8_3','FL8_4','FL8_5','FL8_6','FL8_7','FL9A','FL10'
,'FL11','FL12','FL13','FL14','FL15','FL16','FL17','FL18','FB1_1','FB1_2'
,'FB1_3','FB2','FB3','FB13','FB16_1','FB16_2','FB16_3','FB16_4','FB16_5'
,'FB16_6','FB16_7','FB16_8','FB16_96','FB18','FB19','FB19A_1','FB19A_2'
,'FB19A_3','FB19A_4','FB19A_5','FB19A_96','FB19B_1','FB19B_2','FB19B_3'
,'FB19B_4','FB19B_5','FB19B_96','FB22_1','FB22_2','FB22_3','FB22_4'
,'FB22_5','FB22_6','FB22_7','FB22_8','FB22_9','FB22_10','FB22_11'
,'FB22_12','FB22_96','FB27_1','FB27_2','FB27_3','FB27_4','FB27_5','FB27_6'
,'FB27_7','FB27_8','FB27_9','FB27_96','FB29_1','FB29_2','FB29_3','FB29_4'
,'FB29_5','FB29_6','FB29_96','LN1A','LN1B','LN2_1','LN2_2','LN2_3','LN2_4'
,'GN2','GN3','GN4','GN5']]
'''

train_df=train_df[["DL0","DG6","DL1","MT1A","FL4","DG3","MT10","GN5","GN4","MT2","DG1","GN3"
                   ,"GN2","DG8a","DG4","is_female","train_id"]]
test_df=test_df[["DL0","DG6","DL1","MT1A","FL4","DG3","MT10","GN5","GN4","MT2","DG1","GN3"
                 ,"GN2","DG8a","DG4","test_id"]]

test_df=test_df.dropna()
train_df=train_df.dropna()


'''
"AA3","AA5","AA6","DG3","DG3A","DG14","DL2","DL5","DL27","DL28","MT1A","MT5","MT6","MT6A",
"MT6B","MT7A","MT9","MT11","FF13","MM10B","MM12","MM13","MM14","MM18","MM19","MM20","MM21","MM28","MM30",
"MM34","MM41","IFI5_1","IFI5_2","IFI5_3","IFI24","FL4","FL9A","FL9B","FL9C","FL10","FB2","FB19","FB20",
"FB21","FB24","FB25"

print(train_df.groupby("AA3")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("AA5")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("AA6")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("DG3")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("DG3A")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("DG14")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("DL2")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("DL5")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("DL27")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("DL28")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MT1A")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MT5")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MT6")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MT6A")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MT6B")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MT7A")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MT9")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MT11")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("FF13")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MM10B")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MM12")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MM13")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MM14")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MM18")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MM19")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MM20")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MM21")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MM28")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MM30")["is_female"].agg(['count', 'mean']))

'''
"MM34","MM41","IFI5_1","IFI5_2","IFI5_3","IFI24","FL4","FL9A","FL9B","FL9C","FL10","FB2","FB19","FB20",
"FB21","FB24","FB25"
'''
print(train_df.groupby("MM34")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("MM41")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("IFI5_1")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("IFI5_2")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("IFI5_3")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("IFI24")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("FL4")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("FL9A")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("FL9B")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("FL9C")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("FL10")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("FB2")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("FB19")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("FB20")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("FB21")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("FB24")["is_female"].agg(['count', 'mean']))
print(train_df.groupby("FB25")["is_female"].agg(['count', 'mean']))
'''

#Agrupar cada variable sobre is_female para generar agrupaciones
#print(train_df.groupby("AA7")["is_female"].agg(['count', 'mean']))

#Agrupar valores para dotarlos de mayor peso y correlación
for x in train_df,test_df:
    x.loc[x["DG6"]!=2,"DG6"]=0
    x.loc[x["DL1"]!=7,"DL1"]=0
    x.loc[x["MT1A"]!=2.0,"MT1A"]=0


#print(train_df.groupby("AA7")["is_female"].agg(['count', 'mean']))



#Separamos la columna superviviente en otra matriz de datos y la de test sin el ID de Pasajero
X_train = train_df.drop("is_female", axis=1)
X_train = X_train.drop("train_id", axis=1)
Y_train = train_df["is_female"]
X_test  = test_df.drop("test_id", axis=1).copy()


#Nuevos parametros de Random Forest
'''
random_forest = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 10,   
                                       n_estimators=100, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)

random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
'''

#Mostrar la importancia de cara atributo sobre la probabilidad de ser mujer o no
#importances = pand.DataFrame({'feature':X_train.columns,'importance':nump.round(random_forest.feature_importances_,3)})
#importances = importances.sort_values('importance',ascending=False).set_index('feature')
#print(importances.head(20))

#Visualizar graficamente la importancia de los atributos del Dataset
#importances.plot.bar()

### VISUALIZACIONES ###
'''
#Algoritmo Stochastic Gradient Descent (SGD)
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)

sgd.score(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
#print('Algoritmo Stochastic Gradient Descent (SGD):',acc_sgd)
'''
#Algoritmo Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
#print('Algoritmo Random Forest:',acc_random_forest)
'''
#Algoritmo Regresión Logistica
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
#print('Algoritmo Regresión Logistica:',acc_log)

#Algoritmo K Nearest Neighbor KNN
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, Y_train)  
Y_pred = knn.predict(X_test)  
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
#print('Algoritmo K Nearest Neighbor KNN:',acc_log)


#### Algoritmo Gaussian Naive Bayes
gaussian = GaussianNB() 
gaussian.fit(X_train, Y_train) 
Y_pred = gaussian.predict(X_test)  
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
#print('Algoritmo Gaussian Naive Bayes:',acc_gaussian)

#Algoritmo Perceptron
perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
#print('Algoritmo Perceptron:',acc_perceptron)

#Algoritmo Linear Support Vector Machine(Algoritmo maquinas de Soporte Lineal):
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
#print('Algoritmo Linear Support Vector Machine:',acc_linear_svc)

#Support Vector Machines (Algoritmo maquinas de Soporte Lineal)
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
#print('Precisión Soporte de Vectores:',acc_svc)

#Algoritmo Decision Tree
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, Y_train)  
Y_pred = decision_tree.predict(X_test)  
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
#print('Algoritmo Decision Tree:',acc_decision_tree)

'''

#Mostramos cual de los algoritmos es el más fiable en una matriz de datos
'''
results = pand.DataFrame({
    'Model': ['Support Vector Machines', 'Linear Support Vector Machines', 'KNN', 'Regresión Logistica', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
print(result_df.head(9))
'''

#Afinar el algoritmo Random Forest porque es de los que mejor tasa de acierto tiene

rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
#print("Scores:", scores)
#print("Mean:", scores.mean())
#print("Standard Deviation:", scores.std())


#Mostrar la importancia de cara atributo sobre la probabilidad de ser mujer
importances = pand.DataFrame({'feature':X_train.columns,'importance':nump.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
#print(importances.head(20))

#Visualizar graficamente la importancia de los atributos del Dataset
#importances.plot.bar()


#####AFINAR EL MODELO#####

param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10, 25, 50, 70], "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100, 400, 700, 1000, 1500]}
rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)
clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)
#clf.fit(X_train, Y_train)
#print(clf.bestparams)


#Nuevos parametros de Random Forest
random_forest = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 10,   
                                       n_estimators=100, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)

random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


#####CALCULAR PRECICIÓN UTILIZANDO LOS MODELOS#####

#Matriz de Precisión
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
print(confusion_matrix(Y_train, predictions))

#Precisión del modelo
print("Precision:", precision_score(Y_train, predictions))
print("Recall:",recall_score(Y_train, predictions))

#Puntuación F
print("Preción F",f1_score(Y_train, predictions))


#Visualizar curva de PRECISIÓN

y_scores = random_forest.predict_proba(X_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(Y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plot.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plot.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plot.xlabel("threshold", fontsize=19)
    plot.legend(loc="upper right", fontsize=19)
    plot.ylim([0, 1])

plot.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plot.show()


#Otra opción de visualizar la precisión

def plot_precision_vs_recall(precision, recall):
    plot.plot(recall, precision, "g--", linewidth=2.5)
    plot.ylabel("recall", fontsize=19)
    plot.xlabel("precision", fontsize=19)
    plot.axis([0, 1.5, 0, 1.5])

plot.figure(figsize=(14, 7))
plot_precision_vs_recall(precision, recall)
plot.show()


#Curva ROC AUC
#calcular la tasa de verdaderos positivos y la tasa de falsos positivos

false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)
#plotting entre cada uno de ellos
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plot.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plot.plot([0, 1], [0, 1], 'r', linewidth=4)
    plot.axis([0, 1, 0, 1])
    plot.xlabel('False Positive Rate (FPR)', fontsize=16)
    plot.ylabel('True Positive Rate (TPR)', fontsize=16)

plot.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plot.show()


#Puntuación de ROC AUC
r_a_score = roc_auc_score (Y_train, y_scores) 
print ("ROC-AUC-Score:", r_a_score)

#GENERAR EL NUEVO FICHERO DE PREDICCION

#Crear un DataFrame con las identificaciones de los pasajeros y nuestra predicción sobre si sobrevivieron o no.
submission = pand.DataFrame({'test_id':test_df['test_id'],'is_female':Y_prediction})

filename = 'sample_submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


exit()
#EXPORTACIÓN DEL DATASET FINAL EN OTRO FICHERO CSV
#X_train.to_csv(r'export_Train_DEFINITIVO.csv', index = False, header=True)
#X_test.to_csv(r'export_Test_DEFINITIVO.csv', index = False, header=True)
#train_df.to_csv(r'export_TRAIN2_DEFINITIVO.csv', index = False, header=True)
#test_df.to_csv(r'export_TEST2_DEFINITIVO.csv', index = False, header=True)