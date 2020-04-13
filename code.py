# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:12:04 2020

@author: Amit kumar
"""

import pandas as pd
import numpy as np

data = pd.read_csv("credit_data.csv")
names = list(data)

#Build dummy variables for categorical variables
def Cat_conversion(names):
    for i in names:
        data[i]=data[i].astype("category").cat.codes

Cat_conversion(names)

#Create target variable
X=data.drop('default',1)
y=data.default

#Split train data for cross validation
print("\nusing train_test_split\n")
from sklearn.model_selection import train_test_split
x_train,x_cv,y_train,y_cv = train_test_split(X,y,test_size=0.3)

#Fit model
from sklearn import tree
#using gini criteria
print("\tusing Gini Index")
dt1=tree.DecisionTreeClassifier(criterion='gini',random_state=0
                            ,min_samples_leaf=10
                            ,min_samples_split=10)
dt1.fit(x_train,y_train)

#Predict values for cv data
pred_cv1=dt1.predict(x_cv)

#Evaluate accuracy of model
from sklearn.metrics import accuracy_score

print("\tAccuracy:", round(accuracy_score(y_cv,pred_cv1)*100,2))

#using entropy criteria
print("\n\tusing Entropy")
dt2=tree.DecisionTreeClassifier(criterion='entropy',random_state=0
                            ,min_samples_leaf=10
                            ,min_samples_split=10)
dt2.fit(x_train,y_train)

#Predict values for cv data
pred_cv2=dt2.predict(x_cv)

print("\tAccuracy:", round(accuracy_score(y_cv,pred_cv2)*100,2))

#using k-fold splitting for cross validation
print("\nusing k-fold split\n")
from sklearn.model_selection import KFold

Score_entropy = []
Score_gini = []
kfold= KFold(n_splits=4, random_state=1, shuffle=True)
for train, test in kfold.split(X,y):
    x_train2, x_cv2 = X.iloc[train], X.iloc[test]
    y_train2, y_cv2 = y.iloc[train], y.iloc[test]
    
    #using gini index
    dt1.fit(x_train2,y_train2)
    pred_cv3=dt1.predict(x_cv2)
    Score_entropy.append(accuracy_score(y_cv2,pred_cv3))
    
    #using entropy criteria
    dt2.fit(x_train2,y_train2)
    pred_cv4=dt2.predict(x_cv2)
    Score_gini.append(accuracy_score(y_cv2,pred_cv4))
    
print("\tusing Gini Index")
print("\tAccuracy:", round(np.mean(Score_entropy)*100,2))

print("\n\tusing Entropy")
print("\tAccuracy:", round(np.mean(Score_gini)*100,2))

# to visualize decision tree using graphviz uncomment this part
'''
import pydotplus
import collections

# can ignore next two lines as used because of some path error in graphviz
import os
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

dot_data1 = tree.export_graphviz(dt1,
                                feature_names=list(X),
                                out_file=None,
                                filled=True,
                                rounded=True)
graph1 = pydotplus.graph_from_dot_data(dot_data1)

for edge in graph1.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph1.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
        
graph1.write_png('gini.png')

dot_data2 = tree.export_graphviz(dt2,
                                feature_names=list(X),
                                out_file=None,
                                filled=True,
                                rounded=True)
graph2 = pydotplus.graph_from_dot_data(dot_data2)

for edge in graph2.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph2.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
        
graph2.write_png('entropy.png')
'''
