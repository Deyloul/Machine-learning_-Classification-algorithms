import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
iris=pd.read_csv('C:/Iris.csv')
iris.head()
   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
0   1            5.1           3.5            1.4           0.2  Iris-setosa
1   2            4.9           3.0            1.4           0.2  Iris-setosa
2   3            4.7           3.2            1.3           0.2  Iris-setosa
3   4            4.6           3.1            1.5           0.2  Iris-setosa
4   5            5.0           3.6            1.4           0.2  Iris-setosa
iris['Species'].unique()
array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dtype=object)
iris.describe(include='all')
                Id  SepalLengthCm  ...  PetalWidthCm      Species
count   150.000000     150.000000  ...    150.000000          150
unique         NaN            NaN  ...           NaN            3
top            NaN            NaN  ...           NaN  Iris-setosa
freq           NaN            NaN  ...           NaN           50
mean     75.500000       5.843333  ...      1.198667          NaN
std      43.445368       0.828066  ...      0.763161          NaN
min       1.000000       4.300000  ...      0.100000          NaN
25%      38.250000       5.100000  ...      0.300000          NaN
50%      75.500000       5.800000  ...      1.300000          NaN
75%     112.750000       6.400000  ...      1.800000          NaN
max     150.000000       7.900000  ...      2.500000          NaN

[11 rows x 6 columns]
iris.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 6 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             150 non-null    int64  
 1   SepalLengthCm  150 non-null    float64
 2   SepalWidthCm   150 non-null    float64
 3   PetalLengthCm  150 non-null    float64
 4   PetalWidthCm   150 non-null    float64
 5   Species        150 non-null    object 
dtypes: float64(4), int64(1), object(1)
memory usage: 7.2+ KB
#As we can see above data distribution of data points in each class is equal so Iris is a balanced dataset as the number of data points for every class is 50#

# Removing the unneeded column #
iris.drop(columns="Id",inplace=True)
iris.isnull().sum()  # Checking if there are any missing values #
SepalLengthCm    0
SepalWidthCm     0
PetalLengthCm    0
PetalWidthCm     0
Species          0
dtype: int64
import missingno as msno
Traceback (most recent call last):
  File "<pyshell#15>", line 1, in <module>
    import missingno as msno
ModuleNotFoundError: No module named 'missingno'
import missingno as msno
msno.bar(iris,figsize=(8,6),color='skyblue')
<Axes: >
plt.show()

# Data Visualization #
g=sns.relplot(x='SepalLengthCm',y='SepalWidthCm',data=iris,hue='Species',style='Species')
g.fig.set_size_inches(10,5)
plt.show()
g=sns.relplot(x='PetalLengthCm',y='PetalWidthCm',data=iris,hue='Species',style='Species')
g.fig.set_size_inches(10,5)
plt.show()

# As we can see that the Petal Features are giving a better cluster division compared to the Sepal features. This is an indication that the Petals can help in better and accurate Predictions over the Sepal. #

sns.pairplot(iris,hue="Species")
<seaborn.axisgrid.PairGrid object at 0x000001AC20548BF0>
plt.show()
# from the graph we can see the scatter plot between the any two features and the distributions. from the distributions above peatl length is separating the iris setosa from remaining . from plot between petal length and petal width we can separate the flowers #

# Correlation #
iris.corr()
Traceback (most recent call last):
  File "<pyshell#35>", line 1, in <module>
    iris.corr()
  File "C:\Users\dell i7\AppData\Roaming\Python\Python312\site-packages\pandas\core\frame.py", line 11049, in corr
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
  File "C:\Users\dell i7\AppData\Roaming\Python\Python312\site-packages\pandas\core\frame.py", line 1993, in to_numpy
    result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
  File "C:\Users\dell i7\AppData\Roaming\Python\Python312\site-packages\pandas\core\internals\managers.py", line 1694, in as_array
    arr = self._interleave(dtype=dtype, na_value=na_value)
  File "C:\Users\dell i7\AppData\Roaming\Python\Python312\site-packages\pandas\core\internals\managers.py", line 1753, in _interleave
    result[rl.indexer] = arr
ValueError: could not convert string to float: 'Iris-setosa'
iris.corr(numeric_only=True)
               SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm
SepalLengthCm       1.000000     -0.109369       0.871754      0.817954
SepalWidthCm       -0.109369      1.000000      -0.420516     -0.356544
PetalLengthCm       0.871754     -0.420516       1.000000      0.962757
PetalWidthCm        0.817954     -0.356544       0.962757      1.000000
plt.subplots(figsize = (8,8))
(<Figure size 800x800 with 1 Axes>, <Axes: >)
sns.heatmap(iris.corr(numeric_only=True),annot=True,fmt="f").set_title("Corelation of attributes (petal length,width and sepal length,width) among Iris species")
Text(0.5, 1.0, 'Corelation of attributes (petal length,width and sepal length,width) among Iris species')
plt.show()
# The Sepal Width and Length are not correlated The Petal Width and Length are highly correlated. We will use all the features for training the algorithm and check the accuracy #

# Dividing data into features and labels #

X=iris.iloc[:,0:4].values
y=iris.iloc[:,4].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Building Machine Learning Models #

from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

# Splitting The Data into Training And Testing Dataset #

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

# Random Forest Model #

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
RandomForestClassifier()
Y_prediction = random_forest.predict(X_test)
accuracy_rf=round(accuracy_score(y_test,Y_prediction)* 100, 2)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
cm = confusion_matrix(y_test, Y_prediction)
accuracy = accuracy_score(y_test,Y_prediction)
precision =precision_score(y_test, Y_prediction,average='micro')
recall =  recall_score(y_test, Y_prediction,average='micro')
f1 = f1_score(y_test,Y_prediction,average='micro')
print('Confusion matrix for Random Forest\n',cm)
Confusion matrix for Random Forest
 [[16  0  0]
 [ 0 17  1]
 [ 0  0 11]]
print('accuracy_random_Forest : %.3f' %accuracy)
accuracy_random_Forest : 0.978
print('precision_random_Forest : %.3f' %precision)
precision_random_Forest : 0.978
print('recall_random_Forest : %.3f' %recall)
recall_random_Forest : 0.978
print('f1-score_random_Forest : %.3f' %f1)
f1-score_random_Forest : 0.978

# Logistic Regression: #

logreg = LogisticRegression(solver= 'lbfgs',max_iter=400)
logreg.fit(X_train, y_train)
LogisticRegression(max_iter=400)
Y_pred = logreg.predict(X_test)
accuracy_lr=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)
cm = confusion_matrix(y_test, Y_pred,)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='micro')
recall =  recall_score(y_test, Y_pred,average='micro')
f1 = f1_score(y_test,Y_pred,average='micro')
print('Confusion matrix for Logistic Regression\n',cm)
Confusion matrix for Logistic Regression
 [[16  0  0]
 [ 0 17  1]
 [ 0  0 11]]
print('accuracy_Logistic Regression : %.3f' %accuracy)
accuracy_Logistic Regression : 0.978
print('precision_Logistic Regression : %.3f' %precision)
precision_Logistic Regression : 0.978

# K Nearest Neighbor Model #

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
KNeighborsClassifier(n_neighbors=3)
Y_pred = knn.predict(X_test)
accuracy_knn=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='micro')
recall =  recall_score(y_test, Y_pred,average='micro')
f1 = f1_score(y_test,Y_pred,average='micro')
print('Confusion matrix for KNN\n',cm)
Confusion matrix for KNN
 [[16  0  0]
 [ 0 17  1]
 [ 0  0 11]]
print('accuracy_KNN : %.3f' %accuracy)
accuracy_KNN : 0.978
print('precision_KNN : %.3f' %precision)
precision_KNN : 0.978

# Gaussian Naive Bayes: #

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
GaussianNB()
Y_pred = gaussian.predict(X_test)
accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='micro')
recall =  recall_score(y_test, Y_pred,average='micro')
f1 = f1_score(y_test,Y_pred,average='micro')
print('Confusion matrix for Naive Bayes\n',cm)
Confusion matrix for Naive Bayes
 [[16  0  0]
 [ 0 18  0]
 [ 0  0 11]]
print('accuracy_Naive Bayes: %.3f' %accuracy)
accuracy_Naive Bayes: 1.000
print('precision_Naive Bayes: %.3f' %precision)
precision_Naive Bayes: 1.000

#  Linear Support Vector Machine: #

linear_svc = LinearSVC(max_iter=4000)
linear_svc.fit(X_train, y_train)
LinearSVC(max_iter=4000)
Y_pred = linear_svc.predict(X_test)
accuracy_svc=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='micro')
recall =  recall_score(y_test, Y_pred,average='micro')
f1 = f1_score(y_test,Y_pred,average='micro')
print('Confusion matrix for SVC\n',cm)
Confusion matrix for SVC
 [[16  0  0]
 [ 0 15  3]
 [ 0  0 11]]
print('accuracy_SVC: %.3f' %accuracy)
accuracy_SVC: 0.933
print('precision_SVC: %.3f' %precision)
precision_SVC: 0.933
print('recall_SVC: %.3f' %recall)
recall_SVC: 0.933
print('f1-score_SVC : %.3f' %f1)
f1-score_SVC : 0.933

#  Decision Tree: #

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
DecisionTreeClassifier()
Y_pred = decision_tree.predict(X_test)
accuracy_dt=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)
cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='micro')
recall =  recall_score(y_test, Y_pred,average='micro')
f1 = f1_score(y_test,Y_pred,average='micro')
print('Confusion matrix for DecisionTree\n',cm)
Confusion matrix for DecisionTree
 [[16  0  0]
 [ 0 17  1]
 [ 0  0 11]]
print('accuracy_DecisionTree: %.3f' %accuracy)
accuracy_DecisionTree: 0.978
print('precision_DecisionTree: %.3f' %precision)
precision_DecisionTree: 0.978
print('recall_DecisionTree: %.3f' %recall)
recall_DecisionTree: 0.978
print('f1-score_DecisionTree : %.3f' %f1)
f1-score_DecisionTree : 0.978
from sklearn.tree import plot_tree
plt.figure(figsize = (15,10))
<Figure size 1500x1000 with 0 Axes>
plot_tree(decision_tree.fit(X_train, y_train)  ,filled=True)
[Text(0.4, 0.9, 'x[3] <= 0.75\ngini = 0.664\nsamples = 105\nvalue = [34, 32, 39]'), Text(0.3, 0.7, 'gini = 0.0\nsamples = 34\nvalue = [34, 0, 0]'), Text(0.35, 0.8, 'True  '), Text(0.5, 0.7, 'x[2] <= 4.95\ngini = 0.495\nsamples = 71\nvalue = [0, 32, 39]'), Text(0.45, 0.8, '  False'), Text(0.2, 0.5, 'x[3] <= 1.65\ngini = 0.161\nsamples = 34\nvalue = [0, 31, 3]'), Text(0.1, 0.3, 'gini = 0.0\nsamples = 30\nvalue = [0, 30, 0]'), Text(0.3, 0.3, 'x[1] <= 3.1\ngini = 0.375\nsamples = 4\nvalue = [0, 1, 3]'), Text(0.2, 0.1, 'gini = 0.0\nsamples = 3\nvalue = [0, 0, 3]'), Text(0.4, 0.1, 'gini = 0.0\nsamples = 1\nvalue = [0, 1, 0]'), Text(0.8, 0.5, 'x[3] <= 1.75\ngini = 0.053\nsamples = 37\nvalue = [0, 1, 36]'), Text(0.7, 0.3, 'x[3] <= 1.65\ngini = 0.375\nsamples = 4\nvalue = [0, 1, 3]'), Text(0.6, 0.1, 'gini = 0.0\nsamples = 3\nvalue = [0, 0, 3]'), Text(0.8, 0.1, 'gini = 0.0\nsamples = 1\nvalue = [0, 1, 0]'), Text(0.9, 0.3, 'gini = 0.0\nsamples = 33\nvalue = [0, 0, 33]')]
plt.show()


>>> # Which is the best Model ? #

>>> Results = pd.DataFrame({
...     'Model': [ 'KNN', 
...               'Logistic Regression', 
...               'Random Forest',
...               'Naive Bayes',  
...               ' Support Vector Machine', 
...               'Decision Tree'],
...     'Score': [ acc_knn,
...               acc_log, 
...               acc_random_forest,
...               acc_gaussian,  
...               acc_linear_svc,
...               acc_decision_tree],
...     "Accuracy_score":[accuracy_knn,
...                       accuracy_lr,
...                       accuracy_rf,
...                       accuracy_nb,
...                       accuracy_svc,
...                       accuracy_dt
...                        ]})
>>> Result_df = Results.sort_values(by='Accuracy_score', ascending=False)
>>> Result_df = Result_df.reset_index(drop=True)
>>> Result_df.head(9)
                     Model   Score  Accuracy_score
0              Naive Bayes   94.29          100.00
1                      KNN   96.19           97.78
2      Logistic Regression   98.10           97.78
3            Random Forest  100.00           97.78
4            Decision Tree  100.00           97.78
5   Support Vector Machine   98.10           93.33
>>> 
>>> # As we see best Model is given by Naive Bayes(100% Accuracy). #

>>> plt.subplots(figsize=(12,8))
(<Figure size 1200x800 with 1 Axes>, <Axes: >)
>>> ax=sns.barplot(x='Model',y="Accuracy_score",data=Result_df)
>>> labels = (Result_df["Accuracy_score"])
>>> for i, v in enumerate(labels):
...     ax.text(i, v+1, str(v), horizontalalignment = 'center', size = 15, color = 'black')
