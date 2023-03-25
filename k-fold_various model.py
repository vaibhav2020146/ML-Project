from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('C://Users//91991//Desktop//ML//Project//survey lung cancer.csv')

#randomly shuffle the data
data=data.sample(frac=1).reset_index(drop=True)

X=data.drop('LUNG_CANCER',axis=1)
y=data['LUNG_CANCER']
#we are taking 10 folds
result_logistic=[]
result_bernoulli=[]
result_gaussian=[]
result_randomforest=[]
for folds in range(2,11):
    kfold=KFold(n_splits=folds)
    #apply k-fold cross validation on logistic regression:
    model=LogisticRegression()
    results=cross_val_score(model,X,y,cv=kfold)
    '''results=[]
    for train_index,test_index in kfold.split(X):
        X_train,X_test=X.iloc[train_index],X.iloc[test_index]
        y_train,y_test=y.iloc[train_index],y.iloc[test_index]
        model.fit(X_train,y_train)
        results.append(model.score(X_test,y_test))'''
    #print(results)
    #print("Accuracy for logistic regression: ",results.mean())
    result_logistic.append(results.mean())

    #apply k-fold cross validation on bernoulli naive bayes:
    model=BernoulliNB()
    results=cross_val_score(model,X,y,cv=kfold)
    #print(results)
    #print("Accuracy for bernoulli naive bayes: ",results.mean())
    result_bernoulli.append(results.mean())

    #apply k-fold cross validation on gaussian naive bayes:
    model=GaussianNB()
    results=cross_val_score(model,X,y,cv=kfold)
    #print(results)
    #print("Accuracy for gaussian naive bayes: ",results.mean())
    result_gaussian.append(results.mean())

    #apply k-fold cross validation on random forest:
    model=RandomForestClassifier()
    results=cross_val_score(model,X,y,cv=kfold)
    #print(results)
    #print("Accuracy for random forest: ",results.mean())
    result_randomforest.append(results.mean())


plt.plot(range(2,11),result_logistic,label='logistic regression')
plt.xlabel('Number of folds')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(range(2,11),result_bernoulli,label='bernoulli naive bayes')
plt.xlabel('Number of folds')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(range(2,11),result_gaussian,label='gaussian naive bayes')
plt.xlabel('Number of folds')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(range(2,11),result_randomforest,label='random forest')
plt.xlabel('Number of folds')
plt.ylabel('Accuracy')
plt.legend()
plt.show()