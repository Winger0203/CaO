import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
from xgboost import XGBRegressor,XGBRFRegressor,XGBRFClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,ExtraTreesRegressor,StackingRegressor,BaggingRegressor,GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor,VotingRegressor,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,ARDRegression,PassiveAggressiveRegressor,BayesianRidge
from sklearn.linear_model import TheilSenRegressor,RANSACRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR,LinearSVC
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn import linear_model
from sklearn.linear_model import LinearRegression,Lasso,LassoCV,LassoLars,LassoLarsCV,LassoLarsIC,ElasticNet,ElasticNetCV
import lightgbm as lgb

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples",fontsize=12)
    plt.ylabel("Score",fontsize=12)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Train score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best",fontsize=12,frameon=True)
    return plt


dataset= r'C:\Users\86151\Desktop\CaO\吸附能分类\Train.xlsx'
data=pd.DataFrame(pd.read_excel(dataset))
X=data.values[:,0:-1]
# band gap，即目标值
y= data.values[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2,random_state=1)

# 图一
title = r"Learning Curves (RandomForestRegressor)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, train_size=0.8,random_state=1)
estimator =RandomForestClassifier(n_estimators=150) # 建模
plot_learning_curve(estimator, title, X, y, ylim=(0, 1.01), cv=cv, n_jobs=1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

###设置坐标轴的粗细
ax=plt.gca()#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1.5)###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1.5)####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1.5)###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1.5)####设置上部坐标轴的粗细

# 图二
# title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=3)
# estimator = SVC()  # 建模
# plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=1)
#
plt.show()