import matplotlib.pyplot as plot
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn import preprocessing
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from sklearn.metrics import r2_score

'''dataset= r'11.14.xlsx'
data=pd.DataFrame(pd.read_excel(dataset))
featureData=data.iloc[:,:19]
corMat = DataFrame(featureData.corr())  #corr 求相关系数矩阵
print(corMat)
writer = pd.ExcelWriter('output.xlsx')
corMat.to_excel(writer,'Sheet1')
writer.save()
plt.figure(figsize=(20, 30))
sns.heatmap(corMat, annot=False, vmax=1, square=True, cmap="Blues",linewidths=0)
plot.show()'''


#读取文件
dataset= r'C:\Users\86151\Desktop\CaO\吸附能分类\符号.xlsx'
data=pd.DataFrame(pd.read_excel(dataset))

X=data.values[:,0:-1]
# band gap，即目标值
y= data.values[:,-1]

# 划分训练集与测试集
# 选取百分之二十的数据作为测试集
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3,random_state=0)
# 标准化
# StandardScaler = StandardScaler() # 标准化转换
# StandardScaler.fit(X_train)
# StandardScaler.fit(X_test)
# X_train = StandardScaler.transform(X_train)   # 转换数据集
# X_test = StandardScaler.transform((X_test))

#画出所有特征关于目标值的相关系数排名
'''featureData=data.iloc[:,:-1]
corMat = DataFrame(featureData.corr())  #corr 求相关系数矩阵
print(corMat)
writer = pd.ExcelWriter('shiyong.xlsx')
corMat.to_excel(writer,'Sheet1')
writer.save()'''


#读取原数据集的特征和目标值
'''X=data.values[:195,:14]
for i in range(X.shape[1]):
    X[:,[i]] = preprocessing.MinMaxScaler().fit_transform(X[:,[i]])
y=data.values[:195,14]'''


#读取自定义的训练集和测试集
'''X=data.values[:140,:14]
for i in range(X.shape[1]):
    X[:,[i]] = preprocessing.MinMaxScaler().fit_transform(X[:,[i]])
y=data.values[:140,14]
testX=data.values[140:195,:14]
for i in range(testX.shape[1]):
    testX[:,[i]] = preprocessing.MinMaxScaler().fit_transform(testX[:,[i]])
testy=data.values[140:195,14]'''


#选取4种分类算法
# clf = GradientBoostingClassifier(n_estimators=500)
# clf = RandomForestClassifier(n_estimators=150)
# clf = svm.SVC(C=3522, kernel='rbf', degree=2,probability=True)
# clf=ExtraTreesClassifier(n_estimators=6, max_depth=None,min_samples_split=2, random_state=0)
kernel = 1.0 * RBF([1.0])
clf=GaussianProcessClassifier(max_iter_predict=10,kernel=kernel, warm_start=True)

#使用KFold交叉验证
# for nk in range(2,13):
#  kfolder = KFold(n_splits=nk)
#  score=0
#  for train, test in kfolder.split(X,y):
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
#    clf.fit(X_train,y_train)
#    prey=clf.predict(X_test)
#    true=0
#    for i in range(0,len(y_test)):
#      if prey[i]==y_test[i]:
#          true=true+1
#    score=true/len(y_test)+score
#  print(score/nk)


#画出ROC曲线
clf.fit(X, y)
y_score = clf.fit(X, y).predict_proba(X_test)
fpr,tpr,threshold = roc_curve(y_test, y_score[:,1])
roc_auc = auc(fpr,tpr)

lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
print(y_test)
print(clf.predict(X_test))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


#画出混淆矩阵
clf.fit(X, y)
prey=clf.predict(X_test)
true=0
for i in range(0,len(y_test)):
 if prey[i]==y_test[i]:
     true=true+1
print(true/30)
C = confusion_matrix(y_test, prey, labels=[0,1])
plt.imshow(C, cmap='Blues')
indices = range(len(C))
plt.xticks(indices, [0, 1],fontsize=30)
plt.yticks(indices, [0, 1],fontsize=30)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=30)
for first_index in range(len(C)):    #第几行
    for second_index in range(len(C)):    #第几列
         plt.text(first_index, second_index, C[first_index][second_index],fontsize=30,horizontalalignment='center')


plt.show()
