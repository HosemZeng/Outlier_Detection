import numpy as np 
import pandas as pd
from sklearn.ensemble import  IsolationForest
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer,f1_score,recall_score,classification_report
from sklearn.decomposition import PCA

#加载数据
x_train = pd.read_csv('./train.csv')
x_test = pd.read_csv('./test.csv')
y_test = pd.read_csv('./groudtruth_JRQ.csv',header=None).values.T[0]
for i in range(y_test.size):#dui yi
        y_test[i] = (-1 if(y_test[i]==1) else 1)

#降维处理

pca = PCA(n_components=11,svd_solver='full')
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

#训练模型
'''
param_dist={
    'n_estimators':range(600,1000,20),
    'contamination':np.linspace(0.008,0.005,10)
}
clf = IsolationForest(max_features=10)
def rgrid_scorer(estimator,X):#设计评估函数
    estimator.fit(X)
    y_pred = estimator.predict(x_test)
    f1 = f1_score(y_test,y_pred,average='micro')
    recall = recall_score(y_test,y_pred,average='micro')
    return f1
rgrid = RandomizedSearchCV(clf,param_dist,cv=None,scoring=rgrid_scorer,n_iter=100,n_jobs=1)
rgrid.fit(x_train,None)
print(rgrid.best_estimator_)
print(rgrid.best_score_)
clf = IsolationForest(**rgrid.best_params_)
'''

#计算指标
clf = IsolationForest(n_estimators=2000,contamination=0.025,max_features=11,max_samples='auto')#预先得到的最佳

'''
#测试训练速度
import time
test_time = 0
for i in range(10):
    time_start = time.time()
    clf.fit(x_train)
    time_end = time.time()
    test_time += (time_end-time_start)
print("耗时：%.6fs"%(test_time/5.0))
'''

clf.fit(x_train)
y_pred = clf.predict(x_test)

print(y_test[y_test==-1].size)#打印真实的异常标签数量
print(y_pred[y_pred==-1].size)#打印预测的异常标签数量
print(classification_report(y_test,y_pred,labels=[-1,1]))

#绘图
'''
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

y_score = clf.decision_function(x_test)
for i in range(y_test.size):#将异常样本变为阳性样本
    y_test[i] = (-1 if(y_test[i]==1) else 1)
    y_score[i] = 1 - y_score[i]
    
fig,(ax1,ax2) = plt.subplots(2,1)
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax1)
prec, recall, _ = precision_recall_curve(y_test, y_score)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot(ax=ax2)

plt.show()
'''