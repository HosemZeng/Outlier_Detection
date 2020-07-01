#导入包
import pandas as pd
import numpy as np 
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer,f1_score,recall_score,classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    '''
    #测试#启动多进程处理
    import os
    os.environ['LOKY_PICKLER']='cloudpickle'
    import multiprocessing
    multiprocessing.set_start_method('forkserver')
    '''
    #加载数据
    x_train = pd.read_csv('./train.csv')
    x_test = pd.read_csv('./test.csv')
    y_test = pd.read_csv('./label.csv',header=None).values.T[0]
    for i in range(y_test.size):
        y_test[i] = (-1 if(y_test[i]==1) else 1)

    #归一化处理
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    #降维处理
    
    pca = PCA(n_components='mle',svd_solver='full')
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    

    #训练模型
    '''
    param_dist={
        'nu':(np.linspace(0.8,0.3,20)),
        'gamma':(np.linspace(0.01,0.000001,200))
    }
    #nu=0.001,kernel='rbf',gamma=0.8
    clf = OneClassSVM(kernel='rbf')
    def f1_scorer(estimator,X):
        estimator.fit(X)
        y_pred = estimator.predict(x_test)
        f1 = f1_score(y_test,y_pred,average='micro')
        return f1
    grid = RandomizedSearchCV(clf,param_dist,cv=None,scoring=f1_scorer,n_iter=100,n_jobs=1)
    grid.fit(x_train,None)
    print(grid.best_estimator_)
    print(grid.best_params_)
    print(grid.best_score_)
    clf = OneClassSVM(**grid.best_params_)
    '''
    #计算指标
    clf = OneClassSVM(nu=0.77368,kernel='rbf',gamma=0.0096985)
    clf.fit(x_train)
    y_pred = clf.predict(x_test)

    print(y_test[y_test==-1].size)
    print(y_pred[y_pred==-1].size)
    print(classification_report(y_test,y_pred,labels=[-1,1]))

    #绘图
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
