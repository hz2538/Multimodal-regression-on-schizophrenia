import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from tqdm import tnrange, tqdm_notebook
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, LassoLars, LassoLarsIC 
from sklearn.linear_model import OrthogonalMatchingPursuit, ElasticNet
from matplotlib import pyplot as plt

# if n_subject is 141, substitute weights=[15,14,14,14,14,14,14,14,14,14]

def write_plot(df1, df2, df3, filename, scaler):
    plt.figure()
    plt.plot([0,0.6],[0,0.6])
    plt.scatter(df1['Predict'],df1['True'],color='#97CBFF',marker='v')
    plt.xlabel('$R^2\;Plot$')
    df = pd.DataFrame(scaler.inverse_transform(df1),columns=['Predict','True'])

    with pd.ExcelWriter(filename+'.xlsx') as writer:
        df.to_excel(writer, sheet_name='Predict',index=False)
        df2.to_excel(writer, sheet_name='Score',index=False)
        df3.to_excel(writer, sheet_name='Feature',index=False)
    return df

def importance_map(mat, fig_on_row, tspace, cm):
    for j in range(int(mat.shape[1]/fig_on_row)):
        tmp = mat[:,j*fig_on_row:(j+1)*fig_on_row-1]
        plt.matshow(tmp, cmap=cm)
        plt.title("Feature Ranking/Importance with 10-Fold ("+str(j*fig_on_row+1)+'th to'+str((j+1)*fig_on_row)+'th features)',
                 y=tspace)
        plt.tight_layout
        # plt.colorbar()
    plt.show()
    
def lassoCD(X, y, ll, ul, step, weight, state):
    kf = KFold(n_splits=10,shuffle=True,random_state=state)
    feature=[]
    pred=[]
    true=[]
    r2=[]
    mse=[]
    ilist=np.linspace(ll,ul,step)
    pbar = tnrange(step*10, desc='loop')
    for i in ilist:
        r2_single=[]
        mse_single=[]
        pred_single=[]
        true_single=[]
        feature_single=[]
        for train_index, test_index in kf.split(X):
            y_train, y_test = y[train_index], y[test_index]
            X_train_tmp, X_test_tmp = X[train_index], X[test_index]
            
            clf = Lasso(alpha=i)
            clf.fit(X_train_tmp, np.ravel(y_train))
            feature_index = np.where(clf.coef_>0)[0]
            X_train = X_train_tmp[:,feature_index]
            X_test = X_test_tmp[:,feature_index]

            svr = svm.SVR(kernel='linear')
            svr.fit(X_train, np.ravel(y_train))
            y_test_pred = svr.predict(X_test)
            
            feature_single.append(feature_index)
            pred_single.append(y_test_pred)
            true_single.append(np.ravel(y_test))
            r2_single.append(r2_score(y_test, y_test_pred))
            mse_single.append(mean_squared_error(y_test, y_test_pred))
            pbar.update(1)
        r2.append(r2_single)
        if (r2_single >0):
            break
        r2_single = np.array(r2_single)
        f = np.where(r2_single==max(r2_single))[0][0]
        mse.append(mse_single)
        pred.append(pred_single)
        true.append(true_single)
        feature.append(np.array(feature_single[f]))
#         print(np.array(feature_single)[f])
    r2=np.array(r2)
    r2_mean=np.average(r2,axis=1,weights=weight)
    a = np.where(r2_mean==max(r2_mean))[0][0]
    pred = np.array(pred)[a]
    true = np.array(true)[a]
    mse = np.array(mse)[a]
    feature = np.array(feature)[a]
    print(feature.shape[0])
    alpha = ilist[a]
    r2 = r2[a]
    
    tmp = np.zeros([0,2])
    tmp_ = np.zeros([0,10])
    for i in range(10):
        p = np.expand_dims(pred[i],axis=1)
        t = np.expand_dims(true[i],axis=1)
        tmp1 = np.concatenate([p,t],axis=1)
        tmp = np.concatenate([tmp,tmp1],axis=0)
    df1 = pd.DataFrame(tmp,columns=['Predict','True'])
    df2 = pd.DataFrame({'r2':r2,'mse':mse})
    df3 = pd.DataFrame({'features':feature})

    pbar.close()
    plt.figure()
    plt.plot(ilist,r2_mean)
    plt.xlabel('$alpha$')
    plt.ylabel('$R^2$')
    print('max r2_score=',r2_mean[a],', corresponding alpha=',alpha,a)
    print('number of selected features:',feature.shape[0])
    return df1, df2, df3

def LassoLars(X, y, ll, ul, step, weight, state):
    kf = KFold(n_splits=10,shuffle=True,random_state=state)
    feature=[]
    pred=[]
    true=[]
    r2=[]
    mse=[]
    ilist=np.linspace(ll,ul,step)
    pbar = tnrange(step*10, desc='loop')
    for i in ilist:
        r2_single=[]
        mse_single=[]
        pred_single=[]
        true_single=[]
        feature_single=[]
        for train_index, test_index in kf.split(X):
            y_train, y_test = y[train_index], y[test_index]
            X_train_tmp, X_test_tmp = X[train_index], X[test_index]
            
            clf = LassoLars(alpha=i)
            clf.fit(X_train_tmp, np.ravel(y_train))
            feature_index = np.where(clf.coef_>0)[0]
            X_train = X_train_tmp[:,feature_index]
            X_test = X_test_tmp[:,feature_index]

            svr = svm.SVR(kernel='linear')
            svr.fit(X_train, np.ravel(y_train))
            y_test_pred = svr.predict(X_test)
            
            feature_single.append(feature_index)
            pred_single.append(y_test_pred)
            true_single.append(np.ravel(y_test))
            r2_single.append(r2_score(y_test, y_test_pred))
            mse_single.append(mean_squared_error(y_test, y_test_pred))
            pbar.update(1)
        r2.append(r2_single)
        r2_single = np.array(r2_single)
        f = np.where(r2_single==max(r2_single))[0][0]
        mse.append(mse_single)
        pred.append(pred_single)
        true.append(true_single)
        feature.append(np.array(feature_single[f]))
#         print(np.array(feature_single)[f])
    r2=np.array(r2)
    r2_mean=np.average(r2,axis=1,weights=weight)
    a = np.where(r2_mean==max(r2_mean))[0][0]
    pred = np.array(pred)[a]
    true = np.array(true)[a]
    mse = np.array(mse)[a]
    feature = np.array(feature)[a]
    print(feature.shape[0])
    alpha = ilist[a]
    r2 = r2[a]
    
    tmp = np.zeros([0,2])
    tmp_ = np.zeros([0,10])
    for i in range(10):
        p = np.expand_dims(pred[i],axis=1)
        t = np.expand_dims(true[i],axis=1)
        tmp1 = np.concatenate([p,t],axis=1)
        tmp = np.concatenate([tmp,tmp1],axis=0)
    df1 = pd.DataFrame(tmp,columns=['Predict','True'])
    df2 = pd.DataFrame({'r2':r2,'mse':mse})
    df3 = pd.DataFrame({'features':feature})

    pbar.close()
    plt.figure()
    plt.plot(ilist,r2_mean)
    plt.xlabel('$alpha$')
    plt.ylabel('$R^2$')
    print('max r2_score=',r2_mean[a],', corresponding alpha=',alpha,a)
    print('number of selected features:',feature.shape[0])
    return df1, df2, df3

def lassoLARSIC(data, label, cri, weight, state):
    kf = KFold(n_splits=10, shuffle=True, random_state=state)
    X = data
    y = label
    r2=[]
    mse=[]
    true=[]
    pred=[]
    feature=[]
    for train_index, test_index in kf.split(X):
        y_train, y_test = y[train_index], y[test_index]
        X_train_tmp, X_test_tmp = X[train_index], X[test_index]
        reg = LassoLarsIC(criterion=cri, normalize=False)
        reg.fit(X_train_tmp,y_train)
        feature_index = np.where(reg.coef_>0)[0]
        X_train = X_train_tmp[:,feature_index]
        X_test = X_test_tmp[:,feature_index]
        svr = svm.SVR(kernel='linear')
        svr.fit(X_train, np.ravel(y_train))
        y_test_pred = svr.predict(X_test)
        r2.append(r2_score(y_test, y_test_pred))
        mse.append(mean_squared_error(y_test, y_test_pred))
        feature.append(feature_index)
        pred.append(y_test_pred)
        true.append(np.ravel(y_test))
    feature=np.array(feature)
    pred = np.array(pred)
    true = np.array(true)
    r2=np.array(r2)
    mse = np.array(mse)
    print('mean r2_score=',np.average(r2,weights=weight))
    feature = feature[np.where(r2==max(r2))][0]
    print('number of selected features:',len(feature))
    return pred, true, r2, mse, feature

def elastic_net(data, label, ll1, ul1, step1, ll2, ul2, step2, weight, state):
    kf = KFold(n_splits=10,shuffle=True,random_state=state)
    X = data
    y = label
    r2=[]
    mse=[]
    pred=[]
    true=[]
    ilist=[]
    jlist=[]
    feature = []
    pbar = tnrange(step1*step2*10, desc='loop')
    g1=[]
    g2=[]
    for i in range(step1):
        for j in range(step2):
            g1.append(np.linspace(ll1,ul1,step1)[i])
            g2.append(np.linspace(ll2,ul2,step2)[j])
    g = np.vstack((g1,g2)).T
    for k in range(g.shape[0]):
        r2_single=[]
        mse_single=[]
        pred_single=[]
        true_single=[]
        feature_single=[]
        for train_index, test_index in kf.split(X):
            y_train, y_test = y[train_index], y[test_index]
            X_train_tmp, X_test_tmp = X[train_index], X[test_index]
            
            regr = ElasticNet(random_state=999,alpha=g[k,0],l1_ratio=g[k,1])
            regr.fit(X_train_tmp, np.ravel(y_train))
            feature_index = np.where(regr.coef_>0)[0]
            X_train = X_train_tmp[:,feature_index]
            X_test = X_test_tmp[:,feature_index]

            svr = svm.SVR(kernel='linear')
            svr.fit(X_train, np.ravel(y_train))
            y_test_pred = svr.predict(X_test)
            feature_single.append(feature_index)
            pred_single.append(y_test_pred)
            true_single.append(np.ravel(y_test))
            r2_single.append(r2_score(y_test, y_test_pred))
            mse_single.append(mean_squared_error(y_test, y_test_pred))
            pbar.update(1)
        r2.append(r2_single)
        mse.append(mse_single)
        pred.append(pred_single)
        true.append(true_single)
        feature.append(feature_single)
        ilist.append(g[k,0])
        jlist.append(g[k,1])
    r2=np.array(r2)
    r2_mean=np.average(r2,axis=1,weights=weight)
    pbar.close()
    plt.figure()
    plt.scatter(g[:,0],g[:,1],c=r2_mean,marker='.',cmap='Blues_r')
    plt.xlabel('$alpha$')
    plt.ylabel('$L1 ratio$')
    plt.axis([ll1-(ul1-ll1)*0.05,ul1+(ul1-ll1)*0.05,ll2-(ul2-ll2)*0.05,ul2+(ul2-ll2)*0.05])
    a = np.where(r2_mean==max(r2_mean))[0]
    pred = np.array(pred)[a[0]]
    true = np.array(true)[a[0]]
    r2 = r2[a[0]]
    mse = np.array(mse)[a[0]]
    feature = np.array(feature)[a[0]]
    r = jlist[a[0]]
    a = ilist[a[0]]
    print('max r2_score=',np.max(r2_mean),', corresponding alpha=',
          a,', corresponding ratio=',r)
    feature = feature[np.where(r2==max(r2))][0]
    print('number of selected features:',len(feature))
    return pred, true, r2, mse, feature, a, r

def elastic_net_ratio(data, label, ll, ul, step, al, weight, state):
    kf = KFold(n_splits=10,shuffle=True,random_state=state)
    X = data
    y = label
    r2=[]
    mse=[]
    pred=[]
    true=[]
    ilist=[]
    feature = []
    pbar = tnrange(step*10, desc='loop')
    for i in np.linspace(ll,ul,step):
        r2_single=[]
        mse_single=[]
        pred_single=[]
        true_single=[]
        feature_single=[]
        for train_index, test_index in kf.split(X):
            y_train, y_test = y[train_index], y[test_index]
            X_train_tmp, X_test_tmp = X[train_index], X[test_index]
            
            regr = ElasticNet(random_state=state,alpha=al,l1_ratio=i)
            regr.fit(X_train_tmp, np.ravel(y_train))
            feature_index = np.where(regr.coef_>0)[0]
            X_train = X_train_tmp[:,feature_index]
            X_test = X_test_tmp[:,feature_index]

            svr = svm.SVR(kernel='linear')
            svr.fit(X_train, np.ravel(y_train))
            y_test_pred = svr.predict(X_test)
            feature_single.append(feature_index)
            pred_single.append(y_test_pred)
            true_single.append(np.ravel(y_test))
            r2_single.append(r2_score(y_test, y_test_pred))
            mse_single.append(mean_squared_error(y_test, y_test_pred))
            pbar.update(1)
        r2.append(r2_single)
        mse.append(mse_single)
        pred.append(pred_single)
        true.append(true_single)
        feature.append(feature_single)
        ilist.append(i)
    r2=np.array(r2)
    r2_mean=np.average(r2,axis=1,weights=weight)
    pbar.close()
    plt.figure()
    plt.plot(np.linspace(ll,ul,step),r2_mean)
    plt.xlabel('$L1\; ratio$')
    plt.ylabel('$R^2$')
    a = np.where(r2_mean==max(r2_mean))[0]
    pred = np.array(pred)[a[0]]
    true = np.array(true)[a[0]]
    r2 = r2[a[0]]
    mse = np.array(mse)[a[0]]
    feature = np.array(feature)[a[0]]
    a = ilist[a[0]]
    print('max r2_score=',np.max(r2_mean),', corresponding L1 ratio=',a)
    feature = feature[np.where(r2==max(r2))][0]
    print('number of selected features:',len(feature))
    return pred, true, r2, mse, feature, a

def elastic_net_alpha(data, label, ll, ul, step, ratio, weight, state):
    kf = KFold(n_splits=10,shuffle=True,random_state=state)
    X = data
    y = label
    r2=[]
    mse=[]
    pred=[]
    true=[]
    ilist=[]
    feature = []
    pbar = tnrange(step*10, desc='loop')
    for i in np.linspace(ll,ul,step):
        r2_single=[]
        mse_single=[]
        pred_single=[]
        true_single=[]
        feature_single=[]
        for train_index, test_index in kf.split(X):
            y_train, y_test = y[train_index], y[test_index]
            X_train_tmp, X_test_tmp = X[train_index], X[test_index]
            
            regr = ElasticNet(random_state=state,alpha=i,l1_ratio=ratio)
            regr.fit(X_train_tmp, np.ravel(y_train))
            feature_index = np.where(regr.coef_>0)[0]
            X_train = X_train_tmp[:,feature_index]
            X_test = X_test_tmp[:,feature_index]

            svr = svm.SVR(kernel='linear')
            svr.fit(X_train, np.ravel(y_train))
            y_test_pred = svr.predict(X_test)
            feature_single.append(feature_index)
            pred_single.append(y_test_pred)
            true_single.append(np.ravel(y_test))
            r2_single.append(r2_score(y_test, y_test_pred))
            mse_single.append(mean_squared_error(y_test, y_test_pred))
            pbar.update(1)
        r2.append(r2_single)
        mse.append(mse_single)
        pred.append(pred_single)
        true.append(true_single)
        feature.append(feature_single)
        ilist.append(i)
    r2=np.array(r2)
    r2_mean=np.average(r2,axis=1,weights=weight)
    pbar.close()
    plt.figure()
    plt.plot(np.linspace(ll,ul,step),r2_mean)
    plt.xlabel('alpha')
    plt.ylabel('$R^2$')
    a = np.where(r2_mean==max(r2_mean))[0]
    pred = np.array(pred)[a[0]]
    true = np.array(true)[a[0]]
    r2 = r2[a[0]]
    mse = np.array(mse)[a[0]]
    feature = np.array(feature)[a[0]]
    a = ilist[a[0]]
    print('max r2_score=',np.max(r2_mean),', corresponding alpha=',a)
    feature = feature[np.where(r2==max(r2))][0]
    print('number of selected features:',len(feature))
    return pred, true, r2, mse, feature, a

def omp(data, label, ll, ul, step, weight, state):
    kf = KFold(n_splits=10,shuffle=True,random_state=state)
    X = data
    y = label
    r2=[]
    mse=[]
    pred=[]
    true=[]
    ilist=[]
    feature = []
    pbar = tnrange(step*10, desc='loop')
    for i in np.linspace(ll,ul,step).astype(int):
        r2_single=[]
        mse_single=[]
        pred_single=[]
        true_single=[]
        feature_single=[]
        for train_index, test_index in kf.split(X):
            y_train, y_test = y[train_index], y[test_index]
            X_train_tmp, X_test_tmp = X[train_index], X[test_index]
            
            clf = OrthogonalMatchingPursuit(n_nonzero_coefs=i,normalize=False)
            clf.fit(X_train_tmp, np.ravel(y_train))
            feature_index = np.where(clf.coef_>0)[0]
            X_train = X_train_tmp[:,feature_index]
            X_test = X_test_tmp[:,feature_index]

            svr = svm.SVR(kernel='linear')
            svr.fit(X_train, np.ravel(y_train))
            y_test_pred = svr.predict(X_test)
            feature_single.append(feature_index)
            pred_single.append(y_test_pred)
            true_single.append(np.ravel(y_test))
            r2_single.append(r2_score(y_test, y_test_pred))
            mse_single.append(mean_squared_error(y_test, y_test_pred))
            pbar.update(1)
        r2.append(r2_single)
        mse.append(mse_single)
        pred.append(pred_single)
        true.append(true_single)
        feature.append(feature_single)
        ilist.append(i)
    r2=np.array(r2)
    r2_mean=np.average(r2,axis=1,weights=weight)
    pbar.close()
    plt.figure()
    plt.plot(np.linspace(ll,ul,step),r2_mean)
    plt.xlabel('$non-zero coefficients$')
    plt.ylabel('$R^2$')
    a = np.where(r2_mean==max(r2_mean))[0]
    pred = np.array(pred)[a[0]]
    true = np.array(true)[a[0]]
    r2 = r2[a[0]]
    mse = np.array(mse)[a[0]]
    feature = np.array(feature)[a[0]]
    a = ilist[a[0]]
    print('max r2_score=',np.max(r2_mean),', number of non-zero coefs=',a)
    feature = feature[np.where(r2==max(r2))][0]
    print('number of selected features:',len(feature))
    return pred, true, r2, mse, feature

def reg_tree(data, label, lower_n_features, upper_n_features, weight, state):
    X = data
    y = label
    kf = KFold(n_splits=10,shuffle=True,random_state=state)
    pbar = tnrange((upper_n_features-lower_n_features)*10, desc='loop')
    r2=[]
    mse=[]
    mat=[]
    ilist=[]
    pred=[]
    true=[]
    for train_index, test_index in kf.split(X):
        y_train, y_test = y[train_index], y[test_index]
        X_train_tmp, X_test_tmp = X[train_index], X[test_index]
        r2_single=[]
        mse_single=[]
        i_single=[]
        pred_single=[]
        true_single=[]
        svr = svm.SVR(kernel='linear')
        reg = DecisionTreeRegressor(random_state=999)
        reg.fit(X_train_tmp,y_train)
        importance = np.argsort(reg.feature_importances_)
        mat.append(reg.feature_importances_)
        for i in range(lower_n_features,upper_n_features):
            X_train = X_train_tmp[:,importance[:i]]
            X_test = X_test_tmp[:,importance[:i]]
            svr = svm.SVR(kernel='linear')
            svr.fit(X_train, y_train)
            y_test_pred = svr.predict(X_test)
            pred_single.append(y_test_pred)
            true_single.append(np.ravel(y_test))
            r2_single.append(r2_score(y_test, y_test_pred))
            mse_single.append(mean_squared_error(y_test, y_test_pred))
            i_single.append(i)
            pbar.update(1)
        r2.append(r2_single)
        ilist.append(i_single)
        mse.append(mse_single)
        pred.append(pred_single)
        true.append(true_single)
    r2=np.array(r2)
    r2_mean = np.mean(r2,axis=0)
    a = np.where(r2_mean==max(r2_mean))[0]
    ilist = np.array(ilist)[:,a[0]][0]
    mat = np.array(mat)
    pred = np.array(pred)[:,a[0]]
    true = np.array(true)[:,a[0]]
    mse = np.array(mse)[:,a[0]]
    r2 = r2[:,a[0]]
    feature=np.argsort(np.mean(mat,axis=0))[:ilist]
    pbar.close()
    print('mean r2_score=',np.average(r2,weights=weight),
          ', number of features=',ilist)
    return pred, true, r2, mse, mat, feature

def rfe_(data, label, lower_n_features, upper_n_features, weight, state):
    X = data
    y = label
    kf = KFold(n_splits=10,shuffle=True,random_state=state)
    pbar = tnrange((upper_n_features-lower_n_features)*10, desc='loop')
    r2=[]
    mse=[]
    mat=[]
    ilist=[]
    pred=[]
    true=[]
    for train_index, test_index in kf.split(X):
        y_train, y_test = y[train_index], y[test_index]
        X_train_tmp, X_test_tmp = X[train_index], X[test_index]
        r2_single=[]
        mse_single=[]
        i_single=[]
        pred_single=[]
        true_single=[]
        svr = svm.SVR(kernel='linear')
        rfe = RFE(estimator=svr, n_features_to_select=1, step=1)
        rfe.fit(X_train_tmp, y_train)
        importance = np.argsort(rfe.ranking_)
        mat.append(rfe.ranking_)
        for i in range(lower_n_features,upper_n_features):
            X_train = X_train_tmp[:,importance[:i]]
            X_test = X_test_tmp[:,importance[:i]]
            svr = svm.SVR(kernel='linear')
            svr.fit(X_train, y_train)
            y_test_pred = svr.predict(X_test)
            pred_single.append(y_test_pred)
            true_single.append(np.ravel(y_test))
            r2_single.append(r2_score(y_test, y_test_pred))
            mse_single.append(mean_squared_error(y_test, y_test_pred))
            i_single.append(i)
            pbar.update(1)
        r2.append(r2_single)
        mse.append(mse_single)
        pred.append(pred_single)
        true.append(true_single)
    r2=np.array(r2)
    r2_mean = np.mean(r2,axis=0)
    a = np.where(r2_mean==max(r2_mean))[0]
    ilist = np.array(ilist)[:,a[0]][0]
    mat = np.array(mat)
    pred = np.array(pred)[:,a[0]]
    true = np.array(true)[:,a[0]]
    mse = np.array(mse)[:,a[0]]
    r2 = r2[:,a[0]]
    feature=np.argsort(np.mean(mat,axis=0))[:ilist]
    pbar.close()
    print('mean r2_score=',np.average(r2,weights=weight),
          ', number of features=',ilist)
    return pred, true, r2, mse, mat, feature

def mutual_info(data, label, ll, ul, step, weight, state):
    kf = KFold(n_splits=10,shuffle=True,random_state=state)
    X = data
    y = label
    r2=[]
    mse=[]
    pred=[]
    true=[]
    ilist=[]
    feature = []
    pbar = tnrange(step*10, desc='loop')
    for i in np.linspace(ll,ul,step).astype(int):
        r2_single=[]
        mse_single=[]
        pred_single=[]
        true_single=[]
        feature_single=[]
        for train_index, test_index in kf.split(X):
            y_train, y_test = y[train_index], y[test_index]
            X_train_tmp, X_test_tmp = X[train_index], X[test_index]
            
            mir = mutual_info_regression(X_train_tmp, y_train, discrete_features='auto', 
                                         n_neighbors=i, copy=True, random_state=999)
            feature_index = np.where(mir>0)[0]
            X_train = X_train_tmp[:,feature_index]
            X_test = X_test_tmp[:,feature_index]

            svr = svm.SVR(kernel='linear')
            svr.fit(X_train, np.ravel(y_train))
            y_test_pred = svr.predict(X_test)
            feature_single.append(feature_index)
            pred_single.append(y_test_pred)
            true_single.append(np.ravel(y_test))
            r2_single.append(r2_score(y_test, y_test_pred))
            mse_single.append(mean_squared_error(y_test, y_test_pred))
            pbar.update(1)
        r2.append(r2_single)
        mse.append(mse_single)
        pred.append(pred_single)
        true.append(true_single)
        feature.append(feature_single)
        ilist.append(i)
    r2=np.array(r2)
    r2_mean=np.average(r2,axis=1,weights=weight)
    pbar.close()
    plt.figure()
    plt.plot(np.linspace(ll,ul,step),r2_mean)
    plt.xlabel('$n-neighbors$')
    plt.ylabel('$R^2$')
    a = np.where(r2_mean==max(r2_mean))[0]
    pred = np.array(pred)[a[0]]
    true = np.array(true)[a[0]]
    r2 = r2[a[0]]
    mse = np.array(mse)[a[0]]
    feature = np.array(feature)[a[0]]
    a = ilist[a[0]]
    print('max r2_score=',np.max(r2_mean),', number of neighbors=',a)
    feature = feature[np.where(r2==max(r2))][0]
    print('number of selected features:',len(feature))
    return pred, true, r2, mse, feature