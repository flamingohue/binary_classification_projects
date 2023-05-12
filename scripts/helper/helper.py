import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score,roc_curve, auc
from scipy.optimize import minimize, Bounds, LinearConstraint
from sklearn.model_selection import train_test_split,  StratifiedKFold
from sklearn.ensemble import  RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import collections


def eda_corr(df,feature,title):
    f, ax = plt.subplots(figsize = (8,6))
    plt.title(title, y = 1, size = 12)
    sns.heatmap(df[feature].corr(), square = True, vmax = 0.8)

def eda_corr_diff(df,feature,target,title):
    f, ax = plt.subplots(figsize = (8,6))
    plt.title(title, y = 1, size = 12)
    sns.heatmap(df[df[target] == 0][feature].corr() - df[df[target] == 1][feature].corr(), square = True, vmax = 0.8)
    print(df[df[target] == 0][feature].corr() - df[df[target] == 1][feature].corr())

def fea_str_to_num(df):
    dic = {
        'City' : {
            'Toronto' : 1,
            'Montreal' : 2,
            'Ottawa' : 3
        },
        'Gender' : {
            'Female': 1,
            'Male' : 2
        },
        'CurrencyCode': {
        'CAD' : 1,
        'USD': 2
        },
        'PrefLanguage':{
            'English': 1,
            'French' : 2
        },
        'PrefContact': {
            'Email' : 1,
            'Mobile' : 2,
            'Home_Phone' : 3
        }

    }

    df['City'] = df['City'].map(dic['City'])
    df['Gender'] = df['Gender'].map(dic['Gender'])
    df['CurrencyCode'] = df['CurrencyCode'].map(dic['CurrencyCode'])
    df['PrefLanguage'] = df['PrefLanguage'].map(dic['PrefLanguage'])
    df['PrefContact'] = df['PrefContact'].map(dic['PrefContact'])
    return df

def rescale(raw_lst):
    minMax_scaler = MinMaxScaler()
    raw_lst = np.array(raw_lst).reshape(-1, 1)
    res_lst = minMax_scaler.fit_transform(raw_lst).reshape(1,-1)[0]
    return res_lst

def ks_test_cat(sample1, sample2,input_var):
    sample1_dic = collections.Counter(sample1)
    sample2_dic = collections.Counter(sample2)
    all_cat = sorted(set(sample1_dic + sample2_dic))
    x1_cdf_list = []
    x2_cdf_list = []
    distance = []
    j = 0
    x1_cdf = 0
    x2_cdf = 0
    for i in all_cat:
        x1_cdf += sample1_dic[i]/len(sample1)
        x2_cdf += sample2_dic[i]/len(sample2)
        x1_cdf_list.append(x1_cdf)
        x2_cdf_list.append(x2_cdf)
        distance.append(abs(x1_cdf_list[j] - x2_cdf_list[j]))
        j += 1
    
    d_n = np.max(distance)
    print('The Stats of {} is {}'.format(input_var, d_n))
    return d_n


def feature_eda(df,feature,target):
    #df = fea_str_to_num(df)

    stats = {'feature' : [], 'missing_counts' : []}
    for i in feature:
        print('\n\n',i)
        print(df[i].value_counts(dropna=False))
        stats['feature'].append(i)
        stats['missing_counts'] = df[df[i].isnull()].shape[0]
    stats = pd.DataFrame(stats)
    stats['missing_perc'] = stats['missing_counts']/df.shape[0]
    print('\n\n\nthe missing check for each variables\n',stats)
    
    color = ['red', 'tan', 'lime','grey', 'yellow','blue']
    for i in feature:
        plt.hist([df[df[target]==0][i],df[df[target]==1][i]],label = ['Exit', 'Active'])
        plt.legend(loc ='upper right')
        plt.title(i+' Distribution')
        plt.show()




def check_cluster(df,var_lst,target,angle):
    # example of a normalization

    scaler = MinMaxScaler()
    ax = plt.axes(projection='3d')
    # Data for a three-dimensional line
    zline = scaler.fit_transform(np.array(df[df[target] == 0][var_lst[0]]).reshape(-1, 1))
    print(zline.shape)
    xline = scaler.fit_transform(np.array(df[df[target] == 0][var_lst[1]]).reshape(-1, 1))
    yline = scaler.fit_transform(np.array(df[df[target] == 0][var_lst[2]]).reshape(-1, 1))
    ax.scatter3D(xline, yline, zline, 'gray')

    # Data for a three-dimensional line
    zline = scaler.fit_transform(np.array(df[df[target] == 1][var_lst[0]]).reshape(-1, 1))
    print(zline.shape)
    xline = scaler.fit_transform(np.array(df[df[target] == 1][var_lst[1]]).reshape(-1, 1))
    yline = scaler.fit_transform(np.array(df[df[target] == 1][var_lst[2]]).reshape(-1, 1))
    ax.scatter3D(xline, yline, zline, 'blue')
    ax.view_init(30, angle)
    plt.show()

def model_data_split(df,feature, target, norm_lst = None):
    X_train = df[feature]
    y_train = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,test_size = 0.3, random_state = 0)
    X_train, X_test, y_train, y_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True),y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def timer(start_time = None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(),3600)
        tmin, tsec = divmod(temp_sec, 60)
        print("\nTime Taken: {} hour {} minites and {} seconds" .format(thour, tmin, round(tsec,2)))

def PCA_cluster(df, feature_lst,target,angle):
    pca = PCA(n_components=3)
    pca.fit(df[feature_lst])
    pca_trans = pca.transform(df[feature_lst])   
    pca_trans = pca_trans.T
    df['feature_1'] = pca_trans[0]
    df['feature_2'] = pca_trans[1]
    df['feature_3'] = pca_trans[1]
    check_cluster(df,['feature_1','feature_2','feature_3'],target,angle)



def feature_selection(X_train, Y_train, X_test, Y_test, algo):
    algos = ['LR','RF','XGB','SVM']
    if algo not in algos:
        raise Exception("Alogorithm should be 'LR','RF','XGB','SVM'.")
    
    scaler = MinMaxScaler()
    X_train_transform = scaler.fit_transform(X_train)
    X_train_transform = X_train_transform.reshape(X_train.shape)
    X_train_transform = pd.DataFrame(X_train_transform,columns = X_train.columns)

    X_test_transform = scaler.fit_transform(X_test)
    X_test_transform = X_test_transform.reshape(X_test.shape)
    X_test_transform = pd.DataFrame(X_test_transform,columns = X_test.columns)

    start_time = timer(None)
    if algo == 'LR':

        model = LogisticRegression(C=5, tol = 0.0001,penalty = 'l1',solver = 'saga').fit(X_train_transform, Y_train)
        probs = model.predict(X_test_transform).reshape(-1,1)
        fpr, tpr, threshold = metrics.roc_curve(Y_test,probs)
        final_auc = metrics.auc(fpr, tpr)
        probs_train = model.predict(X_train_transform).reshape(-1,1)
        fpr_train, tpr_train, threshold_train = metrics.roc_curve(Y_train,probs_train)
        final_train_auc = metrics.auc(fpr_train, tpr_train)
        coef_df = pd.DataFrame(data = model.coef_,columns = X_train.columns)
        print('The coefficients for the logistic Regression is\n', coef_df)

    
    elif algo == 'RF': 
        model = RandomForestClassifier(random_state = 42)
    elif algo == 'XGB': 
        model = XGBClassifier(objective = 'binary:logistic', eval_metrics = 'logloss',colsample_bytree = 0.8, alpha = 10, seed = 3)
    elif algo == 'SVM': 
        model = SVC(probability = True, kernel = 'linear',max_iter = 2)

    model.fit(X_train_transform,Y_train)
    scaler = MinMaxScaler()
    
    
    if algo in ['RF','XGB']:
        print('\nFeature Importance')
        print(pd.DataFrame(model.feature_importances_,index = list(X_train.columns), columns = ['Importances']).sort_values('Importances',ascending = False))
        imp = model.feature_importances_
        names = list(X_train.columns)
        imp,names = zip(*sorted(zip(imp,names)))
        plt.barh(range(len(names)), imp, align='center')
        plt.yticks(range(len(names)), names)
        plt.show()
    elif algo == 'SVM':
        imp = model.coef_.flatten()
        imp = [abs(i) for i in imp]
        names = list(X_train.columns)
        imp,names = zip(*sorted(zip(imp,names)))
        plt.barh(range(len(names)), imp, align='center')
        plt.yticks(range(len(names)), names)
        plt.show()
    elif algo == 'LR':
        imp = model.coef_.flatten()
        imp = [abs(i) for i in imp]
        names = list(X_train.columns)
        imp,names = zip(*sorted(zip(imp,names)))
        plt.barh(range(len(names)), imp, align='center')
        plt.yticks(range(len(names)), names)
        plt.show()



    preds = model.predict_proba(X_test_transform)
    probs = preds[:, 1]
    fpr,tpr,threshold = metrics.roc_curve(Y_test,probs)
    final_auc = metrics.auc(fpr,tpr)
    coef_df = pd.DataFrame()

    timer(start_time)
    return model,scaler, probs, final_auc

def Metrics_display_fs(method,probs, y_test,thsld = 0.45):

    preds = np.where(probs>thsld, 1,0)
    scores = pd.DataFrame(data= [accuracy_score(y_test, preds),recall_score(y_test, preds), precision_score(y_test,preds),roc_auc_score(y_test, preds)])

    scores = scores.T
    scores.columns = ['accuracy','recall','precision','roc_auc_score']
    print(scores)


def Metrics_display(method,probs, y_test,thsld = 0.45):
    if method == 'LR':
        probs = probs.reshape(-1,1)
        probs = probs.astype(int)
        scores = pd.DataFrame(data= [accuracy_score(y_test, probs),recall_score(y_test, probs), precision_score(y_test,probs),roc_auc_score(y_test, probs)])    
    else: 
        preds = np.where(probs>thsld, 1,0)
        scores = pd.DataFrame(data= [accuracy_score(y_test, preds),recall_score(y_test, preds), precision_score(y_test,preds),roc_auc_score(y_test, preds)])
    
    scores = scores.T
    scores.columns = ['accuracy','recall','precision','roc_auc_score']
    print(scores)

def Cross_Validation(X_train, Y_train, X_test, Y_test,alpha_lst, col_names, algo, params = None, folds = 5, param_comb = 50):
    algos = ['LR','RF','XGB','SVM']
    if algo not in algos:
        raise Exception("Alogorithm should be 'LR','RF','XGB','SVM'.")
    
    scaler = MinMaxScaler()
    X_train_transform = scaler.fit_transform(X_train)
    X_train_transform = X_train_transform.reshape(X_train.shape)
    X_train_transform = pd.DataFrame(X_train_transform,columns = X_train.columns)

    X_test_transform = scaler.fit_transform(X_test)
    X_test_transform = X_test_transform.reshape(X_test.shape)
    X_test_transform = pd.DataFrame(X_test_transform,columns = X_test.columns)

    skf = StratifiedKFold(n_splits = folds, shuffle = True, random_state = 1001)
    start_time = timer(None)
    if params == None:
        result = dict(zip(alpha_lst, np.repeat(None,len(alpha_lst))))

        for train_index, test_index in skf.split(X_train_transform, Y_train):
            X_subtrain, X_valid = X_train.iloc[train_index, :], X_train.iloc[test_index, : ]
            y_subtrain, y_valid = Y_train.iloc[train_index], Y_train.iloc[test_index]

            for alpha in alpha_lst:
                clf = LogisticRegression(C=alpha, tol = 0.0001).fit(X_subtrain, y_subtrain)
                preds = clf.predict(X_valid).reshape(-1,1)
                fpr, tpr, threshold = metrics.roc_curve(y_valid,preds)
                auc_test = metrics.auc(fpr, tpr)

                if result[alpha] != None:
                    result[alpha].append(auc_test)
                else:
                    result[alpha] = [auc_test]
        result = pd.DataFrame.from_dict(result)
        
        optimal_alpha = result.mean().idxmax()
        print('The optimal penalty term is ', optimal_alpha)
        random_search = LogisticRegression(C=optimal_alpha, tol = 0.0001).fit(X_train_transform, Y_train)
        probs = random_search.predict(X_test_transform).reshape(-1,1)
        fpr, tpr, threshold = metrics.roc_curve(Y_test,probs)
        final_auc = metrics.auc(fpr, tpr)
        probs_train = random_search.predict(X_train_transform).reshape(-1,1)
        fpr_train, tpr_train, threshold_train = metrics.roc_curve(Y_train,probs_train)
        final_train_auc = metrics.auc(fpr_train, tpr_train)
        coef_df = pd.DataFrame(data = random_search.coef_,columns = X_train.columns)
        print('The coefficients for the logistic Regression is\n', coef_df)

    else:
        if algo == 'RF': 
            model = RandomForestClassifier(random_state = 42)
        elif algo == 'XGB': 
            model = XGBClassifier(objective = 'binary:logistic', eval_metrics = 'logloss',colsample_bytree = 0.8, alpha = 10, seed = 3)
        elif algo == 'SVM': 
            model = SVC(probability = True)
        
        random_search = RandomizedSearchCV(model, param_distributions = params, n_iter = param_comb, scoring ='roc_auc',n_jobs = 4, cv = skf.split(X_train, Y_train), verbose = 3, random_state=1001)
        random_search.fit(X_train_transform,Y_train)

        

        if algo in ['LR','RF','XGB']:
            print('\nBest Estimator:')
            print(random_search.best_estimator_)
            print('\nFeature Importance')
            print(pd.DataFrame(random_search.best_estimator_.feature_importances_,index = list(X_train.columns), columns = ['Importances']).sort_values('Importances',ascending = False))
            print('\nBest Parameters:')
            print(random_search.best_params_)
        else:
            print('\nBest Estimator:')
            print(random_search.best_estimator_)


        preds = random_search.predict_proba(X_test_transform)
        probs = preds[:, 1]

        fpr,tpr,threshold = metrics.roc_curve(Y_test,probs)
        final_auc = metrics.auc(fpr,tpr)
        final_train_auc = random_search.cv_results_['mean_test_score'].max()
        coef_df = pd.DataFrame()

    if algo == 'LR':
        coef_df['lambda'] = [optimal_alpha]
    coef_df['Test_AUC'] = [final_auc]
    coef_df['Train_AUC'] = [final_train_auc]
    timer(start_time)
    return random_search,scaler, probs, final_auc, coef_df

def Metrics_display(method,probs, y_test,thsld = 0.5):
    if method == 'LR':
        probs = probs.reshape(-1,1)
        probs = probs.astype(int)
        scores = pd.DataFrame(data= [accuracy_score(y_test, probs),recall_score(y_test, probs), precision_score(y_test,probs),roc_auc_score(y_test, probs)])    
    else: 
        preds = np.where(probs>thsld, 1,0)
        scores = pd.DataFrame(data= [accuracy_score(y_test, preds),recall_score(y_test, preds), precision_score(y_test,preds),roc_auc_score(y_test, preds)])
    
    scores = scores.T
    scores.columns = ['accuracy','recall','precision','roc_auc_score']
    print(scores)


def plot_metrics(probs, method, y_true):

    prob = pd.DataFrame(probs, columns = [method])

    plt.figure(figsize = (8,6))
    plt.plot([0,1], [0,1], 'r--')

    fpr1, tpr1, thresholds1 = roc_curve(y_true, prob)
    roc_auc1 = auc(fpr1, tpr1)

    label = method + ' AUC = ' + '{0:2f}'.format(roc_auc1)
    plt.plot(fpr1,tpr1, c='g', label = label, linewidth = 2)
    plt.xlabel('False Positive Rate', fontsize = 10)
    plt.ylabel('True Positive Rate', fontsize = 10)
    plt.title(method +  '  AUC', fontsize = 10)
    plt.legend(loc = 'lower right',fontsize = 10)