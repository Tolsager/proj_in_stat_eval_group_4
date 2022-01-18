from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.model_selection import train_test_split
from utils import *
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier


def hparam_search(model_type, cv=10):
    seed_everything(1)
    df = pd.read_csv('df3.csv')
    df = df.drop(416, axis=0)
    df = df.sample(frac=1)
    df = df[df['experiment'] != 16]

    df_train, df_test = train_test_split(df, stratify=df['experiment'], test_size=0.2)
    X = df_train.loc[:, 'x1':'z100'].values
    y = df_train['experiment'].values

    sc = StandardScaler()

    param_grid_lr = [
        {'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1],
        'solver': ['saga', 'liblinear']},
        {'penalty': ['l2'],
         'C': [0.001, 0.01, 0.1, 1],
         'solver': ['newton-cg', 'lbfgs', 'sag']},
        {'penalty': ['none'],
         'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']},
        {'penalty': ['elasticnet'],
         'C': [0.001, 0.01, 0.1, 1],
         'l1_ratio': [0.001, 0.01, 0.1]}]

    param_grid_svc = [
        {'C': [0.001, 0.01, 0.1, 1],
         'kernel': ['poly'],
         'degree': [2, 3, 4, 8, 10],
         'coef0': [0, 0.001, 0.01, 0.1]},
        {'C': [0.001, 0.01, 0.1, 1],
         'kernel': ['sigmoid'],
         'coef0': [0, 0.001, 0.01, 0.1]},
        {'C': [0.001, 0.01, 0.1, 1],
         'kernel': ['linear', 'rbf']}]

    param_grid_gb = [
        {'n_estimators': [100, 150, 250]
         , 'learning_rate':[0.001, 0.01, 0.1]
         ,'learning_rate':[0.001, 0.005, 0.01, 0.1]
         ,'subsample': [0.5, 0.7, 0.8, 1]
         ,'max_depth': [8]
         ,'n_iter_no_change':[5, 20, None]
         }
    ]

    for i in range(len(param_grid_lr)):
        param_grid_lr[i]['max_iter'] = [10_000]

    for i in range(len(param_grid_svc)):
        param_grid_svc[i]['max_iter'] = [10_000]

    for i in range(len(param_grid_lr)):
        param_grid_lr[i] = {f'lr__{k}':v for k,v in param_grid_lr[i].items()}

    for i in range(len(param_grid_svc)):
        param_grid_svc[i] = {f'svc__{k}': v for k, v in param_grid_svc[i].items()}

    for i in range(len(param_grid_gb)):
        param_grid_gb[i] = {f'gb__{k}': v for k, v in param_grid_gb[i].items()}

    lr = LogisticRegression()
    pipe_lr = Pipeline(steps=[('sc', sc), ('lr', lr)])

    svc = SVC()
    pipe_svc = Pipeline(steps=[('sc', sc), ('svc', svc)])

    gb = GradientBoostingClassifier(verbose=2)
    pipe_gb = Pipeline(steps=[('sc', sc), ('gb', gb)])
    
    if model_type == 'lr':
        gs = GridSearchCV(pipe_lr, param_grid_lr, scoring='accuracy', cv=cv, verbose=4)
    elif model_type == 'svc':
        gs = GridSearchCV(pipe_svc, param_grid_svc, scoring='accuracy', cv=cv, verbose=4)
    elif model_type == 'gb':
        gs = GridSearchCV(pipe_gb, param_grid_gb, scoring='accuracy', cv=cv, verbose=4, return_train_score=True)
    
    gs.fit(X, y)
    results = {'cv_results_': gs.cv_results_, 'best_score_': gs.best_score_, 'best_params_': gs.best_params_}
    with open(f'{model_type}_gs.pkl', 'w+b') as f:
        pickle.dump(results, f)


hparam_search('lr')
hparam_search('svc')
hparam_search('gb')
