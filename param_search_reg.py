from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import pandas as pd
import pickle
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

seed_everything(1)
df = pd.read_csv('df3.csv')
df = df.drop(416, axis=0)
df = df.sample(frac=1)
df = df[df['experiment'] != 16]

df_train, df_test = train_test_split(df, stratify=df['experiment'], test_size=0.2)
X = df_train.loc[:, 'x1':'z100'].values
y_s = df_train.obstacle.map({'T': 35, 'M': 27.5, 'S': 20}).values
y_d = df_train.loc[:, 'd']
sc = StandardScaler()


def hparam_search():
    param_grid_lr = [
        {'alpha': [0.001, 0.01, 0.1, 1, 10],
         'solver': ['auto', 'sparse_cg', 'sag', 'saga']}]

    param_grid_svr = [
        {'C': [0.001, 0.01, 0.1, 1],
         'kernel': ['poly'],
         'degree': [2, 3, 4, 8, 10],
         'coef0': [0, 0.001, 0.01, 0.1]},
        {'C': [0.001, 0.01, 0.1, 1],
         'kernel': ['sigmoid'],
         'coef0': [0, 0.001, 0.01, 0.1]},
        {'C': [0.001, 0.01, 0.1, 1],
         'kernel': ['linear', 'rbf']}]

    for i in range(len(param_grid_svr)):
        param_grid_svr[i]['max_iter'] = [10_000]

    for i in range(len(param_grid_lr)):
        param_grid_lr[i] = {f'lr__{k}': v for k, v in param_grid_lr[i].items()}

    for i in range(len(param_grid_svr)):
        param_grid_svr[i] = {f'svr__{k}': v for k, v in param_grid_svr[i].items()}

    lr = Ridge()
    pipe_lr = Pipeline(steps=[('sc', sc), ('lr', lr)])

    svr = SVR()
    pipe_svr = Pipeline(steps=[('sc', sc), ('svr', svr)])


    gs = GridSearchCV(pipe_svr, param_grid_svr, scoring='neg_mean_squared_error', cv=10, verbose=4)
    gs.fit(X, y_d)
    results1 = {'CV_results_d': gs.cv_results_, 'best_score_d': gs.best_score_, 'best_params_d': gs.best_params_}
    gs.fit(X, y_s)
    results2 = {'CV_results_s': gs.cv_results_, 'best_score_s': gs.best_score_, 'best_params_s': gs.best_params_}

    with open('gs_svr_d.pkl', 'wb') as f:
        pickle.dump(results1, f)

    with open('gs_svr_s.pkl', 'wb') as f:
        pickle.dump(results2, f)

# hparam_search()


with open('gs_svr_d.pkl', 'rb') as f:
    gsd = pickle.load(f)

with open('gs_svr_s.pkl', 'rb') as f:
    gss = pickle.load(f)

#
# with open('gs_svr_d.pkl', 'rb') as f:
#     gsd = pickle.load(f)
#
# with open('gs_svr_s.pkl', 'rb') as f:
#     gss = pickle.load(f)
#
# # print(gsd['CV_results_d'])
# # print(gsd['best_score_d'])
# print(gsd['best_params_d'])
#
# # print(gss['CV_results_s'])
# # print(gss['best_score_s'])
# print(gss['best_params_s'])