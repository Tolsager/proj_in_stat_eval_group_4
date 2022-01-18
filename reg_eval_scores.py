from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from utils import *
from sklearn.pipeline import Pipeline
from classifier_test import regression_compare
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

obstacle_sizes = [20, 27.5, 35]
obstacle_d = [15, 22.5, 30, 37.5, 45]


def get_eval_scores(model_type, cv_acc=True, test_acc=True, cv=10, save_correct_predictions=False, coords=('x', 'y', 'z')):
    seed_everything(1)
    df = pd.read_csv('df3.csv')
    df = df.drop(416, axis=0)
    df = df.sample(frac=1)
    df = df[df['experiment'] != 16]
    df['obstacle_encode'] = df.obstacle.map({'T': 35, 'M': 27.5, 'S': 20})

    df_train, df_test = train_test_split(df, stratify=df['experiment'], test_size=0.2)
    X_train = df_train.loc[:, 'x1':'z100'].values
    y_train = df_train.loc[:, ['d', 'obstacle_encode']].values
    X_test = df_test.loc[:, 'x1':'z100'].values
    y_test = df_test.loc[:, ['d', 'obstacle_encode']].values
    sc = StandardScaler()

    if model_type == 'ridge':
        gs = pickle.load(open('gs_ridge_d.pkl', 'rb'))
        best_params = gs['best_params_d']
        best_params = {f'{k[4:]}': v for k, v in best_params.items()}
        model1 = Ridge(**best_params)
        gs = pickle.load(open('gs_ridge_s.pkl', 'rb'))
        best_params = gs['best_params_s']
        best_params = {f'{k[4:]}': v for k, v in best_params.items()}
        model2 = Ridge(**best_params)
    elif model_type == 'svr':
        gs = pickle.load(open('gs_svr_d.pkl', 'rb'))
        best_params = gs['best_params_d']
        best_params = {f'{k[5:]}': v for k, v in best_params.items()}
        model1 = SVR(**best_params)
        gs = pickle.load(open('gs_svr_s.pkl', 'rb'))
        best_params = gs['best_params_s']
        best_params = {f'{k[5:]}': v for k, v in best_params.items()}
        model2 = SVR(**best_params)
    elif model_type == 'gbr':
        model1 = GradientBoostingRegressor()
        model2 = GradientBoostingRegressor()
    elif model_type == 'regc':
        gs = pickle.load(open('gs_ridge_d.pkl', 'rb'))
        best_params = gs['best_params_d']
        best_params = {f'{k[4:]}': v for k, v in best_params.items()}
        model1 = Ridge(**best_params)
        model2 = GradientBoostingRegressor()


    squared_errors_d = []
    squared_errors_s = []
    X = [X_train, X_test]
    for i in range(2):
        X_temp = []
        if 'x' in coords:
            X_temp.append(X[i][:, :100])
        if 'y' in coords:
            X_temp.append(X[i][:, 100:200])
        if 'z' in coords:
            X_temp.append(X[i][:, 200:])
        X_temp = np.concatenate(X_temp, axis=1)
        X[i] = X_temp.copy()
    X_train = X[0]
    X_test = X[1]

    # Regression
    if model_type != 'regc':
        pipe1 = Pipeline(steps=[('sc', sc), ('model', model1)])
        pipe2 = Pipeline(steps=[('sc', sc), ('model', model2)])
    if cv_acc:
        ## Regression
        if model_type != 'regc':
            _cv_acc1 = cross_val_score(pipe1, X_train, y_train[:, 0], cv=cv, scoring='neg_mean_squared_error')
            _cv_acc1 = np.mean(_cv_acc1)
            _cv_acc2 = cross_val_score(pipe2, X_train, y_train[:, 1], cv=cv, scoring='neg_mean_squared_error')
            _cv_acc2 = np.mean(_cv_acc2)

        ## Classification
        else:
            k = StratifiedKFold(cv)
            cv_acc_d = []
            cv_acc_s = []
            _cv_acc = []
            for train_idx, val_idx in k.split(X_train, df_train.loc[:, 'experiment'].values):
                X_train_, X_val = X_train[train_idx, :], X_train[val_idx, :]
                y_train_, y_val = y_train[train_idx, :], y_train[val_idx, :]

                X_train_ = sc.fit_transform(X_train_)
                X_val = sc.transform(X_val)

                model1.fit(X_train_, y_train_[:, 0])
                model2.fit(X_train_, y_train_[:, 1])

                d_pred = model1.predict(X_val)
                s_pred = model2.predict(X_val)

                for i in range(len(d_pred)):
                    d_pred[i] = min(obstacle_d, key=lambda x: abs(x - d_pred[i]))
                    s_pred[i] = min(obstacle_sizes, key=lambda x: abs(x - s_pred[i]))

                d_performance = d_pred == y_val[:, 0]
                s_performance = s_pred == y_val[:, 1]

                cv_acc_d.append(np.sum(d_performance) / len(y_val))
                cv_acc_s.append(np.sum(s_performance) / len(y_val))
                _cv_acc.append(np.sum(d_performance & s_performance) / len(y_val))

            cv_acc_d = np.mean(cv_acc_d)
            cv_acc_s = np.mean(cv_acc_s)
            _cv_acc = np.mean(_cv_acc)
    if test_acc:
        ## MSE (Regression)
        if model_type != 'regc':
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            model1.fit(X_train, y_train[:, 0])
            model2.fit(X_train, y_train[:, 1])

            d_pred = model1.predict(X_test)
            s_pred = model2.predict(X_test)

            squared_errors_d.append(np.square(np.subtract(y_test[:, 0], d_pred)))
            squared_errors_s.append(np.square(np.subtract(y_test[:, 1], s_pred)))

            d_mse = (-1)*mean_squared_error(y_test[:, 0], d_pred)
            s_mse = (-1)*mean_squared_error(y_test[:, 1], s_pred)

        ## Accuracy (classification)
        else:
            test_acc_d = []
            test_acc_s = []
            _test_acc = []

            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            model1.fit(X_train, y_train[:, 0])
            model2.fit(X_train, y_train[:, 1])

            d_pred = model1.predict(X_test)
            s_pred = model2.predict(X_test)

            for i in range(len(d_pred)):
                d_pred[i] = min(obstacle_d, key=lambda x: abs(x - d_pred[i]))
                s_pred[i] = min(obstacle_sizes, key=lambda x: abs(x - s_pred[i]))

            d_performance = d_pred == y_test[:, 0]
            s_performance = s_pred == y_test[:, 1]

            test_acc_d.append(np.sum(d_performance) / len(y_test))
            test_acc_s.append(np.sum(s_performance) / len(y_test))
            _test_acc.append(np.sum((d_performance & s_performance)) / len(y_test))


    if save_correct_predictions and model_type == 'regc':
        correct_predictions = d_performance & s_performance
        with open(f'{model_type}_predictions.pkl', 'w+b') as f:
                pickle.dump(correct_predictions, f)

    if cv_acc:
        if model_type != 'regc':
            print(f"{model_type} cross val mse on distance: {_cv_acc1}")
            print(f"{model_type} cross val mse on size: {_cv_acc2}")
        else:
            print(f"{model_type} mean cross validation distance accuracy: {cv_acc_d}")
            print(f"{model_type} mean cross validation size accuracy: {cv_acc_s}")
            print(f"{model_type} mean cross validation accuracy: {_cv_acc}")
    if test_acc:
        if model_type != 'regc':
            print(f"{model_type} test mse on distance: {d_mse}")
            print(f"{model_type} test mse on size: {s_mse}")
        else:
            print(f"{model_type} test accuracy on distance: {test_acc_d[0]}")
            print(f"{model_type} test accuracy on size: {test_acc_s[0]}")
            print(f"{model_type} test accuracy: {_test_acc[0]}")
    print()
    if model_type != 'regc':
        return np.array(squared_errors_d).reshape(-1), np.array(squared_errors_s).reshape(-1)
    else:
        return test_acc_d, test_acc_s


ridge_d, ridge_s = get_eval_scores("ridge", save_correct_predictions=False, cv_acc=False, coords=('x', 'y', 'z'))
svr_d, svr_s = get_eval_scores("svr", save_correct_predictions=False, cv_acc=False, coords=('x', 'y', 'z'))
gbr_d, gbr_s = get_eval_scores("gbr", save_correct_predictions=False, cv_acc=False, coords=('x', 'y', 'z'))

ridge_svr_d = regression_compare(ridge_d, svr_d)
ridge_gbr_d = regression_compare(ridge_d, gbr_d)
svr_gbr_d = regression_compare(svr_d, gbr_d)

ridge_svr_s = regression_compare(ridge_s, svr_s)
ridge_gbr_s = regression_compare(ridge_s, gbr_s)
svr_gbr_s = regression_compare(svr_s, gbr_s)


def pair_wise_test2(p_vals,  names):
    my_cm = np.array([
        [105/256, 165/256, 131/256, 0.5],
        [125 / 256, 180 / 256, 120 / 256, 1],
        [224 / 256, 202 / 256, 151 / 256, 1]

    ])
    my_cm = ListedColormap(my_cm)
    fig, ax = plt.subplots()
    m = len(names)
    p_matrix = np.zeros((m, m))

    counter = 0
    for i in range(len(p_vals)):
        k = i + 1
        while k < len(p_vals):
            print(f"{names[i]} & {names[k]}")
            p = p_vals[counter]
            p_matrix[i, k] = p
            counter += 1
            ax.text(i, k, np.round(p, 3), va='center', ha='center')
            ax.text(k, i, np.round(p, 3), va='center', ha='center')
            if p > 0.05:
                p_matrix[i, k] = 1
            else:
                p_matrix[i, k] = 0.5
            k += 1
    p_matrix += p_matrix.T
    ax.matshow(p_matrix, cmap=my_cm)
    plt.xticks(range(len(names)), labels=names)
    plt.yticks(range(len(names)), labels=names)
    plt.tight_layout()
    plt.show()
    
pair_wise_test2([ridge_svr_d, ridge_gbr_d, svr_gbr_d], ['Ridge', 'SVR', 'GBR'])