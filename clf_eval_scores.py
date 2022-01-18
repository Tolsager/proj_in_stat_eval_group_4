from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from utils import *
from sklearn.pipeline import Pipeline
from LSTM import Model, lstm_cross_validation, DEVICE, DS
from torch.utils.data import DataLoader


def get_eval_scores(model_type, cv_acc=True, test_acc=True, cv=10, save_correct_predictions=False, coords=('x', 'y', 'z')):
    seed_everything(1)
    df = pd.read_csv('df3.csv')
    df = df.drop(416, axis=0)
    df = df.sample(frac=1)
    df = df[df['experiment'] != 16]

    df_train, df_test = train_test_split(df, stratify=df['experiment'], test_size=0.2)
    X_train = df_train.loc[:, 'x1':'z100'].values
    y_train = df_train['experiment'].values
    X_test = df_test.loc[:, 'x1':'z100'].values
    y_test = df_test['experiment'].values
    sc = StandardScaler()

    if model_type == 'lr':
        gs = pickle.load(open('lr_gs.pkl', 'rb'))
        best_params = gs['best_params_']
        best_params = {f'{k[4:]}': v for k, v in best_params.items()}
        model = LogisticRegression(**best_params)
    elif model_type == 'svc':
        gs = pickle.load(open('svc_gs.pkl', 'rb'))
        best_params = gs['best_params_']
        best_params = {f'{k[5:]}': v for k, v in best_params.items()}
        model = SVC(**best_params)
    elif model_type == 'gb':
        model = GradientBoostingClassifier(learning_rate=0.001, max_depth=8, n_estimators=150, subsample=0.8)

    if model_type != 'lstm':
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

        pipe = Pipeline(steps=[('sc', sc), ('model', model)])
        if cv_acc:
            _cv_acc = cross_val_score(pipe, X_train, y_train, cv=cv)
            _cv_acc = np.mean(_cv_acc)
        if test_acc:
            pipe.fit(X_train, y_train)

            _test_acc = pipe.score(X_test, y_test)

    elif model_type == 'lstm':
        y_train = y_train - 1
        if cv_acc:
            performance_dict = lstm_cross_validation(X_train, y_train)
            _cv_acc = []
            for k in performance_dict.keys():
                _cv_acc.append(performance_dict[k])
            _cv_acc = np.mean(_cv_acc)
        if test_acc:
            y_test = y_test - 1
            model = Model(coords)
            model.to(DEVICE)
            model.load_state_dict(torch.load('best_lstm.pt'))

            X_test = DS(X_test, y_test)
            X_test = DataLoader(X_test, batch_size=150)

            correct_predictions = []
            for batch in X_test:
                X_ = batch['X']
                y_ = batch['y']
                X_ = move_to(X_, DEVICE)
                y_ = move_to(y_, DEVICE)

                pred = model(X_)
                correct_pred = pred.argmax(dim=1) == y_
                correct_predictions.append(correct_pred.cpu().numpy())

            _test_acc = np.sum(correct_predictions) / len(y_test)

    if save_correct_predictions:
        if model_type != 'lstm':
            predictions = pipe.predict(X_test)
            correct_predictions = predictions == y_test

        with open(f'{model_type}_predictions.pkl', 'w+b') as f:
                pickle.dump(correct_predictions, f)

    if cv_acc:
        print(f"{model_type} mean cross validation accuracy: {_cv_acc}")
    if test_acc:
        print(f"{model_type} test accuracy: {_test_acc}")
    print()


get_eval_scores('lr', save_correct_predictions=True)
get_eval_scores('svc', save_correct_predictions=True)
get_eval_scores('gb')
get_eval_scores('lstm', save_correct_predictions=True)
