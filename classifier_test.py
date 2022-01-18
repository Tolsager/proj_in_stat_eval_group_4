from scipy.stats import binom, beta
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import t


def mcnemar(preds1, preds2, alpha=0.05):
    n = len(preds1)
    n12 = np.sum(preds1 & ~preds2)
    n21 = np.sum(~preds1 & preds2)

    dif = (n12 - n21) / n

    # confidence interval
    Q = (n**2 * (n+1) * (dif + 1) * (1 - dif))/(n * (n12 + n21) - (n12 - n21)**2)
    f = (dif + 1)/2 * (Q - 1)
    g = (1 - dif)/2 * (Q - 1)
    conf_l = 2*beta.ppf(alpha/2, f, g) - 1
    conf_u = 2*beta.ppf(1 - alpha/2, f, g) - 1

    p_val = 2*binom.cdf(min(n12, n21), n12+n21, 1/2)

    print(f"Significance level: {alpha}")
    print(f"Accuracy cfl1: {sum(preds1) / n}")
    print(f"Accuracy cfl2: {sum(preds2) / n}")
    print(f"Estimated difference in accuracy: {dif}")
    print(f"Confidence interval: [{conf_l}, {conf_u}]")
    print(f"p-value: {p_val}")
    print()
    if p_val > alpha:
        print("We can not reject that the two classifiers have equal accuracy")
    else:
        print(f"We reject that the two classifiers have equal accuracy on a {alpha*100}% significance level ")
    print()
    print()
    return p_val


def regression_compare(zA, zB, alpha=0.05):
    z = zA - zB

    conf_l = t.ppf(alpha, df=len(z)-1, loc=z.mean(), scale=np.sqrt(z.var(ddof=1)/len(z)))
    conf_u = t.ppf(1-alpha, df=len(z)-1, loc=z.mean(), scale=np.sqrt(z.var(ddof=1)/len(z)))

    p_val = 2*t.cdf(-abs(z.mean()), df=len(z)-1, loc=0, scale=np.sqrt(z.var(ddof=1)/len(z)))

    print(f"Significance level: {alpha}")
    print(f"MSE of model 1: {zA.mean()}")
    print(f"MSE of model 2: {zB.mean()}")
    print(f"Confidence interval: [{conf_l}, {conf_u}]")
    print(f"p-value: {p_val}")
    print()
    if p_val > alpha:
        print("We can not reject the null hypothesis")
    else:
        print(f"We reject the null hypothesis on a {alpha*100}% significance level ")
    print()
    print()
    return p_val


def bh_correction(p_vals):
    m = len(p_vals)
    p_indices = np.argsort(p_vals)
    p_back_indices = []
    for i in range(m):
        for k in range(m):
            if i == p_indices[k]:
                p_back_indices.append(k)

    p_vals = np.sort(p_vals)
    p_vals = p_vals * m / np.arange(1, m+1, 1)
    for i in range(1, m):
        if p_vals[i] < p_vals[i-1]:
            p_vals[i-1] = p_vals[i]
    p_vals = np.clip(p_vals, 0, 1)
    p_vals = p_vals[p_back_indices]
    return p_vals


def pair_wise_test2(predictions,  names):
    my_cm = np.array([
        [105/256, 165/256, 131/256, 0.5],
        [224 / 256, 202 / 256, 151 / 256, 1],
        [125 / 256, 180 / 256, 120 / 256, 1]

    ])
    my_cm = ListedColormap(my_cm)
    fig, ax = plt.subplots()
    m = len(names)
    p_matrix = np.zeros((m, m))
    p_vals = []
    for i in range(len(predictions)):
        k = i+1
        while k < len(predictions):
            print(f"{names[i]} & {names[k]}")
            p = mcnemar(predictions[i], predictions[k])
            p_vals.append(p)
            k += 1

    p_vals = bh_correction(p_vals)
    counter = 0
    for i in range(len(predictions)):
        k = i + 1
        while k < len(predictions):
            print(f"{names[i]} & {names[k]}")
            p = p_vals[counter]
            p_matrix[i, k] = p
            counter += 1
            ax.text(i, k, np.round(p, 3), va='center', ha='center')
            ax.text(k, i, np.round(p, 3), va='center', ha='center')
            if p < 0.05:
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


preds_lr = pickle.load(open('lr_predictions.pkl', 'rb'))
preds_svc = pickle.load(open('svc_predictions.pkl', 'rb'))
preds_lstm = pickle.load(open('lstm_predictions.pkl', 'rb'))
preds_lstm = np.concatenate(preds_lstm, axis=0)
preds_gb = pickle.load(open('gb_predictions.pkl', 'rb'))
preds_ridge = pickle.load(open('ridge_gbr_predictions.pkl', 'rb'))

names = ['LogR', 'SVC', 'LSTM', 'GBC', 'RegC']
predictions = [preds_lr, preds_svc, preds_lstm, preds_gb, preds_ridge]
pair_wise_test2(predictions, names)
