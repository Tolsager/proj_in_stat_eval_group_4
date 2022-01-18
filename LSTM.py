import torch.nn as nn
from utils import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import torch


BATCH = 150
EPOCHS = 500
eps = 1e-8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'

class DS(Dataset):
    def __init__(self, X, y, use=('x', 'y', 'z')):
        self.X = X
        self.y = y
        self.use = use

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i, :100]
        y_ = self.X[i, 100:200]
        z = self.X[i, 200:]
        if self.use == ('x', 'y', 'z'):
            coords = np.stack((x, y_, z), axis=1)
        elif self.use == ('x', 'y'):
            coords = np.stack((x, y_), axis=1)
        elif self.use == ('x', 'z'):
            coords = np.stack((x, z), axis=1)
        elif self.use == ('y', 'z'):
            coords = np.stack((y_, z), axis=1)
        return {'X': torch.tensor(coords, dtype=torch.float32), 'y': torch.tensor(self.y[i], dtype=torch.int64)}


class Model(nn.Module):
    def __init__(self, coords):
        super(Model, self).__init__()
        self.rnn1 = nn.LSTM(len(coords), 256, 3, batch_first=True, dropout=0.08)
        self.fc1 = nn.Linear(256, 15)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(256)

    def forward(self, x):
        self.rnn1.flatten_parameters()
        _, (out, _) = self.rnn1(x)
        x = self.batchnorm1(out[-1])
        x = self.fc1(x)
        return x


def lstm_cross_validation(X, y, use=('x', 'y', 'z'), cv=10):
    k = StratifiedKFold(cv, shuffle=False)
    patience_counter = 25
    min_epochs = 150
    max_epochs = 200
    split_count = 0
    performance_dict = {}
    best_overall_acc = 0
    for train_idx, test_idx in k.split(X, y):
        kX_train, kX_test = X[train_idx, :], X[test_idx, :]
        ky_train, ky_test = y[train_idx], y[test_idx]
        ds_train = DS(kX_train, ky_train, use)
        dl_train = DataLoader(ds_train, batch_size=BATCH)
        ds_val = DS(kX_test, ky_test, use)
        dl_val = DataLoader(ds_val, batch_size=BATCH)
        counter = 0
        best_acc = 0
        model = Model(use)
        model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        for epoch in range(EPOCHS):
            if (counter >= patience_counter and epoch >= min_epochs) or (epoch >= max_epochs):
                break
            correct_train = 0
            total_train = 0
            model.train()
            for batch in dl_train:
                X_ = batch['X']
                label = batch['y']
                X_ = move_to(X_, DEVICE)
                label = move_to(label, DEVICE)

                pred = model(X_)

                loss = criterion(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                correct_train += torch.sum(pred.argmax(dim=1) == label)
                total_train += len(label)

            correct_val = 0
            total_val = 0

            with torch.no_grad():
                model.eval()
                for batch in dl_val:
                    X_ = batch['X']
                    label = batch['y']
                    X_ = move_to(X_, DEVICE)
                    label = move_to(label, DEVICE)

                    pred = model(X_)

                    correct_val += torch.sum(pred.argmax(dim=1) == label)
                    total_val += len(label)


                val_acc = correct_val / total_val

                if val_acc > best_acc:
                    counter = 0
                    best_acc = val_acc
                    performance_dict[split_count] = best_acc.item()
                    if best_acc > best_overall_acc:
                        print("Saving new best model")
                        print(f"Split: {split_count}\nEpoch: {epoch}\nTraining accuracy: {correct_train / total_train}\nNew best validation accuracy: {best_acc}")
                        best_overall_acc = best_acc
                        torch.save(model.state_dict(), "best_lstm.pt")
                else:
                    counter += 1

        split_count += 1
    return performance_dict
