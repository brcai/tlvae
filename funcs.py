import numpy as np
import torch.utils.data as data_utils
import sklearn.metrics as metrics
import pandas as pd
import sklearn.preprocessing as prep
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def Mape(y, y_pred):
    res = metrics.mean_absolute_percentage_error(y_true, y_pred)
    return res

def Mae(y, y_pred):
    res = metrics.mean_absolute_error(y_true, y_pred)
    return res

def R_squared():
    res = metrics.r2_score(y_true, y_pred)
    return res

def RMSE():
    res = metrics.mean_squared_error(y_true, y_pred)
    return np.sqrt(res)

def readFile(file):
    dir = "data/"
    fp = open(dir+file)
    d = []
    for line in fp.readlines():
        if line=='\n': continue
        d.append(float(line.replace("\n",'')))
    dt = np.array(d)

    #scaler = StandardScaler()
    scaler = MinMaxScaler()

    scaler.fit(dt.reshape(-1, 1))
    dt = scaler.transform(dt.reshape(-1, 1))
    dt = np.array([i[0] for i in dt])

    return dt

class DeterministicWarmup(object):

    def __init__(self, n_steps, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.increase = self.t_max / n_steps

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.increase

        self.t = self.t_max if t > self.t_max else t

        return self.t

def ReadData(file, batch_size, window_size):
    fp = open("data/"+file+".txt")
    dt = []
    for line in fp.readlines():
        val = line.replace("\n",'')
        dt.append(float(val))
    dt = np.array(dt)
    #dt = np.array([[round(i,3)] for i in dt])

    scalar = prep.MinMaxScaler()
    dt = scalar.fit_transform(dt.reshape(-1, 1))
    dt = np.array([[round(i[0],3)] for i in dt])

    percentile = [0.8,0.1,0.1]
    valid_split = int(len(dt)*percentile[0])
    test_split = int(len(dt)*(percentile[0]+percentile[1]))

    dt_train = dt[:,0][:valid_split]
    dt_valid = dt[:,0][valid_split:test_split]
    dt_test = dt[:,0][test_split:]

    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    x_test = []
    y_test = []

    for i in range(window_size, len(dt_train)):
        x_train.append(dt_train[i-window_size:i])
        y_train.append([dt_train[i]])
    train = data_utils.TensorDataset(torch.tensor(x_train).type(torch.FloatTensor), torch.tensor(y_train).type(torch.FloatTensor))

    for i in range(window_size, len(dt_valid)):
        x_valid.append(dt_valid[i-window_size:i])
        y_valid.append([dt_valid[i]])
    valid = data_utils.TensorDataset(torch.tensor(x_valid).type(torch.FloatTensor), torch.tensor(y_valid).type(torch.FloatTensor))

    for i in range(window_size, len(dt_test)):
        x_test.append(dt_test[i-window_size:i])
        y_test.append([dt_test[i]])
    test = data_utils.TensorDataset(torch.tensor(x_test).type(torch.FloatTensor), torch.tensor(y_test).type(torch.FloatTensor))

    train_loader = data_utils.DataLoader(train, batch_size=batch_size[0], shuffle=False)
    valid_loader = data_utils.DataLoader(valid, batch_size=batch_size[1], shuffle=False)
    test_loader = data_utils.DataLoader(test, batch_size=batch_size[2], shuffle=False)

    return train_loader, valid_loader, test_loader

def ReadData1(file, batch_size, window_size):
    fp = open("dataset/"+file)
    dt = []
    for line in fp.readlines():
        val = line.replace("\n",'')
        dt.append([float(val)])
    dt = np.array(dt)

    scalar = prep.MinMaxScaler()
    dt = scalar.fit_transform(dt)
    dt = np.array([round(i[0],3) for i in dt])

    percentile = [0.6,0.2,0.2]
    valid_split = int(len(dt)*percentile[0])
    test_split = int(len(dt)*(percentile[0]+percentile[1]))

    dt_train = dt[:valid_split]
    dt_valid = dt[valid_split:test_split]
    dt_test = dt[test_split:]

    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    x_test = []
    y_test = []

    for i in range(window_size, len(dt_train)):
        x_train.append(dt_train[i-window_size:i])
        y_train.append([dt_train[i]])
    train = data_utils.TensorDataset(torch.tensor(x_train).type(torch.FloatTensor), torch.tensor(y_train).type(torch.FloatTensor))

    for i in range(window_size, len(dt_valid)):
        x_valid.append(dt_valid[i-window_size:i])
        y_valid.append([dt_valid[i]])
    valid = data_utils.TensorDataset(torch.tensor(x_valid).type(torch.FloatTensor), torch.tensor(y_valid).type(torch.FloatTensor))

    for i in range(window_size, len(dt_test)):
        x_test.append(dt_test[i-window_size:i])
        y_test.append([dt_test[i]])
    test = data_utils.TensorDataset(torch.tensor(x_test).type(torch.FloatTensor), torch.tensor(y_test).type(torch.FloatTensor))

    train_loader = data_utils.DataLoader(train, batch_size=batch_size[0], shuffle=False)
    valid_loader = data_utils.DataLoader(valid, batch_size=batch_size[1], shuffle=False)
    test_loader = data_utils.DataLoader(test, batch_size=batch_size[2], shuffle=False)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    print("test")
    readFile("henon.txt")
    print("end of test")