"""
Created on Fri Oct 23 20:58:38 2020

@author: xuegeeker
@email: xuegeeker@163.com
"""

import numpy as np
import time
import collections
from torch import optim
import torch
from sklearn import metrics, preprocessing
import datetime
import numpy as np
import sys
sys.path.append('./networks')
import network
import train
sys.path.append('./utils')
from generate_pic import aa_and_each_accuracy, sampling1, sampling2, load_dataset, generate_png, generate_iter
import record, extract_samll_cubic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:0')

seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]
ensemble = 1

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')

print('-----Importing Dataset-----')
global Dataset
dataset = 'PC'
Dataset = dataset.upper()
data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE,VALIDATION_SPLIT = load_dataset(Dataset)
print(data_hsi.shape)

ROWS, COLUMNS, BAND = data_hsi.shape
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)
CLASSES_NUM = int(max(gt))
print('The class numbers of the HSI data is:', CLASSES_NUM)

print('-----Importing Setting Parameters-----')
ITER = 2
PATCH_LENGTH = 3
lr, num_epochs, batch_size = 0.0002, 300, 32
loss = torch.nn.CrossEntropyLoss()

img_rows = 2*PATCH_LENGTH+1
img_cols = 2*PATCH_LENGTH+1
INPUT_DIMENSION = data_hsi.shape[2]
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]
VAL_SIZE = int(TRAIN_SIZE)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
SAMPLES_NUM = 200

KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))


data = preprocessing.scale(data)
whole_data = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                         'constant', constant_values=0)

for index_iter in range(ITER):
    print('-----Begining to conduct the ' + str(index_iter + 1) + ' iter training process-----')
    net = network.HResNetAM(BAND, CLASSES_NUM)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    time_1 = int(time.time())
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling1(SAMPLES_NUM, gt)
    _, total_indices = sampling2(gt, 1)
    
    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print('Test size: ', TEST_SIZE)
    VAL_SIZE = int(TRAIN_SIZE)
    print('Validation size: ', VAL_SIZE)
    
    print('-----Selecting Small Pieces from the Original Cube Data-----')
    train_iter, valida_iter, test_iter, all_iter = generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, VAL_SIZE,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt)
    
    print('-----Begining to Train The Model with Training Dataset-----')
    tic1 = time.clock()
    train.train(net, train_iter, valida_iter, loss, optimizer, device, epochs=num_epochs)
    toc1 = time.clock()
    
    pred_test = []
    tic2 = time.clock()
    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)
            net.eval()
            y_hat = net(X)
            pred_test.extend(np.array(net(X).cpu().argmax(axis=1)))
    toc2 = time.clock()
    collections.Counter(pred_test)
    gt_test = gt[test_indices] - 1
    
    overall_acc = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
    confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
    each_acc, average_acc = aa_and_each_accuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])
    
    torch.save(net.state_dict(), "./models/" + str(round(overall_acc, 3)) + '.pt')
    KAPPA.append(kappa)
    OA.append(overall_acc)
    AA.append(average_acc)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[index_iter, :] = each_acc
    
print("--------" + net.name + " Training Finished-----------")
record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                     'records/' + net.name + 'Patch'+ str(2*PATCH_LENGTH+1) + 'Time' + day_str + '_' + Dataset + 'TrainingSamples' + str(SAMPLES_NUM) + 'lr：' + str(lr) + '.txt')

generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices)
