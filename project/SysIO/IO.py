import os
import torch
import random
import numpy as np
import pandas as pd

train_file_map = {'sysmonitor':'sysmonitor_trainset.txt', 'messages':'messages_trainset.txt'}
test_file_map  = {'sysmonitor':'sysmonitor_testset.txt', 'messages':'messages_testset.txt'}

train_path = './datasets/train/'
test_path = './datasets/test/'
result_path = "./results/"

def set_path(path):
    print(path)

    global train_path
    global test_path
    global result_path

    train_path = path[0] + '/'
    test_path = path[2] + '/'
    result_path = path[1] + '/'

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_logs(type):
    train_log = open(train_path+train_file_map[type],"r+", encoding="utf-8").readlines()
    test_log  = open(test_path+test_file_map[type],"r+", encoding="utf-8").readlines()

    for i in range(len(test_log)): 
        test_log[i] = test_log[i].lstrip('\x00')

    concat_log = train_log + test_log

    return train_log, test_log, concat_log

def save_result(sysmonitor_result, messages_result):
    predict_result = pd.concat([messages_result, sysmonitor_result], ignore_index=True, sort=False)
    predict_result.to_csv(result_path+"predict.csv", index=False)
    with open(result_path+"predict_finish.csv", "w") as text_file:
        text_file.write("true\n")
