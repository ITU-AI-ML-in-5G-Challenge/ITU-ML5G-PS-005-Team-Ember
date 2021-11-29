import os.path
import torch
import math
import pandas as pd
import numpy as np
from collections import OrderedDict
from torch.utils.data import TensorDataset

class Feature:
    def __init__(self, options):
        self.options = options
        self.dataset = options['log_name'] + '.feature.dataset.pt'
        self.labels  = options['log_name'] + '.feature.labels.npy'
        self.keys    = options['log_name'] + '.feature.keys.npy'
        self.seqs    = options['log_name'] + '.feature.seqs.npy'

    def load_structured(self, log_structured, log_templates):
        # event key id
        self.key_dict = OrderedDict()
        for idx, row in log_templates.iterrows():
            event_id = row['EventId']
            self.key_dict[event_id] = idx

        # convert to data time format
        date_time = log_structured["Date"]+'-'+log_structured["Time"]
        date_time = date_time.str[:19]

        log_structured["DateTime"] = pd.to_datetime(date_time, errors='raise', format = "%Y-%m-%d-%H:%M:%S") - pd.Timedelta('08:00:00')
        # calculate the time interval since the start time
        log_structured["DateTime"]  = log_structured['DateTime'] - pd.Timestamp("1970-01-01")
        log_structured["TimeSlice"] = log_structured["DateTime"].astype(int) // (10**9 * 300)

        event_dict = OrderedDict()
        for idx, row in log_structured.iterrows():
            time_slice = row['TimeSlice']
            if not time_slice in event_dict:
                event_dict[time_slice] = []
            try: key = self.key_dict[row['EventId']]
            except Exception: key = -1
            event_dict[time_slice].append(key)

        event_df = pd.DataFrame(list(event_dict.items()), columns=['TimeSlice', 'EventSequence'])
        # event_df.to_csv("5g_sequence.csv", index=None)

        return event_df

    def generate_train(self, event_df):
        window_size = self.options["window_size"]
        num_key = len(self.key_dict)
        num_sessions = 0
        inputs = []
        outputs = []
        for idx, row in event_df.iterrows():
            num_sessions += 1
            ln = row['EventSequence']
            line = ln + [num_key] * (window_size + 1 - len(ln))
            for i in range(len(line) - window_size):
                inputs.append(line[i:i + window_size])
                outputs.append(line[i + window_size])
        print('Number of sessions({}): {}'.format("n", num_sessions))
        print('Number of seqs({}): {}'.format("n", len(inputs)))

        dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
        return dataset

    def generate_test(self, event_df):
        num_key = len(self.key_dict)
        test_seqs = OrderedDict()
        test_labels = OrderedDict()
        window_size = self.options["window_size"]
        for idx, row in event_df.iterrows():
            time_slice = row['TimeSlice']
            ln = row['EventSequence']
            line = ln + [num_key] * (window_size + 1 - len(ln))
            for i in range(len(line) - window_size):
                if not time_slice in test_seqs:
                    test_seqs[time_slice] = []
                    test_labels[time_slice] = []

                test_seqs[time_slice].append(line[i:i + window_size])
                test_labels[time_slice].append(line[i + window_size])

        print('Number of timeslice({}): {}'.format("n", len(test_seqs)))

        return test_seqs, test_labels

    def extract(self, log_structured, log_templates, train_log):

        if os.path.isfile(self.dataset) and os.path.isfile(self.seqs) and os.path.isfile(self.labels) and os.path.isfile(self.keys) :
            dataset = torch.load(self.dataset)
            test_seqs = np.load(self.seqs,allow_pickle='TRUE').item()
            test_labels = np.load(self.labels,allow_pickle='TRUE').item()
            self.key_dict = np.load(self.keys,allow_pickle='TRUE').item()

        else:
            train_length = len(train_log)

            train_structured = log_structured.loc[:train_length-1,].copy()
            # print(train_structured)
            print('train_structured', len(train_structured))
            train_df = self.load_structured(train_structured, log_templates)
            dataset = self.generate_train(train_df)

            test_structured = log_structured.loc[train_length:,].copy()
            # print(test_structured)
            print('test_structured', len(test_structured))
            test_df = self.load_structured(test_structured, log_templates)
            test_seqs, test_labels = self.generate_test(test_df)

            torch.save(dataset, self.dataset)
            np.save(self.seqs, test_seqs) 
            np.save(self.labels, test_labels) 
            np.save(self.keys, self.key_dict) 

        return dataset, test_seqs, test_labels

    def get_num_classes(self):
        return len(self.key_dict)+1

    def get_num_candidates(self, ratio):
        return int(-0.3 * math.log(ratio) / ratio * len(self.key_dict))
