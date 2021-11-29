import time
import torch
import pandas as pd

class Tester():
    def __init__(self, options, model):
        self.options = options

        self.model = model
        self.model.eval()

        self.num_candidates = options['num_candidates']

        self.window_size = options['window_size']
        self.input_size = options['input_size']

        self.device = options['device']

    def test(self, test_seqs, test_labels):        
        # Test the model
        result_list = []
        seq_num = 0
        with torch.no_grad():
            for time_slice in test_seqs:
                out_label = 0
                seq_num += len(test_seqs[time_slice])
                for i in range(len(test_seqs[time_slice])):
                    seq = test_seqs[time_slice][i]
                    label = test_labels[time_slice][i]
                    seq = torch.tensor(seq, dtype=torch.float).view(-1, self.window_size, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = self.model(seq, self.device)
                    predicted = torch.argsort(output, 1)[0][-self.num_candidates:]
                    if label not in predicted:
                        out_label = 1
                        break
                result_list.append([self.options['log_name'], time_slice, out_label])

        return seq_num, pd.DataFrame(result_list, columns=['LogName', 'TimeSlice', 'Label'])

    def start_test(self, test_seqs, test_labels):  
        start_time = time.time()

        seq_num, result_df = self.test(test_seqs, test_labels)

        elapsed_time = time.time() - start_time
        print('Test seqs: {}, Test time: {:.2f}s'.format(seq_num, elapsed_time))

        return result_df

