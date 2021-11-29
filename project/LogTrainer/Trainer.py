import time
import os.path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class deeplog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(deeplog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, features, device):
        input0 = features
        h0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        out, _ = self.lstm(input0, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class Trainer():
    def __init__(self, options):
        self.options = options

        self.input_size=options['input_size']
        self.hidden_size=options['hidden_size']
        self.num_layers=options['num_layers']
        self.num_keys=options['num_classes']

        self.window_size = options['window_size']
        self.batch_size = options['batch_size']

        self.device = options['device']
        self.lr_step = options['lr_step']
        self.lr_decay_ratio = options['lr_decay_ratio']
        self.max_epoch = options['max_epoch']

        self.model_path = self.options['log_name'] + '.model.pt'

        model = deeplog(self.input_size,
                        self.hidden_size,
                        self.num_layers,
                        self.num_keys)

        self.model = model.to(self.device)
        self.model.train()

        if options['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=options['lr'],
                                             momentum=0.9)
        elif options['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=options['lr'],
                betas=(0.9, 0.999),
            )
        else:
            raise NotImplementedError
    
        self.start_epoch = 0

    def train(self, epoch):
        start_time = time.time()
        criterion = nn.CrossEntropyLoss()

        total_step = len(self.train_loader)
        train_loss = 0
        for i, (seq, label) in enumerate(self.train_loader):
            # Forward pass
            seq = seq.clone().detach().view(-1, self.window_size, self.input_size).to(self.device)
            output = self.model(seq, self.device)
            loss = criterion(output, label.to(self.device))
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        elapsed_time = time.time() - start_time
        # if (epoch + 1) % (self.max_epoch // 20) == 0:
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print('Epoch: {}, Time: {:.2f}s, Learning Rate: {:.4f}, Loss: {:.4f}'\
              .format(epoch+1, elapsed_time, lr, train_loss/total_step))

    def start_train(self, train_dataset):
        if os.path.isfile(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))

        else:
            self.train_loader = DataLoader(train_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        pin_memory=True)

            self.num_train_log = len(train_dataset)
            print('Find %d train logs' % (self.num_train_log))

            for epoch in range(self.start_epoch, self.max_epoch):
                if epoch == 0:
                    self.optimizer.param_groups[0]['lr'] /= 2
                if epoch in [1, 2, 3, 4, 5]:
                    self.optimizer.param_groups[0]['lr'] *= 2
                if epoch in self.lr_step:
                    self.optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio
                self.train(epoch)

            torch.save(self.model.state_dict(), self.model_path)

        return self.model
