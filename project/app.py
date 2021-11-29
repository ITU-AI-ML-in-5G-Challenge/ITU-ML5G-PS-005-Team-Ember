import sys
import SysIO
import LogParser
import LogFeature
import LogTrainer
import LogTester

def detect_sysmonitor():
    options = dict()

    # load 
    options['log_name'] = 'sysmonitor'

    print(">> Load: sysmonitor")
    train_log, test_log, concat_log = SysIO.load_logs(options['log_name'])

    # parse 
    options['log_format'] = '<Date>T<Time>\+08:00\|<Content>'  # sysmonitor format
    options['regex'] = [
        r'([0-9]*[.])?[0-9]+', # Version
        r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
    ]
    options['st'] = 0.1  # Similarity threshold
    options['depth'] = 3  # Depth of all leaf nodes

    print(">> Parse: sysmonitor")
    log_parser = LogParser.Parser(options)
    log_structured, log_templates = log_parser.parse(concat_log)

    # feature
    options['window_size'] = 10  # Depth of all leaf nodes

    print(">> Feature: sysmonitor")
    log_feature = LogFeature.Feature(options)
    dataset, test_seqs, test_labels = log_feature.extract(log_structured, log_templates, train_log)

    # train
    options['device'] = "cpu"

    options['input_size'] = 1
    options['hidden_size'] = 64
    options['num_layers'] = 2
    options['num_classes'] = log_feature.get_num_classes()

    options['batch_size'] = 2048
    options['optimizer'] = 'adam'
    options['lr'] = 0.001
    options['max_epoch'] = 150
    options['lr_step'] = (100, 120)
    options['lr_decay_ratio'] = 0.1

    print(">> Train: sysmonitor")
    log_train = LogTrainer.Trainer(options)
    moedel = log_train.start_train(dataset)

    # test
    options['num_candidates'] = log_feature.get_num_candidates(0.35)

    print(">> Test: sysmonitor")
    log_tester = LogTester.Tester(options, moedel)
    result_df = log_tester.start_test(test_seqs, test_labels)

    return result_df

def detect_messages():
    options = dict()

    # load 
    options['log_name'] = 'messages'

    print(">> Load: messages")
    train_log, test_log, concat_log = SysIO.load_logs(options['log_name'])

    # parse 
    options['log_format'] = '<Date>T<Time>\+08:00<Content>'  # messages format
    options['regex'] = [
        r'ffff::ffff:ffff:ffff:ffff', # IP
        r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
        r'([0-9]*[.])?[0-9]+', # Version
        r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
    ]
    options['st'] = 0.01  # Similarity threshold
    options['depth'] = 3  # Depth of all leaf nodes

    print(">> Parse: messages")
    log_parser = LogParser.Parser(options)
    log_structured, log_templates = log_parser.parse(concat_log)

    # feature
    options['window_size'] = 10  # Depth of all leaf nodes

    print(">> Feature: messages")
    log_feature = LogFeature.Feature(options)
    dataset, test_seqs, test_labels = log_feature.extract(log_structured, log_templates, train_log)

    # train
    options['device'] = "cpu"

    options['input_size'] = 1
    options['hidden_size'] = 64
    options['num_layers'] = 2
    options['num_classes'] = log_feature.get_num_classes()

    options['batch_size'] = 1024
    options['optimizer'] = 'adam'
    options['lr'] = 0.001
    options['max_epoch'] = 6
    options['lr_step'] = (4, 5)
    options['lr_decay_ratio'] = 0.1

    print(">> Train: messages")
    log_train = LogTrainer.Trainer(options)
    moedel = log_train.start_train(dataset)

    # test
    options['num_candidates'] = log_feature.get_num_candidates(0.32)

    print(">> Test: messages")
    log_tester = LogTester.Tester(options, moedel)
    result_df = log_tester.start_test(test_seqs, test_labels)

    return result_df

if __name__ == '__main__':
    if sys.argv[1:]:
        SysIO.set_path(sys.argv[1:])

    SysIO.seed_everything(seed=1234)
    sysmonitor_result = detect_sysmonitor()
    messages_result = detect_messages()
    SysIO.save_result(sysmonitor_result, messages_result)