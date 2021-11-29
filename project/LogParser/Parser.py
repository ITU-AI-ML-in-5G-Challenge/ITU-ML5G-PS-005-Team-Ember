import os.path
import pandas as pd
from LogParser import TreeParser

class Parser:
    def __init__(self, options):
        self.options = options
        self.log_name = options['log_name']
        self.log_file = options['log_name'] + '.parse.structured.csv'
        self.key_file = options['log_name'] + '.parse.key.csv'

    def parse(self, log_list):
        log_format = self.options['log_format']
        regex      = self.options['regex']
        depth      = self.options['depth']
        st         = self.options['st']

        if os.path.isfile(self.log_file) and os.path.isfile(self.key_file):
            log_structed = pd.read_csv(self.log_file) 
            log_key = pd.read_csv(self.key_file)

        else:
            parser = TreeParser.LogParser(log_format, depth=depth, st=st, rex=regex)

            log_structed, log_key = parser.parse(self.log_name, log_list)
        
            log_structed.to_csv(self.log_file, index=False)
            log_key.to_csv(self.key_file, index=False)

        return log_structed, log_key
