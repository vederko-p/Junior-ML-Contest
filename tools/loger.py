
import json
from copy import deepcopy
import matplotlib.pyplot as plt


class LogModule():
    def __init__(self):
        self.logs = dict()
        self.max_epoch = -1
        
    def add(self, logs):
        self.max_epoch += 1
        self.logs[self.max_epoch] = logs
        return
    
    def save_logs(self, filename):
        with open(filename, 'w') as log_file:
            json.dump(self.logs, log_file)
        return
    
    def load_logs(self, filename):
        with open(filename, 'r') as log_file:
            data = json.load(log_file)
        self.logs = data.copy()
        self.logs = {int(k):i for k,i in self.logs.items()}
        self.max_epoch = len(self.logs.keys()) - 1
        return
    
    def cat(self, logs_list):
        main_loger = LogModule()
        max_epoch = -1
        for log_inst in logs_list:
            if  isinstance(log_inst, str):
                ilog = LogModule()
                ilog.load_logs(log_inst)
            else:
                ilog = deepcopy(log_inst)
            for _,w in ilog.logs.items():
                main_loger.logs[max_epoch + 1] = w
                max_epoch += 1
        return main_loger
    
    def vis_vals(self, vals, title='', xlabel='', ylabel=''):
        fig, ax = plt.subplots()
        ys = []
        for v in vals:
            ys.append([self.logs[ep][v] for ep in range(self.max_epoch+1)])
        x = range(self.max_epoch+1)
        for y,lbl in zip(ys, vals):
            ax.plot(x, y, label=lbl)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        plt.legend()
        plt.show()
