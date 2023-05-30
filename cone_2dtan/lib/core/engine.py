import torch
from core.config import config, update_config
from torch.utils.data import DataLoader
import datasets
from tqdm import tqdm
import math


class Engine(object):
    def __init__(self):
        self.hooks = {}

    def hook(self, name, state):
        if name in self.hooks:
            self.hooks[name](state)

    def train(self, network, iterator, maxepoch, optimizer, scheduler):
        # iterator = DataLoader(train_dataset,
        #                       batch_size=config.TRAIN.BATCH_SIZE,
        #                       shuffle=config.TRAIN.SHUFFLE,
        #                       num_workers=config.WORKERS,
        #                       pin_memory=False,
        #                       collate_fn=datasets.start_end_collate)

        state = {
            'network': network,
            'iterator': iterator,
            'maxepoch': maxepoch,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'epoch': 0,
            't': 0,
            'train': True,
        }

        self.hook('on_start', state)
        while state['epoch'] < state['maxepoch']:
            self.hook('on_start_epoch', state)
            for sample in state['iterator']:
                state['sample'] = sample
                self.hook('on_sample', state)

                def closure():
                    #torch.autograd.set_detect_anomaly(True)
                    loss, output = state['network'](state['sample'],state["epoch"])
                    state['output'] = output
                    state['loss'] = loss
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    self.hook('on_forward', state)
                    # to free memory in save_for_backward
                    state['output'] = None
                    state['loss'] = None
                    return loss

                state['optimizer'].zero_grad()
                state['optimizer'].step(closure)
                self.hook('on_update', state)
                state['t'] += 1
            state['epoch'] += 1
            self.hook('on_end_epoch', state)
        self.hook('on_end', state)
        return state

    def test(self, network, iterator, split, epoch=0, train_t=0):
        # query_id2windowidx = self.pre_filtering(eval_inter_window_dataset)
        # eval_intra_window_dataset.query_id2windowidx = query_id2windowidx
        # iterator = DataLoader(eval_intra_window_dataset,
        #                         batch_size=config.TEST.BATCH_SIZE,
        #                         shuffle=False,
        #                         num_workers=config.WORKERS,
        #                         pin_memory=False,
        #                         collate_fn=datasets.start_end_collate)

        state = {
            'network': network,
            'iterator': iterator,
            'split': split,
            't': 0,
            'train_t': train_t,
            'train': False,
            "epoch": epoch,
        }

        self.hook('on_test_start', state)
        for sample in state['iterator']:
            state['sample'] = sample
            self.hook('on_test_sample', state)

            def closure():
                loss, output = state['network'](state['sample'],state["epoch"])
                state['output'] = output
                state['loss'] = loss
                self.hook('on_test_forward', state)
                # to free memory in save_for_backward
                state['output'] = None
                state['loss'] = None

            closure()
            state['t'] += 1
        self.hook('on_test_end', state)
        return state
