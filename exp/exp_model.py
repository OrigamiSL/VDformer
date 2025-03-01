from data.data_loader import *
from exp.exp_basic import Exp_Basic
from VDformer.Model import VDformer
from thop import profile, clever_format
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_Model(Exp_Basic):
    def __init__(self, args):
        super(Exp_Model, self).__init__(args)

    def _build_model(self):
        model = VDformer(
            self.args.input_len,
            self.args.pred_len[-1],
            self.args.encoder_layer,
            self.args.patch_size,
            self.args.d_model,
            self.args.dropout
        ).float()

        input_x = torch.randn(1, self.args.input_len, self.args.enc_in)
        pred_x = torch.randn(1, self.args.pred_len[-1], self.args.enc_in)
        var_x = torch.zeros([1, self.args.enc_in, self.args.enc_in]).bool()
        flops, params = profile(model, inputs=(input_x, pred_x, var_x))
        flops, params = clever_format([flops, params], '%.3f')
        print(f"flops：{flops}, params：{params}")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args
        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_min,
            'ETTm2': Dataset_ETT_min,
            'weather': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'Traffic': Dataset_Custom,
            'Air': Dataset_Custom,
            'River': Dataset_Custom,
            'HomeC': Dataset_HomeC,
        }
        Data = data_dict[self.args.data]

        size = [args.input_len, args.pred_len[-1]]
        if flag == 'train':
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
        else:
            shuffle_flag = False
            drop_last = True
            batch_size = 1

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=size,
            features=args.features,
            target=args.target,
            ori_target=args.ori_target,
            EMD=args.EMD
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def vali(self, vali_data=None, vali_loader=None):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, pred_x, var_mask) in enumerate(vali_loader):
                pred, true = self._process_one_batch(batch_x, pred_x, var_mask)
                loss = torch.mean((pred - true) ** 2).detach().cpu().numpy()
                total_loss.append(loss)
            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting=None):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()
        train_steps = len(train_loader)

        lr = self.args.learning_rate

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        self.model.train()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, pred_x, var_mask) in enumerate(train_loader):
                model_optim.zero_grad()
                iter_count += 1
                pred, true = self._process_one_batch(batch_x, pred_x, var_mask)
                loss = torch.mean((pred - true) ** 2) + torch.mean(abs(pred - true))
                loss.backward(loss)
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1,
                                                                            torch.mean(loss).item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            vali_loss = self.vali(vali_data, vali_loader)
            test_loss = self.vali(test_data, test_loader)

            print("Pred_len: {0}| Epoch: {1}, Steps: {2} | Total: Vali Loss: {3:.7f} Test Loss: {4:.7f}| "
                  .format(self.args.pred_len[-1], epoch + 1, train_steps, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, (epoch + 1), self.args)

        self.args.learning_rate = lr

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, load=False, write_loss=True, save_loss=True):
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        test_data, test_loader = self._get_data(flag='test')
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        time_now = time.time()

        with torch.no_grad():
            for i, (batch_x, pred_x, var_mask) in enumerate(test_loader):
                pred, true = self._process_one_batch(batch_x, pred_x, var_mask)
                pred = pred.squeeze(0).detach().cpu().numpy()
                true = true.squeeze(0).detach().cpu().numpy()
                np.save(folder_path + 'pred_{}.npy'.format(i), pred)
                np.save(folder_path + 'true_{}.npy'.format(i), true)

        print("inference time: {}".format(time.time() - time_now))

    def _process_one_batch(self, batch_x, pred_x, var_mask):
        batch_x = batch_x.float().to(self.device)
        input_seq = batch_x[:, :self.args.input_len, :]
        batch_y = batch_x[:, -self.args.pred_len[-1]:, :]
        pred_x = pred_x.float().to(self.device)
        var_mask = var_mask.bool().to(self.device)
        pred_data = self.model(input_seq, pred_x, var_mask)
        return pred_data, batch_y
