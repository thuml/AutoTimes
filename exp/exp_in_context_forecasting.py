from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.losses import mape_loss, mase_loss, smape_loss, zero_shot_smape_loss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

# In our in-context learning setting
# the task is to apply a forecaster, trained on a source dataset, to an unseen target dataset
# Additionally, several task demonstrations from the target domain, 
# referred to as time series prompts are available during inference
# Concretely, AutoTimes trains LLMs on the source domain with a larger context length to place the additional time series prompt. 
# See ```Dataset_TSF_ICL``` in ```data_loader.py``` for the construction of time series prompts

warnings.filterwarnings('ignore')

def SMAPE(pred, true):
    return np.mean(200 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))
def MAPE(pred, true):
    return np.mean(np.abs(100 * (pred - true) / (true +1e-8)))

class Exp_In_Context_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_In_Context_Forecast, self).__init__(args)

    def _build_model(self):
        if self.args.data == 'm4':
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        self.device = self.args.gpu
        model = self.model_dict[self.args.model].Model(self.args).to(self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print(n, p.dtype, p.shape)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        
        self.args.root_path = './dataset/tsf'
        self.args.data_path = self.args.test_data_path
        self.args.data = 'tsf'
        test_data2, test_loader2 = self._get_data(flag='test')
        
        self.args.data = 'tsf_icl'
        test_data3, test_loader3 = self._get_data(flag="test")
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion(self.args.loss)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device) 

                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, None, None, None)
                else:
                    outputs = self.model(batch_x, None, None, None)

                loss = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion) # test_loss indicates the result on the source datasets
            test_loss = vali_loss
            test_loss2 = self.vali2(test_data2, test_loader2, zero_shot_smape_loss())  # test_loss2 indicates the result on the target datasets
            test_loss3 = self.vali2(test_data3, test_loader3, zero_shot_smape_loss())  # test_loss3 indicates the result on the target datasets with time series prompts
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Zero Shot Test Loss: {4:.7f} In Context Test Loss: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss2, test_loss3))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + f'checkpoint.pth'

        self.model.load_state_dict(torch.load(best_model_path), strict=False)

        return self.model

    def vali(self, train_loader, vali_loader, criterion):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            # decoder input
            B, _, C = x.shape
            
            outputs = torch.zeros((B, self.args.seq_len, C)).float()  # .to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    for i in range(len(id_list) - 1):
                        outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None, None, None).detach().cpu()
            else:
                for i in range(len(id_list) - 1):
                    outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None, None, None).detach().cpu()
            pred = outputs[:, -self.args.token_len:, :]
            true = torch.from_numpy(np.array(y))
            batch_y_mark = torch.ones(true.shape)
            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        self.model.train()
        return loss

    def vali2(self, vali_data, vali_loader, criterion):
        total_loss = []
        count= []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, None, None, None)
                else:
                    outputs = self.model(batch_x, None, None, None)

                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)

                pred = outputs[:, -self.args.test_pred_len:, :].detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
                count.append(batch_x.shape[0])
        total_loss = np.average(total_loss, weights=count)
        self.model.train()
        
        return total_loss    

    def test_(self, test_loader):
        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, None, None, None)
                else:
                    outputs = self.model(batch_x, None, None, None)

                outputs = outputs[:, -self.args.test_pred_len:, :]
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)
                
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        smape = SMAPE(preds, trues)
        mape = MAPE(preds, trues)
        print('mape:{:4f}, smape:{:.4f}'.format(mape, smape))
        
    def test(self, setting, test=0):
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, best_model_path)), strict=False)

        self.args.data_path = self.args.test_data_path
        
        self.args.root_path = './dataset/tsf'
        self.args.data_path = self.args.test_data_path
        self.args.data = 'tsf'
        test_data, test_loader = self._get_data('test')
        self.args.data = 'tsf_icl'
        test_data2, test_loader2 = self._get_data('test')
        
        print("zero shot forecasting")
        self.test_(test_loader)
        print("in context forecasting")
        self.test_(test_loader2)

