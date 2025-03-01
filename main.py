import argparse
import os
import torch
import numpy as np
import time
import shutil
from utils.metrics import metric
from exp.exp_model import Exp_Model

parser = argparse.ArgumentParser(description='FPPformerV2')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S]; M:multivariate predict multivariate, '
                         'S: univariate predict univariate')
parser.add_argument('--target', type=str, default='None', help='target feature in S or M task')
parser.add_argument('--ori_target', type=str, default='None', help='Default target, determine the EMD'
                                                                   'result order')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--input_len', type=int, default=96, help='input length')
parser.add_argument('--pred_len', type=str, default='96,192,336,720', help='prediction length')

parser.add_argument('--enc_in', type=int, default=7, help='input size')
parser.add_argument('--dec_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=28, help='hidden dims of model')
parser.add_argument('--encoder_layer', type=int, default=3)
parser.add_argument('--patch_size', type=int, default=6, help='patch size')
parser.add_argument('--EMD', action='store_true',
                    help='whether to use EMD as the prediction initialization'
                    , default=False)

parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=5, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=1, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--save_loss', action='store_true', help='whether saving results and checkpoints', default=False)
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--train', action='store_true',
                    help='whether to train'
                    , default=False)
parser.add_argument('--reproducible', action='store_true',
                    help='whether to make results reproducible'
                    , default=False)

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() else False

if args.reproducible:
    np.random.seed(4321)  # reproducible
    torch.manual_seed(4321)
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.deterministic = False

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'target': 'OT', 'root_path': './data/ETT/', 'M': [7, 7], 'S': [1, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'target': 'OT', 'root_path': './data/ETT/', 'M': [7, 7], 'S': [1, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'target': 'OT', 'root_path': './data/ETT/', 'M': [7, 7], 'S': [1, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'target': 'OT', 'root_path': './data/ETT/', 'M': [7, 7], 'S': [1, 1]},
    'ECL': {'data': 'ECL.csv', 'target': 'MT_321', 'root_path': './data/ECL/', 'M': [321, 321], 'S': [1, 1]},
    'Traffic': {'data': 'Traffic.csv', 'target': 'Sensor_861', 'root_path': './data/Traffic/', 'M': [862, 862],
                'S': [1, 1]},
    'weather': {'data': 'weather.csv', 'target': 'OT', 'root_path': './data/weather/', 'M': [21, 21], 'S': [1, 1]},
    'Solar': {'data': 'solar_AL.csv', 'target': '136', 'root_path': './data/Solar/', 'M': [137, 137], 'S': [1, 1]},
    'Air': {'data': 'Air.csv', 'target': 'AH', 'root_path': './data/Air/', 'M': [12, 12], 'S': [1, 1]},
    'River': {'data': 'River.csv', 'target': 'DLDI4__0', 'root_path': './data/River/', 'M': [8, 8], 'S': [1, 1]},
    'HomeC': {'data': 'HomeC.csv', 'target': 'precipProbability', 'root_path': './data/HomeC/', 'M': [29, 29],
              'S': [1, 1]},
}

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.ori_target = data_info['target']
    if args.target == 'None':
        args.target = data_info['target']
    args.root_path = data_info['root_path']
    args.enc_in, args.dec_out = data_info[args.features]

args.target = args.target.replace('/r', '').replace('/t', '').replace('/n', '')
args.pred_len = [int(predl) for predl in args.pred_len.replace(' ', '').split(',')]

lr = args.learning_rate
print('Args in experiment:')
print(args)

Exp = Exp_Model
for ii in range(args.itr):
    if args.train:
        setting = '{}_ft{}_ll{}_pl{}_{}'.format(args.data,
                                                args.features, args.input_len,
                                                args.pred_len, ii)
        print('>>>>>>>start training| pred_len:{}, settings: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.
              format(args.pred_len, setting))
        try:
            exp = Exp(args)  # set experiments
            exp.train(setting)
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from forecasting early')

        print('>>>>>>>testing| pred_len:{}: {}<<<<<<<<<<<<<<<<<'.format(args.pred_len, setting))
        exp = Exp(args)  # set experiments
        exp.test(setting, load=True, write_loss=True, save_loss=args.save_loss)
        torch.cuda.empty_cache()
        args.learning_rate = lr
    else:
        setting = '{}_ft{}_ll{}_pl{}_{}'.format(args.data,
                                                args.features, args.input_len,
                                                args.pred_len, ii)
        print('>>>>>>>testing| pred_len:{} : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.pred_len, setting))
        exp = Exp(args)  # set experiments

        exp.test(setting, load=True, write_loss=True, save_loss=args.save_loss)
        torch.cuda.empty_cache()
        args.learning_rate = lr

path1 = './result.csv'
if not os.path.exists(path1):
    with open(path1, "a") as f:
        write_csv = ['Time', 'Data', 'input_len', 'pred_len', 'MSE', 'MAE']
        np.savetxt(f, np.array(write_csv).reshape(1, -1), fmt='%s', delimiter=',')
        f.flush()
        f.close()

print('>>>>>>>writing results<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
first_setting = '{}_ft{}_ll{}_pl{}_{}'.format(args.data,
                                              args.features, args.input_len,
                                              args.pred_len, 0)
first_folder_path = './results/' + first_setting
num_of_files = len([f for f in os.listdir(first_folder_path) if os.path.isfile(os.path.join(first_folder_path, f))])
num_of_test = num_of_files // 2
print('test windows number: ' + str(num_of_test))

for predl in args.pred_len:
    mses = []
    maes = []
    for i in range(num_of_test):
        pred_total = 0
        true = None
        for ii in range(args.itr):
            setting = '{}_ft{}_ll{}_pl{}_{}'.format(args.data,
                                                    args.features, args.input_len,
                                                    args.pred_len, ii)
            folder_path = './results/' + setting + '/'
            pred_path = folder_path + 'pred_{}.npy'.format(i)
            pred = np.load(pred_path)
            pred_total += pred
            if true is None:
                true_path = folder_path + 'true_{}.npy'.format(i)
                true = np.load(true_path)
        pred = pred_total / args.itr
        mae, mse = metric(pred[:predl, :], true[:predl, :])
        mses.append(mse)
        maes.append(mae)

    mse = np.mean(mses)
    mae = np.mean(maes)
    print('|Mean|mse:{}, mae:{}'.format(mse, mae))
    path = './result.log'
    with open(path, "a") as f:
        f.write('|{}|input_len{}_pred_len{}: '.format(
            args.data, args.input_len, predl) + '\n')
        f.write('mse:{}, mae:{}'.
                format(mse, mae) + '\n')
        f.flush()
        f.close()

    with open(path1, "a") as f:
        f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        f.write(',{},{},{},{},{}'.
                format(args.data, args.input_len, predl
                       , mse, mae) + '\n')
        f.flush()
        f.close()

if not args.save_loss:
    for ii in range(args.itr):
        setting = '{}_ft{}_ll{}_pl{}_{}'.format(args.data,
                                                args.features, args.input_len,
                                                args.pred_len, ii)
        dir_path = os.path.join(args.checkpoints, setting)
        check_path = dir_path + '/' + 'checkpoint.pth'
        if os.path.exists(check_path):
            os.remove(check_path)
            os.removedirs(dir_path)

        folder_path = './results/' + setting
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
