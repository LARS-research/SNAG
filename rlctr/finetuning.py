"""Entry point."""

import argparse
import time
import re
import torch
# import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import graphnas.trainer as trainer
import graphnas.utils.tensor_utils as utils


def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    register_default_args(parser)
    args = parser.parse_args()

    return args


def register_default_args(parser):
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'derive', 'random', 'bayes','mixed','multi','finetuning'],
                        help='train: Training GraphNAS, derive: Deriving Architectures')
    parser.add_argument('--mixed_random_ratio', type=float, default=0.0)
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    parser.add_argument('--save_epoch', type=int, default=2)
    parser.add_argument('--max_save_num', type=int, default=5)

    # controller
    parser.add_argument('--time_budget', type=float, default=5.0, help='time budget(h) for training controller.')
    parser.add_argument('--layers_of_child_model', type=int, default=3)
    parser.add_argument('--shared_initial_step', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])
    parser.add_argument('--entropy_coeff', type=float, default=1e-4)
    parser.add_argument('--shared_rnn_max_length', type=int, default=35)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--search_mode', type=str, default='macro')
    parser.add_argument('--format', type=str, default='two')
    parser.add_argument('--max_epoch', type=int, default=30)

    parser.add_argument('--ema_baseline_decay', type=float, default=0.95)
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--controller_max_step', type=int, default=1,
                        help='step for controller parameters')
    parser.add_argument('--controller_optim', type=str, default='adam')
    parser.add_argument('--controller_lr', type=float, default=3.5e-4,
                        help="will be ignored if --controller_lr_cosine=True")
    parser.add_argument('--controller_grad_clip', type=float, default=0)
    parser.add_argument('--tanh_c', type=float, default=2.5)
    parser.add_argument('--softmax_temperature', type=float, default=5.0)
    parser.add_argument('--derive_num_sample', type=int, default=10)
    parser.add_argument('--hyper_eval_inters', type=int, default=30)
    parser.add_argument('--derive_finally', type=bool, default=True)
    parser.add_argument('--derive_from_history', type=bool, default=True)
    parser.add_argument('--controller_hid', type=int, default=100)

    # child model
    parser.add_argument("--only_one_act_funtion", type=bool, default=False,
                        help="epoch that valid loss bot decrease.")
    parser.add_argument("--early_stop_epoch", type=int, default=50,
                        help="epoch that valid loss bot decrease.")
    parser.add_argument("--shared_params", type=bool, default=False,
                        help="shared_params between child model.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu number")
    # parser.add_argument("--use_jk", type=bool, default=True,
    #                     help="use_jk: add jk in graphnas ")
    parser.add_argument("--without_jk", type=bool, default=False,
                        help="without_jk: remove jk in graphnas ")
    parser.add_argument("--dataset", type=str, default="Citeseer", required=False,
                        help="The input dataset.")
    parser.add_argument("--epochs", type=int, default=800,
                        help="number of training epochs")
    parser.add_argument("--retrain_epochs", type=int, default=800,
                        help="number of training epochs")
    parser.add_argument("--multi_label", type=bool, default=False,
                        help="multi_label or single_label task")
    parser.add_argument("--residual", action="store_false",
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0.6,
                        help="input feature dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--param_file", type=str, default="cora_test.pkl",
                        help="learning rate")
    parser.add_argument("--optim_file", type=str, default="opt_cora_test.pkl",
                        help="optimizer save path")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--max_param', type=float, default=5E6)
    parser.add_argument('--supervised', type=bool, default=False)
    parser.add_argument('--submanager_log_file', type=str, default=f"sub_manager_logger_file_{time.time()}.txt")


def main(args):  # pylint:disable=redefined-outer-name

    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    # args.max_epoch = 1
    # args.controller_max_step = 1
    # args.derive_num_sample = 1
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    utils.makedirs(args.dataset)

    trnr = trainer.Trainer(args)

    if args.mode == 'train':
        print(args)
        trnr.train()
    elif args.mode == 'finetuning':
        print(args)
        trnr.finetuning(actions=actions, hyper=hyper, num=5)
    elif args.mode in ['random','bayes']:
        print(args)
        trnr.random_bayes_search(mode=args.mode)
    elif args.mode =='mixed':
        print(args)
        trnr.mixed_train(random_ratio=args.mixed_random_ratio,max_train_epoch=500)
    elif args.mode == 'multi':
        print(args)
        trnr.mixed_train_multi(random_ratio=args.mixed_random_ratio,max_train_epoch=500)
    else:
        raise Exception(f"[!] Mode not found: {args.mode}")

def process_actions_hyper(actions, hyper):
    #hyper
    actions=actions[1:-1]
    tmp = actions.split(',')
    actions = []
    for i in tmp:
        i = i.lstrip()
        if i == '1' or i == '0':
            actions.append(int(i))
        else:
            actions.append(i[1:-1])

    #hyper
    hyper = hyper[1:-1]
    tmp = re.split(':|,', hyper)
    hyper = {}
    hyper['head_num'] = int(tmp[1])
    hyper['hidden_size'] = int(tmp[3])
    hyper['learning_rate'] = float(tmp[5])
    hyper['optimizer'] = tmp[7].lstrip()[1:-1]
    hyper['weight_decay'] = float(tmp[9])
    return actions, hyper
if __name__ == "__main__":
    args = build_args()
    if args.mode !='finetuning':
        main(args)
    else:

        actions_random_citeseer=[['gcn', 'tanh', 'linear', 'sigmoid', 'gat', 'tanh', 1, 1, 'concat'],
                                 ['generalized_linear', 'relu', 'gcn', 'sigmoid', 'gcn', 'sigmoid', 1, 1, 'concat']]

        actions_citeseer_ENAS_ws=[['gcn', 'elu', 'linear', 'sigmoid', 'sage_sum', 'relu', 1, 0, 'concat'],
                                  ['gcn', 'relu', 'gcn', 'relu6', 'linear', 'leaky_relu', 1, 0, 'lstm'],
                                  ['gat_sym', 'leaky_relu', 'gat_sym', 'tanh', 'linear', 'softplus', 1, 1, 'concat']
                                  ]
        actions_random_pubmed=[['gcn', 'relu6', 'gin', 'softplus', 'generalized_linear', 'softplus', 0, 1, 'concat'],
                               ['geniepath', 'leaky_relu', 'gcn', 'tanh', 'gat_sym', 'sigmoid', 1, 1, 'lstm'],
                               ['generalized_linear', 'elu', 'gat_sym', 'softplus', 'gat_sym', 'leaky_relu', 1, 1, 'lstm']]

        actions_pubmed_ENAS_ws=[['generalized_linear', 'leaky_relu', 'gin', 'softplus', 'sage', 'linear', 0, 1, 'lstm'],
                                ['gcn', 'relu6', 'gin', 'tanh', 'generalized_linear', 'tanh', 1, 0, 'concat'],
                                ['cos', 'elu', 'gat', 'relu6', 'sage', 'linear', 1, 0, 'maxpool']]
        actions_ppi_ENAS_ws=[['gat', 'elu', 'geniepath', 'elu', 'gcn', 'elu', 1, 1, 'lstm'],
                             ['generalized_linear', 'elu', 'gin', 'elu', 'gat', 'elu', 1, 0, 'maxpool'],
                             ['gin', 'elu', 'gat_sym', 'elu', 'sage', 'elu', 0, 1, 'lstm']]
        global actions
        global hyper

        print('*'*50,'Pubmed','*'*50)
        for i in range(3):
            args.dataset = 'Pubmed'
            actions = actions_pubmed_ENAS_ws[i]
            for j in range(30):
                hyper = {}
                hyper['head_num'] = np.random.choice([1, 2, 4, 8])
                hyper['hidden_size'] = np.random.choice([8, 16, 32, 48, 64, 128])
                hyper['learning_rate'] = np.random.uniform(-5, -2)
                hyper['weight_decay'] = np.random.uniform(-5, -3)
                hyper['optimizer'] = np.random.choice(['adagrad', 'adam'])

                print('actions:{}/{}, process:{}/{}'.format(i+1, 5, j+1, 30))
                print('actions:{}'.format(actions))
                print('hyper:{}'.format(hyper))
                args.mode = 'finetuning'
                try:
                    main(args)
                except:
                    continue










