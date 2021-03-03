import os.path as osp
import time
import torchsnooper
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset

from torch_geometric.utils import degree
from torch_geometric.datasets import Planetoid, Coauthor, Amazon,PPI
from torch_geometric.data import DataLoader

from .gnn import GraphNet, GraphNet_GraphNAS
from rlctr.utils.label_split import fix_size_split
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch import cat
import warnings
from rlctr.utils.model_utils import EarlyStop, TopAverage, process_action
warnings.filterwarnings('ignore')
def loader_acc(model, loader,loss_fn):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ys, preds = [], []
    total_loss = 0
    for data_ in loader:
        ys.append(data_.y)
        with torch.no_grad():
            data_ = data_.to(device)
            out = model(data_.x.to(device), data_.edge_index.to(device))
            loss = loss_fn(out, data_.y)
            num_graphs = data_.num_graphs
            total_loss += loss.item() * num_graphs
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro'), total_loss / len(loader.dataset)
def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask
def evaluate_output(output, labels, mask):
    pred = output[mask].max(1)[1]
    acc = pred.eq(labels[mask]).sum().item() / mask.sum().item()
    return acc
def load_data(dataset="Cora", supervised=False, full_data=True):

    '''
    support semi-supervised and supervised
    :param dataset:
    :param supervised:
    :return:
    '''

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    if dataset in ["CS", "Physics"]:
        dataset = Coauthor(path, dataset, T.NormalizeFeatures())
    elif dataset in ["Computers", "Photo"]:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        dataset = Amazon(path, dataset, T.NormalizeFeatures())
    elif dataset in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(path, dataset, split="public")
    elif dataset == 'PPI':
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
        train_dataset = PPI(path, split='train')
        val_dataset = PPI(path, split='val')
        test_dataset = PPI(path, split='test')
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
    data = dataset[0]
    if supervised:
        if full_data:
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.train_mask[:-1000] = 1
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.val_mask[data.num_nodes - 1000: data.num_nodes - 500] = 1
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.test_mask[data.num_nodes - 500:] = 1
        else:
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.train_mask[:1000] = 1
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.val_mask[data.num_nodes - 1000: data.num_nodes - 500] = 1
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.test_mask[data.num_nodes - 500:] = 1
    else:
        print('data_split with 622 split')
        skf = StratifiedKFold(5, shuffle=True)
        idx = [torch.from_numpy(i) for _, i in skf.split(data.y, data.y)]
        split = [cat(idx[:1], 0), cat(idx[1:2], 0), cat(idx[2:], 0)]
        data.train_mask = index_to_mask(split[2], data.num_nodes)
        data.val_mask = index_to_mask(split[1], data.num_nodes)
        data.test_mask = index_to_mask(split[0], data.num_nodes)
    return data



class GeoCitationManager(object):
    def __init__(self, args):
        if hasattr(args, "supervised"):
            self.data = load_data(args.dataset, args.supervised)
        else:
            self.data = load_data(args.dataset)
        self.args = args
        if self.args.dataset in ["Cora", "Citeseer", "Pubmed", "Computers", "Photo", "CS", "Physics"]:
            self.args.in_feats = self.in_feats = self.data.num_features
            self.args.num_class = self.n_classes = self.data.y.max().item() + 1
            device = torch.device('cuda' if args.cuda else 'cpu')
            self.data.to(device)

        self.early_stop_manager = EarlyStop(10)
        self.reward_manager = TopAverage(10)

        self.drop_out = args.in_drop
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.train_graph_index = 0
        self.train_set_length = 10

        self.param_file = args.param_file
        self.shared_params = None

        self.loss_fn = torch.nn.functional.nll_loss

    def evaluate(self, actions=None, format="two"):
        """
        return actions validation acc directly and without training models.
        """
        actions = process_action(actions, format, self.args)
        print("train action:", actions)

        # create model
        model = self.build_gnn(actions)

        try:
            if self.args.cuda:
                model.cuda()
            # use optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            model, val_acc = self.run_model(model, optimizer, self.loss_fn, self.data, epochs=1,
                                            cuda=self.args.cuda, evaluate=True)
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                val_acc = 0
            else:
                raise e
        reward = self.reward_manager.get_reward(val_acc)
        return reward, val_acc

    def train(self, actions=None, format="two", hyperargs=None):
        self.hyperargs = hyperargs
        actions = process_action(actions, format, self.args)
        print("train action:", actions)

        # create model
        model = self.build_gnn(actions)

        try:
            if self.args.cuda:
                model.cuda()
            # use optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            model, val_acc = self.run_model(model, optimizer, self.loss_fn, self.data, self.epochs,
                                            cuda=self.args.cuda)
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                val_acc = 0
            else:
                raise e
        reward = self.reward_manager.get_reward(val_acc)

        return reward, val_acc

    def build_gnn(self, actions):
        if self.args.search_mode == 'graphnas':
            model = GraphNet_GraphNAS(actions, self.in_feats, self.n_classes, drop_out=self.args.in_drop, args=self.args, hyperargs=self.hyperargs)
        else:
            model = GraphNet(actions, self.in_feats, self.n_classes, drop_out=self.args.in_drop, args=self.args, hyperargs=self.hyperargs)

        return model

    def test_with_param(self, actions=None, format="two", with_retrain=False, hyperargs=None):
        self.hyperargs = hyperargs
        return self.train(actions, format, hyperargs)

    def run_model(self, model, optimizer, loss_fn, data, epochs, early_stop=50, return_best=False, cuda=True, need_early_stop=False, show_info=False, evaluate=False):
        if self.hyperargs!=None:
            # use HyperOpt args to fine-tuning derive models.
            lr = 10 ** self.hyperargs['learning_rate']
            w_decay = 10**self.hyperargs['weight_decay']
            if self.hyperargs['optimizer'] == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=w_decay)
            elif self.hyperargs['optimizer'] == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
            elif self.hyperargs['optimizer'] == 'adagrad':
                optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=w_decay)

        if self.args.update_shared == False:
            # training controller over, derive models, without early stop
            early_stop = epochs
        else:
            early_stop = self.args.early_stop_epoch

        dur = []
        begin_time = time.time()
        best_performance = 0
        min_val_loss = float("inf")
        model_val_acc = 0
        print("Number of train data:", data.train_mask.sum())
        early_stop_patient = 0

        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            # forward
            optimizer.zero_grad()
            if evaluate:
                #only get valid acc with fixed params w
                with torch.no_grad():
                    logits = model(data.x, data.edge_index)
                    logits = F.log_softmax(logits, 1)
            else:
                # training models normally
                logits = model(data.x, data.edge_index)
                logits = F.log_softmax(logits, 1)

                loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])

                loss.backward()
                optimizer.step()

            train_acc = evaluate_output(logits, data.y, data.train_mask)
            dur.append(time.time() - t0)

            val_acc = evaluate_output(logits, data.y, data.val_mask)
            test_acc = evaluate_output(logits, data.y, data.test_mask)

            loss = loss_fn(logits[data.val_mask], data.y[data.val_mask])
            val_loss = loss.item()
            if val_loss < min_val_loss:  # and train_loss < min_train_loss
                early_stop_patient = 0
                min_val_loss = val_loss
                model_val_acc = val_acc
                best_performance = test_acc

            else:
                early_stop_patient += 1
                if early_stop_patient >= early_stop:
                    break
            if show_info:
                print(
                    "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                        epoch, loss.item(), np.mean(dur), train_acc, val_acc, test_acc))

        end_time = time.time()
        print("train gnn Cost Time: %.04f " % ((end_time - begin_time) ))
        print(f"val_score:{model_val_acc},test_score:{best_performance},epoch:{epoch}")
        if return_best:
            return model, model_val_acc, best_performance
        else:
            return model, model_val_acc
class GeoCitationManager_PPI(GeoCitationManager):
    def __init__(self, args):
        super(GeoCitationManager_PPI, self).__init__(args)
        self.train_dataset, _, _, self.train_loader, self.val_loader, self.test_loader = load_data(args.dataset, args.supervised)
        self.data = self.train_dataset

        self.args.in_feats = self.in_feats = self.train_dataset.num_features
        self.args.num_class = self.n_classes = self.train_dataset.num_classes
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
    def run_model(self, model, optimizer, loss_fn, data, epochs, early_stop=50, return_best=False, cuda=True, need_early_stop=False, show_info=False, evaluate=False):
        if self.hyperargs != None:
            # use HyperOpt args to fine-tuning derive models.
            lr = 10 ** self.hyperargs['learning_rate']
            w_decay = 10 ** self.hyperargs['weight_decay']
            if self.hyperargs['optimizer'] == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=w_decay)
            elif self.hyperargs['optimizer'] == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
            elif self.hyperargs['optimizer'] == 'adagrad':
                optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=w_decay)

        if self.args.update_shared == False:
            early_stop = epochs
        else:
            early_stop = self.args.early_stop_epoch

        dur = []
        begin_time = time.time()
        best_performance = 0
        min_val_loss = float("inf")

        device = torch.device('cuda' if cuda else 'cpu')

        model_val_acc = 0

        early_stop_patient = 0

        if evaluate:
            val_acc, _ = loader_acc(model, self.val_loader, loss_fn)
            print(' valid_acc:', val_acc)
            return model, val_acc
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs), eta_min=0.005)
        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            # forward
            total_loss = 0

            for data_ in self.train_loader:
                data_ = data_.to(device)
                num_graphs = data_.num_graphs

                optimizer.zero_grad()
                loss = loss_fn(model(data_.x, data_.edge_index), data_.y)
                total_loss += loss.item() * num_graphs
                loss.backward()
                optimizer.step()
            scheduler.step()
            train_loss = total_loss / len(self.train_loader.dataset)
            dur.append(time.time() - t0)

            model.eval()
            train_acc, _ = loader_acc(model, self.train_loader, loss_fn)
            val_acc, val_loss = loader_acc(model, self.val_loader, loss_fn)
            test_acc, test_loss = loader_acc(model, self.test_loader, loss_fn)
            print('train_loss:{},val_loss:{},acc:{},{},{}'.format(train_loss, val_loss, train_acc, val_acc, test_acc))

            if val_loss < min_val_loss:  # and train_loss < min_train_loss
                early_stop_patient = 0
                min_val_loss = val_loss
                model_val_acc = val_acc
                best_performance = test_acc

            else:
                early_stop_patient += 1
                if early_stop_patient >= early_stop:
                    break
            if show_info:
                print(
                    "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                        epoch, val_loss.item(), np.mean(dur), train_acc, val_acc, test_acc))
        end_time = time.time()
        print("train gnn Cost Time: %.04f " % ((end_time - begin_time)))
        print(f"val_score:{model_val_acc},test_score:{best_performance}")
        if return_best:
            return model, model_val_acc, best_performance
        else:
            return model, model_val_acc


