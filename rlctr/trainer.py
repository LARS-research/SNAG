
import time
import numpy as np
import scipy.signal
import torch

import warnings
warnings.filterwarnings('ignore')
import rlctr.utils.tensor_utils as utils
from pyg_file.model_manager import GeoCitationManager,GeoCitationManager_PPI
import hyperopt
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK, rand

logger = utils.get_logger()
def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]
history = []


def scale(value, last_k=10, scale_value=1):
    '''
    scale value into [-scale_value, scale_value], according last_k history
    '''
    max_reward = np.max(history[-last_k:])
    if max_reward == 0:
        return value
    return scale_value / max_reward * value


def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim


class Trainer(object):
    """Manage the training process"""

    def __init__(self, args):
        """
        Constructor for training algorithm.
        Build sub-model manager and controller.
        Build optimizer and cross entropy loss for controller.

        Args:
            args: From command line, picked up by `argparse`.
        """
        self.args = args
        self.controller_step = 0  # counter for controller
        self.cuda = args.cuda
        self.epoch = 0
        self.start_epoch = 0

        self.max_length = self.args.shared_rnn_max_length

        self.with_retrain = False
        self.submodel_manager = None
        self.controller = None
        self.build_model()  # build controller and sub-model

        controller_optimizer = _get_optimizer(self.args.controller_optim)
        self.controller_optim = controller_optimizer(self.controller.parameters(), lr=self.args.controller_lr)
        if self.args.mode == "derive":
            self.load_model()

    def build_model(self):
        self.with_retrain = True
        if self.args.search_mode == 'graphnas': # GraphNAS
            from rlctr.search_space import GraphNAS_SearchSpace
            search_space_cls = GraphNAS_SearchSpace()
            self.search_space = search_space_cls.get_search_space()
            self.action_list = search_space_cls.generate_action_list(self.args.layers_of_child_model)
            if self.args.dataset == 'PPI':
                # OOM
                self.search_space['hidden_units'] = [4, 8, 16, 32, 64, 128]
                self.search_space['number_of_heads'] = [1, 2, 4, 6, 8, 16]

        else:  #SNAG
            from rlctr.search_space import MacroSearchSpace
            search_space_cls = MacroSearchSpace()
            self.search_space = search_space_cls.get_search_space()
            self.action_list = search_space_cls.generate_action_list(self.args.layers_of_child_model)

        # build RNN controller
        from rlctr.graphnas_controller import SimpleNASController

        self.controller = SimpleNASController(self.args, action_list=self.action_list,
                                              search_space=self.search_space,
                                              cuda=self.args.cuda, controller_hid=self.args.controller_hid)

        if self.args.dataset in ["Cora", "Citeseer", "Pubmed", "Computers", "Photo", "CS", "Physics"]:
            self.submodel_manager = GeoCitationManager(self.args)
        elif self.args.dataset == 'PPI':
            self.submodel_manager = GeoCitationManager_PPI(self.args)
        else:
            print('Dataset error!')
            exit()

        if self.cuda:
            self.controller.cuda()

    def form_gnn_info(self, gnn):
         return gnn

    def rb_generate_actions(self,args):
        gnn = []
        for i in range(self.args.layers_of_child_model):
            gnn.append(args['op'+str(i)])
            gnn.append(args['act'+str(i)])
        gnn.append(args['jk0'])
        gnn.append(args['jk1'])
        gnn.append(args['jk_mode'])
        return gnn
    def random_bayes_search(self, mode='random', max_evals=5000):
        rb_search_space = {}
        for i in range(self.args.layers_of_child_model):
            rb_search_space['op' + str(i)] = hp.choice('op' + str(i), self.search_space["gnn_method"])
            rb_search_space['act' + str(i)] = hp.choice('act' + str(i), self.search_space['activate_function'])
        rb_search_space['jk0'] = hp.choice('jk0', self.search_space['use_skip'])
        rb_search_space['jk1'] = hp.choice('jk1', self.search_space['use_skip'])
        rb_search_space['jk_mode'] = hp.choice('jk_mode', self.search_space['jk_mode'])

        def objective_rb_models(args):
            gnn = self.rb_generate_actions(args)
            reward = self.submodel_manager.test_with_param(gnn, format=self.args.format,
                                                           with_retrain=self.with_retrain)
            valid_acc = reward[-1]
            return {'loss': -valid_acc, 'status': STATUS_OK}

        # start training:
        self.args.shared_params = False #share parameters or not.
        self.args.update_shared = True #update shared params in training stage

        print("*" * 35, "training controller({} search)".format(mode), "*" * 35)

        trials = Trials()
        if mode == 'random':
            best = fmin(objective_rb_models, rb_search_space, algo=rand.suggest, max_evals=max_evals, trials=trials)
        elif mode == 'bayes':
            n_startup_jobs = int(max_evals/5)
            best = fmin(objective_rb_models, rb_search_space, algo=partial(tpe.suggest, n_startup_jobs=n_startup_jobs),
                    max_evals=max_evals, trials=trials)
        print("*" * 35, "training controller over ({} search)".format(mode), "*" * 35)
        best_actions = hyperopt.space_eval(rb_search_space, best)
        print('beat actions:', best_actions)
        best_actions = self.rb_generate_actions(best_actions)
        print('beat actions:', best_actions)

        # hyper-parameter finetuning
        global hyper_space
        hyper_space = {'head_num': hp.choice('head_num', [1, 2, 4, 8]),
                       'hidden_size': hp.choice('hidden_size', [8, 16, 32, 48, 64, 128, 256]),
                       'learning_rate': hp.uniform("lr", -5, -2),
                       'weight_decay': hp.uniform("wr", -5, -3),
                       'optimizer': hp.choice('opt', ['adagrad', 'adam']),
                       }
        if self.args.dataset in ["Pubmed", "Computers", "CS"]:
            hyper_space['head_num'] = hp.choice('head_num', [1, 2, 4])
            hyper_space['hidden_size'] = hp.choice('hidden_size', [8, 16, 32, 48, 64])

        def objective(args):
            print('###current hyper:', args)
            reward = self.submodel_manager.test_with_param(best_actions, format=self.args.format,
                                                           with_retrain=self.with_retrain, hyperargs=args)
            valid_acc = reward[-1]
            return {'loss': -valid_acc, 'status': STATUS_OK}

        try:
            trials = Trials()
            n_startup_jobs = int(self.args.hyper_eval_inters/5)
            # bayes hyperopt

            best = fmin(objective, hyper_space, algo=partial(tpe.suggest, n_startup_jobs=n_startup_jobs),
                        max_evals=self.args.hyper_eval_inters, trials=trials)


            hyper = hyperopt.space_eval(hyper_space, best)
            # cal std.
            self.finetuning(best_actions, hyper, num=5)

        except RuntimeError as e:
            if 'CUDA' in str(e):  # usually CUDA Out of Memory
                print(e)
        hyper = hyperopt.space_eval(hyper_space, best)
        print('best_hyper:', hyper)

    def finetuning(self, actions, hyper, num):
        self.args.shared_params = False
        self.args.update_shared = False
        for i in range(num):
            self.submodel_manager.test_with_param(actions, format=self.args.format, hyperargs=hyper)

    def train(self):
        """
        Each epoch consists of two phase:
        - In the first phase, shared parameters are trained to exploration.
        - In the second phase, the controller's parameters are trained.
        """
        global hyper_space
        hyper_space = {'head_num':hp.choice('head_num', [1, 2, 4, 8]),
                        'hidden_size':hp.choice('hidden_size', [32, 64, 128, 256, 512]),
                       'learning_rate': hp.uniform("lr", -5, -2),
                       'weight_decay': hp.uniform("wr", -5, -3),
                       'optimizer': hp.choice('opt', ['adagrad', 'adam']),
                      }
        if self.args.dataset in ["Pubmed", "Computers", "CS"]:
            hyper_space['head_num'] = hp.choice('head_num', [1, 2, 4])
            hyper_space['hidden_size'] = hp.choice('hidden_size',[8,16,32,48,64])
        self.args.shared_parms_dict = {}
        self.args.update_shared = True  # update shared parms dict.
        epoch = 0

        while epoch <= self.args.train_epochs:
            start_time = time.time()

            print("*" * 35, "training shared weights", "*" * 35)
            print('train_shared_step: ', epoch)
            try:
                self.args.update_shared = True #in training controller stage, use fixed params.
                self.train_shared(max_step=self.args.shared_initial_step)
            except RuntimeError as e:
                if 'CUDA' in str(e):  # usually CUDA Out of Memory
                    print(e)


            shared_end_time = time.time()
            shared_cost_time = (shared_end_time - start_time)
            print('\n')
            print('train_shared_step: {}, Cost time: {}'.format(epoch, shared_cost_time))


            print("*" * 35, "training controller", "*" * 35)
            print('controller_step: ', epoch)

            try:
                self.args.update_shared = False #in training controller stage, use fixed params.
                self.train_controller()
            except RuntimeError as e:
                if 'CUDA' in str(e):  # usually CUDA Out of Memory
                    print(e)

            epoch += 1
            controller_end_time = time.time()
            controller_cost_time = (controller_end_time-shared_end_time)

        #self.save_model()

        print("*" * 35, "Training over ", "*" * 35)
        print("*" * 35, "Finetuning", "*" * 35)
        # get final performance
        self.args.update_shared = False #fiexed the params
        best_actions = self.derive(sample_num=self.args.derive_num_sample)
        # best_actions = self.derive_enas(sample_num=self.args.derive_num_sample)

        def objective(args):
            print('####current hyper:', args)
            reward = self.submodel_manager.test_with_param(best_actions, format=self.args.format,
                                                           with_retrain=self.with_retrain, hyperargs=args)
            valid_acc = reward[-1]
            return {'loss': -valid_acc, 'status': STATUS_OK}
        try:
            trials = Trials()
            n_startup_jobs = int(self.args.hyper_eval_inters/5)
            #bayes hyperopt
            best = fmin(objective, hyper_space, algo=partial(tpe.suggest, n_startup_jobs=n_startup_jobs),
                        max_evals=self.args.hyper_eval_inters, trials=trials)
            # best: best hyper parameters.

            hyper = hyperopt.space_eval(hyper_space, best)
            print('best_hyper:', hyper)
            reward = self.submodel_manager.test_with_param(best_actions, format=self.args.format,
                                                           with_retrain=self.with_retrain, hyperargs=hyper)

        except RuntimeError as e:
            if 'CUDA' in str(e):  # usually CUDA Out of Memory
                print(e)
        hyper = hyperopt.space_eval(hyper_space, best)
        print('best_hyper:', hyper)
        reward = self.submodel_manager.test_with_param(best_actions, format=self.args.format,
                                                       with_retrain=self.with_retrain, hyperargs=hyper)
        #get mean and std
        print("*" * 35, "hyperopt over", "*" * 35)
        print("*" * 35, "Cal std", "*" * 35)
        self.finetuning(best_actions, hyper, num=5)



    def train_shared(self, max_step=50, gnn_list=None):
        """
        Args:
            max_step: Used to run extra training steps as a warm-up.
            gnn: If not None, is used instead of calling sample().

        """
        if max_step == 0 or self.args.shared_params == False:  # no train shared
            return

        gnn_list = gnn_list if gnn_list else self.controller.sample(max_step)

        for gnn in gnn_list:
            try:
                _, val_score = self.submodel_manager.train(gnn, format=self.args.format)
                logger.info(f"{gnn}, val_score:{val_score}")
                print('\n')
                print('\n')
            except RuntimeError as e:
                if 'CUDA' in str(e):  # usually CUDA Out of Memory
                    print(e)
                else:
                    raise e



    def get_reward(self, gnn_list, entropies, hidden):
        """
        Computes the reward of a single sampled model on validation data.
        """
        if not isinstance(entropies, np.ndarray):
            entropies = entropies.data.cpu().numpy()
        if isinstance(gnn_list, dict):
            gnn_list = [gnn_list]
        if isinstance(gnn_list[0], list) or isinstance(gnn_list[0], dict):
            pass
        else:
            gnn_list = [gnn_list]  # when structure_list is one structure

        reward_list = []
        for gnn in gnn_list:
            gnn = self.form_gnn_info(gnn)

            reward = self.evaluate(gnn)

            if reward is None:  # cuda error hanppened
                reward = 0
            else:
                reward = reward[1] #reward[1]:val_acc

            reward_list.append(reward)

        if self.args.entropy_mode == 'reward':
            rewards = reward_list + self.args.entropy_coeff * entropies
        elif self.args.entropy_mode == 'regularizer':
            rewards = reward_list * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')

        return rewards, hidden



    def train_controller(self):
        """
            Train controller to find better structure.
        """
        model = self.controller
        model.train()

        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []

        hidden = self.controller.init_hidden(self.args.batch_size)
        total_loss = 0
        for step in range(self.args.controller_max_step):
            # sample graphnas
            start_time = time.time()
            start_time_str = time.strftime("%H:%M:%S")
            structure_list, log_probs, entropies = self.controller.sample(with_details=True)

            # calculate reward
            np_entropies = entropies.data.cpu().numpy()
            results = self.get_reward(structure_list, np_entropies, hidden)
            torch.cuda.empty_cache()

            if results:  # has reward
                rewards, hidden = results
            else:
                continue  # CUDA Error happens, drop structure and step into next iteration

            # discount
            if 1 > self.args.discount > 0:
                rewards = discount(rewards, self.args.discount)

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            history.append(adv)
            adv = scale(adv, scale_value=0.5)
            adv_history.extend(adv)

            adv = utils.get_variable(adv, self.cuda, requires_grad=False)
            # policy loss
            loss = -log_probs * adv
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies

            loss = loss.sum()  # or loss.mean()

            # update
            self.controller_optim.zero_grad()
            loss.backward()

            if self.args.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              self.args.controller_grad_clip)
            self.controller_optim.step()

            total_loss += utils.to_item(loss.data)

            self.controller_step += 1
            torch.cuda.empty_cache()
            end_time = time.time()
            print('start_time:{},end_time:{}, step_time{:.04f}s,'.format(
                start_time_str, time.strftime("%H:%M:%S"), end_time-start_time))
            print('\n')


    def evaluate(self, gnn):
        """
        get validation accuracy with fixed params w.

        """
        self.controller.eval()
        gnn = self.form_gnn_info(gnn)

        if self.args.shared_params: # weight sharing schema.
            results = self.submodel_manager.evaluate(gnn, format=self.args.format)
        else:
            results = self.submodel_manager.test_with_param(gnn, format=self.args.format)
        if results:
            return results
        else:
            return



    def derive(self, sample_num=None):
        """
        sample a serial of structures, and return the best structure with shared params.
        """
        if sample_num is None:
            sample_num = self.args.derive_num_sample

        gnn_list, _, entropies = self.controller.sample(sample_num, with_details=True)

        max_R = 0
        best_actions = None
        filename = self.model_info_filename

        # action_step = 0
        for derive_num in range(sample_num):
            start_time = time.time()
            start_time_str = time.strftime("%H:%M:%S")
            gnn = gnn_list[derive_num]

            print('#'*10, 'derive_process:{}/{}'.format(derive_num, sample_num), '#'*10)

            def objective(args):
                print('####current hyper:', args)
                reward = self.submodel_manager.test_with_param(gnn, format=self.args.format, with_retrain=self.with_retrain, hyperargs=args)
                valid_acc = reward[-1]
                return {'loss': -valid_acc, 'status': STATUS_OK}
            # try:
            trials = Trials()
            best = fmin(objective, hyper_space, algo=rand.suggest, max_evals=self.args.hyper_eval_inters, trials=trials)

            hyper = hyperopt.space_eval(hyper_space, best)
            print('best_hyper:', hyper)
            reward = self.submodel_manager.test_with_param(gnn, format=self.args.format,
                                                               with_retrain=self.with_retrain, hyperargs=hyper)



            if reward is None:  # cuda error hanppened
                continue
            else:
                results = reward[1] #valid acc.

            if results > max_R:
                max_R = results
                best_actions = gnn
                best_hyper = hyper
                end_time = time.time()
                end_time_str = time.strftime("%H:%M:%S")
                print('derive_step:{}, start_time:{}, end_time:{}, step_time{:.04f}s,'.format(derive_num, start_time_str, end_time_str, end_time - start_time))
                print('\n')
        logger.info(f'derive |action:{best_actions} |max_R: {max_R:8.6f}')
        print('derive|actions:{},hyper:{},max_R(max_val_acc):{:.6f},val_acc):{:.6f}'.format(best_actions, best_hyper, max_R, reward[1]))
        return best_actions


    @property
    def model_info_filename(self):
        return f"{self.args.dataset}_{self.args.search_mode}_{self.args.format}_results.txt"

    @property
    def controller_path(self):
        return f'{self.args.dataset}/controller_epoch{self.epoch}_step{self.controller_step}.pth'

    @property
    def controller_optimizer_path(self):
        return f'{self.args.dataset}/controller_epoch{self.epoch}_step{self.controller_step}_optimizer.pth'

    def get_saved_models_info(self):
        paths = glob.glob(os.path.join(self.args.dataset, '*.pth'))
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                name.split(delimiter)[idx].replace(replace_word, ''))
                for name in items if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 1)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 1, 'epoch')
        shared_steps = get_numbers(basenames, '_', 2, 'step', 'shared')
        controller_steps = get_numbers(basenames, '_', 2, 'step', 'controller')

        epochs.sort()
        shared_steps.sort()
        controller_steps.sort()

        return epochs, shared_steps, controller_steps

    def save_model(self):

        torch.save(self.controller.state_dict(), self.controller_path)
        torch.save(self.controller_optim.state_dict(), self.controller_optimizer_path)

        logger.info(f'[*] SAVED: {self.controller_path}')

        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        for epoch in epochs[:-self.args.max_save_num]:
            paths = glob.glob(
                os.path.join(self.args.dataset, f'*_epoch{epoch}_*.pth'))

            for path in paths:
                utils.remove_file(path)

