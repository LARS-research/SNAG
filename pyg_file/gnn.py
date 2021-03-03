import torch
import torch.nn.functional as F
from rlctr.search_space import act_map
from .gnn_layer import GeoLayer
from torch_geometric.nn import JumpingKnowledge,SAGEConv,GCNConv,GATConv,GINConv, global_mean_pool
import copy
from torch.nn import Sequential, ReLU, Linear
from .geniepath import GeniePathLayer

class GraphNet(torch.nn.Module):

    def __init__(self, actions, num_feat, num_label, args, drop_out=0.6, state_num=2, hyperargs=None):
        super(GraphNet, self).__init__()

        self.args = args
        self.num_feat = num_feat
        self.num_label = num_label
        self.dropout = drop_out
        self.layer_nums = self.args.layers_of_child_model

        self.use_skip = []
        self.actions = actions
        self.jk_mode = 'none'
        self.hyperargs = hyperargs

        self.build_model(actions, drop_out, num_feat, num_label, state_num)

    def build_model(self, actions, drop_out, num_feat, num_label, state_num):
        self.layers = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()
        self.acts = []
        self.build_hidden_layers(actions, drop_out, self.layer_nums, num_feat, num_label, state_num,
                                 hyperargs=self.hyperargs, out_channels=self.args.gnn_hidden)


    def generate_layer(self,in_channels, out_channels,  head_num, concat, dropout, gnn_method):
        if gnn_method == 'sage':
            ops = SAGEConv(in_channels, out_channels)
        elif gnn_method == 'gcn':
            ops = GCNConv(in_channels, out_channels)
        elif gnn_method == 'gat':
            ops = GATConv(in_channels, int(out_channels/head_num), heads=head_num, dropout=dropout, concat=concat)

        elif gnn_method == 'gin':
            nn1 = Sequential(Linear(in_channels, out_channels), ReLU(), Linear(out_channels, out_channels))
            ops = GINConv(nn1)

        elif gnn_method in ['gat_sym', 'cos', 'linear', 'generalized_linear']:
            ops = GeoLayer(in_channels, int(out_channels/head_num), heads=head_num, att_type=gnn_method, dropout=dropout, concat=concat)
        elif gnn_method in ['geniepath']:
            ops = GeniePathLayer(in_channels, out_channels)
        elif gnn_method in ['sage_max', 'sage_sum']:
            agg = gnn_method.split('_')[-1]
            ops = GeoLayer(in_channels, out_channels, att_type='const', agg_type=agg, dropout=dropout)

        return ops
    def build_hidden_layers(self, actions, drop_out, layer_nums, num_feat, num_label,
                            state_num=2, head_num=8, out_channels=32, hyperargs=None):

        for i in range(2 * layer_nums, 3 * layer_nums - 1):
            self.use_skip.append(actions[i])
        self.use_skip.append(1)
        self.jk_mode = actions[-1]
        head_num = head_num
        out_channels = out_channels

        if hyperargs != None:
            # in derive state,  with hyperopt
            head_num = hyperargs['head_num']
            out_channels = hyperargs['hidden_size']

        for i in range(layer_nums):

            if i == 0:
                in_channels = num_feat
            else:
                in_channels = out_channels

            if i == layer_nums - 1 and self.args.without_jk:
                out_channels = num_label
                head_num = 1

            # extract layer information
            gnn_method = actions[i * state_num + 0]
            act = actions[i * state_num + 1]

            concat = True


            if self.args.shared_params == False:  # without weight sharing
                tmp_geolayer = self.generate_layer(in_channels, out_channels, head_num,
                                                   concat, drop_out, gnn_method)

                linear_op = torch.nn.Linear(in_channels, out_channels)
            else:  # with weight sharing
                key = "%d_%d_%d_%d_%s_%s" % (i, in_channels, out_channels, head_num, concat, gnn_method)
                if key in self.args.shared_parms_dict:
                    if self.args.update_shared == True:
                        print('load geolayer params: ', key)
                        tmp_geolayer = self.args.shared_parms_dict[key]
                        linear_op = self.args.shared_parms_dict['linear'+key]
                    else:
                        print('copy geolayer params: ', key)
                        tmp_geolayer = copy.deepcopy(self.args.shared_parms_dict[key])
                        linear_op = copy.deepcopy(self.args.shared_parms_dict['linear'+key])

                else:
                    if self.args.update_shared == True:
                        print('insert geolayer params: ', key)
                        tmp_geolayer = self.generate_layer(in_channels, out_channels, head_num, concat, drop_out,
                                                           gnn_method)
                        linear_op = torch.nn.Linear(in_channels, out_channels)

                        self.args.shared_parms_dict[key] = tmp_geolayer
                        self.args.shared_parms_dict['linear' + key] = linear_op
                    else:
                        print('create geolayer params but not insert: ', key)
                        tmp_geolayer = self.generate_layer(in_channels, out_channels, head_num,
                                                           concat, drop_out, gnn_method)
                        linear_op = torch.nn.Linear(in_channels, out_channels)
            self.layers.append(tmp_geolayer)
            self.linears.append(linear_op)
            self.acts.append(act_map(act))



        # parameters sharing of classifier.
        if self.args.shared_params == False:  # without weight sharing
            if self.jk_mode == 'concat':
                jk_func = JumpingKnowledge(mode='cat')
            if self.jk_mode == 'maxpool':
                jk_func = JumpingKnowledge(mode='max')
            if self.jk_mode == 'lstm':
                jk_func = JumpingKnowledge(mode='lstm', channels=out_channels,
                                                num_layers=sum(self.use_skip)).cuda()

            if self.jk_mode in ['lstm','maxpool']:
                final_lin = torch.nn.Linear(out_channels, out_channels).cuda()
            else:
                final_lin = torch.nn.Linear(sum(self.use_skip) * out_channels, out_channels).cuda()

            classifier = torch.nn.Linear(out_channels, self.num_label).cuda()


        else:
            key = ''
            for jk in self.use_skip:
                key += str(jk)
            key += self.jk_mode
            key += str(out_channels)
            key = str(key)
            if str('jk_'+key) in self.args.shared_parms_dict:
                if self.args.update_shared == True:
                    print('load geolayer params: ', key)
                    jk_func = self.args.shared_parms_dict['jk_' + key]
                    final_lin = self.args.shared_parms_dict['linear_' + key]
                    classifier = self.args.shared_parms_dict['cls_' + key]
                else:
                    print('copy geolayer params: ', key)
                    jk_func = copy.deepcopy(self.args.shared_parms_dict['jk_' + key])
                    final_lin = copy.deepcopy(self.args.shared_parms_dict['linear_' + key])
                    classifier = copy.deepcopy(self.args.shared_parms_dict['cls_' + key])
            else:
                if self.jk_mode == 'concat':
                    jk_func = JumpingKnowledge(mode='cat')
                if self.jk_mode == 'maxpool':
                    jk_func = JumpingKnowledge(mode='max')
                if self.jk_mode == 'lstm':
                    jk_func = JumpingKnowledge(mode='lstm', channels=out_channels,
                                                    num_layers=sum(self.use_skip)).cuda()
                if self.jk_mode in ['lstm', 'maxpool']:
                    final_lin = torch.nn.Linear(out_channels, out_channels).cuda()
                else:
                    final_lin = torch.nn.Linear(sum(self.use_skip) * out_channels, out_channels).cuda()

                classifier = torch.nn.Linear(out_channels, self.num_label).cuda()


                if self.args.update_shared == True:
                    print('insert geolayer params: ', key)
                    self.args.shared_parms_dict['jk_' + key] = jk_func
                    self.args.shared_parms_dict['linear_' + key] = final_lin
                    self.args.shared_parms_dict['cls_' + key] = classifier
                else:
                    print('create geolayer params but not insert: ', key)


        self.jk_func = jk_func
        self.final_lin = final_lin
        self.classifier = classifier

    def forward(self, x, edge_index_all):
        output = x
        final_output = []
        for i, (act, layer, linear) in enumerate(zip(self.acts, self.layers,self.linears)):

            output = F.dropout(output, p=self.dropout, training=self.training)
            if self.args.dataset =='PPI':
                if i == self.layer_nums - 1:
                    output = layer(output, edge_index_all) + linear(output)
                else:
                    output = act(layer(output, edge_index_all) + linear(output))
            else: #transductive datasets
                    output = act(layer(output, edge_index_all))
            if self.args.ln:
                layer_norm = torch.nn.LayerNorm(normalized_shape=output.size(), elementwise_affine=False)
                output = layer_norm(output)

            if self.use_skip[i]:
                final_output += [output]
        if self.args.without_jk == False:
            final_output = self.jk_func(final_output)
            output = self.final_lin(final_output)
        else:
            output = self.final_lin(output)
            print('without_jk=True, no skip in this.')
        output = F.dropout(output, p=0.6, training=self.training)
        output = self.classifier(output)

        return output


class GraphNet_GraphNAS(GraphNet):
    def __init__(self, actions, num_feat, num_label, args, drop_out=0.6, state_num=5, hyperargs=None):
        super(GraphNet_GraphNAS, self).__init__(actions, num_feat, num_label, args, drop_out=drop_out, state_num=state_num, hyperargs=hyperargs)

    def build_hidden_layers(self, actions, drop_out, layer_nums, num_feat, num_label,
                            state_num=5, head_num=8, out_channels=32, hyperargs=None):
        state_num = 5
        for i in range(layer_nums):
            if i == 0:
                in_channels = num_feat
            else:
                in_channels = out_channels * head_num

            attention_type = actions[i * state_num + 0]
            aggregator_type = actions[i * state_num + 1]
            act = actions[i * state_num + 2]
            head_num = actions[i * state_num + 3]
            out_channels = actions[i * state_num + 4]
            concat = True
            if i == layer_nums - 1:
                concat = False


            if hyperargs != None:
                # in derive state,  with hyperopt
                head_num = hyperargs['head_num']
                out_channels = hyperargs['hidden_size']


            if self.args.shared_params == False:  # without weight sharing
                tmp_geolayer = GeoLayer(in_channels, out_channels, head_num, concat, dropout=self.dropout,
                               att_type=attention_type, agg_type=aggregator_type, )

                if concat==True:
                    linear_op = torch.nn.Linear(in_channels, out_channels*head_num)
                else:
                    linear_op = torch.nn.Linear(in_channels, out_channels)

            else:  # with weight sharing
                key = "%d_%d_%d_%d_%s_%s_%s" % (i, in_channels, out_channels, head_num, concat, attention_type, aggregator_type)
                if key in self.args.shared_parms_dict:
                    if self.args.update_shared == True:
                        print('load geolayer params: ', key)
                        tmp_geolayer = self.args.shared_parms_dict[key]
                        linear_op = self.args.shared_parms_dict['linear_'+key]

                    else:
                        print('copy geolayer params: ', key)
                        tmp_geolayer = copy.deepcopy(self.args.shared_parms_dict[key])
                        linear_op = copy.deepcopy(self.args.shared_parms_dict['linear_'+key])

                else:
                    if self.args.update_shared == True:
                        print('insert geolayer params: ', key)
                        tmp_geolayer = GeoLayer(in_channels, out_channels, head_num, concat, dropout=self.dropout,
                               att_type=attention_type, agg_type=aggregator_type, )

                        if concat == True:
                            linear_op = torch.nn.Linear(in_channels, out_channels * head_num)
                        else:
                            linear_op = torch.nn.Linear(in_channels, out_channels)

                        self.args.shared_parms_dict[key] = tmp_geolayer
                        self.args.shared_parms_dict['linear_'+key] = linear_op
                    else:
                        print('create geolayer params but not insert: ', key)
                        tmp_geolayer = GeoLayer(in_channels, out_channels, head_num, concat, dropout=self.dropout,
                               att_type=attention_type, agg_type=aggregator_type, )
                        if concat == True:
                            linear_op = torch.nn.Linear(in_channels, out_channels * head_num)
                        else:
                            linear_op = torch.nn.Linear(in_channels, out_channels)

            self.layers.append(tmp_geolayer)
            self.acts.append(act_map(act))
            self.linears.append(linear_op)

        #classifier
        if str('final_lin'+ str(out_channels)) in self.args.shared_parms_dict:
            final_lin = self.args.shared_parms_dict['final_lin' + str(out_channels)]
            classifier = self.args.shared_parms_dict['classifier' + str(out_channels)]
        else:
            final_lin = torch.nn.Linear(out_channels, out_channels).cuda()
            classifier = torch.nn.Linear(out_channels, self.num_label).cuda()
            self.args.shared_parms_dict['final_lin' + str(out_channels)] = final_lin
            self.args.shared_parms_dict['classifier' + str(out_channels)] = classifier

        self.classifier = classifier
        self.final_lin = final_lin

    def forward(self, x, edge_index_all):
        output = x

        for i, (act, layer, linear) in enumerate(zip(self.acts, self.layers, self.linears)):

            output = F.dropout(output, p=self.dropout, training=self.training)
            if i == self.layer_nums - 1:
                output = layer(output, edge_index_all)
            else:
                output = act(layer(output, edge_index_all))

        output = F.elu(self.final_lin(output))
        output = F.dropout(output, p=0.6, training=self.training)
        output = self.classifier(output)

        return output

