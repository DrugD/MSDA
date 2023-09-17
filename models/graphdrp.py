import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool


class GNN(nn.Module):
    def __init__(self,
                 input,
                 output,
                 gnn_type,
                 heads=1,
                 dropout=0.2,
                 feature_pre_dropout=0,
                 activate_func='relu'):
        super(GNN, self).__init__()

        self.gnn_type = gnn_type
        
        if feature_pre_dropout>0:
            self.pre_dropout = nn.Dropout(feature_pre_dropout)
            
        if self.gnn_type == 'GINConvNet':
            nn_core = Sequential(Linear(input, output),
                                 ReLU(), Linear(output, output))
            self.gnn = GINConv(nn_core)
            self.bn = torch.nn.BatchNorm1d(output)
        elif self.gnn_type == 'GCNConv':
            self.gnn = GCNConv(input, output)
        elif self.gnn_type == 'GATConv':
            self.gnn = GATConv(input, output, heads=heads, dropout=dropout)

        if activate_func == 'relu':
            self.activate_func = nn.ReLU()
        elif activate_func == 'elu':
            self.activate_func = nn.ELU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if hasattr(self, 'pre_dropout'):
            x = self.pre_dropout(x)
        
        if self.gnn_type == 'GINConvNet':
            x = self.gnn(x, edge_index)
            x = self.activate_func(x)
            x = self.bn(x)
        elif self.gnn_type == 'GCNConv':
            x = self.gnn(x, edge_index)
            x = self.activate_func(x)
        elif self.gnn_type == 'GATConv':
            x = self.gnn(x, edge_index)
            x = self.activate_func(x)

        data.x = x

        return data


class Drug(nn.Module):
    def __init__(self,
                 module_name,
                 input_drug_feature_dim,
                 output_drug_feature_dim,
                 layer_num,
                 graph_pooling,
                 linear_layers,
                 gnn_layers,
                 dropout):
        super(Drug, self).__init__()

        assert len(
            gnn_layers) == layer_num, 'Number of layer is not same as hyperparameter list.'
        assert graph_pooling in [
            'add', 'max', 'mean', 'max_mean'], 'The type of graph pooling is not right.'

        self.gnn_layers = gnn_layers
        self.linear_layers = linear_layers
        self.graph_pooling = graph_pooling
        self.backbone = nn.Sequential()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        for index, params in enumerate(gnn_layers):
            if module_name[index] == "GATConv":
                self.backbone.add_module(
                    '{0}-{1}'.format(module_name[index], index), GNN(params['intput'], params['output'], module_name[index], heads=params['heads'], dropout=params['dropout'], feature_pre_dropout=params['feature_pre_dropout']))
            else:
                self.backbone.add_module(
                    '{0}-{1}'.format(module_name[index], index), GNN(params['intput'], params['output'], module_name[index]))

        if linear_layers:
            self.linears = nn.Sequential()

            for idx, linear_parameter in enumerate(linear_layers):

                if linear_parameter['operate_name'] == 'linear':
                    self.linears.add_module(
                        '{0}-{1}'.format(linear_parameter['operate_name'], idx), torch.nn.Linear(linear_parameter['param'][0], linear_parameter['param'][1]))

                elif linear_parameter['operate_name'] == 'relu':
                    self.linears.add_module(
                        '{0}-{1}'.format(linear_parameter['operate_name'], idx), self.relu)

                elif linear_parameter['operate_name'] == 'dropout':
                    self.linears.add_module(
                        '{0}-{1}'.format(linear_parameter['operate_name'], idx), nn.Dropout(linear_parameter['param']))

    def forward(self, data):

        data = self.backbone(data)
        x, batch = data.x, data.batch

        if self.graph_pooling == "add":
            x = global_add_pool(x, batch)
        if self.graph_pooling == "max":
            x = gmp(x, batch)
        if self.graph_pooling == "mean":
            x = gap(x, batch)
        if self.graph_pooling == "max_mean":
            x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.linears(x)

        return x


class Cell(nn.Module):
    def __init__(self,
                 input_cell_feature_dim,
                 output_cell_feature_dim,
                 module_name,
                 fc_1_dim,
                 layer_num,
                 dropout,
                 layer_hyperparameter):
        super(Cell, self).__init__()

        self.module_name = module_name

        assert len(
            layer_hyperparameter) == layer_num, 'Number of layer is not same as hyperparameter list.'

        self.backbone = nn.Sequential()

        input_channle = 1
        cell_feature_dim = input_cell_feature_dim

        for index, channel in enumerate(layer_hyperparameter['cnn_channels']):

            self.backbone.add_module('CNN1d-{0}_{1}_{2}'.format(index, input_channle, channel), nn.Conv1d(in_channels=input_channle,
                                                                                                          out_channels=channel,
                                                                                                          kernel_size=layer_hyperparameter['kernel_size'][index]))
            self.backbone.add_module('ReLU-{0}'.format(index), nn.ReLU())
            self.backbone.add_module('Maxpool-{0}'.format(index), nn.MaxPool1d(
                layer_hyperparameter['maxpool1d'][index]))

            input_channle = channel
            cell_feature_dim = int(((
                cell_feature_dim-layer_hyperparameter['kernel_size'][index]) + 1)/layer_hyperparameter['maxpool1d'][index])

        self.cell_output_feature_channel = channel
        self.cell_output_feature_dim = cell_feature_dim
        self.fc_1 = nn.Linear(channel*cell_feature_dim,
                              output_cell_feature_dim)

    def forward(self, x):

        x = self.backbone(x)
        x = x.view(-1, x.shape[1] * x.shape[2])

        x = self.fc_1(x)
        return x


class Fusion(nn.Module):
    def __init__(self,
                 input_dim,
                 fc_1_dim,
                 fc_2_dim,
                 fc_3_dim,
                 dropout,
                 fusion_mode):
        super(Fusion, self).__init__()

        self.fusion_mode = fusion_mode

        if fusion_mode == "concat":
            input_dim = input_dim[0]+input_dim[1]
            self.fc1 = nn.Linear(input_dim, fc_1_dim)

        self.fc2 = nn.Linear(fc_1_dim, fc_2_dim)
        self.fc3 = nn.Linear(fc_2_dim, fc_3_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug, cell):

        if self.fusion_mode == "concat":
            x = torch.cat((drug, cell), 1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = nn.Sigmoid()(x)

        return x


class GraphDRP(torch.nn.Module):
    def __init__(self, config):
        super(GraphDRP, self).__init__()

        self.config = config

        # self.drug_module
        self.init_drug_module(self.config['model']['drug_module'])

        # self.cell_module
        self.init_cell_module(self.config['model']['cell_module'])

        # self.fusion_module
        self.init_fusion_module(self.config['model'])

        # pdb.set_trace()

    def init_drug_module(self, config):
        module_name = config['module_name']
        input_drug_feature_dim = config['input_drug_feature_dim']
        layer_num = config['layer_num']
        graph_pooling = config['graph_pooling']
        dropout = config['dropout']
        output_drug_feature_dim = config['output_drug_feature_dim']
        linear_layers = config['linear_layers'] if config.get(
            'linear_layers') else None
        gnn_layers = config['gnn_layers']

        self.drug_module = Drug(module_name,
                                input_drug_feature_dim,
                                output_drug_feature_dim,
                                layer_num,
                                graph_pooling,
                                linear_layers,
                                gnn_layers,
                                dropout)

    def init_cell_module(self, config):
        input_cell_feature_dim = config['input_cell_feature_dim']
        module_name = config['module_name']
        fc_1_dim = config['fc_1_dim']
        layer_num = config['layer_num']
        dropout = config['transformer_dropout'] if config.get(
            'transformer_dropout') else 0
        layer_hyperparameter = config['layer_hyperparameter']
        output_cell_feature_dim = config['output_cell_feature_dim']

        self.cell_module = Cell(input_cell_feature_dim,
                                output_cell_feature_dim,
                                module_name,
                                fc_1_dim,
                                layer_num,
                                dropout,
                                layer_hyperparameter)

    def init_fusion_module(self, config):
        input_dim = [config['drug_module']['output_drug_feature_dim'],
                     config['cell_module']['output_cell_feature_dim']]

        fc_1_dim = config['fusion_module']['fc_1_dim']
        fc_2_dim = config['fusion_module']['fc_2_dim']
        fc_3_dim = config['fusion_module']['fc_3_dim']
        dropout = config['fusion_module']['dropout']
        fusion_mode = config['fusion_module']['fusion_mode']

        self.fusion_module = Fusion(input_dim,
                                    fc_1_dim,
                                    fc_2_dim,
                                    fc_3_dim,
                                    dropout,
                                    fusion_mode)

    def forward(self, data):

        x_drug = self.drug_module(data)
        x_cell = self.cell_module(data.target[:, None, :])

        x_fusion = self.fusion_module(x_drug, x_cell)

        return x_fusion
