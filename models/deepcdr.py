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
                 dropout=0.1,
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

        self.dropout = nn.Dropout(dropout)
        
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

        x = self.dropout(x)
       
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

        if hasattr(self, 'linears'):
            x = self.linears(x)

        return x


class Cell(nn.Module):
    def __init__(self,
                input_cell_feature_dim,
                output_cell_feature_dim,
                module_name,
                linear_layers):
        super(Cell, self).__init__()

        self.module_name = module_name

        self.backbone = nn.Sequential()
        self.relu = nn.ReLU()
        
        if linear_layers:
            self.backbone = nn.Sequential()

            for idx, linear_parameter in enumerate(linear_layers):

                if linear_parameter['operate_name'] == 'linear':
                    self.backbone.add_module(
                        '{0}-{1}'.format(linear_parameter['operate_name'], idx), torch.nn.Linear(linear_parameter['param'][0], linear_parameter['param'][1]))

                elif linear_parameter['operate_name'] == 'relu':
                    self.backbone.add_module(
                        '{0}-{1}'.format(linear_parameter['operate_name'], idx), self.relu)

                elif linear_parameter['operate_name'] == 'dropout':
                    self.backbone.add_module(
                        '{0}-{1}'.format(linear_parameter['operate_name'], idx), nn.Dropout(linear_parameter['param']))

    def forward(self, x):
        x = x.squeeze()
        x = self.backbone(x)
        return x


class Fusion(nn.Module):
    def __init__(self,
                module_name,
                linear_layers,
                cnn_layers,
                fc_1,
                fusion_mode):
        super(Fusion, self).__init__()

        self.fusion_mode = fusion_mode
        self.relu = nn.ReLU()
        
        self.linear = nn.Sequential()
        self.cnn = nn.Sequential()
        
        for idx, linear_parameter in enumerate(linear_layers):

            if linear_parameter['operate_name'] == 'linear':
                self.linear.add_module(
                    '{0}-{1}'.format(linear_parameter['operate_name'], idx), torch.nn.Linear(linear_parameter['param'][0], linear_parameter['param'][1]))

            elif linear_parameter['operate_name'] == 'relu':
                self.linear.add_module(
                    '{0}-{1}'.format(linear_parameter['operate_name'], idx), self.relu)

            elif linear_parameter['operate_name'] == 'dropout':
                self.linear.add_module(
                    '{0}-{1}'.format(linear_parameter['operate_name'], idx), nn.Dropout(linear_parameter['param']))

            elif linear_parameter['operate_name'] == 'conv1d':
                self.linear.add_module('CNN1d-{0}_{1}_{2}'.format(idx), nn.Conv1d(in_channels=linear_parameter['cnn_channels'][0],
                            out_channels=linear_parameter['cnn_channels'][1],
                            kernel_size=linear_parameter['kernel_size']))
            
            elif linear_parameter['operate_name'] == 'maxpool1d':
                self.linear.add_module('Maxpool-{0}'.format(idx),nn.MaxPool1d(
                                        linear_parameter['param']))
                
                
        for idx, linear_parameter in enumerate(cnn_layers):
           
            if linear_parameter['operate_name'] == 'linear':
                self.cnn.add_module(
                    '{0}-{1}'.format(linear_parameter['operate_name'], idx), torch.nn.Linear(linear_parameter['param'][0], linear_parameter['param'][1]))

            elif linear_parameter['operate_name'] == 'relu':
                self.cnn.add_module(
                    '{0}-{1}'.format(linear_parameter['operate_name'], idx), self.relu)

            elif linear_parameter['operate_name'] == 'dropout':
                self.cnn.add_module(
                    '{0}-{1}'.format(linear_parameter['operate_name'], idx), nn.Dropout(linear_parameter['param']))

            elif linear_parameter['operate_name'] == 'conv1d':
                self.cnn.add_module('CNN1d-{0}'.format(idx), nn.Conv1d(in_channels=linear_parameter['cnn_channels'][0],
                            out_channels=linear_parameter['cnn_channels'][1],
                            kernel_size=linear_parameter['kernel_size']))
            
            elif linear_parameter['operate_name'] == 'maxpool1d':
                self.cnn.add_module('Maxpool-{0}'.format(idx),nn.MaxPool1d(
                                        linear_parameter['param']))
        
        self.fc_1 = nn.Linear(fc_1[0], fc_1[1])             
 
        
    def forward(self, drug, cell):

        if self.fusion_mode == "concat":
            x = torch.cat((drug, cell), 1)
        
        x = self.linear(x)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.fc_1(x)
        
        x = nn.Sigmoid()(x)

        return x


class DeepCDR(torch.nn.Module):
    def __init__(self, config):
        super(DeepCDR, self).__init__()

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
        
        module_name = config['module_name']
        input_cell_feature_dim = config['input_cell_feature_dim']
        output_cell_feature_dim = config['output_cell_feature_dim']
        linear_layers = config['linear_layers'] if config.get(
            'linear_layers') else None
        
        self.cell_module = Cell(input_cell_feature_dim,
                                output_cell_feature_dim,
                                module_name,
                                linear_layers)

    def init_fusion_module(self, config):
        module_name = config['fusion_module']['module_name']
        
        linear_layers = config['fusion_module']['linear_layers']
        cnn_layers = config['fusion_module']['cnn_layers']

        fusion_mode = config['fusion_module']['fusion_mode']
        fc_1 = config['fusion_module']['fc_1']
        
        self.fusion_module = Fusion(module_name,
                                    linear_layers,
                                    cnn_layers,
                                    fc_1,
                                    fusion_mode)

    def forward(self, data):

        x_drug = self.drug_module(data)
        x_cell = self.cell_module(data.target[:, None, :])

        x_fusion = self.fusion_module(x_drug, x_cell)

        return x_fusion
