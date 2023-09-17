import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
import pdb
import numpy as np


class Drug(nn.Module):
    def __init__(self,
                 input_drug_feature_dim,
                 input_drug_feature_channel,
                 layer_hyperparameter,
                 layer_num):
        super(Drug, self).__init__()

        assert len(
            layer_hyperparameter) == layer_num, 'Number of layer is not same as hyperparameter list.'

        self.input_drug_feature_channel = input_drug_feature_channel
        input_channle = input_drug_feature_channel
        drug_feature_dim = input_drug_feature_dim

        self.backbone = nn.Sequential()

        for index, channel in enumerate(layer_hyperparameter['cnn_channels']):

            self.backbone.add_module('CNN1d-{0}_{1}_{2}'.format(index, input_channle, channel), nn.Conv1d(in_channels=input_channle,
                                                                                                          out_channels=channel,
                                                                                                          kernel_size=layer_hyperparameter['kernel_size'][index]))
            self.backbone.add_module('ReLU-{0}'.format(index), nn.ReLU())
            self.backbone.add_module('Maxpool-{0}'.format(index), nn.MaxPool1d(
                layer_hyperparameter['maxpool1d'][index]))
            input_channle = channel
            drug_feature_dim = int(((
                drug_feature_dim-layer_hyperparameter['kernel_size'][index]) + 1)/layer_hyperparameter['maxpool1d'][index])

        self.drug_output_feature_channel = channel
        self.drug_output_feature_dim = drug_feature_dim


    
        self.tCNNs_encode = torch.load("/home/lk/project/ODD/NMI_DRP/data/processed/tCNNs_encode.pth")


     


    def forward(self, data):
        tCNNs_drug_matrix = []
        # pdb.set_trace()
        
        for smile in data.smiles:
            tCNNs_drug_matrix.append(torch.tensor(self.tCNNs_encode[smile]['tCNNs_drug_matrix']).to(data.x.device))
        
        # pdb.set_trace()
        x = torch.stack(tCNNs_drug_matrix, dim=0)
        
        
        if x.shape[1] != self.input_drug_feature_channel:
            x = torch.cat((torch.zeros(
                (x.shape[0], self.input_drug_feature_channel - x.shape[1], x.shape[2]), dtype=torch.float).cuda(), x), 1)
        
        x = self.backbone(x)
        x = x.view(-1, x.shape[1] * x.shape[2])
        return x





class Cell(nn.Module):
    def __init__(self,
                 input_cell_feature_dim,
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

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, x.shape[1] * x.shape[2])
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


class tCNNs(torch.nn.Module):
    def __init__(self, config):
        super(tCNNs, self).__init__()

        self.config = config

        # self.drug_module
        self.init_drug_module(self.config['model']['drug_module'])

        # self.cell_module
        self.init_cell_module(self.config['model']['cell_module'])

        # self.fusion_module
        self.init_fusion_module(self.config['model'])

    def init_drug_module(self, config):
        input_drug_feature_dim = config['input_drug_feature_dim']
        input_drug_feature_channel = config['input_drug_feature_channel']
        layer_hyperparameter = config['layer_hyperparameter']
        layer_num = config['layer_num']

        self.drug_module = Drug(input_drug_feature_dim,
                                input_drug_feature_channel,
                                layer_hyperparameter,
                                layer_num)

    def init_cell_module(self, config):
        input_cell_feature_dim = config['input_cell_feature_dim']
        module_name = config['module_name']
        fc_1_dim = config['fc_1_dim']
        layer_num = config['layer_num']
        dropout = config['transformer_dropout'] if config.get(
            'transformer_dropout') else 0
        layer_hyperparameter = config['layer_hyperparameter']

        self.cell_module = Cell(input_cell_feature_dim,
                                module_name,
                                fc_1_dim,
                                layer_num,
                                dropout,
                                layer_hyperparameter)

    def init_fusion_module(self, config):
        input_dim = [self.drug_module.drug_output_feature_dim * self.drug_module.drug_output_feature_channel,
                     self.cell_module.cell_output_feature_dim * self.cell_module.cell_output_feature_channel]
       
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
        # pdb.set_trace()
        x_drug = self.drug_module(data)
        x_cell = self.cell_module(data.target[:, None, :])
        x_fusion = self.fusion_module(x_drug, x_cell)

        return x_fusion
