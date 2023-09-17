import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool

from model_helper import Embeddings,Encoder_MultipleLayers

class Drug(nn.Module):
    def __init__(self,
                 config):
        super(Drug, self).__init__()
       
        
        self.emb = Embeddings(config['input_drug_feature_dim'],
                         config['gnn_layers']['embed_dim'],
                         50,
                         config['dropout'])

        self.encoder = Encoder_MultipleLayers(config['layer_num'],
                                         config['gnn_layers']['embed_dim'],
                                         config['gnn_layers']['intermediate_dim'],
                                         config['gnn_layers']['head'],
                                         config['gnn_layers']['attention_probs_dropout'],
                                         config['gnn_layers']['hidden_dropout'],)
        self.DeepTTC_encode = torch.load("/home/lk/project/ODD/NMI_DRP/data/processed/DeepTTC_encode.pth")

  
    def forward(self, data):

        DeepTTC_drug_encode = []
        DeepTTC_drug_encode_mask = []
        
        for smile in data.smiles:
            DeepTTC_drug_encode.append(self.DeepTTC_encode[smile]['DeepTTC_drug_encode'].to(data.x.device))
            DeepTTC_drug_encode_mask.append(self.DeepTTC_encode[smile]['DeepTTC_drug_encode_mask'].to(data.x.device))
        
        v = [torch.cat(DeepTTC_drug_encode, dim=0), torch.cat(DeepTTC_drug_encode_mask, dim=0)]
        e = v[0].long()
        e_mask = v[1].long()
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())
        return encoded_layers[:, 0]



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

            elif linear_parameter['operate_name'] == 'conv1d':
                self.backbone.add_module('CNN1d-{0}_{1}_{2}'.format(idx), nn.Conv1d(in_channels=linear_parameter['cnn_channels'][0],
                            out_channels=linear_parameter['cnn_channels'][1],
                            kernel_size=linear_parameter['kernel_size']))
            
            elif linear_parameter['operate_name'] == 'maxpool1d':
                self.backbone.add_module('Maxpool-{0}'.format(idx),nn.MaxPool1d(
                                        linear_parameter['param']))
                        
 
        
    def forward(self, drug, cell):

        if self.fusion_mode == "concat":
            x = torch.cat((drug, cell), 1)
        
        x = self.backbone(x)
        return x


class DeepTTC(torch.nn.Module):
    def __init__(self, config):
        super(DeepTTC, self).__init__()

        self.config = config

        # self.drug_module
        self.init_drug_module(self.config['model']['drug_module'])

        # self.cell_module
        self.init_cell_module(self.config['model']['cell_module'])

        # self.fusion_module
        self.init_fusion_module(self.config['model'])

        # pdb.set_trace()

    def init_drug_module(self, config):
        self.drug_module = Drug(config)

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
        cnn_layers = config['fusion_module']['cnn_layers'] if  config['fusion_module'].get('cnn_layers') else None

        fusion_mode = config['fusion_module']['fusion_mode']
        fc_1 = config['fusion_module']['fc_1'] if  config['fusion_module'].get('fc_1') else None
        
        self.fusion_module = Fusion(module_name,
                                    linear_layers,
                                    cnn_layers,
                                    fc_1,
                                    fusion_mode)

    def forward(self, data):
        # pdb.set_trace()
        x_drug = self.drug_module(data)
        x_cell = self.cell_module(data.target[:, None, :])

        x_fusion = self.fusion_module(x_drug, x_cell)

        return x_fusion


