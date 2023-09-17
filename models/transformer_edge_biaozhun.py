import pdb
from re import X
from matplotlib.pyplot import xkcd
from sympy import xfield
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

'''
1:原版       不使用额外的x特征 和边结构     cell line 使用  cnn1d
2:药物       使用额外的x特征 和边结构       cell line 使用  cnn1d
3:细胞       不使用额外的x特征 和边结构     cell line 使用  transformer
4:全部       使用额外的x特征 和边结构       cell line 使用  transformer
'''


class Drug(nn.Module):
    def __init__(self,
                 input_drug_feature_dim,
                 use_drug_edge,
                 input_drug_edge_dim,
                 fc_1_dim,
                 fc_2_dim,
                 dropout,
                 transformer_dropout,
                 show_attenion=False):
        super(Drug, self).__init__()

        self.use_drug_edge = use_drug_edge
        self.show_attenion = show_attenion
        if use_drug_edge:
            self.gnn1 = GATConv(
                input_drug_feature_dim, input_drug_feature_dim, heads=10, edge_dim=input_drug_feature_dim)
            self.edge_embed = torch.nn.Linear(
                input_drug_edge_dim, input_drug_feature_dim)
        else:
            self.gnn1 = GATConv(input_drug_feature_dim,
                                input_drug_feature_dim, heads=10)

        self.trans_layer_encode_1 = nn.TransformerEncoderLayer(
            d_model=input_drug_feature_dim, nhead=1, dropout=transformer_dropout)
        self.trans_layer_1 = nn.TransformerEncoder(
            self.trans_layer_encode_1, 1)

        self.trans_layer_encode_2 = nn.TransformerEncoderLayer(
            d_model=input_drug_feature_dim*10, nhead=1, dropout=transformer_dropout)
        self.trans_layer_2 = nn.TransformerEncoder(
            self.trans_layer_encode_2, 1)

        self.gnn2 = GCNConv(input_drug_feature_dim*10,
                            input_drug_feature_dim*10)

        self.fc_1 = torch.nn.Linear(input_drug_feature_dim*10*2, fc_1_dim)
        self.fc_2 = torch.nn.Linear(fc_1_dim, fc_2_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        if self.use_drug_edge:
            x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
            edge_embeddings = self.edge_embed(edge_attr.float())
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        x = torch.unsqueeze(x, 1)
        x = self.trans_layer_1(x)
        x = torch.squeeze(x, 1)

        if self.use_drug_edge:
            x = self.gnn1(x, edge_index, edge_attr=edge_embeddings)
        else:
            x = self.gnn1(x, edge_index)

        x = self.relu(x)

        x = torch.unsqueeze(x, 1)
        x = self.trans_layer_2(x)
        x = torch.squeeze(x, 1)

        x = self.gnn2(x, edge_index)
        x = self.relu(x)

        if self.show_attenion:
            self.show_atom_attention(x, data)

        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)

        return x

    def show_atom_attention(self, x, data):
        x_heat = torch.sum(x, 1)

        from rdkit.Chem import Draw
        from rdkit import Chem
        from tqdm import tqdm
        import numpy as np

        for index, i in enumerate(tqdm(data.smiles)):
            if index >= 50:
                break
            m = Chem.MolFromSmiles(i)
            for atom in m.GetAtoms():
                atom.SetProp("atomNote", str(atom.GetIdx()))

            from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
            opts = DrawingOptions()

            opts.includeAtomNumbers = True
            opts.bondLineWidth = 2.8
            draw = Draw.MolToImage(m, size=(600, 600), options=opts)

            smile_name = i.replace('\\', '!').replace('/', '~')

            draw.save('./infer/img/{}.jpg'.format(smile_name))

            heat_item = x_heat.numpy()[np.argwhere(
                data.batch.numpy() == index)]

            with open('./infer/heat/{}.txt'.format(smile_name), 'w') as f:
                for idx, heat in enumerate(heat_item):
                    f.write(str(heat[0])+'\t'+str(idx)+'\n')


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

        if module_name == "Transformer":

            for index, head in enumerate(layer_hyperparameter):
                transformer_encode_layer = nn.TransformerEncoderLayer(
                    d_model=input_cell_feature_dim, nhead=head, dropout=dropout)
                self.backbone.add_module('Transformer-{0}-{1}'.format(index, head), nn.TransformerEncoder(
                    transformer_encode_layer, 1))

            self.fc_1 = nn.Linear(input_cell_feature_dim, fc_1_dim)

        elif module_name == "Conv1d":
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

            self.fc_1 = nn.Linear(cell_feature_dim*channel, fc_1_dim)

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
                 transformer_dropout,
                 fusion_mode):
        super(Fusion, self).__init__()

        self.fusion_mode = fusion_mode

        if fusion_mode == "concat":
            input_dim = input_dim[0]+input_dim[1]

            transformer_encode = nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=1, dropout=transformer_dropout)

            self.transformer_layer = nn.TransformerEncoder(
                transformer_encode, 1)

            self.fc1 = nn.Linear(input_dim, fc_1_dim)

        self.fc2 = nn.Linear(fc_1_dim, fc_2_dim)
        self.fc3 = nn.Linear(fc_2_dim, fc_3_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug, cell):

        if self.fusion_mode == "concat":
            x = torch.cat((drug, cell), 1)

        x = torch.unsqueeze(x, 1)
        x = self.transformer_layer(x)
        x = torch.squeeze(x, 1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        x = nn.Sigmoid()(x)
        return x


class TransE(torch.nn.Module):
    def __init__(self, config):
        super(TransE, self).__init__()

        self.config = config

        # self.drug_module
        self.init_drug_module(self.config['model']['drug_module'])

        # self.cell_module
        self.init_cell_module(self.config['model']['cell_module'])

        # self.fusion_module
        self.init_fusion_module(self.config['model'])

    def init_drug_module(self, config):
        input_drug_feature_dim = config['input_drug_feature_dim']
        input_drug_edge_dim = config['input_drug_edge_dim']
        fc_1_dim = config['fc_1_dim']
        fc_2_dim = config['fc_2_dim']
        dropout = config['dropout'] if config['dropout'] else 0
        transformer_dropout = config['transformer_dropout'] if config['transformer_dropout'] else 0
        use_drug_edge = config['use_drug_edge']

        self.drug_module = Drug(input_drug_feature_dim,
                                use_drug_edge,
                                input_drug_edge_dim,
                                fc_1_dim,
                                fc_2_dim,
                                dropout,
                                transformer_dropout)

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
        input_dim = [config['drug_module']['fc_2_dim'],
                     config['cell_module']['fc_1_dim']]
        fc_1_dim = config['fusion_module']['fc_1_dim']
        fc_2_dim = config['fusion_module']['fc_2_dim']
        fc_3_dim = config['fusion_module']['fc_3_dim']
        dropout = config['fusion_module']['dropout']
        transformer_dropout = config['fusion_module']['transformer_dropout']
        fusion_mode = config['fusion_module']['fusion_mode']

        self.fusion_module = Fusion(input_dim,
                                    fc_1_dim,
                                    fc_2_dim,
                                    fc_3_dim,
                                    dropout,
                                    transformer_dropout,
                                    fusion_mode)

    def forward(self, data):
        x_drug = self.drug_module(data)
        x_cell = self.cell_module(data.target[:, None, :])
        x_fusion = self.fusion_module(x_drug, x_cell)

        return x_fusion

    def getDrugFeature(self, data):
        x_drug = self.drug_module(data)
        # x_cell = self.cell_module(data.target[:, None, :])
        # x_fusion = self.fusion_module(x_drug, x_cell)

        return x_drug