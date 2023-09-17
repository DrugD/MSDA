import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch_scatter import gather_csr, scatter, segment_csr
import pdb
from re import X
from matplotlib.pyplot import xkcd
from sympy import xfield
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

import torch,math
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from typing import Union, Tuple, Optional

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

# GCN-CNN based model

'''
该版本对边的特征嵌入做的比较烂，实际上是用GAT 对边的权重重新按照边的特征做了评估，实际上是 edge weight 而不是 edge feature
'''

def constant(value, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)
            
def zeros(value):
    constant(value, 0.)
    
def glorot(value):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)


class RAGCN(MessagePassing):
    def __init__(self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')
        
        #changd
        self.w_n_q = Linear(self.in_channels, self.in_channels, False, weight_initializer='glorot')
        self.w_n_v = Linear(self.in_channels, self.in_channels, False, weight_initializer='glorot')
        self.w_n_k = Linear(self.in_channels, self.in_channels, False, weight_initializer='glorot')
        self.w_e_q = Linear(self.in_channels, self.in_channels, False, weight_initializer='glorot')
        self.w_e_v = Linear(self.in_channels, self.in_channels, False, weight_initializer='glorot')
        self.w_e_k = Linear(self.in_channels, self.in_channels, False, weight_initializer='glorot')
        self.w4   =  Linear(self.in_channels * 2, self.in_channels, False, weight_initializer='glorot')
        self.w5   =  Linear(self.in_channels, self.in_channels, False, weight_initializer='glorot')
        self.w6   =  Linear(self.in_channels, self.in_channels, False, weight_initializer='glorot')
        self.w7   =  Linear(self.in_channels, self.in_channels, False, weight_initializer='glorot') 
        
        self.LN_1 =nn.LayerNorm(self.in_channels,eps=0,elementwise_affine=True)
        self.LN_2 =nn.LayerNorm(self.in_channels,eps=0,elementwise_affine=True)
        
        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        
        self.w_n_q.reset_parameters()
        self.w_n_v.reset_parameters()
        self.w_n_k.reset_parameters()
        self.w_e_q.reset_parameters()
        self.w_e_v.reset_parameters()
        self.w_e_k.reset_parameters()
        self.w4.reset_parameters()
        self.w5.reset_parameters()
        self.w6.reset_parameters()
        self.w7.reset_parameters()


        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        
        import pdb
        
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
     
        x_src = x_dst = self.lin_src(x).view(-1, H, C)
        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
     
  
      
        # We only want to add self-loops for nodes that appear both as
        # source and target nodes:
        num_nodes = x_src.size(0)
        if x_dst is not None:
            num_nodes = min(num_nodes, x_dst.size(0))
        num_nodes = min(size) if size is not None else num_nodes
        
        
        edge_index, edge_attr = remove_self_loops(
            edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, fill_value=self.fill_value,
            num_nodes=num_nodes)


        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor, edge_attr: OptTensor)  # noqa
        node, edge = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                             size=size)
      

        if self.concat:
            node = node.view(-1, self.heads * self.out_channels)
        else:
            node = node.mean(dim=1)

        if self.bias is not None:
            node += self.bias
       
        return node, edge

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):

 
        size = self.__check_input__(edge_index, size)
        
        coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)
        # source_to_target
        
        
        self.flow = 'target_to_source'
        coll_dict_TTS = self.__collect__(self.__user_args__, edge_index, size, kwargs)
        # target_to_source
        
        
        
        node_K = self.w_n_k(coll_dict['x_j'])
        node_V = self.w_n_v(coll_dict['x_j'])
        
        node_Q = self.w_n_q(kwargs['x'][0])
        
        # K_n = scatter(node_K, coll_dict['edge_index_j'], dim=0, dim_size=coll_dict['size'][1],reduce='add')
        # V_n = scatter(node_K, coll_dict['edge_index_j'], dim=0, dim_size=coll_dict['size'][1],reduce='add')
        # Q_n = node_Q

        edge_K = self.w_e_q(kwargs['edge_attr'])
        edge_V = self.w_e_v(kwargs['edge_attr'])
        
        edge_Q = self.w_e_k(kwargs['edge_attr'])
        
        
        
        K_ij = edge_K.unsqueeze(1) + node_K
        V_ij = edge_V.unsqueeze(1) + node_V
        
        K_scatter = scatter(K_ij, coll_dict['edge_index_j'], dim=0, dim_size=coll_dict['dim_size'],reduce='add')
        V_scatter = scatter(V_ij, coll_dict['edge_index_j'], dim=0, dim_size=coll_dict['dim_size'],reduce='add')
        Q_scatter_edge = scatter(edge_Q.unsqueeze(1), coll_dict['edge_index_j'], dim=0, dim_size=coll_dict['dim_size'],reduce='add')
        
        Q_scatter = Q_scatter_edge + node_Q
        
        QK =  Q_scatter.unsqueeze(3).permute(0,1,3,2) @ K_scatter.unsqueeze(3)
        alphi = F.softmax(QK.div(math.sqrt(self.in_channels)).squeeze(3),1) * V_scatter


        
        
        
        # kwargs['edge_attr']
        # coll_dict['x_j']
        # coll_dict['adj_t']
        # coll_dict['edge_index']
        # coll_dict['edge_index_i']
        # coll_dict['edge_index_j']
        # coll_dict['ptr']
        # coll_dict['index']
        # coll_dict['size']
        # coll_dict['size_i']
        # coll_dict['size_j']
        # coll_dict['dim_size']
        
        # x_j_next = x_j * self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)

        # scatter(x_j_next, index, dim=0, dim_size=size_i,reduce='add')
        
        node_update = kwargs['x'][0] * alphi
        
        # 'target_to_source'
        # 'source_to_target' 
        self.flow = 'source_to_target' 
        coll_dict_update_node = self.__collect__(self.__user_args__, edge_index, size, {'x':node_update,'edge_attr':kwargs['edge_attr']})
        
        edge_repeat = torch.repeat_interleave(kwargs['edge_attr'].unsqueeze(2).permute(0,2,1),self.heads,dim=1)
        edge_node_concat = torch.cat((edge_repeat,coll_dict_update_node['x_j']),dim=2)

        
        m_ij = F.relu(self.w4(edge_node_concat))
        u_ij = self.LN_1(self.w5(m_ij)+kwargs['edge_attr'].unsqueeze(2).permute(0,2,1))
        
        edge_update = self.LN_2(self.w5(u_ij)+kwargs['edge_attr'].unsqueeze(2).permute(0,2,1))
        
        # edge_scatter_update = scatter(edge_update, coll_dict_update_node['edge_index_j'], dim=0, dim_size=coll_dict_update_node['dim_size'],reduce='add')
        edge_index, edge_update = remove_self_loops(edge_index, edge_update)
        
        # 
        
        return node_update, edge_update.mean(dim=1)


    
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
            self.gnn1 = RAGCN(
                input_drug_feature_dim, input_drug_feature_dim, heads=10, edge_dim=input_drug_feature_dim)
            self.gnn2 = RAGCN(
                input_drug_feature_dim, input_drug_feature_dim, heads=10, edge_dim=input_drug_feature_dim)
            # self.gnn3 = RAGCN(
            #     input_drug_feature_dim, input_drug_feature_dim, heads=10, edge_dim=input_drug_feature_dim)
            
            self.edge_embed = torch.nn.Linear(
                input_drug_edge_dim, input_drug_feature_dim)
        else:
            self.gnn1 = RAGCN(input_drug_feature_dim,
                                input_drug_feature_dim, heads=10)

        self.trans_layer_encode_1 = nn.TransformerEncoderLayer(
            d_model=input_drug_feature_dim, nhead=1, dropout=transformer_dropout)
        self.trans_layer_1 = nn.TransformerEncoder(
            self.trans_layer_encode_1, 1)

        self.trans_layer_encode_2 = nn.TransformerEncoderLayer(
            d_model=input_drug_feature_dim*10, nhead=1, dropout=transformer_dropout)
        self.trans_layer_2 = nn.TransformerEncoder(
            self.trans_layer_encode_2, 1)

    
        self.fc_00 = torch.nn.Linear(input_drug_feature_dim*10, input_drug_feature_dim)
        self.fc_01 = torch.nn.Linear(input_drug_feature_dim*10, input_drug_feature_dim)
        self.fc_1 = torch.nn.Linear(input_drug_feature_dim*10*2, fc_1_dim)
        self.fc_2 = torch.nn.Linear(fc_1_dim, fc_2_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        if self.use_drug_edge:
            x, edge_index, batch, edge_attribute = data.x, data.edge_index, data.batch, data.edge_attr
            edge_embeddings = self.edge_embed(edge_attribute.float())
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = torch.unsqueeze(x, 1)
        x = self.trans_layer_1(x)
        x = torch.squeeze(x, 1)
        
        if self.use_drug_edge:
            x, edge = self.gnn1(x, edge_index, edge_attr=edge_embeddings)
        else:
            x = self.gnn1(x, edge_index)

        x = self.relu(x)

        x = torch.unsqueeze(x, 1)
        x = self.trans_layer_2(x)
        x = torch.squeeze(x, 1)

        
        x = self.fc_00(x)
        
        if self.use_drug_edge:
            x, edge = self.gnn2(x, edge_index, edge_attr=edge)
        else:
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


class RelationGraph(torch.nn.Module):
    def __init__(self, config):
        super(RelationGraph, self).__init__()

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
