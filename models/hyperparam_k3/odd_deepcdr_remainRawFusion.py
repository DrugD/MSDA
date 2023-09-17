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
        
        x_feature = self.linear(x)
        x = x_feature.unsqueeze(1)
        x = self.cnn(x)
        
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.fc_1(x)
        
        x = nn.Sigmoid()(x)

        return x_feature, x


class ODDDeepCDR(torch.nn.Module):
    def __init__(self, config):
        super(ODDDeepCDR, self).__init__()

        self.config = config

        # self.drug_module
        self.init_drug_module(self.config['model']['drug_module'])

        # self.cell_module
        self.init_cell_module(self.config['model']['cell_module'])

        # self.fusion_module
        self.init_fusion_module(self.config['model'])

        # ODD
        self.loss_fn = nn.MSELoss()
        self.loss_fn_l1 = nn.L1Loss(reduction='mean')
        
        self.loss_domain = torch.nn.NLLLoss()
        
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=0.0, reduction='mean')
        self.memo = {}
        
    def get_static(self,cell_name_mp):
        self.cell_name_mp = cell_name_mp    
        

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
        self.fusion_module  = Fusion(module_name,
                                    linear_layers,
                                    cnn_layers,
                                    fc_1,
                                    fusion_mode)
        
        self.fusion_module1 = Fusion(module_name,
                                    linear_layers,
                                    cnn_layers,
                                    fc_1,
                                    fusion_mode)
        
        self.fusion_module2 = Fusion(module_name,
                                    linear_layers,
                                    cnn_layers,
                                    fc_1,
                                    fusion_mode)
        
        self.fusion_module3 = Fusion(module_name,
                                    linear_layers,
                                    cnn_layers,
                                    fc_1,
                                    fusion_mode)
        

    def re_idx_feature_by_cell(self, data, data_feature):
                
        re_index = []
        for idx in range(len(data.cell_name)):
            re_index.append(self.cell_name_mp[data.cell_name[idx]])
        return data_feature, re_index
    
    def forward(self, source, target=None, mark=None):
        
        device = source.x.device
        source_x_drug = self.drug_module(source)
        source_x_cell = self.cell_module(source.target[:, None, :])
        
        
        
        mmd_loss = 0
        if self.training == True:
            target_x_drug = self.drug_module(target)
            target_x_cell = self.cell_module(target.target[:, None, :])
            
            self.memo[str(mark)] = source_x_drug
            self.memo['0'] = target_x_drug
            
            target_x_fusion_1_feature, _ = self.fusion_module1(target_x_drug, target_x_cell)
            target_x_fusion_2_feature, _ = self.fusion_module2(target_x_drug, target_x_cell)
            target_x_fusion_3_feature, _ = self.fusion_module3(target_x_drug, target_x_cell)
            
          
            target_x_fusion_1_feature, re_index_t_1 = self.re_idx_feature_by_cell(target, target_x_fusion_1_feature)
            target_x_fusion_2_feature, re_index_t_2 = self.re_idx_feature_by_cell(target, target_x_fusion_2_feature)
            target_x_fusion_3_feature, re_index_t_3 = self.re_idx_feature_by_cell(target, target_x_fusion_3_feature)
            
            if mark ==1:
                source_x_fusion_feature, source_x_fusion_reg = self.fusion_module1(source_x_drug, source_x_cell)
                
                source_x_fusion_feature, re_index_s_1 = self.re_idx_feature_by_cell(source, source_x_fusion_feature)

                mmd_loss_, source_align, target_align = mmd_align(source_x_fusion_feature, target_x_fusion_1_feature, re_index_s_1, re_index_t_1)
                
                mmd_loss += mmd_loss_ * 1
             
                reg_loss = self.loss_fn(source_x_fusion_reg.float(), source.y.view(-1, 1).float().to(device))
                
                rank_loss = self.ranking_loss(source.y.view(-1, 1).float().to(device) , source_x_fusion_reg.float(), torch.ones_like(source_x_fusion_reg.float()))

                main_loss = reg_loss * 0.9 + rank_loss * 0.1

                return main_loss, mmd_loss
            
            if mark ==2:
                source_x_fusion_feature, source_x_fusion_reg = self.fusion_module2(source_x_drug, source_x_cell)
                
                source_x_fusion_feature, re_index_s_1 = self.re_idx_feature_by_cell(source, source_x_fusion_feature)

                mmd_loss_, source_align, target_align = mmd_align(source_x_fusion_feature, target_x_fusion_2_feature, re_index_s_1, re_index_t_2)
                mmd_loss += mmd_loss_ * 0.8
                
                reg_loss = self.loss_fn(source_x_fusion_reg.float(), source.y.view(-1, 1).float().to(device))
                
                rank_loss = self.ranking_loss(source.y.view(-1, 1).float().to(device) , source_x_fusion_reg.float(), torch.ones_like(source_x_fusion_reg.float()))

                main_loss = reg_loss * 0.9 + rank_loss * 0.1

                return main_loss, mmd_loss
            
            if mark ==3:
                source_x_fusion_feature, source_x_fusion_reg = self.fusion_module3(source_x_drug, source_x_cell)
                
                source_x_fusion_feature, re_index_s_1 = self.re_idx_feature_by_cell(source, source_x_fusion_feature)

                mmd_loss_, source_align, target_align = mmd_align(source_x_fusion_feature, target_x_fusion_3_feature, re_index_s_1, re_index_t_3)
                mmd_loss += mmd_loss_ * 0.7
                
                reg_loss = self.loss_fn(source_x_fusion_reg.float(), source.y.view(-1, 1).float().to(device))
                
                rank_loss = self.ranking_loss(source.y.view(-1, 1).float().to(device) , source_x_fusion_reg.float(), torch.ones_like(source_x_fusion_reg.float()))

                main_loss = reg_loss * 0.9 + rank_loss * 0.1

                return main_loss, mmd_loss
             
        else:
            
            _, target_x_fusion_reg_0 = self.fusion_module(source_x_drug, source_x_cell)
            _, target_x_fusion_reg_1 = self.fusion_module1(source_x_drug, source_x_cell)
            _, target_x_fusion_reg_2 = self.fusion_module2(source_x_drug, source_x_cell)
            _, target_x_fusion_reg_3 = self.fusion_module3(source_x_drug, source_x_cell)
            
            
            target_x_fusion_reg_0_reweight = np.dot(1, target_x_fusion_reg_0.cpu()) 
            target_x_fusion_reg_1_reweight = np.dot(1, target_x_fusion_reg_1.cpu()) 
            target_x_fusion_reg_2_reweight = np.dot(1, target_x_fusion_reg_2.cpu()) 
            target_x_fusion_reg_3_reweight = np.dot(1, target_x_fusion_reg_3.cpu()) 
            
            target_x_fusion_reg_mean = torch.Tensor(np.array([target_x_fusion_reg_0_reweight, target_x_fusion_reg_1_reweight, target_x_fusion_reg_2_reweight, target_x_fusion_reg_3_reweight] ).sum(0))/4
        
            return target_x_fusion_reg_0_reweight, target_x_fusion_reg_1_reweight, target_x_fusion_reg_2_reweight, target_x_fusion_reg_3_reweight, target_x_fusion_reg_mean
            

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


def align(source, target, re_index_s, re_index_t):
    align_s_index = []
    align_t_index = []
    
    for idx, item in enumerate(re_index_s):
        if item in re_index_t:
            align_t_index.append(re_index_t.index(item))
            align_s_index.append(idx)
            
    # pdb.set_trace()
    
    source_align = torch.index_select(source, dim=0, index =torch.tensor(align_s_index).to(target.device))
    target_align = torch.index_select(target, dim=0, index =torch.tensor(align_t_index).to(target.device))

    return  source_align, target_align

def mmd_align(source, target, re_index_s, re_index_t, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    
    
    source_align, target_align = align(source, target, re_index_s, re_index_t)
    

    # 1:
    batch_size = int(source_align.size()[0])
    kernels = guassian_kernel(source_align, target_align,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    

    return loss, source_align, target_align


def RSD(Feature_s, Feature_t):
    tradeoff2 = 0.1
    u_s, s_s, v_s = torch.svd(Feature_s.t())
    u_t, s_t, v_t = torch.svd(Feature_t.t())
    p_s, cospa, p_t = torch.svd(torch.mm(u_s.t(), u_t))
    sinpa = torch.sqrt(1-torch.pow(cospa,2))
    return torch.norm(sinpa,1)+tradeoff2*torch.norm(torch.abs(p_s) - torch.abs(p_t), 2)