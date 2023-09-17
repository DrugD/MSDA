import math
import sys

sys.path.insert(0, "/home/lk/project/ODD/MSDA_DRP/models/hyperparam_k3")
sys.path.insert(0, "/home/lk/project/ODD/MSDA_DRP/models")
sys.path.insert(0, "/home/lk/project/ODD/MSDA_DRP")
import pdb
import re
import os
import torch.nn.functional as F
from utils import copyfile
import random
import datetime
from utils import *
from models.hyperparam_k3.odd_deepcdr_remainRawFusion import ODDDeepCDR as ODDDeepCDR_RRF
from models.hyperparam_k3.odd_graphdrp_remainRawFusion import ODDGraphDRP as ODDGraphDRP_RRF
from models.hyperparam_k3.odd_deepttc_remainRawFusion import ODDDeepTTC as ODDDeepTTC_RRF
from models.hyperparam_k3.odd_graphtransdrp_remainRawFusion import ODDGraTransDRP as ODDGraTransDRP_RRF
from models.hyperparam_k3.odd_tcnns_remainRawFusion import ODDtCNNs as ODDtCNNs_RRF
from models.hyperparam_k3.odd_transe_remainRawFusion import ODDTransE as ODDTransE_RRF
from models.odd_deepcdr import ODDDeepCDR
from models.odd_graphdrp import ODDGraphDRP
from models.odd_deepttc import ODDDeepTTC
from models.odd_graphtransdrp import ODDGraTransDRP
from models.odd_tcnns import ODDtCNNs
from models.odd_transe import ODDTransE
from models.relation_graph import RelationGraph
from models.tcnns import tCNNs
from models.deepttc import DeepTTC
from models.deepcdr import DeepCDR
from models.transformer_edge import TransE
from models.graphtransdrp import GraTransDRP
from models.graphdrp import GraphDRP
import argparse
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import torch.nn as nn
import torch
from tqdm import tqdm
from random import shuffle
import pandas as pd
import numpy as np


'''
超参数实验：3个target，k为5
'''


# training function at each epoch
# training function at each epoch
CUDA_LAUNCH_BLOCKING = 1

GDSCv2_100_pretrain_model_dict = {
    'tCNNs': '/home/lk/Data/exp_zs/tCNNs_GDSCv2_drug_classed_100_split_8_1_1_20230627200033/tCNNs.model',
    'GraTransDRP': '/home/lk/Data/exp_zs/GraTransDRP_GDSCv2_drug_classed_100_split_8_1_1_20230612150351/GraTrans.model',
    'DeepCDR': '/home/lk/Data/exp_zs/DeepCDR_GDSCv2_drug_classed_100_split_8_1_1_20230611154310/DeepCDR.model',
    'TransE': '/home/lk/Data/exp_zs/TransEDRP_GDSCv2_drug_classed_100_split_8_1_1_20230611112048/TransE.model',
    'DeepTTC': '/home/lk/Data/exp_zs/DeepTTC_GDSCv2_drug_classed_100_split_8_1_1_20230611154208/DeepTTC.model',
    'GraphDRP': '/home/lk/Data/exp_zs/GAT_GCN_GDSCv2_drug_classed_100_split_8_1_1_20230612150535/GAT_GCN.model'
}

NCI60_30_pretrain_model_dict = {
    'tCNNs': '/home/lk/project/NMI_DRP/exp/tCNNs_NCI60_m2r_30_split_8_1_1_20230810004219/tCNNs.model',
    'GraTransDRP': '/home/lk/project/NMI_DRP/exp/GraTransDRP_NCI60_m2r_30_split_8_1_1_20230809231727/GraTrans.model',
    'DeepCDR': '/home/lk/project/NMI_DRP/exp/DeepCDR_NCI60_m2r_30_split_8_1_1_20230809230113/DeepCDR.model',
    'TransE': '/home/lk/project/NMI_DRP/exp/TransEDRP_NCI60_m2r_30_split_8_1_1_20230809230100/TransE.model',
    'DeepTTC': '/home/lk/Data/exp_zs/DeepTTC_GDSCv2_drug_classed_100_split_8_1_1_20230611154208/DeepTTC.model',
    'GraphDRP': '/home/lk/project/NMI_DRP/exp/GAT_GCN_NCI60_m2r_30_split_8_1_1_20230809230105/GAT_GCN.model'
}



def train(model, device, loaders, optimizer, epoch, log_interval, args):
    model.train()

    avg_loss = []

    len_dataloader = max([len(x) for x in loaders])

    data_source_iter_1 = iter(loaders[0])
    data_source_iter_2 = iter(loaders[1])
    data_source_iter_3 = iter(loaders[2])
    data_target_iter = iter(loaders[3])

    for batch_idx in range(len_dataloader):

        # temp = float(batch_idx + epoch * len_dataloader) / args['num_epoch'] / len_dataloader
        # alpha = 2. / (1. + np.exp(-10 * temp)) - 1

        optimizer.zero_grad()

        # training model using source data

        try:
            data_source_1 = next(data_source_iter_1)
        except Exception as err:
            data_source_iter_1 = iter(loaders[0])
            data_source_1 = next(data_source_iter_1)

        try:
            data_source_2 = next(data_source_iter_2)
        except Exception as err:
            data_source_iter_2 = iter(loaders[1])
            data_source_2 = next(data_source_iter_2)

        try:
            data_source_3 = next(data_source_iter_3)
        except Exception as err:
            data_source_iter_3 = iter(loaders[2])
            data_source_3 = next(data_source_iter_3)
            
        try:
            data_target = next(data_target_iter)
        except Exception as err:
            data_target_iter = iter(loaders[3])
            data_target = next(data_target_iter)

        data_source_1 = data_source_1.to(device)
        data_source_2 = data_source_2.to(device)
        data_source_3 = data_source_3.to(device)
        
        
        from copy import deepcopy
        data_target1 = deepcopy(data_target)
        data_target2 = deepcopy(data_target)
        
        data_target = data_target.to(device)
        data_target1 = data_target1.to(device)
        data_target2 = data_target2.to(device)
        
        
        gamma = 0.1
        # gamma = 0

        reg_loss1, mmd_loss = model(data_source_1, data_target, mark=1)
        # gamma = 2 / (1 + math.exp(-10 * (batch_idx) / (len_dataloader+1) )) - 1
        # print(gamma)
        loss1 = reg_loss1 + mmd_loss * gamma
        loss1.backward()
        optimizer.step()

        reg_loss2, mmd_loss = model(data_source_2, data_target1, mark=2)
        # gamma = 2 / (1 + math.exp(-10 * (batch_idx) / (len_dataloader+1) )) - 1
        loss2 = reg_loss2 + mmd_loss * gamma
        loss2.backward()
        optimizer.step()
        
        reg_loss3, mmd_loss = model(data_source_3, data_target2, mark=3)
        # gamma = 2 / (1 + math.exp(-10 * (batch_idx) / (len_dataloader+1) )) - 1
        loss3 = reg_loss3 + mmd_loss * gamma
        loss3.backward()
        optimizer.step()

        sys.stdout.write('\r epoch: %d, [iter: %d / all %d],  %f %f %f'
                         % (epoch, batch_idx + 1, len_dataloader, reg_loss1.data.cpu().numpy(), reg_loss2.data.cpu().numpy(), reg_loss3.data.cpu().numpy()))
        sys.stdout.flush()

    # pdb.set_trace()
    return np.mean(reg_loss1.data.cpu().numpy()+reg_loss2.data.cpu().numpy()+reg_loss3.data.cpu().numpy())
    # loss_main_1 = loss_fn(output.float(), data_source.y.view(-1, 1).float().to(device))


def predicting(model, device, loader, loader_type, args):
    model.eval()
    total_pred_raw = torch.Tensor()
    total_preds1 = torch.Tensor()
    total_preds2 = torch.Tensor()
    total_preds3 = torch.Tensor()
    total_preds_mean = torch.Tensor()
    total_labels = torch.Tensor()
    print("\nMake prediction for {} samples...".format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):

            data = data.to(device)

            output_raw, output1, output2, output3, output_mean = model(data)

            total_pred_raw = torch.cat(
                (total_pred_raw, torch.tensor(output_raw)), 0)
            
            total_preds1 = torch.cat((total_preds1, torch.tensor(output1)), 0)
            total_preds2 = torch.cat((total_preds2, torch.tensor(output2)), 0)
            total_preds3 = torch.cat((total_preds3, torch.tensor(output3)), 0)
            
            total_preds_mean = torch.cat((total_preds_mean, output_mean), 0)
            total_labels = torch.cat(
                (total_labels, data.y.view(-1, 1).cpu()), 0)

    return [
        [total_labels.numpy().flatten(), total_pred_raw.numpy().flatten()],
        [total_labels.numpy().flatten(), total_preds1.numpy().flatten()],
        [total_labels.numpy().flatten(), total_preds2.numpy().flatten()],
        [total_labels.numpy().flatten(), total_preds3.numpy().flatten()],
        [total_labels.numpy().flatten(), total_preds_mean.numpy().flatten()],
    ]


'''freeze'''


def freeze_model(model):
    # pdb.set_trace()
    for (name, param) in model.named_parameters():
        # print(name)
        if "drug_module." in name or "cell_module." in name or "fusion_module." in name:
            param.requires_grad = False
            param = param.detach()
            # print(name,'is False.')
            # param.grad = None
            # param
            # pdb.set_trace()
        else:
            continue
        # param.requires_grad = False
        # print(name, param.requires_grad)


def dateStr():
    return (
        str(datetime.datetime.now())
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .split(".")[0]
        .replace("_", "")
    )


def main(config, yaml_path):

    test_smiles = [
        'COC1=CC(=CC(=C1OC)OC)/C=C/C(=O)N2CCC=CC2=O', 'C1CC1CONC(=O)C2=C(C(=C(C=C2)F)F)NC3=C(C=C(C=C3)I)Cl', 'C[C@H]1COCCN1C2=NC(=NC3=C2C=CC(=N3)C4=CC(=C(C=C4)OC)CO)N5CCOC[C@@H]5C', 'C1=CC=C(C=C1)CN2C3=CC=CC=C3C(=C(C2=O)C(=O)NCC(=O)O)O', 'CC1=[N+](C2=C(N1CCOC)C(=O)C3=CC=CC=C3C2=O)CC4=NC=CN=C4.[Br-]', 'COC1=CC=C(C=C1)COC2=C(C=C(C=C2)CC3=CN=C(N=C3N)N)OC', 'CCC1=NC(=C(S1)C2=CC(=NC=C2)NC(=O)C3=CC=CC=C3)C4=CC(=CC=C4)C', 'CC1=C(SC(=N1)NC(=O)C)C2=CC(=C(C=C2)Cl)S(=O)(=O)NCCO', 'CC1=NC=C(C=C1)OC2=C(C=C(C=C2)NC3=NC=NC4=C3C=C(C=C4)/C=C/CNC(=O)COC)C', 'COC1=CC(=CC(=C1)C2=CC3=C4C(=CN=C3C=C2)C=CC(=O)N4C5=CC(=C(C=C5)N6CCNCC6)C(F)(F)F)OC', 'CC1=C(C=C(C=C1)C(=O)NC2=CC(=C(C=C2)CN3CCN(CC3)C)C(F)(F)F)C#CC4=CN=C5N4N=CC=C5', 'CC1=CC(=C(C=C1)F)NC(=O)NC2=CC=C(C=C2)C3=C4C(=CC=C3)NN=C4N', 'CC(C)N1C=NC2=C1N=C(N=C2NC3=CC(=CC=C3)NC(=O)C=C)NC4CCC(CC4)N(C)C', 'C1=CC=C(C(=C1)N)NC(=O)C2=CC=C(C=C2)CNC(=O)OCC3=CN=CC=C3', 'CCCCC(C=O)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](CC(C)C)NC(=O)OCC1=CC=CC=C1', 'CC(C)(C)OC(=O)NC1=CC=C(C=C1)C2=CC(=NO2)C(=O)NCCCCCCC(=O)NO', 'C1COCCN1C2=CC(=CC(=C2)C3=NC(=NC=C3)NC4=CC=C(C=C4)N5C=NC(=N5)N6CCOCC6)F', 'C1=CC=C(C=C1)[C@H](COC2=CC3=C(C=C2)NC(=O)N3)NC(=O)C4=CC=CN(C4=O)CC5=CC(=C(C=C5)F)F', 'COC1=C(C=C2C(=C1)N=CN2C3=CC(=C(S3)C#N)OCC4=CC=CC=C4S(=O)(=O)C)OC', 'C1CN(CCN1C2=NC=NC3=C2OC4=CC=CC=C43)C(=S)NCC5=CC6=C(C=C5)OCO6', 'C1=CN(C(=O)N=C1N)[C@H]2[C@H]([C@@H]([C@H](O2)CO)O)O', 'C[C@H](C1=CC=C(C=C1)C(=O)NC2=C3C=CNC3=NC=C2)N.Cl', 'C1=CC=C2C(=C1)C(=CN2)CCNC3=CC=C(C=C3)NC4=CC=NC=C4', 'CCOC1=C(C=C2C(=C1)N=CC(=C2NC3=CC(=C(C=C3)F)Cl)C#N)NC(=O)/C=C/CN(C)C', 'C1=C(C(=O)NC(=O)N1)F', 'C1CN(CCC1N2C3=CC=CC=C3NC2=O)CC4=CC=C(C=C4)C5=NC6=CC7=C(C=C6N=C5C8=CC=CC=C8)N=CN7', 'CC1=CN=C(N1)C2=CN=C(N=C2C3=C(C=C(C=C3)Cl)Cl)NCCNC4=NC=C(C=C4)C#N', 'CNCC1=CC=C(C=C1)C2=C3CCNC(=O)C4=CC(=CC(=C34)N2)F', 'CN(C1=C(C=CC=N1)CNC2=NC(=NC=C2C(F)(F)F)NC3=CC4=C(C=C3)NC(=O)C4)S(=O)(=O)C', 'CC1=CN2C(=O)C=C(N=C2C(=C1)C(C)NC3=CC=CC=C3)N4CCOCC4', 'C1CC2=CC=CC=C2[C@H]1NC3=NC=NC4=C3C=CN4[C@@H]5C[C@H]([C@H](C5)O)COS(=O)(=O)N', 'CC1=CN=C(C(=N1)OC)NS(=O)(=O)C2=C(N=CC=C2)C3=CC=C(C=C3)C4=NN=CO4', 'CN1C=C(C=N1)C2=CC3=C4C(=CN=C3C=C2)C=CC(=O)N4C5=CC6=C(CCN6C(=O)C=C)C=C5', 'CCCS(=O)(=O)NC1=C(C(=C(C=C1)F)C(=O)C2=CNC3=NC=C(C=C23)Cl)F', 'CC[C@@]1(C[C@@H]2C[C@@](C3=C(CCN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)[C@]78CCN9[C@H]7[C@@](C=CC9)([C@H]([C@@]([C@@H]8N6C)(C(=O)OC)O)OC(=O)C)CC)OC)C(=O)OC)O', 'C1=CC=C(C=C1)NS(=O)(=O)C2=CC=CC(=C2)/C=C/C(=O)NO', 'CN1CCN(CC1)C2=CC(=C(C=C2)NC3=NC=C4C(=N3)N(C5=CC=CC=C5C(=O)N4C)C)OC', 'C1CC(=O)NC(=O)C1N2CC3=C(C2=O)C=CC=C3N', 'COC1=C(C=C2C(=C1)N=CN=C2NC3=CC=CC(=C3)C#C)OCCCCCCC(=O)NO', 'COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC', 'CC/C(=C(\\C1=CC=CC=C1)/C2=CC=C(C=C2)OCCN(C)C)/C3=CC=CC=C3', 'CCC1=C(C=C2C(=C1)C(=O)C3=C(C2(C)C)NC4=C3C=CC(=C4)C#N)N5CCC(CC5)N6CCOCC6', 'C1=CC(=C(C(=C1)F)N(C2=NC(=C(C=C2)C(=O)N)C3=C(C=C(C=C3)F)F)C(=O)N)F', 'CC1(CCC(=C(C1)CN2CCN(CC2)C3=CC=C(C=C3)C(=O)NS(=O)(=O)C4=CC(=C(C=C4)N[C@H](CCN5CCOCC5)CSC6=CC=CC=C6)S(=O)(=O)C(F)(F)F)C7=CC=C(C=C7)Cl)C', 'CCC1=CN=CN=C1N2CCN(CC2)CC3=NC4=C(N3)C=C(C=C4)C(F)(F)F', 'CC1=C(C=C(C=C1)NC(=O)C2=CC(=CC=C2)C(C)(C)C#N)NC3=CC4=C(C=C3)N=CN(C4=O)C'
    ]

    test_smiles_pt = ['Belinostat', 'XMD8-85', 'GSK319347A', 'Tamoxifen', 'QL-XII-47', 'IOX2', 'Piperlongumine', 'PF-562271', 'Y-39983', 'Alectinib', 'Ponatinib', 'TAK-715',
                      'JW-7-24-1', 'Vinblastine', 'GW-2580', 'Zibotentan', 'Sepantronium bromide', 'Cytarabine', '5-Fluorouracil', 'Navitoclax', 'Rucaparib', 'JNK-9L', 'Pelitinib']

    Ktop = config["dataset_type"]["K"]

    # Similarity_matrix = np.load("/home/lk/project/NMI_DRP/data/Similarity_matrix.npy",allow_pickle=True).item()
    Similarity_matrix = np.load(
        "/home/lk/project/MSDA_DRP/data/Similarity_matrix_by_wasserstein_GDSCv2.npy", allow_pickle=True).item()

    smiles2pt_mp, cell_name_mp = get_dict_smiles2pt(config["dataset_type"])

    for item in tqdm(test_smiles_pt):
        print("---DA for ", item, '---')

        modeling = MODEL_DICT[config["model_name"]]

        model = modeling(config)

        pretrain_model = GDSCv2_100_pretrain_model_dict[config['model_name'].split('_')[
            0][3:]]

        model = load_weight(pretrain_model, model,
                            "RRF" in config["model_name"])

        model.to(device)

        config['dataset_type']['test'] = item+".pt"
        train_batch = config["batch_size"]["train"]
        val_batch = config["batch_size"]["val"]
        test_batch = config["batch_size"]["test"]
        lr = config["lr"]
        num_epoch = config["num_epoch"]
        log_interval = config["log_interval"]

        work_dir = config["work_dir"]

        date_info = ("_" + dateStr()) if config["work_dir"] != "test" else ""
        work_dir = "/home/lk/project/MSDA_DRP/exp/" + \
            config['marker'] + "/" + work_dir + "_" + item + date_info

        if not os.path.exists("/home/lk/project/MSDA_DRP/exp/" + config['marker']):
            os.mkdir("/home/lk/project/MSDA_DRP/exp/" + config['marker'])

        if not os.path.exists(work_dir):
            os.mkdir(work_dir)

        copyfile(yaml_path, work_dir + "/")
        model_st = config["model_name"]

        # dataset loader
        # domain_list_all = load_dataset_from_smiles(config=config["dataset_type"],config['dataset_type']['test'])
        # domain_list = np.array(domain_list_all)

        # 用药物相似性筛选一波训练集中K个相似的药/不相似的药

        model.get_static(cell_name_mp)
        #
        Target_dataset = load_dataset_from_smiles(
            config["dataset_type"], config['dataset_type']['test'])

        test_aim_sorted = Similarity_matrix[Target_dataset[0].smiles]
        test_aim_sorted = sorted(
            test_aim_sorted.items(),  key=lambda d: d[1], reverse=False)

        test_aim_sorted_del = []

        '''删除在测试集中的topK'''
        for smile_key in test_aim_sorted:

            if smile_key[0] not in test_smiles:
                test_aim_sorted_del.append(smile_key)
                # test_aim_sorted.pop(str(smile_key[0]))
                # del test_aim_sorted[smile_key[0]]

        # pdb.set_trace()

        # test_smiles

        test_aim_sorted_top1 = test_aim_sorted_del[:Ktop]
        test_aim_sorted_top2 = test_aim_sorted_del[Ktop:Ktop*2]
        test_aim_sorted_top3 = test_aim_sorted_del[Ktop*2:Ktop*3]


        source_domain_list_top1 = []
        source_domain_list_top2 = []
        source_domain_list_top3 = []

        for idx in range(len(test_aim_sorted_top1)):
            source_domain_list_top1.append(
                load_dataset_from_smiles(
                    config["dataset_type"],
                    smiles2pt_mp[test_aim_sorted_top1[idx][0]]
                )
            )

        for idx in range(len(test_aim_sorted_top2)):
            source_domain_list_top2.append(
                load_dataset_from_smiles(
                    config["dataset_type"],
                    smiles2pt_mp[test_aim_sorted_top2[idx][0]]
                )
            )

        for idx in range(len(test_aim_sorted_top3)):
            source_domain_list_top3.append(
                load_dataset_from_smiles(
                    config["dataset_type"],
                    smiles2pt_mp[test_aim_sorted_top3[idx][0]]
                )
            )
            
        top1_s_data = ConcatDataset(source_domain_list_top1)
        top2_s_data = ConcatDataset(source_domain_list_top2)
        top3_s_data = ConcatDataset(source_domain_list_top3)

        # source_domain_list_down = []
        # for idx in range(len(test_aim_sorted_down)):
        #     source_domain_list.append(domain_list_all[test_aim_sorted_down[idx][0]])

        # max_batchsize = min(
        #     min( len(x) for x in source_domain_list),
        #     min( len(x) for x in [load_dataset_from_smiles(config["dataset_type"],config['dataset_type']['test'])])
        # )

        # pdb.set_trace()

        # if config['dataset_name'] == "GDSCv2":
        max_batchsize = 256

        loaders_train = []
        for item_dataset in [top1_s_data, top2_s_data,top3_s_data]:
            loaders_train.append(DataLoader(
                item_dataset, batch_size=max_batchsize, shuffle=True, drop_last=True))

        loaders_test = DataLoader(
            Target_dataset, batch_size=max_batchsize, shuffle=False, drop_last=False)

        print("Max Batchsize:", max_batchsize)
        # make data PyTorch mini-batch processing ready

        # val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
        # test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)

        freeze_model(model)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        scheduler = None
        # scheduler = LambdaLR(optimizer, milestones=[150, 350], gamma=0.5)
        # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(num_epoch+1))
        # pdb.set_trace()
        # print("初始化的学习率：", optimizer.defaults['lr'])
        # lr = [0.00003, 0.00001]
        # optimizer = torch.optim.Adam([
        #         {'params': model.drug_module.parameters()},
        #         {'params': model.cell_module.parameters()},
        #         {'params': model.fusion_modules.parameters(), 'lr': lr[1]}
        #     ], lr=lr[0])

        best_mse = 9999
        best_pearson = 1
        best_epoch = -1

        model_file_name = work_dir + "/" + model_st + ".model"
        result_file_name = work_dir + "/" + model_st + ".csv"
        loss_fig_name = work_dir + "/" + model_st + "_loss"
        pearson_fig_name = work_dir + "/" + model_st + "_pearson"

        train_losses = []
        val_losses = []
        val_pearsons = []

        for idx, loader_item in enumerate(loaders_train):
            print("TrainLoader-{} on {} samples...".format(idx,
                  len(loader_item.dataset)))

        print("TestLoader on {} samples...".format(len(loaders_test.dataset)))

        rankingLossFunc = torch.nn.MarginRankingLoss(
            margin=0.0, reduction='mean')

        loaders_train.append(loaders_test)

        for epoch in tqdm(range(num_epoch+1)):

            reg_loss = train(
                model, device,  loaders_train, optimizer, epoch + 1, log_interval, config
            )

            # G, P = predicting(model, device, val_loader, "val", config)
            # ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P)]
            # pdb.set_trace()

            # if ret[1] < best_mse and epoch>10:
            res_test_list = predicting(
                model, device, loaders_test, "test", config)

            with open(result_file_name, "a") as f:
                f.write("\n " + str(epoch))
                # f.write("\n " +str(reg_loss_list))
                # f.write("\n " +str(weight_list))

            for idx, item in enumerate(res_test_list):
                G_test, P_test = item
                # import pdb;pdb.set_trace()
                ret_test = [
                    rmse(G_test, P_test),
                    mse(G_test, P_test),
                    pearson(G_test, P_test),
                    spearman(G_test, P_test),
                    rankingLossFunc(torch.tensor(G_test), torch.tensor(
                        P_test), torch.ones_like(torch.tensor(P_test))).item()
                ]
                # print(ret_test)

                with open(result_file_name, "a") as f:
                    if idx == len(res_test_list)-1:
                        f.write("\n Mean Result:")
                        sys.stdout.write('\r epoch: %d, Test mean loss: %s'
                                         % (epoch, str(ret_test)))
                        sys.stdout.flush()
                    f.write("\n rmse:"+str(ret_test[0]))
                    f.write("\n mse:"+str(ret_test[1]))
                    f.write("\n pearson:"+str(ret_test[2]))
                    f.write("\n spearman:"+str(ret_test[3]))
                    f.write("\n rankingloss:"+str(ret_test[4])+"\n")

            # pdb.set_trace()

            # ret_test = [
            #         rmse(G_test, P_test),
            #         mse(G_test, P_test),
            #         pearson(G_test, P_test),
            #         spearman(G_test, P_test),
            #         rankingLossFunc(torch.tensor(G_test) , torch.tensor(P_test), torch.ones_like(torch.tensor(P_test))).item()
            #     ]
            # print(ret_test)

            train_losses.append(reg_loss)
            val_losses.append(ret_test[1])
            val_pearsons.append(ret_test[2])

            # draw_sort_pred_gt(P, G, title=work_dir + "/val_" +str(epoch))

            draw_sort_pred_gt(
                P_test, G_test, title=work_dir + "/test_" + str(epoch))

            # if ret[1] < best_mse and epoch>10:
            torch.save(model.state_dict(), model_file_name)
            # with open(result_file_name, "a") as f:
            #     f.write("\n".join(map(str, ret_test))+"\n")
            # best_epoch = epoch + 1
            # best_mse = ret[1]
            # best_pearson = ret[2]
            # print(
            #     " rmse improved at epoch ",
            #     best_epoch,
            #     "; best_mse:",
            #     best_mse,
            #     model_st,
            # )

            # else:
            #     print(
            #         " no improvement since epoch ",
            #         best_epoch,
            #         "; best_mse, best pearson:",
            #         best_mse,
            #         best_pearson,
            #         model_st,
            #     )

            draw_loss(train_losses, val_losses, loss_fig_name)
            draw_pearson(val_pearsons, pearson_fig_name)


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def getConfig():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="./config/Transformer_edge_concat_GDSCv2.yaml",
        help="",
    )
    args = parser.parse_args()
    import yaml

    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    return config, args.config


def load_weight(pretrain_model, model, remainRaw=False):
    pre_dict = {}

    pretrain_weight = torch.load(
        pretrain_model, map_location=torch.device('cpu'))

    for key, value in pretrain_weight.items():

        if "drug_module" in key or "cell_module" in key:
            pre_dict[key] = value
        else:
            if remainRaw:
                key_names = [key,
                             key.replace("fusion_module", "fusion_module1"),
                             key.replace("fusion_module", "fusion_module2"),
                             key.replace("fusion_module", "fusion_module3"),
                             ]
                pre_dict[key_names[3]] = value
            else:
                key_names = [key.replace("fusion_module", "fusion_module1"),
                             key.replace("fusion_module", "fusion_module2"),
                             key.replace("fusion_module", "fusion_module3"),
                             ]
            pre_dict[key_names[0]] = value
            pre_dict[key_names[1]] = value
            pre_dict[key_names[2]] = value

    model.load_state_dict(pre_dict, strict=True)

    return model


MODEL_DICT = {
    "ODDTransE": ODDTransE,
    "ODDtCNNs": ODDtCNNs,
    "ODDGraTransDRP": ODDGraTransDRP,
    "ODDDeepTTC": ODDDeepTTC,
    "ODDGraphDRP": ODDGraphDRP,
    "ODDDeepCDR": ODDDeepCDR,
    "ODDTransE_RRF": ODDTransE_RRF,
    "ODDtCNNs_RRF": ODDtCNNs_RRF,
    "ODDGraTransDRP_RRF": ODDGraTransDRP_RRF,
    "ODDDeepTTC_RRF": ODDDeepTTC_RRF,
    "ODDGraphDRP_RRF": ODDGraphDRP_RRF,
    "ODDDeepCDR_RRF": ODDDeepCDR_RRF
}

if __name__ == "__main__":
    config, yaml_path = getConfig()
    seed_torch(config["seed"])

    cuda_name = config["cuda_name"]

    print("CPU/GPU: ", torch.cuda.is_available())

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

    #  [ODDTransE, ODDtCNNs, ODDGraTransDRP, ODDDeepTTC, ODDGraphDRP, ODDDeepCDR,
    #             ODDTransE_RRF, ODDtCNNs_RRF, ODDGraTransDRP_RRF, ODDDeepTTC_RRF, ODDGraphDRP_RRF, ODDDeepCDR_RRF][
    # ODDTransE
    # ODDtCNNs
    # ODDGraTransDRP
    # ODDDeepTTC
    # ODDGraphDRP
    # ODDDeepCDR

    # checkClassMatch = re.search(config['model_name'], modeling.__module__, re.IGNORECASE)
    # assert re.search(config['model_name'], modeling.__module__, re.IGNORECASE) is not None

    # pretrain_model = '/home/lk/project/ODD/NMI_DRP/exp_zs/TransEDRP_GDSCv2_drug_classed_100_split_8_1_1_20230611112048/TransE.model'

    main(config, yaml_path)
