import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/home/lk/project/MSDA_DRP')

from utils import *
import argparse
import pdb
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, "./")


res_dict = {
    "DeepCDR_res_dict": {
        'Ponatinib': ['0.0718', '0.0051', '0.6674', '0.5204', '0.0653'],
        'Rucaparib': ['0.0236', '0.0005', '0.4127', '0.4106', '0.0074'],
        '5-Fluorouracil': ['0.0448', '0.0020', '0.4734', '0.4538', '0.0295'],
        'PF-562271': ['0.0478', '0.0022', '0.3692', '0.3441', '0.0413'],
        'Cytarabine': ['0.0917', '0.0084', '0.4156', '0.3993', '0.0835'], 
        'GSK319347A': ['0.0268', '0.0007', '0.2737', '0.2681', '0.0011'],
        'Sepantronium bromide': ['0.1849', '0.0341', '0.2473', '0.2369', '0.1762'],
        'JNK-9L': ['0.0742', '0.0055', '0.2677', '0.2645', '0.0696'], 
        'Y-39983': ['0.0662', '0.0043', '0.4809', '0.4886', '0.0007'],
        'IOX2': ['0.0418', '0.0017', '0.2763', '0.3368', '0.0335'], 
        'JW-7-24-1': ['0.0290', '0.0008', '0.6684', '0.6635', '0.0078'], 'Pelitinib': ['0.0627', '0.0039', '0.3698', '0.3526', '0.0495'], 'Tamoxifen': ['0.0215', '0.0004', '0.5120', '0.5215', '0.0138'], 'Alectinib': ['0.0352', '0.0012', '0.2814', '0.3669', '0.0038'], 'XMD8-85': ['0.0320', '0.0010', '0.4686', '0.4817', '0.0128'], 'Belinostat': ['0.0624', '0.0038', '0.4934', '0.5058', '0.0516'], 'Piperlongumine': ['0.0537', '0.0028', '0.4054', '0.3752', '0.0495'], 'Zibotentan': ['0.1309', '0.0171', '0.5084', '0.4980', '0.0'], 'TAK-715': ['0.0215', '0.0004', '0.7078', '0.7096', '0.0047'], 'QL-XII-47': ['0.0618', '0.0038', '0.5634', '0.5646', '0.0517'], 'Navitoclax': ['0.0583', '0.0034', '0.3441', '0.3075', '0.0404'], 'GW-2580': ['0.0666', '0.0044', '0.4755', '0.4953', '0.0001'], 'Vinblastine': ['0.0395', '0.0015', '0.5339', '0.5319', '0.0273']
    },
    "DeepTTC_res_dict": {
        'Ponatinib': ['0.0725', '0.0052', '0.6465', '0.5188', '0.0655'], 'Rucaparib': ['0.0290', '0.0008', '0.4914', '0.4872', '0.0030'], '5-Fluorouracil': ['0.0444', '0.0019', '0.4122', '0.3832', '0.0082'], 'PF-562271': ['0.0747', '0.0055', '0.3936', '0.4003', '0.0004'], 'Cytarabine': ['0.0864', '0.0074', '0.3996', '0.4219', '0.0049'], 'GSK319347A': ['0.1583', '0.0250', '0.3179', '0.3156', '0.0'], 'Sepantronium bromide': ['0.1741', '0.0303', '0.1638', '0.0942', '0.1647'], 'JNK-9L': ['0.0756', '0.0057', '0.3403', '0.3441', '0.0703'], 'Y-39983': ['0.0545', '0.0029', '0.4514', '0.4273', '0.0014'], 'IOX2': ['0.0559', '0.0031', '0.4024', '0.5040', '0.0519'], 'JW-7-24-1': ['0.0340', '0.0011', '0.4713', '0.4351', '0.0094'], 'Pelitinib': ['0.0659', '0.0043', '0.4322', '0.4444', '0.0537'], 'Tamoxifen': ['0.0218', '0.0004', '0.4852', '0.5047', '0.0041'], 'Alectinib': ['0.0359', '0.0012', '0.2830', '0.3299', '0.0038'], 'XMD8-85': ['0.0256', '0.0006', '0.4884', '0.4756', '0.0101'], 'Belinostat': ['0.0745', '0.0055', '0.3543', '0.3995', '0.0643'], 'Piperlongumine': ['0.0250', '0.0006', '0.3812', '0.3593', '0.0078'], 'Zibotentan': ['0.0780', '0.0060', '0.5811', '0.5782', '4.1701'], 'TAK-715': ['0.0424', '0.0018', '0.4079', '0.3933', '0.0016'], 'QL-XII-47': ['0.0434', '0.0018', '0.3680', '0.3686', '0.0238'], 'Navitoclax': ['0.0534', '0.0028', '0.2129', '0.1869', '0.0326'], 'GW-2580': ['0.0880', '0.0077', '0.4967', '0.5335', '3.9968'], 'Vinblastine': ['0.0324', '0.0010', '0.5368', '0.5179', '0.0170']
    },
    "GraTrans_res_dict": {
        'Ponatinib': ['0.0679', '0.0046', '0.7276', '0.5612', '0.0621'],
        'Rucaparib': ['0.0379', '0.0014', '0.4635', '0.4690', '0.0014'],
        '5-Fluorouracil': ['0.0409', '0.0016', '0.5564', '0.5422', '0.0072'],
        'PF-562271': ['0.0612', '0.0037', '0.3671', '0.3607', '0.0549'],
        'Cytarabine': ['0.0661', '0.0043', '0.4251', '0.4422', '0.0056'],
        'GSK319347A': ['0.1308', '0.0171', '0.3896', '0.3973', '0.0'],
        'Sepantronium bromide': ['0.1144', '0.0130', '0.4482', '0.4034', '0.1037'],
        'JNK-9L': ['0.0611', '0.0037', '0.2580', '0.2969', '0.0551'],
        'Y-39983': ['0.0648', '0.0042', '0.5805', '0.5721', '0.0005'],
        'IOX2': ['0.0598', '0.0035', '0.3180', '0.3811', '0.0497'],
        'JW-7-24-1': ['0.0343', '0.0011', '0.6011', '0.5734', '0.0057'],
        'Pelitinib': ['0.0502', '0.0025', '0.4489', '0.4458', '0.0358'],
        'Tamoxifen': ['0.0249', '0.0006', '0.5425', '0.5740', '0.0024'],
        'Alectinib': ['0.0256', '0.0006', '0.3699', '0.4207', '0.0079'],
        'XMD8-85': ['0.0354', '0.0012', '0.4217', '0.4651', '0.0083'],
        'Belinostat': ['0.0577', '0.0033', '0.6502', '0.7200', '0.0489'],
        'Piperlongumine': ['0.0250', '0.0006', '0.4048', '0.3949', '0.0084'],
        'Zibotentan': ['0.1868', '0.0349', '0.4569', '0.4479', '0.0'],
        'TAK-715': ['0.0214', '0.0004', '0.6742', '0.6736', '0.0087'],
        'QL-XII-47': ['0.0381', '0.0014', '0.6263', '0.6288', '0.0234'],
        'Navitoclax': ['0.0607', '0.0036', '0.3831', '0.3202', '0.0442'],
        'GW-2580': ['0.0671', '0.0045', '0.5332', '0.5722', '0.0001'],
        'Vinblastine': ['0.0363', '0.0013', '0.4835', '0.4855', '0.0181']
    },

    "tCNNs_res_dict": {
        'Ponatinib': ['0.0799', '0.0063', '0.6143', '0.4824', '0.0734'],
        'Rucaparib': ['0.0223', '0.0004', '0.4449', '0.4458', '0.0085'],
        '5-Fluorouracil': ['0.0660', '0.0043', '0.4035', '0.3673', '0.0538'],
        'PF-562271': ['0.0404', '0.0016', '0.5503', '0.5440', '0.0345'],
        'Cytarabine': ['0.0550', '0.0030', '0.3755', '0.3680', '0.0419'],
        'GSK319347A': ['0.0488', '0.0023', '0.4078', '0.3845', '0.0002'],
        'Sepantronium bromide': ['0.1177', '0.0138', '0.3446', '0.3410', '0.1064'],
        'JNK-9L': ['0.0684', '0.0046', '0.4376', '0.4909', '0.0640'],
        'Y-39983': ['0.0575', '0.0033', '0.6667', '0.6678', '0.0006'],
        'IOX2': ['0.0358', '0.0012', '0.3215', '0.3819', '0.0264'],
        'JW-7-24-1': ['0.0308', '0.0009', '0.6027', '0.5894', '0.0082'],
        'Pelitinib': ['0.0490', '0.0024', '0.4364', '0.4381', '0.0337'],
        'Tamoxifen': ['0.0468', '0.0021', '0.4714', '0.5061', '0.0004'],
        'Alectinib': ['0.0719', '0.0051', '0.4697', '0.5476', '0.0007'],
        'XMD8-85': ['0.0291', '0.0008', '0.5212', '0.4914', '0.0043'],
        'Belinostat': ['0.0567', '0.0032', '0.4871', '0.5171', '0.0446'],
        'Piperlongumine': ['0.0249', '0.0006', '0.4374', '0.4456', '0.0160'],
        'Zibotentan': ['0.0878', '0.0077', '0.5628', '0.5612', '1.0808'],
        'TAK-715': ['0.0714', '0.0051', '0.6098', '0.5906', '0.0001'],
        'QL-XII-47': ['0.0398', '0.0015', '0.5057', '0.5301', '0.0214'],
        'Navitoclax': ['0.0585', '0.0034', '0.0911', '0.0852', '0.0142'],
        'GW-2580': ['0.0613', '0.0037', '0.4446', '0.5029', '0.0002'],
        'Vinblastine': ['0.0442', '0.0019', '0.5228', '0.5286', '0.0318']
    },

    "TransEDRP_res_dict": {
        "Ponatinib": ['0.0727', '0.0052', '0.6840', '0.5121', '0.0666'],
        "Rucaparib": ['0.0268', '0.0007', '0.5629', '0.5303', '0.0031'],
        "5-Fluorouracil": ['0.0381', '0.0014', '0.5837', '0.5680', '0.0082'],
        "PF-562271": ['0.0585', '0.0034', '0.4496', '0.4431', '0.0534'],
        "Cytarabine": ['0.0753', '0.0056', '0.4613', '0.4577', '0.0637'],
        "GSK319347A":  ['0.1186', '0.0140', '0.3999', '0.3828', '0.0'],
        "Sepantronium bromide": ['0.1661', '0.0276', '0.3751', '0.3630', '0.1573'],
        "JNK-9L": ['0.0490', '0.0024', '0.4381', '0.4730', '0.0422'],
        "Y-39983": ['0.0586', '0.0034', '0.6318', '0.6316', '0.0006'],
        "IOX2": ['0.0770', '0.0059', '0.3747', '0.4684', '0.0726'],
        "JW-7-24-1": ['0.0578', '0.0033', '0.4687', '0.4421', '0.0012'],
        "Pelitinib": ['0.0511', '0.0026', '0.5070', '0.4819', '0.0377'],
        "Tamoxifen": ['0.0220', '0.0004', '0.5153', '0.5406', '0.0037'],
        "Alectinib": ['0.0256', '0.0006', '0.4178', '0.4813', '0.0068'],
        "XMD8-85": ['0.0299', '0.0008', '0.5075', '0.5033', '0.0069'],
        "Belinostat": ['0.0413', '0.0017', '0.5958', '0.6401', '0.0251'],
        "Piperlongumine": ['0.0352', '0.0012', '0.5419', '0.5627', '0.0019'],
        "Zibotentan":  ['0.1339', '0.0179', '0.5630', '0.5741', '0.0'],
        "TAK-715": ['0.0339', '0.0011', '0.6286', '0.6056', '0.0016'],
        "QL-XII-47": ['0.0416', '0.0017', '0.4800', '0.4848', '0.0092'],
        "Navitoclax": ['0.0757', '0.0057', '0.3994', '0.3536', '0.0621'],
        "GW-2580": ['0.0875', '0.0076', '0.4596', '0.4981', '7.1912'],
        "Vinblastine": ['0.0602', '0.0036', '0.5138', '0.5058', '0.0504']
    }
}


def getConfig():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="/home/lk/project/MSDA_DRP/config/DeepCDR/DeepCDR_GDSCv2_odd.yaml",
        help="",
    )
    args = parser.parse_args()
    import yaml

    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    return config, args.config

# root_path = "/home/lk/project/MSDA_DRP/exp/exp_odd_gratrans"
# root_path = "/home/lk/project/MSDA_DRP/exp/exp_odd_deepcdr_new"
# root_path = "/home/lk/project/MSDA_DRP/exp/exp_odd_deepcdr"
# root_path = "/home/lk/project/MSDA_DRP/exp/exp_odd_deepttc"
root_path = "/home/lk/project/MSDA_DRP/exp/exp_odd_tcnns"
metrixs = [
    [], [], [], [], []
]
metrixs_raw = [
    [], [], [], [], []
]
drug_map = {}

Ktop = 35
model = "tCNNs"

Compare_dict = res_dict[model+"_res_dict"]
# TransEDRP
# TransEDRP
# tCNNs
# GraTrans
# DeepCDR
# DeepTTC

# nanDict = []

for item in os.listdir(root_path):
    # if "test10" in item:
    #  and "top80"  in item and "remain"  in item
    if "K35_10epoch" in item and "remain"  in item:
        cvs_root_path = os.path.join(root_path, item)+"/" + model + ".csv"

        df = pd.read_csv(cvs_root_path, encoding="utf-8")
        data_ = df[-5:]

        # pdb.set_trace()
        drug_name = item.split('__')[0].split('_')[-2:-1]
        drug_name = '_'.join([str(i) for i in drug_name])
        # drug_name = item.split('_')[1]
        if Compare_dict.get(drug_name) is None:
            print(drug_name)
            continue
        
        # if np.array(data_).tolist()[0][0] == ' rmse:nan':
        #     nanDict.append(drug_name)
            
        for idx, item_ in enumerate(np.array(data_).tolist()):
            metrixs[idx].append(float(item_[0].split(':')[1]))

        for idx, item_ in enumerate(np.array([float(x) for x in Compare_dict[drug_name]]).tolist()):

            metrixs_raw[idx].append(item_)
        
        # epoch1_100 = [  [float(x[0].split(":")[1]) for x in np.array(df[16:21]).tolist()] ,
        #                           [float(x[0].split(":")[1]) for x in np.array(data_).tolist()] ]
        epoch0_100 = [[float(x) for x in Compare_dict[drug_name]],
                      [round(float(x[0].split(":")[1]), 4) for x in np.array(data_).tolist()]]

        epoch0 = np.array([float(x) for x in Compare_dict[drug_name]])
        a1 = np.array([[1, 1, -1, -1, 1], [-1, -1, 1, 1, -1]])
        a2 = np.array(epoch0_100)
        improve = (np.sum(a1*a2, axis=0)/np.abs(a2[0]))
        where_are_nan = np.isnan(improve)
        where_are_inf = np.isinf(improve)

        # nan替换成0,inf替换成nan
        improve[where_are_nan] = 0
        improve[where_are_inf] = 0

        drug_map[drug_name] = {'raw': epoch0_100,
                               'improve': round(np.mean(improve), 5)}

# pdb.set_trace()
# top K


config, yaml_path = getConfig()

# Similarity_matrix = np.load("/home/lk/project/NMI_DRP/data/Similarity_matrix_GDSCv2.npy",allow_pickle=True).item()
Similarity_matrix = np.load(
    "/home/lk/project/MSDA_DRP/data/Similarity_matrix_by_wasserstein_GDSCv2.npy", allow_pickle=True).item()


smiles2pt_mp, cell_name_mp = get_dict_smiles2pt(config["dataset_type"])

# model.get_static(cell_name_mp)
#

# 举一个例子：


for drug_name in drug_map:

    test_drug = drug_name+'.pt'

    Target_dataset = load_dataset_from_smiles(
        config["dataset_type"], test_drug)

    test_aim_sorted = Similarity_matrix[Target_dataset[0].smiles]
    test_aim_sorted = sorted(test_aim_sorted.items(),
                             key=lambda d: d[1], reverse=True)

    test_smiles = [
        'COC1=CC(=CC(=C1OC)OC)/C=C/C(=O)N2CCC=CC2=O', 'C1CC1CONC(=O)C2=C(C(=C(C=C2)F)F)NC3=C(C=C(C=C3)I)Cl', 'C[C@H]1COCCN1C2=NC(=NC3=C2C=CC(=N3)C4=CC(=C(C=C4)OC)CO)N5CCOC[C@@H]5C', 'C1=CC=C(C=C1)CN2C3=CC=CC=C3C(=C(C2=O)C(=O)NCC(=O)O)O', 'CC1=[N+](C2=C(N1CCOC)C(=O)C3=CC=CC=C3C2=O)CC4=NC=CN=C4.[Br-]', 'COC1=CC=C(C=C1)COC2=C(C=C(C=C2)CC3=CN=C(N=C3N)N)OC', 'CCC1=NC(=C(S1)C2=CC(=NC=C2)NC(=O)C3=CC=CC=C3)C4=CC(=CC=C4)C', 'CC1=C(SC(=N1)NC(=O)C)C2=CC(=C(C=C2)Cl)S(=O)(=O)NCCO', 'CC1=NC=C(C=C1)OC2=C(C=C(C=C2)NC3=NC=NC4=C3C=C(C=C4)/C=C/CNC(=O)COC)C', 'COC1=CC(=CC(=C1)C2=CC3=C4C(=CN=C3C=C2)C=CC(=O)N4C5=CC(=C(C=C5)N6CCNCC6)C(F)(F)F)OC', 'CC1=C(C=C(C=C1)C(=O)NC2=CC(=C(C=C2)CN3CCN(CC3)C)C(F)(F)F)C#CC4=CN=C5N4N=CC=C5', 'CC1=CC(=C(C=C1)F)NC(=O)NC2=CC=C(C=C2)C3=C4C(=CC=C3)NN=C4N', 'CC(C)N1C=NC2=C1N=C(N=C2NC3=CC(=CC=C3)NC(=O)C=C)NC4CCC(CC4)N(C)C', 'C1=CC=C(C(=C1)N)NC(=O)C2=CC=C(C=C2)CNC(=O)OCC3=CN=CC=C3', 'CCCCC(C=O)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](CC(C)C)NC(=O)OCC1=CC=CC=C1', 'CC(C)(C)OC(=O)NC1=CC=C(C=C1)C2=CC(=NO2)C(=O)NCCCCCCC(=O)NO', 'C1COCCN1C2=CC(=CC(=C2)C3=NC(=NC=C3)NC4=CC=C(C=C4)N5C=NC(=N5)N6CCOCC6)F', 'C1=CC=C(C=C1)[C@H](COC2=CC3=C(C=C2)NC(=O)N3)NC(=O)C4=CC=CN(C4=O)CC5=CC(=C(C=C5)F)F', 'COC1=C(C=C2C(=C1)N=CN2C3=CC(=C(S3)C#N)OCC4=CC=CC=C4S(=O)(=O)C)OC', 'C1CN(CCN1C2=NC=NC3=C2OC4=CC=CC=C43)C(=S)NCC5=CC6=C(C=C5)OCO6', 'C1=CN(C(=O)N=C1N)[C@H]2[C@H]([C@@H]([C@H](O2)CO)O)O', 'C[C@H](C1=CC=C(C=C1)C(=O)NC2=C3C=CNC3=NC=C2)N.Cl', 'C1=CC=C2C(=C1)C(=CN2)CCNC3=CC=C(C=C3)NC4=CC=NC=C4', 'CCOC1=C(C=C2C(=C1)N=CC(=C2NC3=CC(=C(C=C3)F)Cl)C#N)NC(=O)/C=C/CN(C)C', 'C1=C(C(=O)NC(=O)N1)F', 'C1CN(CCC1N2C3=CC=CC=C3NC2=O)CC4=CC=C(C=C4)C5=NC6=CC7=C(C=C6N=C5C8=CC=CC=C8)N=CN7', 'CC1=CN=C(N1)C2=CN=C(N=C2C3=C(C=C(C=C3)Cl)Cl)NCCNC4=NC=C(C=C4)C#N', 'CNCC1=CC=C(C=C1)C2=C3CCNC(=O)C4=CC(=CC(=C34)N2)F', 'CN(C1=C(C=CC=N1)CNC2=NC(=NC=C2C(F)(F)F)NC3=CC4=C(C=C3)NC(=O)C4)S(=O)(=O)C', 'CC1=CN2C(=O)C=C(N=C2C(=C1)C(C)NC3=CC=CC=C3)N4CCOCC4', 'C1CC2=CC=CC=C2[C@H]1NC3=NC=NC4=C3C=CN4[C@@H]5C[C@H]([C@H](C5)O)COS(=O)(=O)N', 'CC1=CN=C(C(=N1)OC)NS(=O)(=O)C2=C(N=CC=C2)C3=CC=C(C=C3)C4=NN=CO4', 'CN1C=C(C=N1)C2=CC3=C4C(=CN=C3C=C2)C=CC(=O)N4C5=CC6=C(CCN6C(=O)C=C)C=C5', 'CCCS(=O)(=O)NC1=C(C(=C(C=C1)F)C(=O)C2=CNC3=NC=C(C=C23)Cl)F', 'CC[C@@]1(C[C@@H]2C[C@@](C3=C(CCN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)[C@]78CCN9[C@H]7[C@@](C=CC9)([C@H]([C@@]([C@@H]8N6C)(C(=O)OC)O)OC(=O)C)CC)OC)C(=O)OC)O', 'C1=CC=C(C=C1)NS(=O)(=O)C2=CC=CC(=C2)/C=C/C(=O)NO', 'CN1CCN(CC1)C2=CC(=C(C=C2)NC3=NC=C4C(=N3)N(C5=CC=CC=C5C(=O)N4C)C)OC', 'C1CC(=O)NC(=O)C1N2CC3=C(C2=O)C=CC=C3N', 'COC1=C(C=C2C(=C1)N=CN=C2NC3=CC=CC(=C3)C#C)OCCCCCCC(=O)NO', 'COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC', 'CC/C(=C(\\C1=CC=CC=C1)/C2=CC=C(C=C2)OCCN(C)C)/C3=CC=CC=C3', 'CCC1=C(C=C2C(=C1)C(=O)C3=C(C2(C)C)NC4=C3C=CC(=C4)C#N)N5CCC(CC5)N6CCOCC6', 'C1=CC(=C(C(=C1)F)N(C2=NC(=C(C=C2)C(=O)N)C3=C(C=C(C=C3)F)F)C(=O)N)F', 'CC1(CCC(=C(C1)CN2CCN(CC2)C3=CC=C(C=C3)C(=O)NS(=O)(=O)C4=CC(=C(C=C4)N[C@H](CCN5CCOCC5)CSC6=CC=CC=C6)S(=O)(=O)C(F)(F)F)C7=CC=C(C=C7)Cl)C', 'CCC1=CN=CN=C1N2CCN(CC2)CC3=NC4=C(N3)C=C(C=C4)C(F)(F)F', 'CC1=C(C=C(C=C1)NC(=O)C2=CC(=CC=C2)C(C)(C)C#N)NC3=CC4=C(C=C3)N=CN(C4=O)C'
    ]

    test_aim_sorted_del = []

    '''删除在测试集中的topK'''
    for smile_key in test_aim_sorted:

        if smile_key[0] not in test_smiles:
            test_aim_sorted_del.append(smile_key)
            # test_aim_sorted.pop(str(smile_key[0]))
            # del test_aim_sorted[smile_key[0]]

    # pdb.set_trace()

    test_aim_sorted_top = test_aim_sorted_del[1:1+Ktop]
    drug_map[drug_name]['topk'] = test_aim_sorted_top


improve = []
similiry = []

print(np.mean(metrixs_raw, 1))
print(np.mean(metrixs, 1))
print(len(metrixs[0]))
imp_mean = 0
for i in drug_map:
    if drug_map[i]['improve'] == 0.0:
        continue
    imp_mean += drug_map[i]['improve']
    improve.append(drug_map[i]['improve'])
    # pdb.set_trace()
    similiry.append(np.mean([drug_map[i]['topk'][0][1]]))
    print(i, '\t', drug_map[i]['improve'], '\t', drug_map[i]['topk'][0][1])


# xmin = min(improve)
# xmax=max(improve)
# for i, x in enumerate(improve):
#     improve[i] = (x-xmin) / (xmax-xmin)


xmin = min(similiry)
xmax = max(similiry)
for i, x in enumerate(similiry):
    similiry[i] = (x-xmin) / (xmax-xmin)


plt.figure(figsize=(10, 10), dpi=100)

similiry = [-x for x in similiry]
plt.scatter(similiry, improve)
# plt.scatter(improve, similiry)
plt.xlabel('cost distance')
# plt.xlabel('similiry')
plt.ylabel('improve')

plt.savefig(
    "/home/lk/project/MSDA_DRP/result/{}_cost_topK_{}_improve.jpg".format(model, Ktop))
# plt.savefig("/home/lk/project/ODD/NMI_DRP/result/simility_topK_{}_improve.jpg".format(Ktop))

print(pearson(improve, similiry))
pdb.set_trace()

# K1  [0.08019055 0.0088903  0.42634929 0.42362967 0.01519748]
#     [0.08020064 0.0088927  0.42640685 0.42369376 0.01517559]
#     [0.08020054 0.00889265 0.42627984 0.42362674 0.01519257]

# K3  [0.06526923 0.00527457 0.50175728 0.49681443 0.01209974]
#     [0.06527468 0.0052763  0.50168342 0.49677893 0.01210131]

# K5  array([0.06264258, 0.0047959 , 0.48837023, 0.48413619, 0.01375839])

# K10 array([0.05473766, 0.0034791 , 0.49866046, 0.50196942, 0.01909889])

# K15 array([0.05706712, 0.0038948 , 0.51201627, 0.510809  , 0.02250435])

# K20 [0.05876012 0.00424694 0.45174406 0.45345949 0.01947629]
#     [0.05869205 0.00414074 0.50640806 0.50573957 0.02438102]

# K25 [0.05925139 0.00425756 0.51542293 0.51372925 0.02463982]
# K35 [0.0590667  0.00423162 0.52732409 0.52726751 0.02507983]
# K50 [0.05924674 0.00423116 0.52573874 0.52632341 0.02550996]


'''

import pdb
import numpy as np
import pandas as pd
import os,sys


for KK in ['K1_','K5_','K10_',"K15_","K20_","K50_"]:
    root_path =  "/home/lk/project/ODD/NMI_DRP/exp_mmd"
    metrixs = [
                
            ]


    for item in os.listdir(root_path):
        if KK in item:
            cvs_root_path = os.path.join(root_path,item)+"/ODDTransE.csv"
        
            df = pd.read_csv(cvs_root_path,encoding="utf-8")
            
            data_1 = df[-5:]
            data_2 = df[-27:-22]
            data_3 = df[-49:-44]

            temp = [
                [],[],[],[],[]
            ]
            
            for idx, item in enumerate(np.array(data_1).tolist()):
                temp[idx].append(float(item[0].split(':')[1]))
            for idx, item in enumerate(np.array(data_2).tolist()):
                temp[idx].append(float(item[0].split(':')[1]))
            for idx, item in enumerate(np.array(data_3).tolist()):
                temp[idx].append(float(item[0].split(':')[1]))
                
            # pdb.set_trace()
            
            metrixs.append(np.mean(temp,1))
    print(KK[:-1])      
    print(np.mean(np.array(metrixs),0)) 
    # print(len(metrixs))
pdb.set_trace()


# K1
# [0.08032175 0.00892451 0.42459487 0.42126994 0.01521619]
# K5
# [0.06261167 0.0047924  0.4908584  0.48646969 0.01374447]
# K10
# [0.05485653 0.00349611 0.49823476 0.50048533 0.01913364]
# K15
# [0.05685454 0.00388618 0.50866372 0.50737358 0.02234534]
# K20
# [0.05868725 0.0042403  0.45324148 0.45398905 0.01943183]
# K50
# [0.05935383 0.00425024 0.52627877 0.52650173 0.02566245]


'''
