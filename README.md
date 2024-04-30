# MSDA
Paper: "Zero-shot Learning for Preclinical Drug Screening"

# How to Run?

1.  CUDA & Pytorch & Main PIP
   
      torch                    1.10.1+cu113
    
      torch-cluster            1.6.0
    
      torch-geometric          2.2.0
    
      torch-scatter            2.0.9
    
      torch-sparse             0.6.13
    
      torch-spline-conv        1.2.1
    
      torch-tb-profiler        0.4.1
    
      rdkit                    2023.3.1
    
      nvidia-cudnn-cu11        8.5.0.96
    

3. Change your root path in mainODD.py and mainODD_NCI60.py
   
      　3 ROOTDIR = "YOURROOTPATH"
   
4. Run mainODD.py for GDSCv2  


5. Run mainODD_NCI60.py for CellMiner 

        python [../mainODD.py] --config [../config/DeepCDR/DeepCDR_GDSCv2_odd_remainRawFusion.yaml]

# Overall 
![image](https://github.com/DrugD/MSDA/assets/37626451/b68f1977-63a3-487e-bc94-fb7002f08d6a)

# Ablation study
<center>
  <img src="https://github.com/DrugD/MSDA/assets/37626451/7a410375-66ae-4b5c-a9ee-70fbd4bce5ed" width=160/>
  <img src="https://github.com/DrugD/MSDA/assets/37626451/78c2b583-e7c2-42b6-a12b-b314e420befd" width=160/>
  <img src="https://github.com/DrugD/MSDA/assets/37626451/60adbe93-2f17-49c8-85b5-714c80c87ed2" width=160/>
  <img src="https://github.com/DrugD/MSDA/assets/37626451/35835b34-b2d1-4a83-b135-912cf3deba64" width=160/>
  <img src="https://github.com/DrugD/MSDA/assets/37626451/ce12d91f-2409-4225-92a8-3d3baf72efd2" width=160/>
</center>
