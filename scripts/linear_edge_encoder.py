import os
import pickle
import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_edge_encoder
from PGTNet4SSD.PGTNetutils import eventlog_name_provider 

@register_edge_encoder('LinearEdge')
class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        
        if cfg.ssd:
            class_file_dir = os.path.dirname(os.path.abspath(__file__))        
            parent_dir = os.path.dirname(os.path.dirname(class_file_dir))
            raw_dataset_dir =  os.path.join(parent_dir, 'PGTNet4SSD', 'raw_dataset')
            dataset_class_name = cfg.dataset.format.split('-')[1]
            event_log_name = eventlog_name_provider(dataset_class_name)
            edge_dim_path = os.path.join(raw_dataset_dir, 
                                         event_log_name + '#edge_dim.pkl')
            with open(edge_dim_path, 'rb') as f:
                self.in_dim =  pickle.load(f)
        else:        
            if cfg.dataset.name in ['MNIST', 'CIFAR10']:
                self.in_dim = 1            
            # Achtung: followng part is added to the original implementation        
            elif cfg.dataset.name in ['BPIC12CWcycletimeprediction' ,
                                      'BPIC12Ocycletimeprediction',
                                      'BPIC12Wcycletimeprediction']:
                self.in_dim = 68 
            elif cfg.dataset.name in ['BPIC12cycletimeprediction' ,
                                      'BPIC12Ccycletimeprediction']:
                self.in_dim = 77
            elif cfg.dataset.name == 'BPIC12Acycletimeprediction':
                self.in_dim = 69
            elif cfg.dataset.name == 'BPIC13Ccycletimeprediction':
                self.in_dim = 93          
            elif cfg.dataset.name == 'BPIC13Icycletimeprediction':
                self.in_dim = 115
            elif cfg.dataset.name == 'BPIC20Dcycletimeprediction':
                self.in_dim = 17
            elif cfg.dataset.name == 'BPIC20Icycletimeprediction':
                self.in_dim = 407
            elif cfg.dataset.name == 'ENVPERMITcycletimeprediction':
                self.in_dim = 120
            elif cfg.dataset.name == 'HELPDESKcycletimeprediction':
                self.in_dim = 477
            elif cfg.dataset.name == 'HOSPITALcycletimeprediction':
                self.in_dim = 150
            elif cfg.dataset.name == 'SEPSIScycletimeprediction':
                self.in_dim = 250
            elif cfg.dataset.name == 'Trafficfinescycletimeprediction':
                self.in_dim = 267
            elif cfg.dataset.name == 'BPIC15M1cycletimeprediction':
                self.in_dim = 219 
            elif cfg.dataset.name == 'BPIC15M2cycletimeprediction':
                self.in_dim = 142 
            elif cfg.dataset.name == 'BPIC15M3cycletimeprediction':
                self.in_dim = 209
            elif cfg.dataset.name == 'BPIC15M4cycletimeprediction':
                self.in_dim = 141 
            elif cfg.dataset.name == 'BPIC15M5cycletimeprediction':
                self.in_dim = 208 
            # extra condition for ablation study:
            elif 'ablation' in cfg.dataset.name:
                self.in_dim = 6 
            
            # and now back to the original implementation.
            else:
                raise ValueError("Input edge feature dim is required to be"
                                 "hardset or refactored to use a cfg option.")
            
        self.encoder = torch.nn.Linear(self.in_dim, emb_dim)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr.view(-1, self.in_dim))
        return batch