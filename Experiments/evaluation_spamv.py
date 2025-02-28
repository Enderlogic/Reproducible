import random

import anndata
import pandas
import torch
import sys
import os

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from Methods.SpatialGlue.preprocess import lsi
from Methods.SpaMV.spamv import SpaMV
from Methods.SpaMV.utils import pca, clr_normalize_each_cell, ST_preprocess
import scanpy as sc
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
result = pandas.DataFrame(
    columns=['Dataset', 'method', 'epoch', 'ari', 'mi', 'nmi', 'ami', 'hom', 'vme', 'average', 'jaccard 1', 'jaccard 2',
             'jaccard', 'moran I'])
for dataset in ['6_Mouse_Embryo', '4_Human_Lymph_Node', '5_Mouse_Brain']:
    data_omics1 = sc.read_h5ad('../Dataset/' + dataset + '/adata_RNA.h5ad')
    data_omics1 = ST_preprocess(data_omics1)
    data_omics1 = anndata.AnnData(pca(data_omics1, n_comps=30 if dataset == '4_Human_Lymph_Node' else 50),
                                  obs=data_omics1.obs, obsm=data_omics1.obsm)

    if dataset == '4_Human_Lymph_Node':
        data_omics2 = sc.read_h5ad('../Dataset/' + dataset + '/adata_ADT.h5ad')
        data_omics2 = clr_normalize_each_cell(data_omics2)
        sc.pp.scale(data_omics2)
        data_omics2 = anndata.AnnData(pca(data_omics2, n_comps=30), obs=data_omics2.obs, obsm=data_omics2.obsm)
        data_omics2.obsm['X_pca'] = data_omics2.X
        neighborhood_depth = 2
        n_cluster = 10
    elif dataset == '5_Mouse_Brain':
        data_omics2 = sc.read_h5ad('../Dataset/' + dataset + '/adata_peaks_normalized.h5ad')
        data_omics2 = anndata.AnnData(data_omics2.obsm['X_lsi'], obs=data_omics2.obs, obsm=data_omics2.obsm)
        data_omics2.obsm['X_lsi'] = data_omics2.X
        neighborhood_depth = 2
        n_cluster = 15
    elif dataset == '6_Mouse_Embryo':
        data_omics2 = sc.read_h5ad('../Dataset/' + dataset + '/adata_peaks.h5ad')
        if 'X_lsi' not in data_omics2.obsm.keys():
            lsi(data_omics2, use_highly_variable=False, n_components=50)
        data_omics2 = anndata.AnnData(data_omics2.obsm['X_lsi'], obs=data_omics2.obs, obsm=data_omics2.obsm)
        data_omics2.obsm['X_lsi'] = data_omics2.X
        neighborhood_depth = 2
        n_cluster = 14
    else:
        raise ValueError('Unknown dataset')
    data_omics2 = data_omics2[data_omics1.obs_names, :]
    if dataset == '4_Human_Lymph_Node':
        weights = [10, 1]
    elif dataset == '5_Mouse_Brain':
        weights = [1, 10]
    else:
        weights = [1, 10]
    omics_names = ['Transcriptomics', 'Proteomics' if dataset == '4_Human_Lymph_Node' else 'Epigenomics']
    for i in range(10):
        seed = random.randint(1, 10000)
        print('data: {}, iteration: {}, seed {}'.format(dataset, i + 1, seed))
        wandb.init()
        wandb.login()
        model = SpaMV([data_omics1, data_omics2], weights=weights, interpretable=False, random_seed=seed,
                      neighborhood_depth=neighborhood_depth, neighborhood_embedding=4, recon_types=['gauss', 'gauss'],
                      omics_names=omics_names, max_epochs=800, n_cluster=n_cluster, test_mode=True, device=device,
                      result=result)
        result = model.train(dataset)
        wandb.finish()

        result.to_csv('../Results/Evaluation_SpaMV.csv', index=False)
        print(result.tail(1))
