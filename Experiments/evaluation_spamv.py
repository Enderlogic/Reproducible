import random

import anndata
import pandas
import torch
import sys
import os
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from Methods.SpatialGlue.preprocess import lsi
from Methods.SpaMV.spamv import SpaMV
from Methods.SpaMV.utils import pca, clr_normalize_each_cell, ST_preprocess, clustering
import scanpy as sc
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
result = pandas.DataFrame(
    columns=['Dataset', 'method', 'epoch', 'ari', 'mi', 'nmi', 'ami', 'hom', 'vme', 'average', 'jaccard 1', 'jaccard 2',
             'moran I'])
for dataset in ['5_Mouse_Brain', '4_Human_Lymph_Node', '6_Mouse_Embryo']:
    data_omics1 = sc.read_h5ad('../Dataset/' + dataset + '/adata_RNA.h5ad')
    data_omics1 = ST_preprocess(data_omics1, scale=False if dataset == '6_Mouse_Embryo' else True)
    data_omics1 = anndata.AnnData(pca(data_omics1, n_comps=30 if dataset == '4_Human_Lymph_Node' else 50),
                                  obs=data_omics1.obs, obsm=data_omics1.obsm)
    data_omics1.obsm['X_pca'] = data_omics1.X

    if dataset == '4_Human_Lymph_Node':
        data_omics2 = sc.read_h5ad('../Dataset/' + dataset + '/adata_ADT.h5ad')
        data_omics2 = data_omics2[data_omics1.obs_names, :]
        data_omics2 = clr_normalize_each_cell(data_omics2)
        sc.pp.scale(data_omics2)
        data_omics2 = anndata.AnnData(pca(data_omics2, n_comps=30), obs=data_omics2.obs, obsm=data_omics2.obsm)
        data_omics2.obsm['X_pca'] = data_omics2.X
        n_cluster = 10
        omics_names = ['Transcriptomics', 'Proteomics']
    elif dataset == '5_Mouse_Brain':
        data_omics2 = sc.read_h5ad('../Dataset/' + dataset + '/adata_peaks_normalized.h5ad')
        data_omics2 = data_omics2[data_omics1.obs_names, :]
        data_omics2 = anndata.AnnData(data_omics2.obsm['X_lsi'], obs=data_omics2.obs, obsm=data_omics2.obsm)
        data_omics2.obsm['X_lsi'] = data_omics2.X
        n_cluster = 15
        omics_names = ['Transcriptomics', 'Epigenomics']
    elif dataset == '6_Mouse_Embryo':
        data_omics2 = sc.read_h5ad('../Dataset/' + dataset + '/adata_peaks.h5ad')
        data_omics2 = data_omics2[data_omics1.obs_names, :]
        lsi(data_omics2, use_highly_variable=False, n_components=51)
        data_omics2 = anndata.AnnData(data_omics2.obsm['X_lsi'], obs=data_omics2.obs, obsm=data_omics2.obsm)
        data_omics2.obsm['X_lsi'] = data_omics2.X
        n_cluster = 14
        omics_names = ['Transcriptomics', 'Epigenomics']
    elif dataset == '11_ccRCC_Y7_T':
        data_omics2 = sc.read_h5ad('../Dataset/' + dataset + '/adata_SM.h5ad')
        data_omics2 = data_omics2[data_omics1.obs_names, :]
        sc.pp.filter_genes(data_omics2, min_cells=10)
        sc.pp.normalize_total(data_omics2, target_sum=1e4)
        sc.pp.log1p(data_omics2)
        sc.pp.scale(data_omics2)
        data_omics2 = anndata.AnnData(pca(data_omics2, n_comps=50), obs=data_omics2.obs, obsm=data_omics2.obsm)
        data_omics2.obsm['X_pca'] = data_omics2.X
        n_cluster = 10
        omics_names = ['Transcriptomics', 'Metabolomics']
    else:
        raise ValueError('Unknown dataset')

    if dataset == '4_Human_Lymph_Node':
        weights = [1, 1]
    elif dataset == '5_Mouse_Brain':
        weights = [1, 1]
    elif dataset == '6_Mouse_Embryo':
        weights = [1, 1]
    else:
        weights = [1, 1]
    max_epochs = 1600
    for i in range(10):
        seed = random.randint(1, 10000)
        print('data: {}, iteration: {}, seed {}'.format(dataset, i + 1, seed))
        wandb.init(project=dataset, id=str(seed))
        wandb.login()
        model = SpaMV([data_omics1, data_omics2], weights=weights, interpretable=False, random_seed=seed,
                      learning_rate=1e-4, neighborhood_embedding=10, recon_types=['gauss', 'gauss'],
                      omics_names=omics_names, max_epochs=max_epochs, n_cluster=n_cluster, test_mode=True,
                      device=device, result=result)
        result = model.train(dataset)
        wandb.finish()
        result.to_csv('../Results/Evaluation_SpaMV.csv', index=False)
        print(result.tail(1))
        if i == 0:
            data_omics1.obsm['SpaMV'] = model.get_embedding()
            clustering(data_omics1, key='SpaMV', add_key='SpaMV', n_clusters=10, method='mclust', use_pca=True)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            sc.pp.neighbors(data_omics1, use_rep='SpaMV', n_neighbors=10)
            sc.tl.umap(data_omics1)
            sc.pl.umap(data_omics1, color='SpaMV', ax=axes[0], s=20, show=False)
            sc.pl.embedding(data_omics1, color='SpaMV', basis='spatial', ax=axes[1], s=25, show=False,
                            title='SpaMV\nARI: {:.3f}'.format(
                                adjusted_rand_score(data_omics1.obs['SpaMV'], data_omics1.obs['cluster'])))
            plt.tight_layout()
            plt.savefig('../Results/' + dataset + '/SpaMV.pdf')
            data_omics1.write_h5ad('../Results/' + dataset + '/adata_SpaMV.h5ad')
