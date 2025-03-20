import random

import anndata
import matplotlib.pyplot as plt
import torch
import sys
import os
from sklearn.metrics import adjusted_rand_score

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from Methods.SpaMV.spamv import SpaMV
from Methods.SpaMV.utils import pca, clr_normalize_each_cell, clustering
import scanpy as sc
import wandb

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
for dataset in ['1_Simulation', '2_Simulation', '3_Simulation']:
    data_rna = sc.read_h5ad('../Dataset/' + dataset + '/adata_RNA.h5ad')
    data_pro = sc.read_h5ad('../Dataset/' + dataset + '/adata_ADT.h5ad')

    sc.pp.normalize_total(data_rna, target_sum=1e4)
    sc.pp.log1p(data_rna)
    sc.pp.scale(data_rna)
    adata_rna = anndata.AnnData(pca(data_rna, n_comps=50), obs=data_rna.obs, obsm=data_rna.obsm)
    data_pro = clr_normalize_each_cell(data_pro)
    sc.pp.scale(data_pro)
    adata_pro = anndata.AnnData(pca(data_pro, n_comps=50), obs=data_pro.obs, obsm=data_pro.obsm)

    wandb.init(project=dataset)
    wandb.login()
    model = SpaMV([adata_rna, adata_pro], betas=[20, 20], interpretable=False, neighborhood_embedding=0,
                  max_epochs=800, random_seed=random.randint(1, 10000), recon_types=['gauss', 'gauss'],
                  omics_names=['Transcriptomics', 'Proteomics'])
    model.train()
    wandb.finish()

    output = model.get_embedding()
    for emb_type in ['all', 'shared', 'Transcriptomics', 'Proteomics']:
        name = 'SpaMV_' + emb_type
        if emb_type == 'all':
            adata_rna.obsm[name] = output
            n_clusters = 10
        elif emb_type == 'shared':
            adata_rna.obsm[name] = output[:, :32]
            if dataset == '1_Simulation':
                n_clusters = 8
            elif dataset == '2_Simulation':
                n_clusters = 6
            elif dataset == '3_Simulation':
                n_clusters = 4
            else:
                raise ValueError('Unknown dataset: {}'.format(dataset))
        else:
            if dataset == '1_Simulation':
                n_clusters = 2
            elif dataset == '2_Simulation':
                n_clusters = 3
            elif dataset == '3_Simulation':
                n_clusters = 4
            else:
                raise ValueError('Unknown dataset: {}'.format(dataset))
            adata_rna.obsm[name] = output[:, 32:64] if emb_type == 'Transcriptomics' else output[:, 64:]
        clustering(adata_rna, key=name, add_key=name, n_clusters=n_clusters, method='mclust', use_pca=True)
        sc.pp.neighbors(adata_rna, use_rep=name, n_neighbors=30, key_added=name)
    fig, ax_list = plt.subplots(2, 4, figsize=(24, 10))
    i = 0
    for emb_type in ['all', 'shared', 'Transcriptomics', 'Proteomics']:
        name = 'SpaMV_' + emb_type
        sc.tl.umap(adata_rna, neighbors_key=name)
        sc.pl.umap(adata_rna, color=name, ax=ax_list[0][i], s=30, show=False)
        sc.pl.embedding(adata_rna, color=name, basis='spatial', ax=ax_list[1][i], s=250, show=False,
                        title=name + '\nARI: {:.3f}'.format(adjusted_rand_score(adata_rna.obs['cluster'], adata_rna.obs[
                            name])) if emb_type == 'all' else name)
        i += 1
    plt.tight_layout(w_pad=0.3)
    plt.savefig('../Results/' + dataset + '/Spamv.pdf')
    plt.show()
