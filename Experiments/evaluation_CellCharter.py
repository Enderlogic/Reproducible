import warnings
warnings.filterwarnings("ignore")
import random
import os
import scanpy as sc
import cellcharter as cc
import pandas
import sys
sys.path.append('/home/makx/Reproducible-main')
from Methods.SpaMV.utils import pca, clr_normalize_each_cell, ST_preprocess
from Methods.SpaMV.metrics import compute_moranI, compute_jaccard, compute_supervised_scores
import scvi
import matplotlib as mpl
import squidpy as sq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
from matplotlib.colors import ListedColormap
import h5py
import pickle
import anndata
from lightning.pytorch import seed_everything


result = pandas.DataFrame(
    columns=['Dataset', 'method', 'ari', 'mi', 'nmi', 'ami', 'hom', 'vme', 'average', 'jaccard 1', 'jaccard 2',
             'moran I'])
early_stopping_kwargs = {
    "early_stopping_metric": "val_unweighted_loss",
    "threshold": 0,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}


for dataset in ['4_Human_Lymph_Node','5_Mouse_Brain']:
    data_omics1 = sc.read_h5ad('Dataset/' + dataset + '/adata_RNA.h5ad')
    sc.pp.filter_genes(data_omics1, min_counts=3)
    data_omics1.layers["counts"] = data_omics1.X.copy()  # preserve counts
    sc.pp.normalize_total(data_omics1, target_sum=1e4)
    sc.pp.log1p(data_omics1)
    data_omics1 = ST_preprocess(data_omics1, n_top_genes=3000, n_comps=50)
    data_omics1 = data_omics1[:, data_omics1.var.highly_variable]

    if dataset == '4_Human_Lymph_Node':
        data_omics2 = sc.read_h5ad('Dataset/' + dataset + '/adata_ADT.h5ad')
        data_omics2.layers["counts"] = data_omics2.X.copy()
        data_omics2 = clr_normalize_each_cell(data_omics2)
        sc.pp.scale(data_omics2)
        # data_omics2 = anndata.AnnData(pca(data_omics2, n_comps=30), obs=data_omics2.obs, obsm=data_omics2.obsm)
        # data_omics2.obsm['X_pca'] = data_omics2.X
        adata = sc.concat([data_omics1, data_omics2], axis=1, merge='same')
        adata.obs['sample'] = 'Human_lymph_node'
        data_omics1.obs['sample'] = 'Human_lymph_node'
        data_omics2.obs['sample'] = 'Human_lymph_node'
        data_omics1.obs['sample'] = pd.Categorical(data_omics1.obs['sample'])
        data_omics2.obs['sample'] = pd.Categorical(data_omics2.obs['sample'])
        adata.obs['sample'] = pd.Categorical(adata.obs['sample'])
    elif dataset == '5_Mouse_Brain':
        data_omics2 = sc.read_h5ad('Dataset/' + dataset + '/adata_peaks_normalized.h5ad')
        data_omics2.layers["counts"] = data_omics2.X.copy()
        # data_omics2 = anndata.AnnData(data_omics2.obsm['X_lsi'], obs=data_omics2.obs, obsm=data_omics2.obsm)
        # data_omics2.obsm['X_lsi'] = data_omics2.X
        adata = sc.concat([data_omics1, data_omics2], axis=1, merge='same')
        adata.obs['sample'] = 'P22_mouse_brain'
        data_omics1.obs['sample'] = 'P22_mouse_brain'
        data_omics2.obs['sample'] = 'P22_mouse_brain'
        data_omics1.obs['sample'] = pd.Categorical(data_omics1.obs['sample'])
        data_omics2.obs['sample'] = pd.Categorical(data_omics2.obs['sample'])
        adata.obs['sample'] = pd.Categorical(adata.obs['sample'])
    elif dataset == '6_Mouse_Embryo':
        data_omics2 = sc.read_h5ad('Dataset/' + dataset + '/adata_peaks.h5ad')
        data_omics2.layers["counts"] = data_omics2.X.copy()
        # data_omics2 = anndata.AnnData(data_omics2.obsm['X_lsi'], obs=data_omics2.obs, obsm=data_omics2.obsm)
        # data_omics2.obsm['X_lsi'] = data_omics2.X
        adata = sc.concat([data_omics1, data_omics2], axis=1, merge='same')
        adata.obs['sample'] = 'mouse_embryo'
        data_omics1.obs['sample'] = 'mouse_embryo'
        data_omics2.obs['sample'] = 'mouse_embryo'
        data_omics1.obs['sample'] = pd.Categorical(data_omics1.obs['sample'])
        data_omics2.obs['sample'] = pd.Categorical(data_omics2.obs['sample'])
        adata.obs['sample'] = pd.Categorical(adata.obs['sample'])
    else:
        raise ValueError('Unknown dataset')
    
    adata.uns['spatial'] = {s: {} for s in adata.obs['sample'].unique()}
    adata.obs['cluster'] = data_omics2.obs['cluster'].copy()
    condition_key = 'sample'
    conditions = ['Human_lymph_node']
    for i in range(10):
        seed = random.randint(1, 10000)
        seed_everything(seed)
        print('data: {}, iteration: {}, seed {}'.format(dataset, i + 1, seed))
        if dataset == '4_Human_Lymph_Node':
            model_omics1 = scvi.model.SCVI.setup_anndata(
            data_omics1,
            layer="counts",
            batch_key='sample',)
            model_omics1 = scvi.model.SCVI(data_omics1)
            model_omics1.train(early_stopping=True, enable_progress_bar=True)
            data_omics1.obsm['X_latent'] = model_omics1.get_latent_representation()

            model_omics2 = cc.tl.TRVAE(data_omics2, condition_key=condition_key, conditions=conditions,)
            model_omics2.train(
            n_epochs=500,
            alpha_epoch_anneal=200,
            early_stopping_kwargs=early_stopping_kwargs )
            data_omics2.obsm['X_latent'] = model_omics2.get_latent(data_omics2.X, data_omics2.obs['sample'])
        elif dataset == '5_Mouse_Brain' or dataset == '6_Mouse_Embryo':
            model_omics1 = scvi.model.SCVI.setup_anndata(
            data_omics1,
            layer="counts",
            batch_key='sample',)
            model_omics1 = scvi.model.SCVI(data_omics1)
            model_omics1.train(early_stopping=True, enable_progress_bar=True)
            data_omics1.obsm['X_latent'] = model_omics1.get_latent_representation()

            model_omics2 = scvi.model.SCVI.setup_anndata(
            data_omics2,
            layer="counts",
            batch_key='sample',)
            model_omics2 = scvi.model.SCVI(data_omics2)
            model_omics2.train(early_stopping=True, enable_progress_bar=True)
            data_omics2.obsm['X_latent'] = model_omics2.get_latent_representation()
        else:
            raise ValueError('Unknown dataset')
        adata.obsm['X_latent'] = np.concatenate([data_omics1.obsm['X_latent'], data_omics2.obsm['X_latent']], axis=1)
        sq.gr.spatial_neighbors(adata, library_key='sample', coord_type='grid', n_neighs=4, n_rings=1)
        cc.gr.aggregate_neighbors(adata, n_layers=4, use_rep='X_latent')
        if dataset == '4_Human_Lymph_Node':
            gmm = cc.tl.Cluster(
            n_clusters=10,
            random_state=seed,)
            gmm.fit(adata, use_rep='X_cellcharter')
            adata.obs['X_cellcharter'] = gmm.predict(adata, use_rep='X_cellcharter')
        elif dataset == '5_Mouse_Brain':
            gmm = cc.tl.Cluster(
            n_clusters=15,
            random_state=seed,)
            gmm.fit(adata, use_rep='X_cellcharter')
            adata.obs['X_cellcharter'] = gmm.predict(adata, use_rep='X_cellcharter')
            digit_labels = list(adata.obs['X_cellcharter'])
            for i in range(len(digit_labels)):
                if digit_labels[i] not in [0,1,2,4,5,6,7,8,11]:
                    digit_labels[i] = 100
            adata.obs['X_cellcharter'] = digit_labels
            adata.obs['X_cellcharter'] = adata.obs["X_cellcharter"].astype('category')
            adata.obs['X_cellcharter'] = adata.obs['X_cellcharter'].cat.rename_categories({0: 'CP2',
                                                   1: 'L5',
                                                   2: 'CP1',
                                                   4: 'L4',
                                                   5: 'ccg/aco',
                                                   6: 'ACB',
                                                   7: 'L1-L3',
                                                    8: 'L6a/b',
                                                   11: 'VL',
                                                    100: '1_others'
                                                         })
        elif dataset == '6_Mouse_Embryo':
            gmm = cc.tl.Cluster(
            n_clusters=14,
            random_state=seed,)
            gmm.fit(adata, use_rep='X_cellcharter')
            adata.obs['X_cellcharter'] = gmm.predict(adata, use_rep='X_cellcharter')
   
        else:
            raise ValueError('Unknown dataset')

        scores = compute_supervised_scores(adata,'X_cellcharter')
        data_omics1.obsm['X_cellcharter'] = adata.obsm['X_cellcharter'].copy()
        data_omics2.obsm['X_cellcharter'] = adata.obsm['X_cellcharter'].copy()
        jaccard1 = compute_jaccard(data_omics1, 'X_cellcharter')
        jaccard2 = compute_jaccard(data_omics2, 'X_cellcharter')
        moranI = compute_moranI(adata, 'X_cellcharter')
        result.loc[len(result)] = [dataset, 'CellCharter', scores['ari'],
                                   scores['mi'], scores['nmi'], scores['ami'],
                                   scores['hom'], scores['vme'],scores['average'], 
                                   jaccard1, jaccard2,moranI]
        result.to_csv('Results/Evaluation_CellCharter.csv', index=False)
        print(result.tail(1))



