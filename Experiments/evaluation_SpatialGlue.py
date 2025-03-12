import os.path
import sys
sys.path.append('/home/makx/Reproducible-main')
import random
import anndata
import pandas
import torch
from Methods.SpatialGlue.preprocess import pca, construct_neighbor_graph, lsi
from Methods.SpaMV.metrics import compute_moranI, compute_jaccard, compute_supervised_scores
from Methods.SpaMV.utils import clr_normalize_each_cell
import scanpy as sc
from matplotlib import pyplot as plt
from Methods.SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue
from Methods.SpatialGlue.utils import clustering
from sklearn.metrics import adjusted_rand_score

os.environ['R_HOME'] = "/home/makx/anaconda3/envs/torch2/lib/R"
os.environ['R_USER'] = "/home/makx/anaconda3/envs/torch2/lib/python3.10/site-packages/rpy2"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
result = pandas.DataFrame(
    columns=['Dataset', 'method',  'ari', 'mi', 'nmi', 'ami', 'hom', 'vme', 'average', 'jaccard 1', 'jaccard 2',
             'moran I'])
for dataset in ['4_Human_Lymph_Node', '5_Mouse_Brain', '6_Mouse_Embryo']:
    data_omics1 = sc.read_h5ad('Dataset/' + dataset + '/adata_RNA.h5ad')
    data_omics1.var_names_make_unique()
    sc.pp.filter_genes(data_omics1, min_cells=10)
    sc.pp.highly_variable_genes(data_omics1, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(data_omics1, target_sum=1e4)
    sc.pp.log1p(data_omics1)
    sc.pp.scale(data_omics1)

    data_omics1_high =  data_omics1[:, data_omics1.var['highly_variable']]
    data_omics1.obsm['feat'] = pca(data_omics1_high, n_comps=50)

    if dataset == '4_Human_Lymph_Node':
        data_omics2 = sc.read_h5ad('Dataset/' + dataset + '/adata_ADT.h5ad')
        data_omics2.var_names_make_unique()
        data_omics2 = clr_normalize_each_cell(data_omics2)
        sc.pp.scale(data_omics2)
        data_omics2.obsm['feat'] = pca(data_omics2, n_comps=data_omics2.n_vars-1)
        data_type = '10x'
        n_clusters = 10
    elif dataset == '5_Mouse_Brain':
        data_omics2 = sc.read_h5ad('Dataset/' + dataset + '/adata_peaks_normalized.h5ad')
        data_omics2.var_names_make_unique()
        data_omics2 = data_omics2[data_omics1.obs_names].copy() # .obsm['X_lsi'] represents the dimension reduced feature
        sc.pp.highly_variable_genes(data_omics2, flavor="seurat_v3", n_top_genes=3000)
        lsi(data_omics2, use_highly_variable=False, n_components=51)
        data_omics2.obsm['feat'] = data_omics2.obsm['X_lsi'].copy()
        data_type = 'Spatial-epigenome-transcriptome'
        n_clusters = 15
    elif dataset == '6_Mouse_Embryo':
        data_omics2 = sc.read_h5ad('Dataset/' + dataset + '/adata_peaks.h5ad')
        data_omics2.var_names_make_unique()
        data_omics2 = data_omics2[data_omics1.obs_names].copy() # .obsm['X_lsi'] represents the dimension reduced feature
        sc.pp.highly_variable_genes(data_omics2, flavor="seurat_v3", n_top_genes=3000)
        lsi(data_omics2, use_highly_variable=False, n_components=51)
        data_omics2.obsm['feat'] = data_omics2.obsm['X_lsi'].copy()
        data_type = 'Spatial-epigenome-transcriptome'
        n_clusters = 14
    else:
        raise ValueError('Unknown dataset')
    for i in range(10):
        seed = random.randint(1, 10000)
        print('data: {}, iteration: {}, seed {}'.format(dataset, i + 1, seed))

        data = construct_neighbor_graph(data_omics1, data_omics2, datatype=data_type)

        model = Train_SpatialGlue(data, datatype=data_type, device=device, random_seed=seed)

        output = model.train()
        data_omics1.obsm['SpatialGlue'] = output['SpatialGlue']
        clustering(data_omics1, key='SpatialGlue', add_key='SpatialGlue', n_clusters=n_clusters, method='mclust', use_pca=True)
        print(data_omics1.obs['SpatialGlue'])
        scores = compute_supervised_scores(data_omics1,'SpatialGlue') 
        data_omics2.obsm['SpatialGlue'] = output['SpatialGlue']
        jaccard1 = compute_jaccard(data_omics1, 'SpatialGlue')
        jaccard2 = compute_jaccard(data_omics2, 'SpatialGlue')
        moranI = compute_moranI(data_omics1, 'SpatialGlue')
        result.loc[len(result)] = [dataset, 'SpatialGlue', scores['ari'],
                                   scores['mi'], scores['nmi'], scores['ami'],
                                   scores['hom'], scores['vme'],scores['average'], 
                                   jaccard1, jaccard2,moranI]
        result.to_csv('Results/Evaluation_SpatialGlue.csv', index=False)
        print(result.tail(1))


    