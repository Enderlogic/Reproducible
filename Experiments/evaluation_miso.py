from Methods.miso.utils import *
from Methods.miso import Miso
from PIL import Image
import pandas as pd
import numpy as np
import scanpy as sc
import os
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

Image.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None
import torch
import random
import anndata as ad
import sklearn
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
import torch.nn.functional as F

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from Methods.miso.utils import compute_jaccard, pca
from Methods.SpatialGlue.preprocess import lsi

# 结果列表
results = []

columns = ['Dataset', 'method', 'ARI', 'MI', 'NMI', 'AMI', 'HOM', 'VME', 'Average', 'jaccard_RNA', 'jaccard_ADT',
           'moranI']
all_results = pd.DataFrame(columns=columns)

for dataset in [ '4_Human_Lymph_Node', '5_Mouse_Brain', '6_Mouse_Embryo']:
    data_omics1 = sc.read_h5ad('../Dataset/' + dataset + '/adata_RNA.h5ad')
    LayerName = list(data_omics1.obs['cluster'])
    adata_omics1 = preprocess(data_omics1, modality='rna')  # 模型输入：ndarray格式，降维前

    sc.pp.normalize_total(data_omics1, target_sum=1e4)
    sc.pp.log1p(data_omics1)
    sc.pp.scale(data_omics1)
    data_omics1 = ad.AnnData(pca(data_omics1, n_comps=30 if dataset == '4_Human_Lymph_Node' else 50),
                             obs=data_omics1.obs, obsm=data_omics1.obsm)  # 降维求指标
    data_omics1.obsm['X_pca'] = data_omics1.X

    if dataset == '4_Human_Lymph_Node':
        data_omics2 = sc.read_h5ad('../Dataset/' + dataset + '/adata_ADT.h5ad')
        adata_omics2 = preprocess(data_omics2, modality='protein')
        sc.pp.normalize_total(data_omics2, target_sum=1e4)
        sc.pp.log1p(data_omics2)
        sc.pp.scale(data_omics2)
        data_omics2.obsm['X_pca'] = data_omics2.X
        n_cluster = 10

    elif dataset == '5_Mouse_Brain':
        data_omics2 = sc.read_h5ad('../Dataset/' + dataset + '/adata_peaks_normalized.h5ad')
        # data_omics2 = ad.AnnData(data_omics2.obsm['X_lsi'], obs=data_omics2.obs, obsm=data_omics2.obsm)
        adata_omics2 = preprocess(data_omics2, modality='atac')
        n_cluster = 15

    elif dataset == '6_Mouse_Embryo':
        data_omics2 = sc.read_h5ad('../Dataset/' + dataset + '/adata_peaks.h5ad')
        adata_omics2 = preprocess(data_omics2, modality='atac')
        if 'X_lsi' not in data_omics2.obsm.keys():
            lsi(data_omics2, use_highly_variable=False, n_components=50)
        data_omics2 = ad.AnnData(data_omics2.obsm['X_lsi'], obs=data_omics2.obs, obsm=data_omics2.obsm)
        data_omics2.obsm['X_lsi'] = data_omics2.X
        n_cluster = 14
    else:
        raise ValueError('Unknown dataset')

    omics_names = ['Transcriptomics', 'Proteomics' if dataset == '4_Human_Lymph_Node' else 'Epigenomics']

    for i in range(10):

        seed = random.randint(1, 1000)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        print('data: {}, iteration: {}, seed {}'.format(dataset, i + 1, seed))

        if torch.cuda.is_available():
            device = 'cuda'
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("CUDA is available. GPU:", torch.cuda.get_device_name(0))
        else:
            device = 'cpu'
            print("CUDA is not available. Using CPU.")

        if isinstance(LayerName[0], bytes):
            LayerName = [item.decode('utf-8') for item in LayerName]
        # if using a subset of modality-specific terms, the "ind_views" parameter should be a list with values entries to the indices of the modalities to be included, e.g.,  ind_views=[0,2] if including RNA and image features
        # if using a subset of interaction terms, the "combs" parameter should be a list of tuples with entries to the indices of the modalities for each interaction, e.g. combs = [(0,1),(0,2)] if including the RNA-protein and RNA-image interaction terms
        # model = Miso([rna,protein,image_emb],ind_views='all',combs='all',sparse=False,device=device)

        model = Miso([adata_omics1, adata_omics2], ind_views=[0, 1], combs=[(0, 1)], sparse=False, device=device)

        model.train()
        np.save('emb.npy', model.emb)

        # 获取聚类结果
        clusters = model.cluster(n_clusters=n_cluster)
        data_omics2.obs['clusters'] = clusters
        cluster = LayerName
        cluster_learned = clusters
        data_omics1.obsm['emb'] = model.emb  # 存储聚类标签,anndata格式
        data_omics2.obsm['emb'] = model.emb
        ari = adjusted_rand_score(cluster, cluster_learned)
        mi = mutual_info_score(cluster, cluster_learned)
        nmi = normalized_mutual_info_score(cluster, cluster_learned)
        ami = adjusted_mutual_info_score(cluster, cluster_learned)
        hom = homogeneity_score(cluster, cluster_learned)
        vme = v_measure_score(cluster, cluster_learned)
        average = (ari + mi + nmi + ami + hom + vme) / 6

        # 计算Jaccard
        jaccard_RNA = compute_jaccard(data_omics1, 'emb')
        jaccard_ATAC = compute_jaccard(data_omics2, 'emb')
        moranI = compute_moranI(data_omics2, 'clusters')

        results_row = {
            'Dataset': dataset,
            'method': 'miso',
            'ARI': ari,
            'MI': mi,
            'NMI': nmi,
            'AMI': ami,
            'HOM': hom,
            'VME': vme,
            'Average': average,
            'jaccard_RNA': jaccard_RNA,
            'jaccard_ADT': jaccard_ATAC,
            'moranI': moranI
        }

        for key, value in results_row.items():
            print(f"{key}: {value}")

        all_results = all_results.append(results_row, ignore_index=True)

print("Results saved to 'miso_evaluation_results.csv'")
all_results.to_csv('miso_evaluation_results.csv')
