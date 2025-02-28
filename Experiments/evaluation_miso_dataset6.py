from miso.utils import *
from miso import Miso
from PIL import Image
import pandas as pd
import numpy as np
import scanpy as sc
import os
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
from miso.utils import compute_jaccard, pca

# 结果列表
results = []

columns = ['ARI', 'MI', 'NMI', 'AMI', 'HOM', 'VME', 'Average', 'jaccard_RNA', 'jaccard_ADT', 'moranI']
all_results = pd.DataFrame(columns=columns)

n = 10

for i in range(n):

    print(f"Run {i + 1}:")

    seed = random.randint(1, 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("CUDA is available. GPU:", torch.cuda.get_device_name(0))
    else:
        device = 'cpu'
        print("CUDA is not available. Using CPU.")

    data_RNA = sc.read_h5ad("adata_RNA.h5ad")
    LayerName = list(data_RNA.obs['cluster'])
    rna = preprocess(data_RNA, modality='rna')  # 模型输入：ndarray格式
    sc.pp.normalize_total(data_RNA, target_sum=1e4)  # 总量归一化
    sc.pp.log1p(data_RNA)  # 对数转换
    sc.pp.scale(data_RNA)  # 标准化
    adata_RNA = ad.AnnData(pca(data_RNA, n_comps=50), obs=data_RNA.obs, obsm=data_RNA.obsm)
    adata_RNA.obsm['X_pca'] = adata_RNA.X

    adata_ATAC = sc.read_h5ad("adata_peaks.h5ad")
    atac = preprocess(adata_ATAC, modality='atac')  # 模型输入：ndarray格式
    if 'X_lsi' not in adata_ATAC.obsm.keys():
        lsi(adata_ATAC, use_highly_variable=False, n_components=50)
    adata_ATAC = anndata.AnnData(adata_ATAC.obsm['X_lsi'], obs=adata_ATAC.obs, obsm=adata_ATAC.obsm)
    adata_ATAC.obsm['X_lsi'] = adata_ATAC.X

    n_cluster = 14

    if isinstance(LayerName[0], bytes):
        LayerName = [item.decode('utf-8') for item in LayerName]
    # if using a subset of modality-specific terms, the "ind_views" parameter should be a list with values entries to the indices of the modalities to be included, e.g.,  ind_views=[0,2] if including RNA and image features
    # if using a subset of interaction terms, the "combs" parameter should be a list of tuples with entries to the indices of the modalities for each interaction, e.g. combs = [(0,1),(0,2)] if including the RNA-protein and RNA-image interaction terms
    # model = Miso([rna,protein,image_emb],ind_views='all',combs='all',sparse=False,device=device)
    # 使用 RNA 和 ATAC数据进行训练，只考虑 RNA 和 ATAC之间的交互
    model = Miso([rna, atac], ind_views=[0, 1], combs=[(0, 1)], sparse=False, device=device)

    model.train()
    np.save('emb.npy', model.emb)

    # 获取聚类结果
    clusters = model.cluster(n_clusters=14)
    adata_ATAC.obs['clusters'] = clusters
    cluster = LayerName
    cluster_learned = clusters
    adata_RNA.obsm['emb'] = model.emb
    adata_ATAC.obsm['emb'] = model.emb
    ari = adjusted_rand_score(cluster, cluster_learned)
    mi = mutual_info_score(cluster, cluster_learned)
    nmi = normalized_mutual_info_score(cluster, cluster_learned)
    ami = adjusted_mutual_info_score(cluster, cluster_learned)
    hom = homogeneity_score(cluster, cluster_learned)
    vme = v_measure_score(cluster, cluster_learned)
    average = (ari + mi + nmi + ami + hom + vme) / 6

    # 计算Jaccard
    jaccard_RNA = compute_jaccard(adata_RNA, 'emb')
    jaccard_ATAC = compute_jaccard(adata_ADT, 'emb')
    moranI = compute_moranI(adata_ADT, 'clusters')

    results_row = {
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
    # 打印每次的结果
    for key, value in results_row.items():
        print(f"{key}: {value}")

    # 添加这一行到结果 DataFrame
    all_results = all_results.append(results_row, ignore_index=True)

# 添加索引
all_results.index = [str(i + 1) for i in range(n)]
print("Results saved to 'miso_evaluation_results1.csv'")
all_results.to_csv('miso_evaluation_results1.csv', index_label="evaluation")
