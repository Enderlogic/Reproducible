import os
from Methods.miso.utils import *
import pandas as pd
import numpy as np
import scanpy as sc
from Methods.cosmos import cosmos
import h5py
import anndata as ad
import warnings

warnings.filterwarnings('ignore')
random_seed = 20
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
from Methods.miso.utils import compute_jaccard
from Methods.SpaMV.utils import mclust_R
import sys

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)


# 结果列表
results = []

columns = ['Dataset', 'method', 'ARI', 'MI', 'NMI', 'AMI', 'HOM', 'VME', 'Average', 'jaccard_RNA', 'jaccard_ADT',
           'moranI']
all_results = pd.DataFrame(columns=columns)

for dataset in ['4_Human_Lymph_Node', '5_Mouse_Brain', '6_Mouse_Embryo']:
    adata_omics1 = sc.read_h5ad('../Dataset/' + dataset + '/adata_RNA.h5ad')

    df_data_RNA = adata_omics1.X.astype('float64')  # 提取基因表达矩阵
    loc_RNA = np.array(adata_omics1.obsm['spatial'])  # 获取RNA数据的空间位置信息
    LayerName_RNA = list(adata_omics1.obs['cluster'])  # 获取RNA的LayerName
    # 如果LayerName是字节类型，需要进行解码
    if isinstance(LayerName_RNA[0], bytes):
        LayerName_RNA = [item.decode('utf-8') for item in LayerName_RNA]

    if dataset == '4_Human_Lymph_Node':
        adata_omics2 = sc.read_h5ad('../Dataset/' + dataset + '/adata_ADT.h5ad')
        LayerName_RNA = list(adata_omics2.obs['cluster'])  # 获取ATAC的LayerName
        n_cluster = 10

    elif dataset == '5_Mouse_Brain':
        adata_omics2 = sc.read_h5ad('../Dataset/' + dataset + '/adata_peaks_normalized.h5ad')
        n_cluster = 15
    elif dataset == '6_Mouse_Embryo':
        adata_omics2 = sc.read_h5ad('../Dataset/' + dataset + '/adata_peaks.h5ad')
        n_cluster = 14
    else:
        raise ValueError('Unknown dataset')

    LayerName = LayerName_RNA

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

        # COSMOS training
        cosmos_comb = cosmos.Cosmos(adata1=adata_omics1, adata2=adata_omics2)
        cosmos_comb.preprocessing_data(n_neighbors=10)
        cosmos_comb.train(spatial_regularization_strength=0.01, z_dim=50,
                          lr=1e-3, wnn_epoch=500, total_epoch=1000, max_patience_bef=10, max_patience_aft=30, min_stop=200,
                          random_seed=random_seed, gpu=0, regularization_acceleration=True, edge_subset_sz=1000000)
        weights = cosmos_comb.weights
        embedding = cosmos_comb.embedding
        adata_omics2.obsm['emb']=embedding

        # Clustering of COSMOS integration

        adata_omics2 = mclust_R(adata_omics2,used_obsm='emb',num_cluster=n_cluster)
        cluster_learned = adata_omics2.obs['mclust']
        adata_omics2.obs['clusters'] = cluster_learned
        cluster = LayerName
        adata_omics1.obsm['emb'] = embedding  # 存储聚类标签,anndata格式
        ari = adjusted_rand_score(cluster, cluster_learned)
        mi = mutual_info_score(cluster, cluster_learned)
        nmi = normalized_mutual_info_score(cluster, cluster_learned)
        ami = adjusted_mutual_info_score(cluster, cluster_learned)
        hom = homogeneity_score(cluster, cluster_learned)
        vme = v_measure_score(cluster, cluster_learned)
        average = (ari + mi + nmi + ami + hom + vme) / 6

        # 计算Jaccard
        jaccard_RNA = compute_jaccard(adata_omics1, 'emb')
        jaccard_ATAC = compute_jaccard(adata_omics2, 'emb')
        moranI = compute_moranI(adata_omics2, 'clusters')

        results_row = {
            'Dataset': dataset,
            'method': 'cosmos',
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

print("Results saved to 'cosmos_evaluation_results.csv'")
all_results.to_csv('cosmos_evaluation_results.csv')