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

for dataset in ['1_Simulation', '2_Simulation', '3_Simulation']:
    adata_omics1 = sc.read_h5ad('../Dataset/' + dataset + '/adata_RNA.h5ad')
    adata_omics2 = sc.read_h5ad('../Dataset/' + dataset + '/adata_ADT.h5ad')
    df_data_RNA = adata_omics1.X.astype('float64')  # 提取基因表达矩阵
    LayerName = list(adata_omics1.obs['cluster'])  # 获取RNA的LayerName
    n_cluster = 10
    # 如果LayerName是字节类型，需要进行解码
    if isinstance(LayerName[0], bytes):
        LayerName = [item.decode('utf-8') for item in LayerName]

    for i in range(1):
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
        from rpy2.robjects import r

        r('Sys.setlocale(locale="UTF-8")')  # 设置R控制台输出编码为UTF-8
        r('Sys.setenv(LANG="en_US.UTF-8")')  # 设置环境变量

        adata_omics2 = mclust_R(adata_omics2,used_obsm='emb',num_cluster=n_cluster)
        cluster_learned = adata_omics2.obs['mclust']
        adata_omics2.obs['clusters'] = cluster_learned
        adata_omics1.obs['cosmos'] = cluster_learned.astype('category')  # 确保cluster_learned是聚类结果
        cluster = LayerName

        adata_omics1.obsm['emb'] = embedding  # 存储聚类标签,anndata格式
        fig, ax_list = plt.subplots(1, 2, figsize=(8, 4))
        sc.pp.neighbors(adata_omics1, use_rep='emb', key_added='cosmos', n_neighbors=30)
        sc.tl.umap(adata_omics1, neighbors_key='cosmos')
        sc.pl.umap(adata_omics1, color='cosmos', ax=ax_list[0], s=60, show=False)
        sc.pl.embedding(adata_omics1, basis='spatial', color='cosmos', ax=ax_list[1], title='cosmos\n' + 'ARI: {:.3f}'.format(
            adjusted_rand_score(cluster, cluster_learned)), s=200, show=False)

        plt.tight_layout(w_pad=0.3)
        result_folder = '../Results/' + dataset + '/'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder, exist_ok=True)
        plt.savefig(result_folder + 'cosmos.pdf')
        plt.show()