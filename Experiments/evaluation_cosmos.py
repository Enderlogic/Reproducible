from Methods.miso.utils import *
from Methods.miso import Miso
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from umap import UMAP
import sklearn
import seaborn as sns
from Methods.cosmos import cosmos
from Methods.cosmos.pyWNN import pyWNN
import h5py
import anndata as ad
import warnings
warnings.filterwarnings('ignore')
random_seed = 20
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
from Methods.miso.utils import compute_jaccard, pca
from Methods.SpatialGlue.preprocess import lsi
import sys
# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)



# ##  Preparation of data

# ### Importing the data



# data_mat = h5py.File('./ATAC_RNA_Seq_MouseBrain_RNA_ATAC.h5', 'r')
# df_data_RNA = np.array(data_mat['X_RNA']).astype('float64')     # gene count matrix
# df_data_ATAC= np.array(data_mat['X_ATAC']).astype('float64')  # protein count matrix
# loc = np.array(data_mat['Pos']).astype('float64')
# LayerName = list(data_mat['LayerName'])
# LayerName = [item.decode("utf-8") for item in LayerName]


# 加载RNA数据
adata_RNA = ad.read_h5ad("D:/AI_Sophomore/winter2025/COSMOS-main/Tutorials/adata_RNA.h5ad")

# 打印 obs 中的所有列名
# print(adata_RNA.obs.columns)

df_data_RNA = adata_RNA.X.astype('float64')  # 提取基因表达矩阵
loc_RNA = np.array(adata_RNA.obsm['spatial'])  # 获取RNA数据的空间位置信息
LayerName_RNA = list(adata_RNA.obs['cluster'])  # 获取RNA的LayerName
# 如果LayerName是字节类型，需要进行解码
if isinstance(LayerName_RNA[0], bytes):
    LayerName_RNA = [item.decode('utf-8') for item in LayerName_RNA]

# 加载ATAC数据
adata_ATAC = ad.read_h5ad("D:/AI_Sophomore/winter2025/COSMOS-main/Tutorials/adata_ADT.h5ad")
df_data_ATAC = adata_ATAC.X.astype('float64')  # 提取ATAC数据矩阵
loc_ATAC = np.array(adata_ATAC.obsm['spatial'])  # 获取ATAC数据的空间位置信息
LayerName_ATAC = list(adata_ATAC.obs['cluster'])  # 获取ATAC的LayerName
# 如果LayerName是字节类型，需要进行解码
if isinstance(LayerName_ATAC[0], bytes):
    LayerName_ATAC = [item.decode('utf-8') for item in LayerName_ATAC]

adata1 = sc.AnnData(df_data_RNA, dtype="float64")
adata1.obsm['spatial'] = loc_RNA
adata1.obs['LayerName'] = LayerName_RNA
adata1.obs['x_pos'] = np.array(loc_RNA)[:,0]
adata1.obs['y_pos'] = np.array(loc_RNA)[:,1]

adata2 = sc.AnnData(df_data_ATAC, dtype="float64")
adata2.obsm['spatial'] = loc_ATAC
adata2.obs['LayerName'] = LayerName_ATAC
adata2.obs['x_pos'] = np.array(loc_ATAC)[:,0]
adata2.obs['y_pos'] = np.array(loc_ATAC)[:,1]


# ### Visualizing spatial positions with annotation

# In[8]:

LayerName = LayerName_ATAC
adata_new = adata1.copy()
adata_new.obs["LayerName"]=adata_new.obs["LayerName"].astype('category')


matplotlib.rcParams['font.size'] = 12.0
fig, axes = plt.subplots(1, 1, figsize=(3.5,3))
sz = 20
plot_color=['#D1D1D1','#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', \
            '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#ffd8b1', '#800000', '#aaffc3', '#808000', '#000075', '#000000', '#808080', '#ffffff', '#fffac8']

domains="LayerName"
num_celltype=len(adata_new.obs[domains].unique())
adata_new.uns[domains+"_colors"]=list(plot_color[:num_celltype])
titles = 'Annotation '
ax=sc.pl.scatter(adata_new,alpha=1,x="x_pos",y="y_pos",color=domains,title=titles ,color_map=plot_color,show=False,size=sz,ax = axes)
ax.axis('off')

## COSMOS training
cosmos_comb = cosmos.Cosmos(adata1=adata1,adata2=adata2)
cosmos_comb.preprocessing_data(n_neighbors = 10)
cosmos_comb.train(spatial_regularization_strength=0.01, z_dim=50,
         lr=1e-3, wnn_epoch = 500, total_epoch=1000, max_patience_bef=10, max_patience_aft=30, min_stop=200,
         random_seed=random_seed, gpu=0, regularization_acceleration=True, edge_subset_sz=1000000)
weights = cosmos_comb.weights
df_embedding = pd.DataFrame(cosmos_comb.embedding)


# Clustering of COSMOS integration

adata_new = adata1.copy()
embedding_adata = sc.AnnData(df_embedding)
sc.pp.neighbors(embedding_adata, n_neighbors=50, use_rep='X')

# Manualy setting resolution for clustering
res = 1.0
sc.tl.louvain(embedding_adata, resolution=res)
adata_new.obs['Cluster_cosmos'] = list(embedding_adata.obs["louvain"].cat.codes)
adata_new.obs["Cluster_cosmos"]=adata_new.obs["Cluster_cosmos"].astype('category')

# Relabeling clusters with layer names
digit_labels = list(adata_new.obs['Cluster_cosmos'])
for i in range(len(digit_labels)):
    if digit_labels[i] not in [0,1,2,4,5,6,7,8,11]:
        digit_labels[i] = 100
adata_new.obs['Cluster_cosmos'] = digit_labels
adata_new.obs["Cluster_cosmos"]=adata_new.obs["Cluster_cosmos"].astype('category')
adata_new.obs['Cluster_cosmos'].cat.rename_categories({0: 'CP2',
                                                   1: 'L5',
                                                   2: 'CP1',
                                                   4: 'L4',
                                                   5: 'ccg/aco',
                                                   6: 'ACB',
                                                   7: 'L1-L3',
                                                    8: 'L6a/b',
                                                   11: 'VL',
                                                    100: '1_others'
                                                         }, inplace=True)

ordered_cluster = np.unique(list(adata_new.obs['Cluster_cosmos']))
adata_new.obs['Cluster_cosmos'] = adata_new.obs['Cluster_cosmos'].cat.reorder_categories(ordered_cluster, ordered=True)

# Calculating ARI
opt_cluster_cosmos = list(adata_new.obs['Cluster_cosmos'])
opt_ari_cosmos = sklearn.metrics.adjusted_rand_score(LayerName, opt_cluster_cosmos)
opt_ari_cosmos = round(opt_ari_cosmos, 2)

        cluster = LayerName
        cluster_learned = opt_cluster_cosmos
        adata1.obsm['emb'] = embedding_adata  # 存储聚类标签,anndata格式
        adata2.obsm['emb'] = embedding_adata
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
# # Ploting figures
# matplotlib.rcParams['font.size'] = 8.0
# fig, axes = plt.subplots(1, 2, figsize=(6,2))
# sz = 10
# plot_color=['#D1D1D1','#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', \
#             '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#ffd8b1', '#800000', '#aaffc3', '#808000', '#000075', '#000000', '#808080', '#ffffff', '#fffac8']
# domains="LayerName"
# num_celltype=len(adata_new.obs[domains].unique())
# adata_new.uns[domains+"_colors"]=list(plot_color[:num_celltype])
# titles = 'Annotation'
# ax=sc.pl.scatter(adata_new,alpha=1,x="x_pos",y="y_pos",color=domains,title=titles ,color_map=plot_color,show=False,size=sz,ax = axes[0])
# ax.axis('off')
#
# plot_color_1=['#D1D1D1','#e6194b', '#3cb44b', '#bcf60c','#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', \
#              '#fabebe', '#008080', '#e6beff', '#9a6324', '#ffd8b1', '#800000', '#aaffc3', '#808000', '#000075', '#000000', '#808080', '#ffffff', '#fffac8']
# domains="Cluster_cosmos"
# num_celltype=len(adata_new.obs[domains].unique())
# adata_new.uns[domains+"_colors"]=list(plot_color_1[:num_celltype])
# titles = 'COSMOS, ARI = ' + str(opt_ari_cosmos)
# ax=sc.pl.scatter(adata_new,alpha=1,x="x_pos",y="y_pos",color=domains,title=titles ,color_map=plot_color_1,show=False,size=sz,ax = axes[1])
# ax.axis('off')
# plt.tight_layout()
#
#
# # ### UMAP visualization of COSMOS integration
#
# # In[6]:
#
#
# # UMAP visualization
# umap_2d = UMAP(n_components=2, init='random', random_state=random_seed, min_dist = 0.3,n_neighbors=30)
# umap_pos = umap_2d.fit_transform(df_embedding)
# adata_new.obs['cosmos_umap_pos_x'] = umap_pos[:,0]
# adata_new.obs['cosmos_umap_pos_y'] = umap_pos[:,1]
#
# # Ploting figures
# matplotlib.rcParams['font.size'] = 8.0
# fig, axes = plt.subplots(1, 1, figsize=(4,3))
# sz = 10
# domains="Cluster_cosmos"
# num_celltype=len(adata_new.obs[domains].unique())
# adata_new.uns[domains+"_colors"]=list(plot_color_1[:num_celltype])
# titles = 'UMAP of COSMOS'
# ax=sc.pl.scatter(adata_new,alpha=1,x="cosmos_umap_pos_x",y="cosmos_umap_pos_y",color=domains,title=titles ,color_map=plot_color_1,show=False,size=sz,ax = axes)
# ax.axis('off')
# plt.tight_layout()
# plt.show()
# plt.savefig('output_image.png')  # Save the figure to a file
#
# # ### Pseudo-spatiotemporal map (pSM) from COSMOS integration
# #
#
# # In[7]:
#
#
# # Calculating pseudo-times
# embedding_adata.uns['iroot'] = np.flatnonzero(adata_new.obs["Cluster_cosmos"] == 'CP2')[0]
# sc.tl.diffmap(embedding_adata)
# sc.tl.dpt(embedding_adata)
# pSM_values_cosmos = embedding_adata.obs['dpt_pseudotime'].to_numpy()
#
#
# # Ploting figures
# matplotlib.rcParams['font.size'] = 8.0
# fig, axes = plt.subplots(1, 1, figsize=(3,3))
# sz = 10
# x = np.array(adata_new.obs['x_pos'])
# y = np.array(adata_new.obs['y_pos'])
# ax_temp = axes
# im = ax_temp.scatter(x, y, s=20, c=pSM_values_cosmos, marker='.', cmap='coolwarm',alpha = 1)
# ax_temp.axis('off')
# ax_temp.set_title('dpt_pseudotime of COSMOS')
# plt.tight_layout()


# ### Showing modality weights of two omics in COSMOS integration

# In[8]:


# def plot_weight_value(alpha, label, modality1='RNA', modality2='ATAC',order = None):
#     df = pd.DataFrame(columns=[modality1, modality2, 'label'])
#     df[modality1], df[modality2] = alpha[:, 0], alpha[:, 1]
#     df['label'] = label
#     df = df.set_index('label').stack().reset_index()
#     df.columns = ['label_COSMOS', 'Modality', 'Modality weights']
#     matplotlib.rcParams['font.size'] = 8.0
#     fig, axes = plt.subplots(1, 1, figsize=(5,3))
#     ax = sns.violinplot(data=df, x='label_COSMOS', y='Modality weights', hue="Modality",
#                 split=True, inner="quart", linewidth=1, show=False, orient = 'v', order=order)
#     plt.tight_layout(w_pad=0.05)
#     ax.set_title(modality1 + ' vs ' + modality2)
#     ax.set_xticklabels(order, rotation = 45)
# ordered_cluster = np.unique(LayerName)
# layer_type = ordered_cluster
# index_all = [np.array([i for i in range(len(LayerName)) if LayerName[i] == ordered_cluster[0]])]
# for k in range(1,len(layer_type)):
#     temp_idx = np.array([i for i in range(len(LayerName)) if LayerName[i] == ordered_cluster[k]])
#     index_all.append(temp_idx)
#
# wghts_mean = np.mean(weights[index_all[0],:],0)
# for k in range(1,len(ordered_cluster)):
#     wghts_mean_temp = np.mean(weights[index_all[k],:],0)
#     wghts_mean = np.vstack([wghts_mean, wghts_mean_temp])
# df_wghts_mean = pd.DataFrame(wghts_mean,columns = ['w1','w2'],index = ordered_cluster)
#
# df_sort_mean = df_wghts_mean.sort_values(by=['w1'])
# plot_weight_value(weights, np.array(adata_new.obs['LayerName']), order = list(df_sort_mean.index))

