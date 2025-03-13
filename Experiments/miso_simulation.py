import os
import sys
import random
import anndata
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from Methods.miso.utils import pca, preprocess
from Methods.SpaMV.utils import clustering
from Methods.miso import Miso
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
import logging

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARN)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for dataset in ['1_Simulation', '2_Simulation', '3_Simulation']:
    data_rna = sc.read_h5ad('../Dataset/' + dataset + '/adata_RNA.h5ad')
    data_pro = sc.read_h5ad('../Dataset/' + dataset + '/adata_ADT.h5ad')

    LayerName = list(data_rna.obs['cluster'])

    adata_rna = preprocess(data_rna, modality='rna')  # 模型输入：ndarray格式，降维前
    sc.pp.normalize_total(data_rna, target_sum=1e4)
    sc.pp.log1p(data_rna)
    sc.pp.scale(data_rna)
    data_rna.obsm['feat'] = pca(data_rna, n_comps=50)

    adata_pro = preprocess(data_pro, modality='protein')
    sc.pp.normalize_total(data_pro, target_sum=1e4)
    sc.pp.log1p(data_pro)
    sc.pp.scale(data_pro)
    data_pro.obsm['feat'] = pca(data_pro, n_comps=50)

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

        if isinstance(LayerName[0], bytes):
            LayerName = [item.decode('utf-8') for item in LayerName]
        # if using a subset of modality-specific terms, the "ind_views" parameter should be a list with values entries to the indices of the modalities to be included, e.g.,  ind_views=[0,2] if including RNA and image features
        # if using a subset of interaction terms, the "combs" parameter should be a list of tuples with entries to the indices of the modalities for each interaction, e.g. combs = [(0,1),(0,2)] if including the RNA-protein and RNA-image interaction terms
        # model = Miso([rna,protein,image_emb],ind_views='all',combs='all',sparse=False,device=device)

        model = Miso([adata_rna, adata_pro], ind_views=[0, 1], combs=[(0, 1)], sparse=False, device=device)

        model.train()
        data_rna.obsm['miso'] = model.emb
        clustering(data_rna, key='miso', add_key='miso', n_clusters=10, method='mclust', use_pca=True)

        fig, ax_list = plt.subplots(1, 2, figsize=(8, 4))
        sc.pp.neighbors(data_rna, use_rep='miso', key_added='miso', n_neighbors=30)
        sc.tl.umap(data_rna, neighbors_key='miso')
        sc.pl.umap(data_rna, color='miso', ax=ax_list[0], s=60, show=False)
        sc.pl.embedding(data_rna, basis='spatial', color='miso', ax=ax_list[1], title='miso\n' + 'ARI: {:.3f}'.format(adjusted_rand_score(data_rna.obs['miso'], data_rna.obs['cluster'])), s=200, show=False)

        plt.tight_layout(w_pad=0.3)
        result_folder = '../Results/' + dataset + '/'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder, exist_ok=True)
        plt.savefig(result_folder + 'miso.pdf')
        plt.show()