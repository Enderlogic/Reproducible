import random

import pandas
import torch
import sys
import os

from matplotlib import pyplot as plt
from pandas import DataFrame
from scanpy.plotting import embedding

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from Methods.SpaMV.spamv import SpaMV
from Methods.SpaMV.utils import plot_embedding_results
from Methods.SpaMV.metrics import compute_topic_coherence, compute_topic_diversity
import scanpy as sc
import wandb

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

dataset = '7_ME13_1'
data_folder = '../Dataset/' + dataset
adata_ac = sc.read_h5ad(data_folder + '/adata_H3K27ac_ATAC.h5ad')
adata_ac = adata_ac[:, (adata_ac.X > 1).sum(0) > adata_ac.n_obs / 100]
sc.pp.normalize_total(adata_ac)
sc.pp.log1p(adata_ac)
sc.pp.highly_variable_genes(adata_ac, flavor='seurat', subset=False, n_top_genes=1000)
adata_ac.var.loc[
    ['Hand2', 'Gfi1b', 'Wwp2', 'Sox2', 'Hoxc4', 'Wnt7b', 'Sall1', 'Dtx4', 'Nprl3', 'Nfe2'], 'highly_variable'] = True
adata_ac = adata_ac[:, adata_ac.var['highly_variable']]

adata_me3 = sc.read_h5ad(data_folder + '/adata_H3K27me3_ATAC.h5ad')
adata_me3 = adata_me3[:, (adata_me3.X > 1).sum(0) > adata_me3.n_obs / 100]
sc.pp.normalize_total(adata_me3)
sc.pp.log1p(adata_me3)
sc.pp.highly_variable_genes(adata_me3, flavor='seurat', subset=False, n_top_genes=1000)
adata_me3.var.loc[
    ['Hand2', 'Gfi1b', 'Wwp2', 'Sox2', 'Hoxc4', 'Wnt7b', 'Sall1', 'Dtx4', 'Nprl3', 'Nfe2'], 'highly_variable'] = True
adata_me3 = adata_me3[:, adata_me3.var['highly_variable']]

seed = random.randint(1, 10000)
# seed = 0
print('data: {}, seed {}'.format(dataset, seed))
wandb.init(project=dataset)
wandb.login()
model = SpaMV([adata_ac, adata_me3], weights=[1, 1], betas=[2, 2], zs_dim=15, zp_dims=[10, 10], interpretable=True,
              random_seed=seed, learning_rate=1e-2, dropout_prob=0, neighborhood_embedding=10, recon_types=['nb', 'nb'],
              omics_names=['H3K27ac', 'H3K27me3'], max_epochs=800, device=device)
model.train(dataset, size=100)
wandb.finish()
zs = model.get_separate_embedding()
for key, value in zs.items():
    adata_ac.obs = DataFrame(value.detach().cpu().numpy())
    embedding(adata_ac, color=adata_ac.obs.columns, basis='spatial', ncols=5, show=False, size=150, vmax='p99')
    plt.savefig('../Results/' + dataset + '/SpaMV_' + key + '.pdf')
z, w = model.get_embedding_and_feature_by_topic(map=True)
tc0 = compute_topic_coherence(adata_ac, w[0], topk=20)
td0 = compute_topic_diversity(w[0], topk=20)
tc1 = compute_topic_coherence(adata_me3, w[1], topk=20)
td1 = compute_topic_diversity(w[1], topk=20)
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
adata_ac.obs['SpaMV'] = z.idxmax(1)
sc.pl.embedding(adata_ac, color='SpaMV', basis='spatial', size=300, ax=ax,
                title='SpaMV\nTC on H3K27ac: {:.3f}, TD on H3K27ac: {:.3f}\nTC on H3K27me3: {:.3f}, TD on H3K27me3: {:.3f}'.format(
                    tc0, td0, tc1, td1), show=False)
plt.tight_layout()
plt.savefig('../Results/' + dataset + '/SpaMV_cluster.pdf')

plot_embedding_results([adata_ac, adata_me3], ['H3K27ac', 'H3K27me3'], z, w, folder_path='../Results/' + dataset + '/',
                       file_name='SpaMV_topics.pdf', size=300)
z.to_csv('../Results/' + dataset + '/SpaMV_z.csv')
w[0].to_csv('../Results/' + dataset + '/SpaMV_w_H3K27ac.csv')
w[1].to_csv('../Results/' + dataset + '/SpaMV_w_H3K27me3.csv')
