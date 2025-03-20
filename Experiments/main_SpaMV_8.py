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
from Methods.SpaMV.utils import plot_embedding_results, clr_normalize_each_cell
from Methods.SpaMV.metrics import compute_topic_coherence, compute_topic_diversity
import scanpy as sc
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset = '8_ccRCC_R114_T'
data_folder = '../Dataset/' + dataset
omics_names = ['Transcriptomics', "Metabolomics"]
recon_types = ['nb', 'nb']

adata_rna = sc.read_h5ad(data_folder + '/adata_RNA.h5ad')
adata_rna = adata_rna[:, (adata_rna.X > 1).sum(0) > adata_rna.n_obs / 100]
sc.pp.filter_genes(adata_rna, min_cells=10)
sc.pp.filter_cells(adata_rna, min_counts=200)
sc.pp.normalize_total(adata_rna)
sc.pp.log1p(adata_rna)
sc.pp.highly_variable_genes(adata_rna, subset=True, flavor='seurat', n_top_genes=1000)

adata_sm = sc.read_h5ad(data_folder + '/adata_SM.h5ad')
adata_sm = adata_sm[:, (adata_sm.X > 1).sum(0) > adata_sm.n_obs / 100]
adata_sm = adata_sm[adata_sm.obs_names.intersection(adata_rna.obs_names)]
sc.pp.normalize_total(adata_sm)
sc.pp.log1p(adata_sm)
data = [adata_rna, adata_sm]

seed = random.randint(1, 10000)
print('data: {}, seed {}'.format(dataset, seed))
wandb.init(project=dataset)
wandb.login()
model = SpaMV(data, weights=[1, 1], zs_dim=15, zp_dims=[10, 10], interpretable=True, random_seed=seed,
              learning_rate=1e-2, dropout_prob=0, neighborhood_embedding=10, recon_types=recon_types,
              omics_names=omics_names, max_epochs=800, device=device)
model.train(dataset, size=50)
wandb.finish()
zs = model.get_separate_embedding()
for key, value in zs.items():
    data[0].obs = DataFrame(value.detach().cpu().numpy())
    embedding(data[0], color=data[0].obs.columns, basis='spatial', ncols=5, show=False, size=150, vmax='p99')
    plt.savefig('../Results/' + dataset + '/SpaMV_' + key + '_stage3.pdf')
z, w = model.get_embedding_and_feature_by_topic(map=True)
tc0 = compute_topic_coherence(data[0], w[0], topk=5 if omics_names[0] == 'Proteomics' else 20)
td0 = compute_topic_diversity(w[0], topk=5 if omics_names[0] == 'Proteomics' else 20)
tc1 = compute_topic_coherence(data[1], w[1], topk=5 if omics_names[1] == 'Proteomics' else 20)
td1 = compute_topic_diversity(w[1], topk=5 if omics_names[1] == 'Proteomics' else 20)
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
data[0].obs['SpaMV'] = z.idxmax(1)
sc.pl.embedding(data[0], color='SpaMV', basis='spatial', size=300, ax=ax,
                title='SpaMV\nTC on {}: {:.3f}, TD on {}: {:.3f}\nTC on {}: {:.3f}, TD on {}: {:.3f}'.format(
                    omics_names[0], tc0, omics_names[0], td0, omics_names[1], tc1, omics_names[1], td1),
                show=False)
plt.tight_layout()
plt.savefig('../Results/' + dataset + '/SpaMV_cluster.pdf')

plot_embedding_results(data, omics_names, z, w, folder_path='../Results/' + dataset + '/',
                       file_name='SpaMV_topics.pdf', size=100)
z.to_csv('../Results/' + dataset + '/SpaMV_z.csv')
w[0].to_csv('../Results/' + dataset + '/SpaMV_w_' + omics_names[0] + '.csv')
w[1].to_csv('../Results/' + dataset + '/SpaMV_w_' + omics_names[1] + '.csv')
