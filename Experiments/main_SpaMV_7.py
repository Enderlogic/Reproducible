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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset = '7_ME13_1'
data_folder = '../Dataset/' + dataset
omics_names = []
recon_types = []
weights = []
data = []
for filename in os.listdir(data_folder):
    omics_names.append(filename.split('.')[0].split('_')[1])
    recon_types.append('nb')
    weights.append(1)
    d = sc.read_h5ad(data_folder + '/' + filename)
    sc.pp.filter_genes(d, min_cells=10)
    d = d[:, (d.X > 1).sum(0) > d.n_obs / 100]
    sc.pp.highly_variable_genes(d, n_top_genes=3000, flavor='seurat')
    d.var.loc[['Hand2', 'Gfi1b', 'Wwp2', 'Sox2', 'Hoxc4', 'Wnt7b', 'Sall1', 'Dtx4', 'Nprl3',
               'Nfe2'], 'highly_variable'] = True
    d = d[:, d.var.highly_variable]
    sc.pp.filter_cells(d, min_genes=1)
    data.append(d)
data[0] = data[0][data[0].obs_names.intersection(data[1].obs_names)]
data[1] = data[1][data[0].obs_names]
seed = random.randint(1, 10000)
# seed = 2606
print('data: {}, seed {}'.format(dataset, seed))
wandb.init()
wandb.login()
model = SpaMV(data, weights=weights, interpretable=True, random_seed=seed, learning_rate=1e-2, dropout_prob=0,
              neighborhood_embedding=0, recon_types=recon_types, omics_names=omics_names, max_epochs=800, device=device)
model.train(dataset, size=100)
wandb.finish()
zs = model.get_separate_embedding()
for key, value in zs.items():
    data[0].obs = DataFrame(value.detach().cpu().numpy())
    embedding(data[0], color=data[0].obs.columns, basis='spatial', ncols=5, show=False, size=150, vmax='p99')
    plt.savefig('../Results/' + dataset + '/SpaMV_' + key + '.pdf')
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
                       file_name='SpaMV_topics.pdf', size=150)
z.to_csv('../Results/' + dataset + '/SpaMV_z.csv')
w[0].to_csv('../Results/' + dataset + '/SpaMV_w_' + omics_names[0] + '.csv')
w[1].to_csv('../Results/' + dataset + '/SpaMV_w_' + omics_names[1] + '.csv')
