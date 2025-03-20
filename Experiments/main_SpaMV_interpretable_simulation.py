import os
import random
import sys

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import torch
from pandas import DataFrame
from scanpy.plotting import embedding

import wandb
from Methods.SpaMV.metrics import compute_topic_coherence, compute_topic_diversity
from Methods.SpaMV.spamv import SpaMV
from Methods.SpaMV.utils import plot_embedding_results, clr_normalize_each_cell

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
result = pd.DataFrame(columns=['Method', 'Dataset', 'Modality', 'Topic Coherence', 'Topic Diversity'])

# ts = None
for dataset in ['1_Simulation', '2_Simulation', '3_Simulation']:
    data_rna = sc.read_h5ad('../Dataset/' + dataset + '/adata_RNA.h5ad')
    sc.pp.normalize_total(data_rna)
    sc.pp.log1p(data_rna)
    data_pro = sc.read_h5ad('../Dataset/' + dataset + '/adata_ADT.h5ad')
    data_pro = clr_normalize_each_cell(data_pro)

    seed = random.randint(1, 10000)
    print(seed)
    wandb.init(project=dataset)
    wandb.login()
    model = SpaMV([data_rna, data_pro], zs_dim=10, zp_dims=[10, 10], weights=[1, 1], betas=[10, 10], interpretable=True,
                  random_seed=seed, neighborhood_embedding=0, recon_types=['nb', 'nb'],
                  omics_names=['Transcriptomics', 'Proteomics'], device=device)
    model.train(dataset)
    wandb.finish()

    zs = model.get_separate_embedding()
    for key, value in zs.items():
        data_rna.obs = DataFrame(value.detach().cpu().numpy())
        embedding(data_rna, color=data_rna.obs.columns, basis='spatial', ncols=5, show=False, size=200, vmax='p99')
        plt.savefig('../Results/' + dataset + '/' + key + '_stage3.pdf')
    z, w = model.get_embedding_and_feature_by_topic(map=False)
    plot_embedding_results([data_rna, data_pro], ['Transcriptomics', 'Proteomics'], z, w, save=True, show=False,
                           corresponding_features=True, folder_path='results/Simulation/',
                           file_name='spamv_interpretable_' + dataset + '_False.pdf', full=False, size=300)
    z, w = model.get_embedding_and_feature_by_topic(map=True)
    tc0 = compute_topic_coherence(data_rna, w[0], topk=10)
    td0 = compute_topic_diversity(w[0], topk=10)
    tc1 = compute_topic_coherence(data_pro, w[1], topk=5)
    td1 = compute_topic_diversity(w[1], topk=5)
    result.loc[len(result)] = ['STAMP', dataset, 'Transcriptomics', tc0, td0]
    result.loc[len(result)] = ['STAMP', dataset, 'Proteomics', tc1, td1]
    plot_embedding_results([data_rna, data_pro], ['Transcriptomics', 'Proteomics'], z, w, save=True, show=False,
                           corresponding_features=True, folder_path='../Results/' + dataset + '/',
                           file_name='SpaMV_topics.pdf', full=False, size=300)
    data_rna.obs['SpaMV'] = z.idxmax(1)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sc.pl.embedding(data_rna, basis='spatial', color='SpaMV', show=False,
                    title='SpaMV\nTC on RNA: {:.3f}, TD on RNA: {:.3f}\nTC on Protein: {:.3f}, TD on Protein: {:.3f}'.format(
                        tc0, td0, tc1, td1), size=600, ax=ax)
    plt.tight_layout()
    plt.savefig('../Results/' + dataset + '/cluster.pdf')
    plt.close()
result.to_csv('../Results/SpaMV_simulation_interpretable.csv', index=False)
