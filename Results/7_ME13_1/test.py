from pandas import read_csv
import scanpy as sc
import matplotlib.pyplot as plt

z = read_csv('SpaMV_z.csv', index_col=0)
w = [read_csv('SpaMV_w_H3K27ac.csv', index_col=0), read_csv('SpaMV_w_H3K27me3.csv', index_col=0)]

adata_ac = sc.read_h5ad('../../Dataset/7_ME13_1/adata_H3K27ac_ATAC.h5ad')
sc.pp.filter_genes(adata_ac, min_cells=10)
adata_ac = adata_ac[:, (adata_ac.X > 1).sum(0) > adata_ac.n_obs / 100]
sc.pp.normalize_total(adata_ac)
sc.pp.log1p(adata_ac)
adata_me3 = sc.read_h5ad('../../Dataset/7_ME13_1/adata_H3K27me3_ATAC.h5ad')
sc.pp.filter_genes(adata_me3, min_cells=10)
adata_me3 = adata_me3[:, (adata_me3.X > 1).sum(0) > adata_me3.n_obs / 100]
sc.pp.normalize_total(adata_me3)
sc.pp.log1p(adata_me3)
topk = 20
for topic1 in w[0].columns:
    features1 = w[0].nlargest(topk, topic1).index.tolist()
    for topic2 in w[1].columns:
        if topic1 != topic2:
            features2 = w[1].nlargest(topk, topic2).index.tolist()
            common_features = set(features1).intersection(set(features2))
            if len(common_features) > 0:
                for f in common_features:
                    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                    adata_ac.obs[topic1] = z[topic1].values
                    adata_ac.obs[topic2] = z[topic2].values
                    sc.pl.embedding(adata_ac, basis='spatial', color=topic1, show=False, ax=axes[0, 0], size=200,
                                    vmax='p99')
                    sc.pl.embedding(adata_ac, basis='spatial', color=topic2, show=False, ax=axes[0, 1], size=200,
                                    vmax='p99')
                    sc.pl.embedding(adata_ac, basis='spatial', color=f, show=False, ax=axes[1, 0], size=200,
                                    cmap='coolwarm', vmax='p99')
                    sc.pl.embedding(adata_me3, basis='spatial', color=f, show=False, ax=axes[1, 1], size=200,
                                    cmap='coolwarm', vmax='p99')
                    plt.tight_layout()
                    # plt.savefig(f+'.pdf')
                    plt.show()
                    a = 1
a = 1
