import os
from time import time
import sys
sys.path.append('/home/makx/Reproducible-main')
import pandas
import torch
from Methods.spaMultiVAE.spaMultiVAE import SPAMULTIVAE
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
import scanpy as sc
from Methods.spaMultiVAE.preprocess import normalize, geneSelection
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import anndata
import random
from Methods.SpaMV.metrics import compute_moranI, compute_jaccard, compute_supervised_scores

# torch.manual_seed(42)

### refine clustering labels by the majority of neighbors
def refine(sample_id, pred, dis, shape="square"):
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values()
        nbs = dis_tmp.iloc[0:(num_nbs + 1)]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if (v_c.loc[self_pred] < num_nbs / 2) and (np.max(v_c) > num_nbs / 2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
        if (i + 1) % 1000 == 0:
            print("Processed", i + 1, "lines")
    return np.array(refined_pred)
if __name__ == "__main__":

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(
        description='Spatial dependency-aware variational autoencoder for spatial multi-omics data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--select_genes', default=0, type=int)
    parser.add_argument('--select_proteins', default=0, type=int)
    parser.add_argument('--batch_size', default="auto")
    parser.add_argument('--maxiter', default=200, type=int)
    parser.add_argument('--train_size', default=0.95, type=float)
    parser.add_argument('--patience', default=200, type=int)
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--gene_noise', default=0, type=float)
    parser.add_argument('--protein_noise', default=0, type=float)
    parser.add_argument('--dropoutE', default=0, type=float,
                        help='dropout probability for encoder')
    parser.add_argument('--dropoutD', default=0, type=float,
                        help='dropout probability for decoder')
    parser.add_argument('--encoder_layers', nargs="+", default=[128, 64], type=int)
    parser.add_argument('--GP_dim', default=2, type=int)
    parser.add_argument('--Normal_dim', default=18, type=int)
    parser.add_argument('--gene_decoder_layers', nargs="+", default=[128], type=int)
    parser.add_argument('--protein_decoder_layers', nargs="+", default=[128], type=int)
    parser.add_argument('--dynamicVAE', default=True, type=bool,
                        help='whether to use dynamicVAE to tune the value of beta, if setting to false, then beta is fixed to initial value')
    parser.add_argument('--init_beta', default=10, type=float, help='initial coefficient of the KL loss')
    parser.add_argument('--min_beta', default=4, type=float, help='minimal coefficient of the KL loss')
    parser.add_argument('--max_beta', default=25, type=float, help='maximal coefficient of the KL loss')
    parser.add_argument('--KL_loss', default=0.025, type=float, help='desired KL_divergence value')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--grid_inducing_points', default=True, type=bool,
                        help='whether to generate grid inducing points or use k-means centroids on locations as inducing points')
    parser.add_argument('--inducing_point_steps', default=19, type=int)
    parser.add_argument('--inducing_point_nums', default=None, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--loc_range', default=20., type=float)
    parser.add_argument('--kernel_scale', default=20., type=float)
    parser.add_argument('--model_file', default='model.pt')
    parser.add_argument('--gene_denoised_counts_file', default='gene_denoised_counts.txt')
    parser.add_argument('--protein_denoised_counts_file', default='protein_denoised_counts.txt')
    parser.add_argument('--protein_sigmoid_file', default='protein_sigmoid.txt')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    
    result = pandas.DataFrame(
    columns=['Dataset', 'method', 'ari', 'ari_refined', 'mi', 'mi_refined', 
             'nmi', 'nmi_refined', 'ami', 'ami_refined' , 'hom', 'hom_refined', 'vme', 'vme_refined', 
             'average','average_refined', 'jaccard 1', 'jaccard 2', 'moran I', 'moran I_refined'])
    for dataset in ['4_Human_Lymph_Node']:
        data_omics1 = sc.read_h5ad('Dataset/' + dataset + '/adata_RNA.h5ad')
        sc.pp.highly_variable_genes(
        data_omics1,
        n_top_genes=3000,
        subset=True,
        flavor="seurat_v3",
        )
        sc.pp.pca(data_omics1, n_comps=50)
        if dataset == '4_Human_Lymph_Node':
            data_omics2 = sc.read_h5ad('Dataset/' + dataset + '/adata_ADT.h5ad')
            sc.pp.pca(data_omics2, n_comps=30)
        else:
            raise ValueError('Unknown dataset')
    loc_adata = data_omics1.obsm['spatial']
    if args.batch_size == "auto":
        if data_omics1.shape[0] <= 1024:
            args.batch_size = 128
        elif data_omics1.shape[0] <= 2048:
            args.batch_size = 256
        else:
            args.batch_size = 512
    else:
        args.batch_size = int(args.batch_size)

    print(args)
    # importantGenes_omics = geneSelection(data_omics1.X, n=500, plot=False)
    # adata_omics1 = data_omics1[:, importantGenes_omics]
    scaler = MinMaxScaler()
    loc_adata = scaler.fit_transform(loc_adata) * args.loc_range

    # We provide two ways to generate inducing point, argument "grid_inducing_points" controls whether to choice grid inducing or k-means
    # One way is grid inducing points, argument "inducing_point_steps" controls number of grid steps, the resulting number of inducing point is (inducing_point_steps+1)^2
    # Another way is k-means on the locations, argument "inducing_point_nums" controls number of inducing points
    if args.grid_inducing_points:
        eps = 1e-5
        initial_inducing_points = np.mgrid[0:(1 + eps):(1. / args.inducing_point_steps),
                                  0:(1 + eps):(1. / args.inducing_point_steps)].reshape(2, -1).T * args.loc_range
        print(initial_inducing_points.shape)
    else:
        loc_kmeans = KMeans(n_clusters=args.inducing_point_nums, n_init=100).fit(loc_adata)
        # np.savetxt("location_centroids.txt", loc_kmeans.cluster_centers_, delimiter=",")
        # np.savetxt("location_kmeans_labels.txt", loc_kmeans.labels_, delimiter=",", fmt="%i")
        initial_inducing_points = loc_kmeans.cluster_centers_

    data_omics1_new = data_omics1.copy()
    data_omics1_new = normalize(data_omics1_new,
                       size_factors=True,
                       normalize_input=True,
                       logtrans_input=True)
    data_omics2_new = data_omics2.copy()
    data_omics2_new = normalize(data_omics2_new,
                       size_factors=False,
                       normalize_input=True,
                       logtrans_input=True)
    data_omics2_no_scale = data_omics2.copy()
    data_omics2_no_scale = normalize(data_omics2_no_scale,
                                      size_factors=False,
                                      normalize_input=False,
                                      logtrans_input=True)

    for i in range(10):
        seed = random.randint(1, 10000)
        torch.manual_seed(seed)
        print('data: {}, iteration: {}, seed {}'.format(dataset, i + 1, seed))

        # Fit GMM model to the protein counts and use the smaller component as the initial values as protein background prior
        gm = GaussianMixture(n_components=2, covariance_type="diag", n_init=20).fit(data_omics2_no_scale.X.toarray())
        back_idx = np.argmin(gm.means_, axis=0)
        protein_log_back_mean = np.log(np.expm1(gm.means_[back_idx, np.arange(data_omics2_no_scale.n_vars)]))
        protein_log_back_scale = np.sqrt(gm.covariances_[back_idx, np.arange(data_omics2_no_scale.n_vars)])
        print("protein_back_mean shape", protein_log_back_mean.shape)

        model = SPAMULTIVAE(gene_dim=data_omics1_new.n_vars, protein_dim=data_omics2_new.n_vars, GP_dim=args.GP_dim,
                        Normal_dim=args.Normal_dim,
                        encoder_layers=args.encoder_layers, gene_decoder_layers=args.gene_decoder_layers,
                        protein_decoder_layers=args.protein_decoder_layers,
                        gene_noise=args.gene_noise, protein_noise=args.protein_noise, encoder_dropout=args.dropoutE,
                        decoder_dropout=args.dropoutD,
                        fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points,
                        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=data_omics1_new.n_obs,
                        KL_loss=args.KL_loss, dynamicVAE=args.dynamicVAE,
                        init_beta=args.init_beta, min_beta=args.min_beta, max_beta=args.max_beta,
                        protein_back_mean=protein_log_back_mean, protein_back_scale=protein_log_back_scale,
                        dtype=torch.float64, device=args.device)

        print(str(model))
        t0 = time()
        model.train_model(pos=loc_adata, gene_ncounts=data_omics1_new.X, gene_raw_counts=data_omics1_new.raw.X.toarray(),
                          gene_size_factors=data_omics1_new.obs.size_factors,
                          protein_ncounts=data_omics2_new.X, protein_raw_counts=data_omics2_new.raw.X.toarray(),
                          lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
                          num_samples=args.num_samples,
                          train_size=args.train_size, maxiter=args.maxiter, patience=args.patience, save_model=False,
                          model_weights=args.model_file)
        print('Training time: %d seconds.' % int(time() - t0))
       
        final_latent_file = 'Results/final_latent'+ '_' + str(i) + '.txt'
        final_latent = model.batching_latent_samples(X=loc_adata, gene_Y=data_omics1_new.X, protein_Y=data_omics2_new.X, batch_size=args.batch_size)
        np.savetxt(final_latent_file, final_latent, delimiter=",")
    

        # gene_denoised_counts, protein_denoised_counts, protein_sigmoid = model.batching_denoise_counts(X=loc_adata,
        #                                                                                                gene_Y=adata_omics1_new.X,
        #                                                                                                protein_Y=adata_omics2_new.X,
        #                                                                                                batch_size=args.batch_size,
        #                                                                                             n_samples=25)
        # # np.savetxt(args.gene_denoised_counts_file, gene_denoised_counts, delimiter=",")
        # np.savetxt(args.protein_denoised_counts_file, protein_denoised_counts, delimiter=",")
        # np.savetxt(args.protein_sigmoid_file, protein_sigmoid, delimiter=",")

        pos = data_omics1_new.obsm['spatial']
        pred = KMeans(n_clusters=10, n_init=100).fit_predict(final_latent)
        # np.savetxt("clustering_labels.txt", pred, delimiter=",", fmt="%i")


        dis = pairwise_distances(pos, metric="euclidean", n_jobs=-1).astype(np.double)
        pred_refined = refine(np.arange(pred.shape[0]), pred, dis, shape="hexagon")
        data_omics1_new.obs['spaMultiVAE'] = pred
        data_omics1_new.obs['spaMultiVAE_refined'] = pred_refined
        data_omics1_new.obsm['spaMultiVAE'] = final_latent
        data_omics2_new.obsm['spaMultiVAE'] = final_latent
        scores = compute_supervised_scores(data_omics1_new,'spaMultiVAE')
        scores_refined = compute_supervised_scores(data_omics1_new,'spaMultiVAE_refined')
        jaccard1 = compute_jaccard(data_omics1_new, 'spaMultiVAE')
        jaccard2 = compute_jaccard(data_omics2_new, 'spaMultiVAE')
        moranI = compute_moranI(data_omics1_new, 'spaMultiVAE')
        moranI_refined = compute_moranI(data_omics1_new, 'spaMultiVAE_refined')
        result.loc[len(result)] = [dataset, 'spaMultiVAE', scores['ari'], scores_refined['ari'],
                                   scores['mi'], scores_refined['mi'], scores['nmi'], scores_refined['nmi'], 
                                   scores['ami'], scores_refined['ami'],scores['hom'], scores_refined['hom'],
                                   scores['vme'], scores_refined['vme'], scores['average'], scores_refined['average'], 
                                   jaccard1, jaccard2, moranI, moranI_refined]
        result.to_csv('Results/Evaluation_spaMultiVAE.csv', index=False)
        print(result.tail(1))
    

