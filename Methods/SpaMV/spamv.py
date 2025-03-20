"""Main module."""
from typing import List

import numpy
import numpy as np
import pyro
import scanpy
import scanpy.plotting
import torch
import torch.nn.functional as F

import wandb
from anndata import AnnData
from matplotlib import pyplot as plt
from pandas import DataFrame
from pyro.infer import TraceMeanField_ELBO
from pyro.infer.enum import get_importance_trace
from pyro.poutine import scale, trace
from scanpy.plotting import embedding
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from torch.distributions.kl import kl_divergence
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm
from .metrics import compute_moranI, compute_jaccard, compute_supervised_scores, compute_topic_coherence, \
    compute_topic_diversity
from .model import spamv
from .layers import Measurement
from .utils import adjacent_matrix_preprocessing, get_init_bg, log_mean_exp, clustering


def pairwise_distances(x):
    # x should be two dimensional
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / sigma)


class SpaMV:
    def __init__(self, adatas: List[AnnData], interpretable: bool, zp_dims: List[int] = None, zs_dim: int = None,
                 weights: List[float] = None, betas: List[float] = None, recon_types: List[str] = None,
                 omics_names: List[str] = None, device: torch.device = None, hidden_dim: int = None, heads: int = 1,
                 neighborhood_depth: int = 2, neighborhood_embedding: int = 10, random_seed: int = 0,
                 max_epochs: int = 800, dropout_prob: float = .2, min_kl: float = 1, max_kl: float = 1,
                 learning_rate: float = None, folder_path: str = None, early_stopping: bool = True, patience: int = 200,
                 n_cluster: int = 10, test_mode: bool = False, result: DataFrame = None, plot: bool = False):
        pyro.clear_param_store()
        pyro.set_rng_seed(random_seed)
        torch.manual_seed(random_seed)
        self.n_omics = len(adatas)
        if zs_dim is None:
            zs_dim = 10 if interpretable else 32
        elif zs_dim <= 0:
            raise ValueError("zs_dim must be a positive integer")
        self.zs_dim = zs_dim
        if zp_dims is None:
            zp_dims = [10 if interpretable else 32 for _ in range(self.n_omics)]
        elif min(zp_dims) < 0:
            raise ValueError("all elements in zp_dims must be non-negative integers")
        self.zp_dims = zp_dims
        if weights is None:
            weights = [1 for _ in range(self.n_omics)]
        elif min(weights) < 0:
            raise ValueError("all elements in weights must be non-negative")
        self.weights = weights
        if betas is None:
            betas = [1 for _ in range(self.n_omics)]
        elif min(betas) < 0:
            raise ValueError("all elements in betas must be non-negative")
        self.betas = betas
        if recon_types is None:
            recon_types = ["nb" for _ in range(self.n_omics)] if interpretable else ["gauss", "gauss"]
        else:
            for recon_type in recon_types:
                if recon_type not in ['zinb', 'nb', 'gauss']:
                    raise ValueError("recon_type must be 'nb' or 'zinb' or 'gauss'")
        self.recon_types = recon_types
        if hidden_dim is None:
            hidden_dim = 128 if interpretable else 256
        elif hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        self.hidden_dim = hidden_dim
        self.omics_names = ["omics_{}".format(i) for i in range(self.n_omics)] if omics_names is None else omics_names
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if learning_rate is None:
            learning_rate = 1e-2 if interpretable else 1e-4
        self.learning_rate = learning_rate
        print(self.device)
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.neighborhood_depth = neighborhood_depth
        self.neighborhood_embedding = neighborhood_embedding
        self.adatas = adatas
        self.n_obs = adatas[0].shape[0]
        self.data_dims = [data.shape[1] for data in adatas]
        self.interpretable = interpretable
        self.max_epochs = max_epochs
        self.min_kl = min_kl
        self.max_kl = max_kl
        self.learning_rate = learning_rate
        self.folder_path = folder_path
        self.early_stopping = early_stopping
        self.patience = patience
        self.n_cluster = n_cluster
        self.test_mode = test_mode
        self.result = result
        self.plot = plot
        self.pretrain_epoch = 200
        self.epoch = 0
        self.epoch2 = 0
        self.meaningful_dimensions = {}

        self.x = [torch.tensor(data.X.toarray() if issparse(data.X) else data.X, device=self.device, dtype=torch.float)
                  for data in adatas]
        self.edge_index = adjacent_matrix_preprocessing(adatas, neighborhood_depth, neighborhood_embedding, self.device)

        self.init_bg_means = get_init_bg(self.x) if interpretable else None
        self.model = spamv(self.data_dims, zs_dim, zp_dims, self.init_bg_means, weights, hidden_dim, recon_types,
                           heads, interpretable, self.device, self.omics_names, dropout_prob)

    def train(self, dataname=None, size=200):
        if dataname is None:
            dataname = ''
        self.model = self.model.to(self.device)
        if self.early_stopping:
            self.early_stopper = EarlyStopper(patience=self.patience)

        pbar = tqdm(range(self.max_epochs), position=0, leave=True)
        loss_fn = lambda model, guide: TraceMeanField_ELBO(num_particles=1).differentiable_loss(
            scale(model, 1 / self.n_obs), scale(guide, 1 / self.n_obs), self.x, self.edge_index)
        with trace(param_only=True) as param_capture:
            loss = loss_fn(self.model.model, self.model.guide)
        params = set(site["value"].unconstrained() for site in param_capture.trace.nodes.values())
        optimizer = Adam(params, lr=self.learning_rate, betas=(.9, .999), weight_decay=0)
        params_zs = set(
            site["value"].unconstrained() for site in param_capture.trace.nodes.values() if 'zp' not in site['name'])
        optimizer_zs = Adam(params_zs, lr=self.learning_rate, betas=(.9, .999), weight_decay=0)
        for self.epoch in pbar:
            if self.epoch == self.pretrain_epoch:
                self.early_stopper.min_training_loss = np.inf
                if self.interpretable:
                    zs = self.get_separate_embedding()
                    for key, value in zs.items():
                        self.adatas[0].obs = DataFrame(value.detach().cpu().numpy())
                        embedding(self.adatas[0], color=self.adatas[0].obs.columns, basis='spatial', ncols=5,
                                  show=False, size=size, vmax='p99')
                        plt.savefig('../Results/' + dataname + '/' + key + '_stage1.pdf')
                        plt.close()
            if self.epoch >= self.pretrain_epoch:
                if self.epoch % 100 == 0 or self.epoch == self.pretrain_epoch:
                    n_epochs = 100
                    self.measurement = Measurement(self.zp_dims, self.hidden_dim, self.data_dims, self.recon_types,
                                                   self.omics_names, self.interpretable).to(self.device)
                    optimizer_measurement = Adam(self.measurement.parameters(), lr=self.learning_rate, betas=(.9, .999),
                                                 weight_decay=0)
                else:
                    n_epochs = 1
                for epoch_measurement in range(n_epochs):
                    self.measurement.train()
                    optimizer_measurement.zero_grad()
                    measurement_loss = self.get_measurement_loss()
                    measurement_loss.backward()
                    clip_grad_norm_(self.measurement.parameters(), 5)
                    optimizer_measurement.step()

            # train the model
            self.model.train()
            optimizer.zero_grad()
            loss = self.get_elbo()
            loss.backward()
            clip_grad_norm_(params, 5)
            optimizer.step()
            pbar.set_description(f"Epoch Loss:{loss:.3f}")

            if self.early_stopping:
                if self.early_stopper.early_stop(loss):
                    print("Early Stopping")
                    break

            if (self.epoch + 1) % 200 == 0 and self.test_mode:
                if self.interpretable:
                    z, w = self.get_embedding_and_feature_by_topic(map=False)
                    if self.result is not None:
                        self.result.loc[len(self.result)] = [dataname, 'SpaMV', self.epoch,
                                                             compute_topic_coherence(self.adatas[0], w[0],
                                                                                     topk=5 if self.omics_names[
                                                                                                   0] == 'Proteomics' else 20),
                                                             compute_topic_diversity(w[0], topk=5 if self.omics_names[
                                                                                                         0] == 'Proteomics' else 20),
                                                             compute_topic_coherence(self.adatas[1], w[1],
                                                                                     topk=5 if self.omics_names[
                                                                                                   1] == 'Proteomics' else 20),
                                                             compute_topic_diversity(w[1], topk=5 if self.omics_names[
                                                                                                         1] == 'Proteomics' else 20)]
                else:
                    if 'cluster' in self.adatas[0].obs:
                        z = self.get_embedding()
                        # for emb_type in ['all', 'shared']:
                        for emb_type in ['all']:
                            print("embedding type:", emb_type)
                            if emb_type == 'all':
                                # self.adatas[0].obsm[emb_type] = F.normalize(z, p=2, eps=1e-12, dim=1).detach().cpu().numpy()
                                self.adatas[0].obsm[emb_type] = z
                                self.adatas[0].obsm['zs+zp1'] = self.adatas[0].obsm[emb_type][:,
                                                                :self.zs_dim + self.zp_dims[0]]
                                self.adatas[1].obsm['zs+zp2'] = numpy.concatenate((self.adatas[0].obsm[emb_type][:,
                                                                                   :self.zs_dim],
                                                                                   self.adatas[0].obsm[emb_type][:,
                                                                                   -self.zp_dims[1]:]), axis=1)
                            else:
                                self.adatas[0].obsm[emb_type] = z[:, :self.zs_dim]
                                self.adatas[0].obsm['zs+zp1'] = self.adatas[0].obsm[emb_type]
                                self.adatas[1].obsm['zs+zp2'] = self.adatas[0].obsm[emb_type]
                            clustering(self.adatas[0], key=emb_type, add_key=emb_type, n_clusters=self.n_cluster,
                                       method='mclust', use_pca=True)
                            scores = compute_supervised_scores(self.adatas[0], emb_type)
                            scores_rename = {emb_type + " " + key: value for key, value in scores.items()}
                            wandb.log(scores_rename, step=self.epoch)
                            print("ari: ", str(scores['ari']), "\naverage: ", str(scores['average']))

                            jaccard1 = compute_jaccard(self.adatas[0], 'zs+zp1')
                            jaccard2 = compute_jaccard(self.adatas[1], 'zs+zp2')
                            wandb.log({emb_type + " jaccard1": jaccard1, emb_type + " jaccard2": jaccard2},
                                      step=self.epoch)
                            moranI = compute_moranI(self.adatas[0], emb_type)
                            wandb.log({emb_type + " moran I": moranI}, step=self.epoch)
                            if self.result is not None:
                                self.result.loc[len(self.result)] = [dataname, 'SpaMV', self.epoch, scores['ari'],
                                                                     scores['mi'], scores['nmi'], scores['ami'],
                                                                     scores['hom'], scores['vme'], scores['average'],
                                                                     jaccard1, jaccard2, moranI]
                    if self.plot:
                        self.adatas[0].obsm['SpaMV'] = self.get_embedding()
                        clustering(self.adatas[0], key='SpaMV', add_key='SpaMV', n_clusters=self.n_cluster,
                                   method='mclust', use_pca=True)
                        scanpy.pl.embedding(self.adatas[0], color='SpaMV', basis='spatial', show=False)
                        plt.tight_layout()
                        plt.savefig('../Results/' + dataname + '/SpaMV_' + str(self.epoch) + '.pdf')
        if self.interpretable:
            zs = self.get_separate_embedding()
            for key, value in zs.items():
                self.adatas[0].obs = DataFrame(value.detach().cpu().numpy())
                embedding(self.adatas[0], color=self.adatas[0].obs.columns, basis='spatial', ncols=5, show=False,
                          size=size, vmax='p99')
                plt.savefig('../Results/' + dataname + '/' + key + '_stage2.pdf')
                plt.close()
            pbar = tqdm(range(self.epoch, 600 + self.epoch), position=0, leave=True)
            self.early_stopper.min_training_loss = np.inf
            zps = self.model.get_private_latent(self.x, self.edge_index, False)

            scanpy.pp.neighbors(self.adatas[0], use_rep='spatial')
            for i in range(self.n_omics):
                self.meaningful_dimensions['zp_' + self.omics_names[i]] = self.get_meaningful_dimensions(zps[i])
            for self.epoch2 in pbar:
                # train shared model
                self.model.train()
                optimizer_zs.zero_grad()
                loss = self.get_elbo_shared()
                loss.backward()
                clip_grad_norm_(params_zs, 5)
                optimizer_zs.step()
                pbar.set_description(f"Epoch Loss:{loss:.3f}")

                if self.early_stopping:
                    if self.early_stopper.early_stop(loss):
                        print("Early Stopping")
                        break
            zs = self.model.get_shared_embedding(self.x, self.edge_index)
            self.meaningful_dimensions['zs'] = self.get_meaningful_dimensions(zs)
        return self.result

    def _kl_weight(self):
        kl = self.min_kl + self.epoch / self.max_epochs * (self.max_kl - self.min_kl)
        if kl > self.max_kl:
            kl = self.max_kl
        return kl

    def get_meaningful_dimensions(self, z):
        self.adatas[0].obsm['z'] = z.detach().cpu().numpy()
        morans_i = scanpy.metrics.morans_i(self.adatas[0], obsm='z')
        if morans_i.min() < .3:
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(morans_i.reshape(-1, 1))
            return np.where(kmeans.labels_ == np.argmax(kmeans.cluster_centers_))[0]
        else:
            return np.array(range(z.shape[1]))

    def _kl_weight2(self):
        kl = self.min_kl + (self.epoch2 - self.epoch) / self.max_epochs * (self.max_kl - self.min_kl)
        if kl > self.max_kl:
            kl = self.max_kl
        return kl

    def HSIC(self, x, y, s_x=1, s_y=1):
        m, _ = x.shape  # batch size
        K = GaussianKernelMatrix(x, s_x)
        L = GaussianKernelMatrix(y, s_y)
        H = torch.eye(m, device=self.device) - 1.0 / m * torch.ones((m, m), device=self.device)
        HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
        return HSIC

    def get_elbo(self):
        annealing_factor = self._kl_weight()
        elbo_particle = 0
        model_trace, guide_trace = get_importance_trace('flat', torch.inf, scale(self.model.model, 1 / self.n_obs),
                                                        scale(self.model.guide, 1 / self.n_obs),
                                                        (self.x, self.edge_index), {}, detach=False)
        for name, model_site in model_trace.nodes.items():
            if model_site["type"] == "sample":
                if model_site["is_observed"]:
                    elbo_particle = elbo_particle + model_site["log_prob_sum"]
                    wandb.log({name: -model_site["log_prob_sum"].item()}, step=self.epoch)
                else:
                    guide_site = guide_trace.nodes[name]
                    entropy_term = (log_mean_exp(torch.stack(
                        [guide_trace.nodes["zs_" + self.omics_names[i]]["fn"].log_prob(guide_site["value"]) for i in
                         range(len(self.data_dims))])) * guide_site['scale']).sum() if "zs" in name else guide_site[
                        'log_prob_sum']
                    elbo_particle += (model_site["log_prob_sum"] - entropy_term) * annealing_factor
                    wandb.log({name: (-model_site["log_prob_sum"] + entropy_term.sum()).item()}, step=self.epoch)
        if self.epoch >= self.pretrain_epoch:
            self.measurement.eval()
            output = self.measurement(self.model.get_private_latent(self.x, self.edge_index, True))
            for i in range(self.n_omics):
                for j in range(self.n_omics):
                    if i != j:
                        name = "from_" + self.omics_names[i] + "_to_" + self.omics_names[j]
                        if self.interpretable:
                            loss_measurement = -((self.x[j].sum(1, keepdim=True) * output[name]).var(
                                0)).mean() * model_trace.nodes[
                                                   'recon_' + self.omics_names[i] + '_from_' + self.omics_names[i]][
                                                   'log_prob_sum'].detach() * self.betas[i] / 10
                        else:
                            loss_measurement = output[name].std(0).sum() * self.betas[i]
                        elbo_particle -= loss_measurement
                        wandb.log({name + '_std': loss_measurement}, step=self.epoch)
        wandb.log({"Loss": -elbo_particle.item()}, step=self.epoch)
        return -elbo_particle

    def get_elbo_shared(self):
        annealing_factor = self._kl_weight2()
        elbo_particle = 0
        model_trace, guide_trace = get_importance_trace('flat', torch.inf, scale(self.model.model, 1 / self.n_obs),
                                                        scale(self.model.guide, 1 / self.n_obs),
                                                        (self.x, self.edge_index), {}, detach=False)
        for name, model_site in model_trace.nodes.items():
            if model_site["type"] == "sample":
                if model_site["is_observed"]:
                    elbo_particle = elbo_particle + model_site["log_prob_sum"]
                    wandb.log({name: -model_site["log_prob_sum"].item()}, step=self.epoch2)
                elif 'zs' in name:
                    guide_site = guide_trace.nodes[name]
                    entropy_term = (log_mean_exp(torch.stack(
                        [guide_trace.nodes["zs_" + self.omics_names[i]]["fn"].log_prob(guide_site["value"]) for i in
                         range(len(self.data_dims))])) * guide_site['scale']).sum()
                    elbo_particle += (model_site["log_prob_sum"] - entropy_term) * annealing_factor
                    wandb.log({name: (-model_site["log_prob_sum"] + entropy_term.sum()).item()}, step=self.epoch2)
                    omics_name = name.split("_")[1]
                    for on in self.omics_names:
                        if on != omics_name:
                            # loss_hsic = self.HSIC(guide_site['fn'].mean,
                            #                       guide_trace.nodes["zp_" + on]["fn"].mean.detach()[:,
                            #                       self.meaningful_dimensions['zp_' + on]]) * self.n_obs * np.sqrt(
                            #     max(self.data_dims)) * self.weights[self.omics_names.index(on)]
                            loss_hsic = self.HSIC(guide_site['fn'].mean,
                                                  guide_trace.nodes["zp_" + on]["fn"].mean.detach()[:,
                                                  self.meaningful_dimensions['zp_' + on]]) * self.n_obs * np.sqrt(
                                self.data_dims[self.omics_names.index(on)] * self.betas[self.omics_names.index(on)])
                            wandb.log({"HSIC between zs " + omics_name + " and zp " + on: loss_hsic.item()},
                                      step=self.epoch2)
                            elbo_particle -= loss_hsic

        names = list(model_trace.nodes.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if names[i].startswith('zs') and names[j].startswith('zs'):
                    kl_zs = kl_divergence(guide_trace.nodes[names[i]]['fn'].base_dist,
                                          guide_trace.nodes[names[j]]['fn'].base_dist).sum() * \
                            guide_trace.nodes[names[i]]['scale']
                    wandb.log({"KL_" + names[i] + "_" + names[j]: kl_zs.item()}, step=self.epoch2)
                    elbo_particle -= kl_zs
        wandb.log({"Loss": -elbo_particle.item()}, step=self.epoch2)
        return -elbo_particle

    def get_measurement_loss(self):
        output = self.measurement(self.model.get_private_latent(self.x, self.edge_index, False))
        loss = 0
        for i in range(self.n_omics):
            for j in range(self.n_omics):
                if i != j:
                    name = "from_" + self.omics_names[i] + "_to_" + self.omics_names[j]
                    if self.interpretable:
                        # output[name] = output[name] / output[name].sum(1, keepdim=True)
                        # loss += mse_loss(output[name], self.x[j].div(self.x[j].sum(1, keepdim=True))) * \
                        #         output[name].shape[0] * output[name].shape[1]
                        # output[name] = self.x[j].sum(1, keepdim=True) * output[name] / output[name].sum(1, keepdim=True)
                        output[name] = self.x[j].sum(1, keepdim=True) * output[name]
                        loss += mse_loss(output[name], self.x[j])
                    else:
                        loss += mse_loss(output[name], self.x[j])
        return loss

    def save(self, path):
        self.model.save(path)

    def load(self, path, map_location=torch.device('cpu')):
        self.model.load(path, map_location=map_location)

    def get_separate_embedding(self):
        return self.model.get_separate_embedding(self.x, self.edge_index)

    def get_embedding(self):
        '''
        This function is used to get the embeddings. The returned embedding is stored in a pandas dataframe object if
        the model is in interpretable mode. Shared embeddings will be present in the first zs_dim columns, and private
        embeddings will be present in the following columns given their input orders.

        For example, if the input data is [data1, data2] and the shared latent dimension and both private latent
        dimensions are all 5, (i.e., zs_dim=5, zp_dim[0]=5, zp_dim[1]=5). Then the first 5 columns in returned dataframe
        will be the shared embeddings, and the following 5 columns will be the private embeddings for data1, and the
        last 5 columns will be the private embeddings for data2.
        '''
        z_mean = self.model.get_embedding(self.x, self.edge_index)
        if self.interpretable:
            columns_name = ["Shared topic {}".format(i + 1) for i in range(self.zs_dim)]
            for i in range(self.n_omics):
                columns_name += [self.omics_names[i] + ' private topic {}'.format(j + 1) for j in
                                 range(self.zp_dims[i])]
            spot_topic = DataFrame(z_mean.detach().cpu().numpy(), columns=columns_name)
            spot_topic.set_index(self.adatas[0].obs_names, inplace=True)
            return spot_topic
        else:
            return F.normalize(z_mean, p=2, eps=1e-12, dim=1).detach().cpu().numpy()

    def get_embedding_and_feature_by_topic(self, map=False):
        '''
        This function is used to get the feature by topic. The returned list contains feature by topic for each modality
        according to their input order. The row names in the returned dataframes are the feature names in the
        corresponding modality, and the column names are the topic names.

        For example, if the input data is [data1, data2] and the shared latent dimension and both private latent are all
        5. Assume, data1 is RNA modality and data2 is Protein modality. Then feature_topics[0] would be the feature by
        topic matrix for RNA, and each row represents a gene and each column represents a topic. The topic names are
        defined in the same way as the get_embedding() function. That is, Topics 1-5 are shared topics, Topics 6-10 are
        private topics for modality 1 (RNA), and Topics 11-15 are private topics for modality 2 (Protein).
        '''
        if self.interpretable:
            z_mean = self.model.get_embedding(self.x, self.edge_index)
            columns_name = ["Shared topic {}".format(i + 1) for i in range(self.zs_dim)]
            for i in range(self.n_omics):
                columns_name += [self.omics_names[i] + ' private topic {}'.format(j + 1) for j in
                                 range(self.zp_dims[i])]
            spot_topic = DataFrame(z_mean.detach().cpu().numpy(), columns=columns_name)
            spot_topic.set_index(self.adatas[0].obs_names, inplace=True)
            feature_topics = self.model.get_feature_by_topic()
            for i in range(len(self.zp_dims)):
                feature_topics[i] = DataFrame(feature_topics[i],
                                              columns=["Shared topic {}".format(j + 1) for j in range(self.zs_dim)] + [
                                                  self.omics_names[i] + " private topic {}".format(j + 1) for j in
                                                  range(self.zp_dims[i])], index=self.adatas[i].var_names)
            if map:
                spot_topic, feature_topics = self.merge_and_prune(spot_topic, feature_topics)
            return spot_topic, feature_topics
        else:
            raise Exception("This function can only be used with interpretable mode.")

    def merge_and_prune(self, spot_topic, feature_topic):
        # prune noisy topics
        meaningful_topics = ["Shared topic {}".format(i + 1) for i in self.meaningful_dimensions['zs']]
        for i in range(self.n_omics):
            meaningful_topics += [self.omics_names[i] + " private topic {}".format(j + 1) for j in
                                  self.meaningful_dimensions['zp_' + self.omics_names[i]]]
        spot_topic = spot_topic[meaningful_topics]
        for i in range(self.n_omics):
            existing_topics = [col for col in meaningful_topics if col in feature_topic[i].columns]
            feature_topic[i] = feature_topic[i][existing_topics]
        return spot_topic, feature_topic


class EarlyStopper:
    def __init__(self, patience=50, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_training_loss = np.inf

    def early_stop(self, training_loss):
        if training_loss < self.min_training_loss:
            self.min_training_loss = training_loss
            self.counter = 0
        elif training_loss > (self.min_training_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def reset(self):
        self.counter = 0
        self.min_training_loss = np.Inf
