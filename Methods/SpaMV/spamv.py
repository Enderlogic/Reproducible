"""Main module."""
from typing import List

import anndata
import numpy
import numpy as np
import pyro
import scanpy
import scanpy.plotting
import torch
import torch.nn.functional as F

import wandb
from anndata import AnnData
from pandas import DataFrame
from pyro.infer import TraceMeanField_ELBO
from pyro.infer.enum import get_importance_trace
from pyro.poutine import scale, trace
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm
from .metrics import compute_moranI, compute_jaccard, compute_supervised_scores, compute_topic_coherence
from .model import spamv
from .layers import Distinguished_Decoder
from .utils import adjacent_matrix_preprocessing, get_init_bg, plot_embedding_results, log_mean_exp, clustering, \
    compute_similarity


class SpaMV:
    def __init__(self, adatas: List[AnnData], zp_dims: List[int] = None, zs_dim: int = 5, weights: List[float] = None,
                 recon_types: List[str] = None, omics_names: List[str] = None, beta: List[float] = None,
                 device: torch.device = None, hidden_dim: int = 128, heads: int = 1, neighborhood_depth: int = 3,
                 neighborhood_embedding: int = 10, interpretable: bool = True, random_seed: int = 1214,
                 max_epochs: int = 800, dropout_prob: float = .2, min_kl: float = 1, max_kl: float = 1,
                 learning_rate: float = None, folder_path: str = None, early_stopping: bool = True, patience: int = 150,
                 n_cluster: int = 10, test_mode: bool = False, result: DataFrame = None):
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
        elif min(weights) <= 0:
            raise ValueError("all elements in weights must be positive")
        self.weights = weights
        if recon_types is None:
            recon_types = ["gauss" for _ in range(self.n_omics)] if interpretable else ['nb' for _ in
                                                                                        range(self.n_omics)]
        else:
            for recon_type in recon_types:
                if recon_type not in ['zinb', 'nb', 'gauss']:
                    raise ValueError("recon_type must be 'nb' or 'zinb' or 'gauss'")
        if learning_rate is None:
            learning_rate = 1e-2 if interpretable else 1e-4
        self.recon_types = recon_types
        self.omics_names = ["omics_{}".format(i) for i in range(self.n_omics)] if omics_names is None else omics_names
        self.beta = [1, 1] if beta is None else beta
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        self.pretrain_epoch = 10

        self.x = [torch.tensor(data.X.toarray() if issparse(data.X) else data.X, device=self.device, dtype=torch.float)
                  for data in adatas]
        self.edge_index = adjacent_matrix_preprocessing(adatas, neighborhood_depth, neighborhood_embedding, self.device)

        self.init_bg_means = get_init_bg(self.x) if interpretable else None
        self.model = spamv(self.data_dims, zs_dim, zp_dims, self.init_bg_means, weights, hidden_dim, recon_types,
                           heads, interpretable, self.device, self.omics_names, dropout_prob)

    def train(self, dataname=None, size=400):
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
        # self.distinguished_decoder = Distinguished_Decoder(self.zp_dims, self.hidden_dim, self.data_dims,
        #                                                    self.recon_types, self.omics_names,
        #                                                    self.interpretable).to(self.device)
        # optimizer_distinguished_decoder = Adam(self.distinguished_decoder.parameters(),
        #                                        lr=self.learning_rate, betas=(.9, .999), weight_decay=0)

        for self.epoch in pbar:
            if self.epoch == self.pretrain_epoch:
                self.early_stopper.min_training_loss = np.inf
            if self.epoch >= self.pretrain_epoch:
                if self.epoch % 100 == 0 or self.epoch == self.pretrain_epoch:
                    n_epochs = 100
                    self.distinguished_decoder = Distinguished_Decoder(self.zp_dims, self.hidden_dim, self.data_dims,
                                                                       self.recon_types, self.omics_names,
                                                                       self.interpretable).to(self.device)
                    optimizer_distinguished_decoder = Adam(self.distinguished_decoder.parameters(),
                                                           lr=self.learning_rate, betas=(.9, .999), weight_decay=0)
                else:
                    n_epochs = 1
                for epoch_distinguished_decoder in range(n_epochs):
                    self.distinguished_decoder.train()
                    optimizer_distinguished_decoder.zero_grad()
                    distinguished_loss = self.get_distinguished_loss()
                    distinguished_loss.backward()
                    clip_grad_norm_(self.distinguished_decoder.parameters(), 5)
                    optimizer_distinguished_decoder.step()
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
                    if self.test_mode:
                        if self.interpretable:
                            z, w = self.get_embedding_and_feature_by_topic(map=True)
                            self.adatas[0].obs['spamv'] = DataFrame(MinMaxScaler().fit_transform(z),
                                                                    columns=z.columns, index=z.index).idxmax(1)
                            plot_embedding_results(self.adatas, self.omics_names, z, w, folder_path=self.folder_path,
                                                   file_name='spamv_' + str(self.epoch + 1) + '.pdf', size=size)
                        else:
                            z = self.get_embedding()
                            # for emb_type in ['all', 'shared']:
                            for emb_type in ['all']:
                                print("embedding type", emb_type)
                                if emb_type == 'all':
                                    self.adatas[0].obsm[emb_type] = F.normalize(z, p=2, eps=1e-12,
                                                                                dim=1).detach().cpu().numpy()
                                    self.adatas[0].obsm['zs+zp1'] = self.adatas[0].obsm[emb_type][:,
                                                                    :self.zs_dim + self.zp_dims[0]]
                                    self.adatas[1].obsm['zs+zp2'] = numpy.concatenate((
                                        self.adatas[0].obsm[emb_type][
                                        :, :self.zs_dim],
                                        self.adatas[0].obsm[emb_type][
                                        :, -self.zp_dims[1]:]),
                                        axis=1)
                                else:
                                    self.adatas[0].obsm[emb_type] = F.normalize(z[:, :self.zs_dim], p=2, eps=1e-12,
                                                                                dim=1).detach().cpu().numpy()
                                    self.adatas[0].obsm['zs+zp1'] = self.adatas[0].obsm[emb_type]
                                    self.adatas[1].obsm['zs+zp2'] = self.adatas[0].obsm[emb_type]

                                jaccard1 = compute_jaccard(self.adatas[0], 'zs+zp1')
                                jaccard2 = compute_jaccard(self.adatas[1], 'zs+zp2')
                                wandb.log({emb_type + " jaccard1": jaccard1, emb_type + " jaccard2": jaccard2},
                                          step=self.epoch)
                                clustering(self.adatas[0], key=emb_type, add_key=emb_type,
                                           n_clusters=self.n_cluster, method='mclust', use_pca=True)
                                moranI = compute_moranI(self.adatas[0], emb_type)
                                wandb.log({emb_type + " moran I": moranI}, step=self.epoch)

                                print("jaccard 1: ", str(jaccard1), "jaccard 2:", str(jaccard2), "moran I: ",
                                      str(moranI))

                                if 'cluster' in self.adatas[0].obs:
                                    scores = compute_supervised_scores(self.adatas[0], emb_type)
                                    scores_rename = {emb_type + " " + key: value for key, value in scores.items()}
                                    wandb.log(scores_rename, step=self.epoch)
                                    print("ari: ", str(scores['ari']), "\naverage: ", str(scores['average']))
                                else:
                                    scores = {key: np.nan for key in
                                              ['ari', 'mi', 'nmi', 'ami', 'hom', 'vme', 'average']}
                                if self.result is not None:
                                    self.result.loc[len(self.result)] = [dataname, 'SpaMV', self.epoch, scores['ari'],
                                                                         scores['mi'], scores['nmi'], scores['ami'],
                                                                         scores['hom'], scores['vme'],
                                                                         scores['average'], jaccard1, jaccard2,
                                                                         moranI]
                    break

            if (self.epoch + 1) % 50 == 0 and self.test_mode:
                if self.interpretable:
                    z, w = self.get_embedding_and_feature_by_topic(map=False)
                    # self.adatas[0].obs['spamv'] = DataFrame(MinMaxScaler().fit_transform(z), columns=z.columns,
                    #                                         index=z.index).idxmax(1)
                    plot_embedding_results(self.adatas, self.omics_names, z, w, folder_path=self.folder_path,
                                           file_name='spamv_' + dataname + '_' + str(self.epoch + 1) + '.pdf')
                    if self.result is not None:
                        self.result.loc[len(self.result)] = [self.zp_dims[0], self.zs_dim, self.hidden_dim, self.heads,
                                                             self.max_kl, self.beta[0], self.beta[1],
                                                             self.learning_rate, self.weights[0] / self.weights[1],
                                                             self.epoch, compute_topic_coherence(self.adatas[0], w[0],
                                                                                                 topk=5 if
                                                                                                 self.omics_names[
                                                                                                     0] == 'Proteomics' else 20),
                                                             compute_topic_coherence(self.adatas[1], w[1], topk=5 if
                                                             self.omics_names[1] == 'Proteomics' else 20)]
                else:
                    # z = self.get_separate_embedding()
                    # for key, item in z.items():
                    #     temp = anndata.AnnData(item.detach().cpu().numpy())
                    #     temp.obsm['spatial'] = self.adatas[0].obsm['spatial']
                    #     temp.obsm[key] = item.detach().cpu().numpy()
                    #     clustering(temp, key=key, add_key=key, n_clusters=10)
                    #     scanpy.plotting.embedding(temp, color=key, basis='spatial', title=key + '_' + str(self.epoch),
                    #                               size=400)
                    if 'cluster' in self.adatas[0].obs:
                        z = self.get_embedding()
                        # for emb_type in ['all', 'shared']:
                        for emb_type in ['all']:
                            print("embedding type:", emb_type)
                            if emb_type == 'all':
                                self.adatas[0].obsm[emb_type] = F.normalize(z, p=2, eps=1e-12,
                                                                            dim=1).detach().cpu().numpy()
                                self.adatas[0].obsm['zs+zp1'] = self.adatas[0].obsm[emb_type][:,
                                                                :self.zs_dim + self.zp_dims[0]]
                                self.adatas[1].obsm['zs+zp2'] = numpy.concatenate((self.adatas[0].obsm[emb_type][:,
                                                                                   :self.zs_dim],
                                                                                   self.adatas[0].obsm[emb_type][:,
                                                                                   -self.zp_dims[1]:]), axis=1)
                            else:
                                self.adatas[0].obsm[emb_type] = F.normalize(z[:, :self.zs_dim], p=2, eps=1e-12,
                                                                            dim=1).detach().cpu().numpy()
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
                                self.result.loc[len(self.result)] = [emb_type, self.epoch, scores['ari'], scores['mi'],
                                                                     scores['nmi'], scores['ami'], scores['hom'],
                                                                     scores['vme'], scores['average'], jaccard1,
                                                                     jaccard2, moranI]

                        # z = self.get_separate_embedding()
                        # zs = torch.cat([z['zs_' + self.omics_names[0]], z['zs_' + self.omics_names[1]]], dim=0)
                        # zdata = anndata.AnnData(zs.detach().cpu().numpy())
                        # zdata.obs['omics'] = [self.omics_names[0]] * self.adatas[0].n_obs + [self.omics_names[1]] * \
                        #                      self.adatas[1].n_obs
                        # sc.pp.neighbors(zdata)
                        # sc.tl.umap(zdata)
                        # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                        # sc.pl.umap(zdata, color='omics', show=False, ax=ax)
                        # plt.tight_layout()
                        # plt.savefig(
                        #     'Results/4_Human_Lymph_Node/umap_shared_omics_' + str(self.epoch) + '.pdf')
                        # plt.show()
                        # plt.close()
        return self.result

    def _kl_weight(self):
        kl = self.min_kl + self.epoch / self.max_epochs * (self.max_kl - self.min_kl)
        if kl > self.max_kl:
            kl = self.max_kl
        return kl

    def get_elbo(self):
        self.model = self.model.to(self.device)
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
            z = self.model.get_private_latents(self.x, self.edge_index, train_eval=True)
            self.distinguished_decoder.eval()
            output = self.distinguished_decoder(z.split(self.zp_dims, dim=1))
            for i in range(self.n_omics):
                for j in range(self.n_omics):
                    if i != j:
                        name = "from_" + self.omics_names[i] + "_to_" + self.omics_names[j]
                        if self.interpretable:
                            output[name] = output[name] / output[name].sum(1, keepdim=True)
                            loss_measurement = output[name].std(0).mean() * output[name].shape[1] * np.sqrt(
                                z.shape[0]) * self.beta[i]
                        else:
                            loss_measurement = output[name].std(0).mean() * np.sqrt(z.shape[0]) * self.beta[i]
                        elbo_particle -= loss_measurement
                        wandb.log({name + '_std': loss_measurement}, step=self.epoch)
        wandb.log({"Loss": -elbo_particle.item()}, step=self.epoch)
        return -elbo_particle

    def get_distinguished_loss(self):
        z = self.model.get_private_latents(self.x, self.edge_index, train_eval=False)
        output = self.distinguished_decoder(z.split(self.zp_dims, dim=1))
        loss = 0
        for i in range(self.n_omics):
            for j in range(self.n_omics):
                if i != j:
                    name = "from_" + self.omics_names[i] + "_to_" + self.omics_names[j]
                    if self.interpretable:
                        # output[name] = output[name] / output[name].sum(1, keepdim=True)
                        # loss += mse_loss(output[name], self.x[j].div(self.x[j].sum(1, keepdim=True))) * \
                        #         output[name].shape[0] * output[name].shape[1]
                        output[name] = self.x[j].sum(1, keepdim=True) * output[name] / output[name].sum(1, keepdim=True)
                        loss += mse_loss(output[name], self.x[j])
                    else:
                        loss += mse_loss(output[name], self.x[j])
        return loss

    def save(self, path):
        self.model.save(path)

    def load(self, path, map_location=torch.device('cpu')):
        self.model.load(path, map_location=map_location)

    def get_separate_embedding(self):
        return self.model.get_separate_latents(self.x, self.edge_index)

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
        z_mean = self.model.get_embedding(self.x, self.edge_index, train_eval=False)
        if self.interpretable:
            columns_name = ["Shared topic {}".format(i + 1) for i in range(self.zs_dim)]
            for i in range(self.n_omics):
                columns_name += [self.omics_names[i] + ' private topic {}'.format(j + 1) for j in
                                 range(self.zp_dims[i])]
            spot_topic = DataFrame(z_mean.detach().cpu().numpy(), columns=columns_name)
            spot_topic.set_index(self.adatas[0].obs_names, inplace=True)
            return spot_topic
        else:
            # return F.normalize(z_mean, p=2, eps=1e-12, dim=1).detach().cpu().numpy()
            return z_mean

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
            z_mean = self.model.get_embedding(self.x, self.edge_index, train_eval=False)
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

    def merge_and_prune(self, topic_spot, feature_topics, threshold=0.8):
        self.adatas[0].obsm['topics'] = topic_spot.values
        scanpy.pp.neighbors(self.adatas[0], use_rep='spatial')
        morans_i = scanpy.metrics.morans_i(self.adatas[0], obsm='topics')
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(morans_i.reshape(-1, 1))
        topic_spot = topic_spot[topic_spot.columns[kmeans.labels_ == np.argmax(kmeans.cluster_centers_)]]
        for i in range(len(feature_topics)):
            feature_topics[i] = feature_topics[i][feature_topics[i].columns.intersection(topic_spot.columns)]
        topic_spot_update = topic_spot.copy()
        feature_topics_update = feature_topics.copy()

        while True:
            similarity_spot, similarity_feature = compute_similarity(topic_spot_update, feature_topics_update)
            if similarity_spot.stack().max() > threshold:
                topic1, topic2 = similarity_spot.stack().idxmax()
                print('The pattern between', topic1, 'and', topic2,
                      'is similar, with a cosine similarity of {:.3f}. Therefore, we decided to prune'.format(
                          similarity_spot.stack().max()), topic2 + '.')
            elif similarity_feature.stack().max() > threshold:
                topic1, topic2 = similarity_feature.stack().idxmax()
                print('The weight vectors between', topic1, 'and', topic2,
                      'are similar, with an average cosine similarity of {:.3f}. Therefore we decided to prune'.format(
                          similarity_feature.stack().max()), topic2 + '.')
            else:
                break
            if topic1.split(' ')[0] == topic2.split(' ')[0]:
                topic_spot_update[topic1] += topic_spot_update[topic2]
                if 'Shared' in topic2:
                    for i in range(self.n_omics):
                        feature_topics_update[i][topic1] += feature_topics_update[i][topic2]
                else:
                    for i in range(self.n_omics):
                        if topic2 in feature_topics_update[i].columns:
                            feature_topics_update[i][topic1] += feature_topics_update[i][topic2]
            topic_spot_update = topic_spot_update.drop(topic2, axis=1)
            if 'Shared' in topic2:
                for i in range(self.n_omics):
                    feature_topics_update[i] = feature_topics_update[i].drop(topic2, axis=1)
            else:
                for i in range(self.n_omics):
                    if topic2 in feature_topics_update[i].columns:
                        feature_topics_update[i] = feature_topics_update[i].drop(topic2, axis=1)
            continue
        return topic_spot_update, feature_topics_update


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
