# Taken from: https://github.com/AI4CTS/E2Usd

import abc
import os
import sys

sys.path.append(os.path.dirname(__file__))
import numpy
import numpy as np
import math
import torch.nn as nn
import torch
from sklearn import mixture

params = {
    "batch_size": 1,
    "channels": 30,
    "win_size": 256,
    "win_type": 'rect',
    "depth": 1,
    "nb_steps": 20,
    "in_channels": 1,
    "kernel_size": 3,
    "lr": 0.003,
    "out_channels": 4,
    "reduced_size": 80,
    "cuda": False,
    "gpu": 0,
    "M": 20,
    "N": 4
}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return np.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index]


def compact(series):
    compacted = []
    pre = series[0]
    compacted.append(pre)
    for e in series[1:]:
        if e != pre:
            pre = e
            compacted.append(e)
    return compacted


def remove_duplication(series):
    result = []
    for e in series:
        if e not in result:
            result.append(e)
    return result


def reorder_label(label):
    label = np.array(label)
    ordered_label_set = remove_duplication(compact(label))
    idx_list = [np.argwhere(label == e) for e in ordered_label_set]
    for i, idx in enumerate(idx_list):
        label[idx] = i
    return label


def all_normalize(data_tensor):
    mean = np.mean(data_tensor)
    var = np.var(data_tensor)
    i = 0
    for channel in data_tensor[0]:
        data_tensor[0][i] = (channel - mean) / math.sqrt(var)
        i += 1
    return data_tensor


class BasicClusteringClass:
    def __init__(self, params):
        pass

    @abc.abstractmethod
    def fit(self, X):
        pass


class BasicEncoderClass:
    def __init__(self, params):
        self._set_parmas(params)

    @abc.abstractmethod
    def _set_parmas(self, params):
        pass

    @abc.abstractmethod
    def fit(self, X):
        pass

    @abc.abstractmethod
    def encode(self, X, win_size, step):
        pass


def add_noise(data, mean=0, std=0.05):
    noise = torch.randn(data.size()) * std + mean
    return data + noise


def apply_scaling(data, mean=1.0, std=0.1):
    scaling_factor = torch.normal(mean=mean, std=std, size=data.size())
    return data * scaling_factor


# 随机选择数据增强方法


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class NetworkDDEM(torch.nn.Module):
    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size):
        super(NetworkDDEM, self).__init__()
        moving_ks = 5
        self.decompsition = series_decomp(moving_ks)

        self.trend_cnn = nn.Conv1d(in_channels, reduced_size, kernel_size=kernel_size)
        self.seasonal_cnn = nn.Conv1d(in_channels, reduced_size, kernel_size=kernel_size)

        self.trend_cnn.requires_grad_(False)
        self.seasonal_cnn.requires_grad_(False)
        self.trend_pooling = torch.nn.AdaptiveMaxPool1d(1)
        self.seasonal_pooling = torch.nn.AdaptiveMaxPool1d(1)

        self.linear_trend = torch.nn.Linear(reduced_size, out_channels)
        self.linear_seasonal = torch.nn.Linear(reduced_size, out_channels)

        self.linear = torch.nn.Linear(out_channels * 2, out_channels)

        self.trade_off_freq = 33

    def forward(self, x):
        low_specx = torch.fft.rfft(x, dim=-1)
        low_specx = low_specx[:, :, :self.trade_off_freq]
        x = torch.fft.irfft(low_specx, dim=-1) * self.trade_off_freq / x.size(-1)
        seasonal_init, trend_init = self.decompsition(x)

        trend_x, seasonal_x = (self.trend_cnn(trend_init)), (self.seasonal_cnn(seasonal_init))
        trend_x_reduced, seasonal_x_reduced = self.trend_pooling(trend_x), self.seasonal_pooling(seasonal_x)

        trend_x_reduced, seasonal_x_reduced = trend_x_reduced.squeeze(2), seasonal_x_reduced.squeeze(2)

        trend_x_embedding, seasonal_x_embedding = (self.linear_trend(trend_x_reduced)), (
            self.linear_seasonal(seasonal_x_reduced))

        embedding = torch.concat([trend_x_embedding, seasonal_x_embedding], dim=-1)
        embedding = self.linear(embedding)

        return embedding, trend_x_embedding, seasonal_x_embedding


class fncc_loss(torch.nn.modules.loss._Loss):

    def __init__(self, win_size, M, N, win_type):
        super(fncc_loss, self).__init__()
        self.win_size = win_size
        self.win_type = win_type
        self.M = M
        self.N = N
        self.total = 0
        self.total_condition = 0

    def forward(self, batch, encoder, save_memory=False):
        M = self.M
        N = self.N
        length_pos_neg = self.win_size
        total_length = batch.size(2)
        center_list = []
        center_trend_list = []
        center_seasonal_list = []
        loss1 = 0

        total_embeddings = []
        total_trend_embeddings = []
        total_seasonal_embeddings = []

        for i in range(M):
            random_pos = np.random.randint(0, high=total_length - length_pos_neg * 2 + 1, size=1)
            rand_samples = [batch[0, :, i: i + length_pos_neg] for i in range(random_pos[0], random_pos[0] + N)]

            intra_sample = torch.stack(rand_samples)

            embeddings, trend_x_embedding, seasonal_x_embedding = encoder(
                intra_sample)  # ([4, 4]) N / embedding_channel
            total_embeddings.append(embeddings)
            total_trend_embeddings.append(trend_x_embedding)
            total_seasonal_embeddings.append(seasonal_x_embedding)

            size_representation = embeddings.size(1)

            for i in range(N):
                for j in range(N):
                    if j <= i:
                        continue
                    else:

                        similarity_embedding = torch.bmm(
                            embeddings[i].view(1, 1, size_representation),
                            embeddings[j].view(1, size_representation, 1))
                        loss1_term = -torch.mean(torch.nn.functional.logsigmoid(
                            similarity_embedding))
                        loss1 += loss1_term

            center = torch.mean(embeddings, dim=0)
            center_trend = torch.mean(trend_x_embedding, dim=0)
            center_seasonal = torch.mean(seasonal_x_embedding, dim=0)
            center_list.append(center)
            center_seasonal_list.append(center_seasonal)
            center_trend_list.append(center_trend)

        loss2 = 0
        smi = []
        loss2_item = []
        totalnumber = 0
        for i in range(M):
            for ii in range(N):
                for j in range(M):
                    if j <= i:
                        continue
                    for jj in range(N):
                        totalnumber += 1
                        similarity_trend = torch.bmm(
                            total_trend_embeddings[i][ii].view(1, 1, size_representation),
                            total_trend_embeddings[j][jj].view(1, size_representation, 1))
                        similarity_seasonal = torch.bmm(
                            total_seasonal_embeddings[i][ii].view(1, 1, size_representation),
                            total_seasonal_embeddings[j][jj].view(1, size_representation, 1))

                        loss2_term = torch.bmm(
                            total_embeddings[i][ii].view(1, 1, size_representation),
                            total_embeddings[j][jj].view(1, size_representation, 1))

                        smi_value = similarity_trend * similarity_seasonal
                        smi.append(smi_value.item())
                        loss2_item.append(loss2_term)

        sorted_indices = sorted(range(len(smi)), key=lambda k: smi[k])
        half_index = len(sorted_indices) // 2
        indices_of_smallest_half = sorted_indices[:half_index]

        for idx in indices_of_smallest_half:
            loss2 += loss2_item[idx]

        loss1 = (loss1) / (M * N * (N - 1) / 2)
        loss2 = (loss2) / (totalnumber / 2)
        loss = loss1 + loss2
        return loss


class BasicEncoder():
    def encode(self, X):
        pass

    def save(self, X):
        pass

    def load(self, X):
        pass


class DDEM(BasicEncoder):
    def __init__(self, win_size, batch_size, nb_steps, lr,
                 channels, depth, reduced_size, out_channels, kernel_size,
                 in_channels, cuda, gpu, M, N, win_type):
        self.network = self.__create_network(in_channels, channels, depth, reduced_size,
                                             out_channels, kernel_size, cuda, gpu)

        self.win_type = win_type
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = fncc_loss(
            win_size, M, N, win_type
        )
        params_to_update = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params_to_update, lr=lr)

        self.loss_list = []

    def __create_network(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):

        network = NetworkDDEM(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        network.double()
        if cuda:
            network.cuda(gpu)
        return network

    def fit(self, X, save_memory=False, verbose=False):
        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)

        train_torch_dataset = Dataset(X)

        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )
        i = 0
        while i < self.nb_steps:

            for batch in train_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                loss = self.loss(batch, self.network, save_memory=False)
                loss.backward()
                self.optimizer.step()
                i += 1
                if i >= self.nb_steps:
                    break

        return self.network

    def encode(self, X, batch_size=500):
        print(X.shape)
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.network = self.network.eval()

        count = 0
        with torch.no_grad():
            for batch in test_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                features[
                count * batch_size: (count + 1) * batch_size
                ] = self.network(batch)[0].cpu()
                count += 1

        return features

    def encode_window(self, X, win_size=128, batch_size=500, window_batch_size=10000, step=10):
        num_batch, num_channel, length = numpy.shape(X)
        num_window = int((length - win_size) / step) + 1
        embeddings = numpy.empty((num_batch, self.out_channels, num_window))

        for b in range(num_batch):
            for i in range(math.ceil(num_window / window_batch_size)):
                masking = numpy.array([X[b, :, j:j + win_size] for j in range(step * i * window_batch_size,
                                                                              step * min((i + 1) * window_batch_size,
                                                                                         num_window), step)])
                embeddings[b, :, i * window_batch_size: (i + 1) * window_batch_size] = numpy.swapaxes(
                    self.encode(masking[:], batch_size=batch_size), 0, 1)
        return embeddings[0].T

    def set_params(self, compared_length, batch_size, nb_steps, lr,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, batch_size,
            nb_steps, lr, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self


class E2USD_Adaper(BasicEncoderClass):
    def _set_parmas(self, params):
        self.hyperparameters = params
        if 'compared_length' in self.hyperparameters:
            del self.hyperparameters['compared_length']
        self.encoder = DDEM(**self.hyperparameters)

    def fit(self, X):
        _, dim = X.shape
        X = np.transpose(np.array(X[:, :], dtype=float)).reshape(1, dim, -1)
        X = all_normalize(X)
        self.encoder.fit(X, save_memory=True, verbose=False)

    def encode(self, X, win_size, step):
        _, dim = X.shape
        X = np.transpose(np.array(X[:, :], dtype=float)).reshape(1, dim, -1)
        X = all_normalize(X)
        embeddings = self.encoder.encode_window(X, win_size=win_size, step=step)
        return embeddings

    def encode_one(self, X, win_size, step):
        _, dim = X.shape
        X = np.transpose(np.array(X[:, :], dtype=float)).reshape(1, dim, -1)
        X = all_normalize(X)
        embeddings = self.encoder.encode_window(X, win_size=win_size, step=step)
        return embeddings


class DPGMM(BasicClusteringClass):
    def __init__(self, n_states, alpha=1e3):
        self.alpha = alpha
        if n_states is not None:
            self.n_states = n_states
        else:
            self.n_states = 20

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, np.newaxis]

        n_samples = X.shape[0]
        if n_samples == 0:
            return np.empty(0, dtype=int)

        n_components = max(1, min(self.n_states, n_samples))

        dpgmm = mixture.BayesianGaussianMixture(
            init_params="kmeans",
            n_components=n_components,
            covariance_type="full",
            weight_concentration_prior=self.alpha,
            weight_concentration_prior_type="dirichlet_process",
            max_iter=1000,
        )
        return dpgmm.fit_predict(X)


class E2USD:
    def __init__(self, win_size, step, encoder, clustering_component, verbose=False):
        self.__win_size = win_size
        self.__step = step
        self.__offset = int(win_size / 2)
        self.__encoder = encoder
        self.__clustering_component = clustering_component

    def fit(self, X, win_size, step):

        self.__length = X.shape[0]
        self.fit_encoder(X)
        self.__encode(X, win_size, step)
        self.__cluster()
        self.__assign_label()
        return self

    def predict(self, X, win_size, step):
        self.__length = X.shape[0]
        self.__step = step
        self.__encode(X, win_size, step)
        self.__cluster()
        self.__assign_label()
        return self

    def set_step(self, step):
        self.__step = step

    def set_clustering_component(self, clustering_obj):
        self.__clustering_component = clustering_obj
        return self

    def fit_encoder(self, X):
        self.__encoder.fit(X)
        return self

    def predict_without_encode(self, X, win_size, step):
        self.__cluster()
        self.__assign_label()
        return self

    def __encode(self, X, win_size, step):
        self.__embeddings = self.__encoder.encode(X, win_size, step)

    def __cluster(self):
        self.__embedding_label = reorder_label(self.__clustering_component.fit(self.__embeddings))

    def __assign_label(self):
        hight = len(set(self.__embedding_label))
        weight_vector = np.ones(shape=(2 * self.__offset)).flatten()
        self.__state_seq = self.__embedding_label
        vote_matrix = np.zeros((self.__length, hight))
        i = 0
        for l in self.__embedding_label:
            vote_matrix[i:i + self.__win_size, l] += weight_vector
            i += self.__step
        self.__state_seq = np.array([np.argmax(row) for row in vote_matrix])

    def save_encoder(self):
        pass

    def online_threshold_cluster(self, X, win_size, step, tau, ratio):
        self.__length = X.shape[0]
        self.__step_threshold = step

        miner = 1 - self.delta
        maxer = 1 + self.delta * ratio
        label = []
        total_clusetring = 0
        for i in range(0, self.__length - win_size, step):
            now_x = X[i:i + win_size]
            now_win_embedding = self.__encode_one(now_x)
            if self.last_win_embedding is None:
                self.last_win_embedding = now_win_embedding
                self.last_win_state = self.__cluster_one(now_win_embedding)
                label.append(self.last_win_state)
                total_clusetring += 1
            else:
                similarity = np.dot(self.last_win_embedding, now_win_embedding.T)

                if similarity >= tau:
                    label.append(self.last_win_state)
                    tau = tau * maxer
                else:
                    new_win_state = self.__cluster_one(now_win_embedding)
                    if new_win_state != self.last_win_state:
                        self.last_win_embedding = now_win_embedding
                        self.last_win_state = new_win_state
                        tau = tau * maxer
                    else:
                        tau = tau * miner
                    label.append(self.last_win_state)
                    total_clusetring += 1
        label_np = np.array(label)
        self.threshold_label = label_np
        self.__assign_label_threshold()
        return label_np, total_clusetring

    def load_encoder(self):
        pass

    def save_result(self, path):
        pass

    def load_result(self, path):
        pass

    def plot(self, path):
        pass

    @property
    def embeddings(self):
        return self.__embeddings

    @property
    def state_seq(self):
        return self.__state_seq

    @property
    def embedding_label(self):
        return self.__embedding_label

    @property
    def velocity(self):
        return self.__velocity

    @property
    def change_points(self):
        return self.__change_points