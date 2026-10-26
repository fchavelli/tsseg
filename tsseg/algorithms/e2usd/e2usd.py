# Taken from: https://github.com/AI4CTS/E2Usd (no licence published).
# Dataset, moving_avg and series_decomp come from projects under the Apache
# License 2.0 and the MIT License: see LICENSE in this directory.

"""E2USD encoder, loss and clustering, from the reference implementation.

Taken from https://github.com/AI4CTS/E2Usd, which publishes no licence: the
modules of its ``E2USD`` package are merged here, unchanged except where a
docstring or a comment says otherwise. :class:`Dataset` comes from the code
of Franceschi et al. (2019),
https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
(Apache License 2.0), and :class:`moving_avg` and :class:`series_decomp` from
Autoformer, https://github.com/thuml/Autoformer (MIT licence); their licences,
the ``NOTICE`` of Franceschi et al. and the changes made to their code are in
``LICENSE``, in this directory. Public entry point:
:class:`tsseg.algorithms.E2USDDetector`, which uses :class:`E2USD_Adaper`,
:class:`DPGMM` and :class:`E2USD`.

Tensors and arrays follow the PyTorch layout ``(batch, channels, length)``,
except where a docstring says otherwise.
"""

import abc
import os
import sys

sys.path.append(os.path.dirname(__file__))
import math

import numpy
import numpy as np
import torch
import torch.nn as nn
from sklearn import mixture

#: Default hyperparameters of :class:`DDEM`, keyed by its argument names.
#: ``E2USDDetector`` overrides all of them but ``channels`` (30) and
#: ``win_type`` (``"rect"``), which have no effect, ``M`` (20) and ``N`` (4).
params = {
    "batch_size": 1,
    "channels": 30,
    "win_size": 256,
    "win_type": "rect",
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
    "N": 4,
}


# Dataset: taken, as in E2Usd, from utils.py of
# https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
# Copyright 2019 Jean-Yves Franceschi, Aymeric Dieuleveut, Martin Jaggi,
# licensed under the Apache License, Version 2.0. Modified: np.shape for
# numpy.shape, docstring removed. Licence and NOTICE: LICENSE in this directory.


class Dataset(torch.utils.data.Dataset):
    """PyTorch dataset whose samples are the entries of an array.

    Parameters
    ----------
    dataset : np.ndarray
        The samples, along the first axis: ``dataset[i]`` is sample ``i``.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return np.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index]


def compact(series):
    """Collapse each run of equal consecutive values into one value.

    Parameters
    ----------
    series : sequence
        A non-empty sequence of comparable values.

    Returns
    -------
    list
        The value of each run, in order: ``[3, 3, 1, 1, 3]`` gives
        ``[3, 1, 3]``.
    """
    compacted = []
    pre = series[0]
    compacted.append(pre)
    for e in series[1:]:
        if e != pre:
            pre = e
            compacted.append(e)
    return compacted


def remove_duplication(series):
    """Distinct values of a sequence, in order of first appearance.

    Parameters
    ----------
    series : sequence
        The values.

    Returns
    -------
    list
        Each value once: ``[3, 3, 1, 1, 3, 2]`` gives ``[3, 1, 2]``.
    """
    result = []
    for e in series:
        if e not in result:
            result.append(e)
    return result


def reorder_label(label):
    """Renumber labels from 0 in order of first appearance.

    Parameters
    ----------
    label : array-like of int, shape (n,)
        The labels, non-empty.

    Returns
    -------
    np.ndarray of shape (n,)
        A copy of ``label`` where the first label is 0, the next new one 1,
        and so on: ``[3, 3, 1, 1, 3, 2]`` gives ``[0, 0, 1, 1, 0, 2]``.
    """
    label = np.array(label)
    ordered_label_set = remove_duplication(compact(label))
    idx_list = [np.argwhere(label == e) for e in ordered_label_set]
    for i, idx in enumerate(idx_list):
        label[idx] = i
    return label


def all_normalize(data_tensor):
    """Z-normalise a series with one mean and one variance for all channels.

    Parameters
    ----------
    data_tensor : np.ndarray of shape (1, n_channels, n_timepoints)
        The series, normalised in place. The mean and the variance are those
        of all its values; a constant series has zero variance and becomes
        NaN.

    Returns
    -------
    np.ndarray
        ``data_tensor`` itself.
    """
    mean = np.mean(data_tensor)
    var = np.var(data_tensor)
    i = 0
    for channel in data_tensor[0]:
        data_tensor[0][i] = (channel - mean) / math.sqrt(var)
        i += 1
    return data_tensor


class BasicClusteringClass:
    """Interface of the clustering step of :class:`E2USD`.

    ``abc.abstractmethod`` is not enforced: the class has no ``ABCMeta``
    metaclass.

    Parameters
    ----------
    params : object
        Ignored.
    """

    def __init__(self, params):
        pass

    @abc.abstractmethod
    def fit(self, X):
        """Cluster the rows of ``X``; does nothing here.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The samples to cluster.

        Returns
        -------
        None
            Subclasses return the label of each row, an ``np.ndarray`` of
            shape ``(n_samples,)``.
        """
        pass


class BasicEncoderClass:
    """Interface of the encoders used by :class:`E2USD`.

    The constructor hands ``params`` to ``_set_parmas`` (spelling of the
    reference code), which subclasses implement to build the encoder.
    ``abc.abstractmethod`` is not enforced: the class has no ``ABCMeta``
    metaclass.

    Parameters
    ----------
    params : dict
        Hyperparameters of the encoder.
    """

    def __init__(self, params):
        self._set_parmas(params)

    @abc.abstractmethod
    def _set_parmas(self, params):
        """Build the encoder from ``params``; does nothing here."""
        pass

    @abc.abstractmethod
    def fit(self, X):
        """Train the encoder on the series ``X``; does nothing here.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels)
            The training series.
        """
        pass

    @abc.abstractmethod
    def encode(self, X, win_size, step):
        """Embed the sliding windows of ``X``; does nothing here.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels)
            The series.
        win_size : int
            Length of the windows.
        step : int
            Distance between the starts of consecutive windows.

        Returns
        -------
        None
            Subclasses return one embedding per window, an ``np.ndarray`` of
            shape ``(n_windows, n_features)``.
        """
        pass


def add_noise(data, mean=0, std=0.05):
    """Add Gaussian noise: data augmentation of the reference code, unused.

    Parameters
    ----------
    data : torch.Tensor
        The values.
    mean : float, default=0
        Mean of the noise.
    std : float, default=0.05
        Standard deviation of the noise, drawn from PyTorch's global
        generator.

    Returns
    -------
    torch.Tensor
        ``data`` plus the noise, as a new tensor.
    """
    noise = torch.randn(data.size()) * std + mean
    return data + noise


def apply_scaling(data, mean=1.0, std=0.1):
    """Multiply by Gaussian factors: data augmentation of the reference code, unused.

    Parameters
    ----------
    data : torch.Tensor
        The values.
    mean : float, default=1.0
        Mean of the factors.
    std : float, default=0.1
        Standard deviation of the factors, drawn from PyTorch's global
        generator.

    Returns
    -------
    torch.Tensor
        ``data`` times the factors, as a new tensor.
    """
    scaling_factor = torch.normal(mean=mean, std=std, size=data.size())
    return data * scaling_factor


# Reference code: "randomly choose a data augmentation method"; neither
# function above is called.


# moving_avg and series_decomp: taken from layers/Autoformer_EncDec.py of
# https://github.com/thuml/Autoformer, Copyright (c) 2021 THUML @ Tsinghua
# University, MIT License. Modified: moving_avg works on (batch, channels,
# length) tensors along the last axis, without permuting; docstrings removed.
# Licence: LICENSE in this directory.


class moving_avg(nn.Module):
    """Moving average along the last axis, with the edge values repeated.

    Parameters
    ----------
    kernel_size : int
        Number of points averaged.
    stride : int
        Distance between the starts of consecutive averages.
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        """Pad each end of ``x`` with its edge value, then average.

        Each end gets ``(kernel_size - 1) // 2`` copies of its edge value.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, channels, length)
            The series.

        Returns
        -------
        torch.Tensor of shape (batch, channels, n_averages)
            The averages. With ``stride=1`` and an odd ``kernel_size``,
            ``n_averages`` is ``length``.
        """
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x


class series_decomp(nn.Module):
    """Split a series into a trend, its moving average, and the residual.

    Parameters
    ----------
    kernel_size : int
        Number of points of the moving average; odd, for the trend to have
        the length of the series.
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        """Decompose ``x``.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, channels, length)
            The series.

        Returns
        -------
        res : torch.Tensor of shape (batch, channels, length)
            The residual ``x - moving_mean`` (the seasonal part).
        moving_mean : torch.Tensor of shape (batch, channels, length)
            The moving average of ``x`` (the trend).
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class NetworkDDEM(torch.nn.Module):
    """Decomposed dual-view embedding network (DDEM) of E2USD.

    Named ``DDEM`` in the reference code (``networks/network.py``); renamed
    here, where :class:`DDEM` is the encoder that trains it.

    Parameters
    ----------
    in_channels : int
        Number of channels of the series.
    channels : int
        Unused.
    depth : int
        Unused.
    reduced_size : int
        Number of output channels of each of the two convolutions.
    out_channels : int
        Dimension of the embeddings.
    kernel_size : int
        Kernel size of the two convolutions, at most the length of a window
        after the low-pass filter (see Notes).

    Notes
    -----
    The forward pass keeps the ``trade_off_freq = 33`` lowest frequencies of
    the real Fourier transform of each window and transforms back without a
    target length, scaling by ``33 / length``: a window of ``length >= 64``
    points becomes 64 points; a shorter one keeps its length if even, and
    loses one point if odd. :class:`series_decomp` with a moving average over
    5 points splits it into a trend and a seasonal part (the residual). Each
    part goes through its own ``Conv1d`` (``in_channels`` to
    ``reduced_size`` channels, no padding), a max pooling over time and a
    linear layer to ``out_channels``; a last linear layer maps the two
    concatenated embeddings to ``out_channels``. The two convolutions are
    frozen (``requires_grad_(False)``) at their random initial weights: only
    the linear layers are trained.
    """

    def __init__(
        self, in_channels, channels, depth, reduced_size, out_channels, kernel_size
    ):
        super(NetworkDDEM, self).__init__()
        moving_ks = 5
        self.decompsition = series_decomp(moving_ks)

        self.trend_cnn = nn.Conv1d(in_channels, reduced_size, kernel_size=kernel_size)
        self.seasonal_cnn = nn.Conv1d(
            in_channels, reduced_size, kernel_size=kernel_size
        )

        self.trend_cnn.requires_grad_(False)
        self.seasonal_cnn.requires_grad_(False)
        self.trend_pooling = torch.nn.AdaptiveMaxPool1d(1)
        self.seasonal_pooling = torch.nn.AdaptiveMaxPool1d(1)

        self.linear_trend = torch.nn.Linear(reduced_size, out_channels)
        self.linear_seasonal = torch.nn.Linear(reduced_size, out_channels)

        self.linear = torch.nn.Linear(out_channels * 2, out_channels)

        self.trade_off_freq = 33

    def forward(self, x):
        """Embed a batch of windows.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, in_channels, length)
            The windows.

        Returns
        -------
        embedding : torch.Tensor of shape (batch, out_channels)
            The embeddings of the windows.
        trend_x_embedding : torch.Tensor of shape (batch, out_channels)
            The embeddings of their trends.
        seasonal_x_embedding : torch.Tensor of shape (batch, out_channels)
            The embeddings of their seasonal parts.
        """
        low_specx = torch.fft.rfft(x, dim=-1)
        low_specx = low_specx[:, :, : self.trade_off_freq]
        x = torch.fft.irfft(low_specx, dim=-1) * self.trade_off_freq / x.size(-1)
        seasonal_init, trend_init = self.decompsition(x)

        trend_x, seasonal_x = (
            (self.trend_cnn(trend_init)),
            (self.seasonal_cnn(seasonal_init)),
        )
        trend_x_reduced, seasonal_x_reduced = (
            self.trend_pooling(trend_x),
            self.seasonal_pooling(seasonal_x),
        )

        trend_x_reduced, seasonal_x_reduced = (
            trend_x_reduced.squeeze(2),
            seasonal_x_reduced.squeeze(2),
        )

        trend_x_embedding, seasonal_x_embedding = (
            (self.linear_trend(trend_x_reduced)),
            (self.linear_seasonal(seasonal_x_reduced)),
        )

        embedding = torch.concat([trend_x_embedding, seasonal_x_embedding], dim=-1)
        embedding = self.linear(embedding)

        return embedding, trend_x_embedding, seasonal_x_embedding


class fncc_loss(torch.nn.modules.loss._Loss):
    """False negative cancellation contrastive loss (FNCC) of E2USD.

    Parameters
    ----------
    win_size : int
        Length of the windows.
    M : int
        Number of random positions drawn at each call.
    N : int
        Number of windows at each position, shifted by one point.
    win_type : str
        Stored, unused.

    Notes
    -----
    Each call draws ``M`` positions ``p`` uniformly in
    ``[0, length - 2 * win_size]`` from NumPy's global generator and embeds,
    at each, the ``N`` windows starting at ``p, ..., p + N - 1``. The first
    term is the mean, over the pairs of windows of the same position, of
    ``-log(sigmoid(<e_i, e_j>))``, which pulls their embeddings together. For
    each pair of windows from two different positions, the product of the dot
    products of their trend and of their seasonal embeddings measures their
    similarity; the second term is the sum of the dot products ``<e_i, e_j>``
    over the half of these pairs with the lowest similarity, divided by half
    the number of pairs (their mean when that number is even), which pushes
    them apart. The loss is the sum of the two terms. The other half, the pairs
    most likely to come from the same state (false negatives), is left out.
    The attributes ``total`` and ``total_condition`` are unused.
    """

    def __init__(self, win_size, M, N, win_type):
        super(fncc_loss, self).__init__()
        self.win_size = win_size
        self.win_type = win_type
        self.M = M
        self.N = N
        self.total = 0
        self.total_condition = 0

    def forward(self, batch, encoder, save_memory=False):
        """Compute the loss on random windows of a series.

        Parameters
        ----------
        batch : torch.Tensor of shape (batch, n_channels, length)
            The series; only ``batch[0]`` is used. ``length`` must be at
            least ``2 * win_size``.
        encoder : torch.nn.Module
            The network that embeds the windows, such as
            :class:`NetworkDDEM`: it returns the embeddings and the trend and
            seasonal embeddings.
        save_memory : bool, default=False
            Unused.

        Returns
        -------
        torch.Tensor
            The loss, a scalar.
        """
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
            random_pos = np.random.randint(
                0, high=total_length - length_pos_neg * 2 + 1, size=1
            )
            rand_samples = [
                batch[0, :, i : i + length_pos_neg]
                for i in range(random_pos[0], random_pos[0] + N)
            ]

            intra_sample = torch.stack(rand_samples)

            embeddings, trend_x_embedding, seasonal_x_embedding = encoder(
                intra_sample
            )  # ([4, 4]) N / embedding_channel
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
                            embeddings[j].view(1, size_representation, 1),
                        )
                        loss1_term = -torch.mean(
                            torch.nn.functional.logsigmoid(similarity_embedding)
                        )
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
                            total_trend_embeddings[i][ii].view(
                                1, 1, size_representation
                            ),
                            total_trend_embeddings[j][jj].view(
                                1, size_representation, 1
                            ),
                        )
                        similarity_seasonal = torch.bmm(
                            total_seasonal_embeddings[i][ii].view(
                                1, 1, size_representation
                            ),
                            total_seasonal_embeddings[j][jj].view(
                                1, size_representation, 1
                            ),
                        )

                        loss2_term = torch.bmm(
                            total_embeddings[i][ii].view(1, 1, size_representation),
                            total_embeddings[j][jj].view(1, size_representation, 1),
                        )

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


class BasicEncoder:
    """Base class of :class:`DDEM`, whose methods do nothing."""

    def encode(self, X):
        """Do nothing (overridden by :meth:`DDEM.encode`)."""
        pass

    def save(self, X):
        """Do nothing."""
        pass

    def load(self, X):
        """Do nothing."""
        pass


class DDEM(BasicEncoder):
    """Encoder of E2USD: trains a :class:`NetworkDDEM` and embeds windows.

    Parameters
    ----------
    win_size : int
        Length of the training windows of the loss.
    batch_size : int
        Batch size of the training data loader. Training uses a single
        sample, the whole series, so this value has no effect.
    nb_steps : int
        Number of optimisation steps.
    lr : float
        Learning rate of the Adam optimiser.
    channels : int
        Unused (passed to :class:`NetworkDDEM`, which ignores it).
    depth : int
        Unused (passed to :class:`NetworkDDEM`, which ignores it).
    reduced_size : int
        Number of output channels of the two convolutions of the network.
    out_channels : int
        Dimension of the embeddings.
    kernel_size : int
        Kernel size of the two convolutions of the network.
    in_channels : int
        Number of channels of the series.
    cuda : bool
        Whether to run on a CUDA device.
    gpu : int
        Index of that device.
    M : int
        Number of random positions of each step of the loss.
    N : int
        Number of windows at each position.
    win_type : str
        Stored, unused.

    Attributes
    ----------
    network : NetworkDDEM
        The network, in double precision.
    loss : fncc_loss
        The training loss.
    optimizer : torch.optim.Adam
        Optimiser of the trainable parameters of the network, those of its
        linear layers.
    loss_list : list
        Empty, never filled.
    """

    def __init__(
        self,
        win_size,
        batch_size,
        nb_steps,
        lr,
        channels,
        depth,
        reduced_size,
        out_channels,
        kernel_size,
        in_channels,
        cuda,
        gpu,
        M,
        N,
        win_type,
    ):
        self.network = self.__create_network(
            in_channels,
            channels,
            depth,
            reduced_size,
            out_channels,
            kernel_size,
            cuda,
            gpu,
        )

        self.win_type = win_type
        self.architecture = ""
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = fncc_loss(win_size, M, N, win_type)
        params_to_update = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params_to_update, lr=lr)

        self.loss_list = []

    def __create_network(
        self,
        in_channels,
        channels,
        depth,
        reduced_size,
        out_channels,
        kernel_size,
        cuda,
        gpu,
    ):
        network = NetworkDDEM(
            in_channels, channels, depth, reduced_size, out_channels, kernel_size
        )
        network.double()
        if cuda:
            network.cuda(gpu)
        return network

    def fit(self, X, save_memory=False, verbose=False):
        """Train the network for ``nb_steps`` steps.

        Each step evaluates the loss on random windows of the series (see
        :class:`fncc_loss`) and updates the linear layers.

        Parameters
        ----------
        X : np.ndarray of shape (1, in_channels, n_timepoints)
            The training series, in float64, with ``n_timepoints`` at least
            ``2 * win_size``.
        save_memory : bool, default=False
            Unused.
        verbose : bool, default=False
            Unused.

        Returns
        -------
        NetworkDDEM
            The trained network.
        """
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
        """Embed a set of windows, without gradients.

        The network is put in evaluation mode. The reference code prints the
        shape of ``X`` here; this copy does not.

        Parameters
        ----------
        X : np.ndarray of shape (n_windows, in_channels, length)
            The windows, in float64, without NaN: with NaN the data loader
            uses batches of one window, but the rows of the result are still
            indexed by batches of ``batch_size``.
        batch_size : int, default=500
            Number of windows embedded at once.

        Returns
        -------
        np.ndarray of shape (n_windows, out_channels)
            The embeddings.
        """
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
                features[count * batch_size : (count + 1) * batch_size] = self.network(
                    batch
                )[0].cpu()
                count += 1

        return features

    def encode_window(
        self, X, win_size=128, batch_size=500, window_batch_size=10000, step=10
    ):
        """Embed the sliding windows of a series.

        Parameters
        ----------
        X : np.ndarray of shape (n_series, in_channels, n_timepoints)
            The series, in float64.
        win_size : int, default=128
            Length of the windows.
        batch_size : int, default=500
            Passed to :meth:`encode`.
        window_batch_size : int, default=10000
            Number of windows cut out of the series at once.
        step : int, default=10
            Distance between the starts of consecutive windows.

        Returns
        -------
        np.ndarray of shape (n_windows, out_channels)
            The embeddings of the ``(n_timepoints - win_size) // step + 1``
            windows of the first series, starting at ``0, step, 2 * step``,
            and so on. The other series are embedded, then dropped.
        """
        num_batch, num_channel, length = numpy.shape(X)
        num_window = int((length - win_size) / step) + 1
        embeddings = numpy.empty((num_batch, self.out_channels, num_window))

        for b in range(num_batch):
            for i in range(math.ceil(num_window / window_batch_size)):
                masking = numpy.array(
                    [
                        X[b, :, j : j + win_size]
                        for j in range(
                            step * i * window_batch_size,
                            step * min((i + 1) * window_batch_size, num_window),
                            step,
                        )
                    ]
                )
                embeddings[
                    b, :, i * window_batch_size : (i + 1) * window_batch_size
                ] = numpy.swapaxes(self.encode(masking[:], batch_size=batch_size), 0, 1)
        return embeddings[0].T

    def set_params(
        self,
        compared_length,
        batch_size,
        nb_steps,
        lr,
        channels,
        depth,
        reduced_size,
        out_channels,
        kernel_size,
        in_channels,
        cuda,
        gpu,
    ):
        """Rebuild the encoder with new hyperparameters (broken).

        Calls ``__init__`` with 12 positional arguments, ``compared_length``
        in the place of ``win_size``, whereas it requires 15: any call raises
        ``TypeError`` (``M``, ``N`` and ``win_type`` are missing), as in the
        reference code.

        Parameters
        ----------
        compared_length : int
            The ``win_size`` of the class.
        batch_size, nb_steps, lr, channels, depth, reduced_size
            As the parameters of the class.
        out_channels, kernel_size, in_channels, cuda, gpu
            As the parameters of the class.

        Returns
        -------
        DDEM
            ``self``, if the call succeeded.
        """
        self.__init__(
            compared_length,
            batch_size,
            nb_steps,
            lr,
            channels,
            depth,
            reduced_size,
            out_channels,
            kernel_size,
            in_channels,
            cuda,
            gpu,
        )
        return self


class E2USD_Adaper(BasicEncoderClass):
    """Adapter that gives a :class:`DDEM` the interface used by :class:`E2USD`.

    The class name keeps the spelling of the reference code.

    Parameters
    ----------
    params : dict
        Keyword arguments of :class:`DDEM`, such as :data:`params`. A
        ``"compared_length"`` key is removed from the dict itself.

    Attributes
    ----------
    hyperparameters : dict
        ``params``.
    encoder : DDEM
        The encoder.
    """

    def _set_parmas(self, params):
        """Build the :class:`DDEM` encoder from ``params``."""
        self.hyperparameters = params
        if "compared_length" in self.hyperparameters:
            del self.hyperparameters["compared_length"]
        self.encoder = DDEM(**self.hyperparameters)

    def fit(self, X):
        """Normalise a series with :func:`all_normalize` and train the encoder.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels)
            The training series, with ``n_timepoints`` at least twice the
            ``win_size`` of the encoder.
        """
        _, dim = X.shape
        X = np.transpose(np.array(X[:, :], dtype=float)).reshape(1, dim, -1)
        X = all_normalize(X)
        self.encoder.fit(X, save_memory=True, verbose=False)

    def encode(self, X, win_size, step):
        """Normalise a series with :func:`all_normalize` and embed its windows.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels)
            The series.
        win_size : int
            Length of the windows.
        step : int
            Distance between the starts of consecutive windows.

        Returns
        -------
        np.ndarray of shape (n_windows, out_channels)
            The embeddings of the ``(n_timepoints - win_size) // step + 1``
            windows (see :meth:`DDEM.encode_window`).
        """
        _, dim = X.shape
        X = np.transpose(np.array(X[:, :], dtype=float)).reshape(1, dim, -1)
        X = all_normalize(X)
        embeddings = self.encoder.encode_window(X, win_size=win_size, step=step)
        return embeddings

    def encode_one(self, X, win_size, step):
        """Same as :meth:`encode`.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels)
            The series.
        win_size : int
            Length of the windows.
        step : int
            Distance between the starts of consecutive windows.

        Returns
        -------
        np.ndarray of shape (n_windows, out_channels)
            The embeddings of the windows.
        """
        _, dim = X.shape
        X = np.transpose(np.array(X[:, :], dtype=float)).reshape(1, dim, -1)
        X = all_normalize(X)
        embeddings = self.encoder.encode_window(X, win_size=win_size, step=step)
        return embeddings


class DPGMM(BasicClusteringClass):
    """Dirichlet process Gaussian mixture on the window embeddings.

    Parameters
    ----------
    n_states : int or None
        Number of components of the mixture, an upper bound on the number of
        clusters; ``None`` means 20.
    alpha : float, default=1e3
        Weight concentration prior of the Dirichlet process
        (``weight_concentration_prior`` of
        ``sklearn.mixture.BayesianGaussianMixture``).
    """

    def __init__(self, n_states, alpha=1e3):
        self.alpha = alpha
        if n_states is not None:
            self.n_states = n_states
        else:
            self.n_states = 20

    def fit(self, X):
        """Fit the mixture on ``X`` and return the component of each row.

        ``sklearn.mixture.BayesianGaussianMixture`` with full covariances,
        k-means initialisation, a Dirichlet process prior and at most 1000
        iterations; it has no ``random_state``, so it draws from NumPy's
        global generator. Changes from the reference code: the number of
        components is capped at the number of rows, and an empty ``X`` gives
        an empty result.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            The samples, without NaN (``ValueError`` otherwise).

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The component of each sample, not renumbered: the values can
            skip integers.
        """
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
    """E2USD pipeline: embed sliding windows, cluster them, vote per point.

    Parameters
    ----------
    win_size : int
        Length of the windows, as voted on. Must be even: each window votes
        with a vector of ``2 * (win_size // 2)`` ones, so an odd value raises
        ``ValueError`` when the labels are assigned.
    step : int
        Distance between the starts of consecutive windows, as voted on.
    encoder : BasicEncoderClass
        Object with ``fit(X)`` and ``encode(X, win_size, step)``, such as
        :class:`E2USD_Adaper`.
    clustering_component : BasicClusteringClass
        Object with ``fit(embeddings)`` that returns one label per window,
        such as :class:`DPGMM`.
    verbose : bool, default=False
        Unused.

    Notes
    -----
    The window labels are renumbered from 0 in order of first appearance.
    Each window adds one vote for its label to each point it covers, and
    each point takes the label with the most votes (the smallest label on a
    tie). Change from the reference code: the points after the last window,
    fewer than ``step``, take the label of the last window instead of 0.
    """

    def __init__(self, win_size, step, encoder, clustering_component, verbose=False):
        self.__win_size = win_size
        self.__step = step
        self.__offset = int(win_size / 2)
        self.__encoder = encoder
        self.__clustering_component = clustering_component

    def fit(self, X, win_size, step):
        """Train the encoder on ``X``, then label its points.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels)
            The series.
        win_size : int
            Length of the windows embedded. The votes use the ``win_size``
            of the constructor: the two should be equal.
        step : int
            Distance between the windows embedded. The votes use the
            ``step`` of the constructor (or of :meth:`set_step`): the two
            should be equal.

        Returns
        -------
        E2USD
            ``self``; the labels are in :attr:`state_seq`.
        """
        self.__length = X.shape[0]
        self.fit_encoder(X)
        self.__encode(X, win_size, step)
        self.__cluster()
        self.__assign_label()
        return self

    def predict(self, X, win_size, step):
        """Label the points of ``X`` with the encoder as it is.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels)
            The series.
        win_size : int
            Length of the windows embedded. The votes use the ``win_size``
            of the constructor: the two should be equal.
        step : int
            Distance between the windows, for the embedding and the votes.

        Returns
        -------
        E2USD
            ``self``; the labels are in :attr:`state_seq`.
        """
        self.__length = X.shape[0]
        self.__step = step
        self.__encode(X, win_size, step)
        self.__cluster()
        self.__assign_label()
        return self

    def set_step(self, step):
        """Set the ``step`` used by the votes.

        Parameters
        ----------
        step : int
            Distance between the starts of consecutive windows.
        """
        self.__step = step

    def set_clustering_component(self, clustering_obj):
        """Replace the clustering component.

        Parameters
        ----------
        clustering_obj : BasicClusteringClass
            The new clustering component.

        Returns
        -------
        E2USD
            ``self``.
        """
        self.__clustering_component = clustering_obj
        return self

    def fit_encoder(self, X):
        """Train the encoder on ``X`` only.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels)
            The series.

        Returns
        -------
        E2USD
            ``self``.
        """
        self.__encoder.fit(X)
        return self

    def predict_without_encode(self, X, win_size, step):
        """Cluster the embeddings of the last series encoded and vote again.

        Raises ``AttributeError`` if no series was encoded.

        Parameters
        ----------
        X, win_size, step
            Unused.

        Returns
        -------
        E2USD
            ``self``.
        """
        self.__cluster()
        self.__assign_label()
        return self

    def __encode(self, X, win_size, step):
        self.__embeddings = self.__encoder.encode(X, win_size, step)

    def __cluster(self):
        self.__embedding_label = reorder_label(
            self.__clustering_component.fit(self.__embeddings)
        )

    def __assign_label(self):
        hight = len(set(self.__embedding_label))
        weight_vector = np.ones(shape=(2 * self.__offset)).flatten()
        self.__state_seq = self.__embedding_label
        vote_matrix = np.zeros((self.__length, hight))
        i = 0
        for l in self.__embedding_label:
            vote_matrix[i : i + self.__win_size, l] += weight_vector
            i += self.__step
        self.__state_seq = np.array([np.argmax(row) for row in vote_matrix])
        # Change from the reference code, where the points after the last
        # window (fewer than ``step``) get no vote, hence label 0: they take
        # the label of the last window, which alone covers the point before.
        covered = (len(self.__embedding_label) - 1) * self.__step + self.__win_size
        self.__state_seq[covered:] = self.__embedding_label[-1]

    def save_encoder(self):
        """Do nothing."""
        pass

    def online_threshold_cluster(self, X, win_size, step, tau, ratio):
        """Online labelling with an adaptive similarity threshold (broken).

        Kept from the reference code. It reads attributes and calls methods
        that do not exist (``delta``, ``last_win_embedding``,
        ``__encode_one``, ``__cluster_one``, ``__assign_label_threshold``):
        any call raises ``AttributeError``. As written, the first window is
        clustered and becomes the reference. A later window whose embedding
        has a dot product of at least ``tau`` with that of the reference
        takes its state, and ``tau`` grows; otherwise the window is
        clustered, and becomes the reference if its state differs (``tau``
        grows) or ``tau`` shrinks. ``tau`` grows by the factor
        ``1 + delta * ratio`` and shrinks by ``1 - delta``.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels)
            The series.
        win_size : int
            Length of the windows.
        step : int
            Distance between the starts of consecutive windows.
        tau : float
            Initial similarity threshold.
        ratio : float
            Factor of ``delta`` in the growth of ``tau``.

        Returns
        -------
        label : np.ndarray
            The label of each window, if the call succeeded.
        total_clusetring : int
            The number of windows clustered.
        """
        self.__length = X.shape[0]
        self.__step_threshold = step

        miner = 1 - self.delta
        maxer = 1 + self.delta * ratio
        label = []
        total_clusetring = 0
        for i in range(0, self.__length - win_size, step):
            now_x = X[i : i + win_size]
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
        """Do nothing."""
        pass

    def save_result(self, path):
        """Do nothing; ``path`` is unused."""
        pass

    def load_result(self, path):
        """Do nothing; ``path`` is unused."""
        pass

    def plot(self, path):
        """Do nothing; ``path`` is unused."""
        pass

    @property
    def embeddings(self):
        """Embeddings of the windows of the last series encoded.

        ``np.ndarray`` of shape ``(n_windows, out_channels)``.
        """
        return self.__embeddings

    @property
    def state_seq(self):
        """Label of each point of the last series labelled, after the votes.

        ``np.ndarray`` of shape ``(n_timepoints,)``.
        """
        return self.__state_seq

    @property
    def embedding_label(self):
        """Label of each window, numbered from 0 in order of first appearance.

        ``np.ndarray`` of shape ``(n_windows,)``.
        """
        return self.__embedding_label

    @property
    def velocity(self):
        """Never set: reading it raises ``AttributeError``."""
        return self.__velocity

    @property
    def change_points(self):
        """Never set: reading it raises ``AttributeError``."""
        return self.__change_points
