# Taken from: https://github.com/Lab-ANT/Time2State/

import abc
import math

import numpy as np
import torch
from sklearn import mixture

params_LSE = {
    "batch_size": 1,
    "channels": 30,
    "win_size": 256,
    "win_type": 'rect',  # {rect, hanning}
    "depth": 10,
    "nb_steps": 20,
    "in_channels": 1,
    "kernel_size": 3,
    "lr": 0.003,
    "out_channels": 4,
    "reduced_size": 80,
    "cuda": torch.cuda.is_available(),
    "gpu": 0,
    "M": 10,
    "N": 4
}


def _normalize(X, mode='channel'):
    if mode == 'channel':
        for i in range(X.shape[1]):
            max = np.max(X[:, i])
            min = np.min(X[:, i])
            try:
                X[:, i] = (X[:, i] - min) / (max - min)
            except ZeroDivisionError:
                pass
    elif mode == 'all':
        max = np.max(X)
        min = np.min(X)
        X = np.true_divide(X - min, max - min)

    return X


def _all_normalize(data_tensor):
    mean = np.mean(data_tensor)
    var = np.var(data_tensor)
    i = 0
    for channel in data_tensor[0]:
        data_tensor[0][i] = (channel - mean) / math.sqrt(var)
        i += 1
    return data_tensor


def _compact(series):
    '''
    Compact Time Series.
    '''
    compacted = []
    pre = series[0]
    compacted.append(pre)
    for e in series[1:]:
        if e != pre:
            pre = e
            compacted.append(e)
    return compacted


def _remove_duplication(series):
    '''
    Remove duplication.
    '''
    result = []
    for e in series:
        if e not in result:
            result.append(e)
    return result


def _reorder_label(label):
    # Start from 0.
    label = np.array(label)
    ordered_label_set = _remove_duplication(_compact(label))
    idx_list = [np.argwhere(label == e) for e in ordered_label_set]
    for i, idx in enumerate(idx_list):
        label[idx] = i
    return label


class BasicClusteringClass:

    def __init__(self, params):
        pass

    @abc.abstractmethod
    def fit(self, X):
        pass


class DPGMM(BasicClusteringClass):

    def __init__(self, n_states, alpha=1e3):
        self.alpha = alpha
        if n_states is not None:
            self.n_states = n_states
        else:
            self.n_states = 20

    def fit(self, X):
        dpgmm = mixture.BayesianGaussianMixture(init_params='kmeans',
                                                n_components=min(self.n_states, X.shape[0]),
                                                covariance_type="full",
                                                weight_concentration_prior=self.alpha,  # alpha
                                                weight_concentration_prior_type='dirichlet_process',
                                                max_iter=1000).fit(X)
        return dpgmm.predict(X)


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


def hanning_tensor(X):
    length = X.size(2)
    weight = (1 - np.cos(2 * math.pi * np.arange(length) / length)) / 2
    weight = torch.tensor(weight)
    if X.is_cuda:
        return weight.cuda(X.get_device()) * X
    return weight * X


class LSELoss(torch.nn.modules.loss._Loss):
    """
    LSE loss for representations of time series.

    Parameters
    ----------
    win_size : even integer.
        Size of the sliding window.

    M : integer.
        Number of inter-state samples.

    N : integer.
        Number of intra-state samples.

    win_type : {'rect', 'hanning'}.
        window function.
    """

    def __init__(self, win_size, M, N, win_type):
        super(LSELoss, self).__init__()
        self.win_size = win_size
        self.win_type = win_type
        self.M = M
        self.N = N
        # temperature parameter
        # self.tau = 1
        # self.lambda1 = 1

    def forward(self, batch, encoder, save_memory=False):
        M = self.M
        N = self.N
        length_pos_neg = self.win_size

        total_length = batch.size(2)
        center_list = []
        loss1 = 0
        for i in range(M):
            random_pos = np.random.randint(0, high=total_length - length_pos_neg * 2 + 1, size=1)
            rand_samples = [batch[0, :, i: i + length_pos_neg] for i in range(random_pos[0], random_pos[0] + N)]
            # print(random_pos)
            if self.win_type == 'hanning':
                embeddings = encoder(hanning_tensor(torch.stack(rand_samples)))
            else:
                embeddings = encoder(torch.stack(rand_samples))
            # print(embeddings.shape)
            size_representation = embeddings.size(1)

            for i in range(N):
                for j in range(N):
                    if j <= i:
                        continue
                    else:
                        # loss1 += -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                        #     embeddings[i].view(1, 1, size_representation),
                        #     embeddings[j].view(1, size_representation, 1))/self.tau))
                        loss1 += -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                            embeddings[i].view(1, 1, size_representation),
                            embeddings[j].view(1, size_representation, 1))))
            center = torch.mean(embeddings, dim=0)
            center_list.append(center)

        loss2 = 0
        for i in range(M):
            for j in range(M):
                if j <= i:
                    continue
                # loss2 += -torch.mean(torch.nn.functional.logsigmoid(-torch.bmm(
                #     center_list[i].view(1, 1, size_representation),
                #     center_list[j].view(1, size_representation, 1))/self.tau))
                loss2 += -torch.mean(torch.nn.functional.logsigmoid(-torch.bmm(
                    center_list[i].view(1, 1, size_representation),
                    center_list[j].view(1, size_representation, 1))))

        loss = loss1 / (M * N * (N - 1) / 2) + loss2 / (M * (M - 1) / 2)
        # loss = loss2/(M*(M-1)/2)
        return loss


def hanning_numpy(X):
    length = X.shape[2]
    weight = (1 - np.cos(2 * math.pi * np.arange(length) / length)) / 2
    # weight = np.cos(2*math.pi*np.arange(length)/length)+0.5
    return weight * X


class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.

    Parameters
    ----------
    chomp_size : int
        Number of elements to remove.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """

    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Kernel size of the applied non-residual convolutions.
    dilation : int
        Dilation parameter of non-residual convolutions.
    final : bool, default=False
        If ``True``, disables the last activation function.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalCNN(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of channels processed in the network and of output channels.
    depth : int
        Depth of the network.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Kernel size of the applied non-residual convolutions.
    """

    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of channels manipulated in the causal CNN.
    depth : int
        Depth of the causal CNN.
    reduced_size : int
        Fixed length to which the output time series is reduced by the
        adaptive pooling layer.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Kernel size of the applied non-residual convolutions.
    """

    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size):
        super(CausalCNNEncoder, self).__init__()
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear = torch.nn.Linear(reduced_size, out_channels)
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze, linear
        )

    def forward(self, x):
        return self.network(x)


class Dataset(torch.utils.data.Dataset):
    """
    PyTorch wrapper for a numpy dataset.

    Parameters
    ----------
    dataset : numpy.ndarray
        Array representing the dataset.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return np.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index]


class BasicEncoder():
    def encode(self, X):
        pass

    def save(self, X):
        pass

    def load(self, X):
        pass


class CausalConv_LSE(BasicEncoder):
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
        self.loss = LSELoss(win_size, M, N, win_type)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.98, -1)
        self.loss_list = []

    def __create_network(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        network = CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        network.double()
        if cuda:
            network.cuda(gpu)
        return network

    def fit(self, X, y=None, save_memory=False, verbose=False):
        # _, dim = X.shape
        # X = np.transpose(np.array(X, dtype=float)).reshape(1, dim, -1)

        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)

        train_torch_dataset = Dataset(X)
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )

        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs

        # Encoder training
        while i < self.nb_steps:
            if verbose:
                print('Epoch: ', epochs + 1)
            for batch in train_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                loss = self.loss(batch, self.network, save_memory=save_memory)
                loss.backward()
                self.optimizer.step()
                i += 1
                if i >= self.nb_steps:
                    break
            # self.scheduler.step()
            epochs += 1

        return self.network

    def encode(self, X, batch_size=500):
        """
        Outputs the representations associated to the input by the encoder.

        Parameters
        ----------
        X : numpy.ndarray
            Testing set.
        batch_size : int, default=500
            Size of batches used for splitting the test data to avoid out of
            memory errors when using CUDA. Ignored if the testing set contains
            time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths
        varying = bool(np.isnan(np.sum(X)))

        test = Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        features = np.zeros((np.shape(X)[0], self.out_channels))
        self.network = self.network.eval()

        count = 0
        with torch.no_grad():
            for batch in test_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                # if self.win_type=='hanning':
                #     batch = hanning_tensor(batch)
                features[
                count * batch_size: (count + 1) * batch_size
                ] = self.network(batch).cpu().numpy()
                count += 1

        self.network = self.network.train()
        return features

    def encode_window(self, X, win_size=128, batch_size=500, window_batch_size=10000, step=10):
        """
        Outputs the representations associated to the input by the encoder,
        for each subseries of the input of the given size (sliding window
        representations).

        Parameters
        ----------
        X : numpy.ndarray
            Testing set.
        win_size : int, default=128
            Size of the sliding window.
        batch_size : int, default=500
            Size of batches used for splitting the test data to avoid
            out-of-memory errors when using CUDA.
        window_batch_size : int, default=10000
            Number of windows processed per batch when calling ``encode`` to
            save RAM.
        step : int, default=10
            Step length of the sliding window.
        """
        # _, dim = X.shape
        # X = np.transpose(np.array(X, dtype=float)).reshape(1, dim, -1)

        num_batch, num_channel, length = np.shape(X)
        num_window = int((length - win_size) / step) + 1
        embeddings = np.empty((num_batch, self.out_channels, num_window))

        for b in range(num_batch):
            for i in range(math.ceil(num_window / window_batch_size)):
                masking = np.array([X[b, :, j:j + win_size] for j in range(step * i * window_batch_size,
                                                                           step * min((i + 1) * window_batch_size,
                                                                                      num_window), step)])
                if self.win_type == 'hanning':
                    masking = hanning_numpy(masking)
                # print(masking.shape,step*i*window_batch_size, step*min((i+1)* window_batch_size, num_window))
                embeddings[b, :, i * window_batch_size: (i + 1) * window_batch_size] = np.swapaxes(
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


class CausalConv_LSE_Adaper(BasicEncoderClass):
    def _set_parmas(self, params):
        self.hyperparameters = params
        self.encoder = CausalConv_LSE(**self.hyperparameters)

    def fit(self, X):
        _, dim = X.shape
        X = np.transpose(np.array(X[:, :], dtype=float)).reshape(1, dim, -1)
        X = _all_normalize(X)
        self.encoder.fit(X, save_memory=True, verbose=False)
        # self.loss_list = self.encoder.loss_list

    def encode(self, X, win_size, step):
        _, dim = X.shape
        X = np.transpose(np.array(X[:, :], dtype=float)).reshape(1, dim, -1)
        X = _all_normalize(X)
        embeddings = self.encoder.encode_window(X, win_size=win_size, step=step)
        return embeddings


class Time2State:
    def __init__(self, win_size, step, encoder, clustering_component, verbose=False):
        """
        Initialize Time2State.

        Parameters
        ----------
        win_size : even integer.
            The size of sliding window.

        step : integer.
            The step size of sliding window.

        encoder_class : object.
            The instance of encoder.

        clustering_class: object.
            The instance of clustering component.
        """

        # The window size must be an even number.
        if win_size % 2 != 0:
            raise ValueError('Window size must be even.')

        self.__win_size = win_size
        self.__step = step
        self.__offset = int(win_size / 2)
        self.__encoder = encoder
        self.__clustering_component = clustering_component

    def fit(self, X, win_size, step):
        """
        Fit Time2State.

        Parameters
        ----------
        X : {ndarray} of shape (n_samples, n_features)

        win_size : even integer.
            The size of sliding window.

        step : integer.
            The step size of sliding window.

        Returns
        -------
        self : object
            Fitted Time2State.
        """

        self.__length = X.shape[0]
        self.fit_encoder(X)
        self.__encode(X, win_size, step)
        self.__cluster()
        self.__assign_label()
        return self

    def predict(self, X, win_size, step):
        """
        Find state sequence for X.

        Parameters
        ----------
        X : {ndarray} of shape (n_samples, n_features)

        win_size : even integer.
            The size of sliding window.

        step : integer.
            The step size of sliding window.

        Returns
        -------
        self : object
            Fitted Time2State.
        """
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
        self.__embedding_label = _reorder_label(self.__clustering_component.fit(self.__embeddings))

    def __assign_label(self):
        hight = len(set(self.__embedding_label))
        # weight_vector = np.concatenate([np.linspace(0,1,self.__offset),np.linspace(1,0,self.__offset)])
        weight_vector = np.ones(shape=(2 * self.__offset)).flatten()
        self.__state_seq = self.__embedding_label
        vote_matrix = np.zeros((self.__length, hight))
        i = 0
        for l in self.__embedding_label:
            vote_matrix[i:i + self.__win_size, l] += weight_vector
            i += self.__step
        self.__state_seq = np.array([np.argmax(row) for row in vote_matrix])
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
