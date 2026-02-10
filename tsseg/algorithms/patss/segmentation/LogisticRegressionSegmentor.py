
import numpy as np
import multiprocessing
from typing import List, Union

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression

from baselines.patss.embedding import PatternBasedEmbedding
from baselines.patss.segmentation.Segmentor import Segmentor


class LogisticRegressionSegmentor(Segmentor):
    """
    Segments a time series based on an embedding matrix in two steps.

    First, a KMeans clustering model is fitted on the embedding, which will
    provide a discrete clustering (i.e., every observation in the time series
    will be assigned a discrete cluster label). The number of clusters `K` is
    decided based on the silhouette method. The discrete clustering serves as
    an initial model of where the different semantic segments occur.

    Second, the discrete clustering is fed to a logistic regression model. This
    model will consequently learn to which segment a given embedding of the
    pattern-based embedding belongs. Because this is a probabilistic model, we
    can also retrieve the probabilities of a given observation belong to some
    specific segment, thereby obtaining a probabilistic segmentation.

    Parameters
    ----------
    n_segments : int or List[int], default=[2, 3, 4, 5, 6, 7, 8, 9]
        The number of segments that should be computed. If only an integer is
        given, then this is the exact number of segments. Otherwise, if a list
        of integers is given, the number of segments with the largest silhouette
        score will be used.
    regularization : float, default=0.1
        The regularization factor used for fitting Logistic Regression.
    k_means_kwargs: dict, default={'n_init': 'auto', 'init': 'k-means++'}
        Additional arguments to pass to the sklearn k-means clustering.
    n_jobs : int, default = 1
        The number of jobs that may run in parallel. This is used to compute the
        K-means clustering for multiple number of segments at the same time.

    Attributes
    ----------
    k_means\_ : sklearn.KMeans
        The fitted K-Means clustering model.
    logistic_regression\_  : sklearn.LogisticRegression
        The fitted logistic regression model.
    """
    def __init__(self,
                 n_segments: Union[List[int], int] = None,
                 regularization: float = 0.1,
                 k_means_kwargs=None,
                 n_jobs: int = 1):
        self.n_segments: List[int] = \
            list(range(2, 10)) if n_segments is None else \
            [n_segments] if isinstance(n_segments, int) else \
            n_segments
        self.regularization = regularization
        self.k_means_kwargs = k_means_kwargs or {'n_init': 'auto', 'init': 'k-means++'}
        self.n_jobs = n_jobs

    def fit(self, pattern_based_embedding: PatternBasedEmbedding, y=None) -> 'LogisticRegressionSegmentor':
        """
        Fits this segmentor with the given pattern-based embedding through
        logistic regression supervised by K-Means clustering. (1) K-Means
        clustering will be performed for all ``K in self.n_segments``. If there
        are at least two values and ``self.n_jobs > 1``, then these clusterings
        will be performed in parallel. (2) The best clustering is selected
        using the silhouette method, which gives an unsupervised estimate of
        how well the clustering is. The obtained clustering gives a general
        idea of the different segments in the data. (3) A logistic regression
        model is trained using the clustering as labels, thereby enabling to
        learn a more fine-grained version of the semantic segmentation, as
        well as learning the probability distribution over the semantic
        segments.

        Parameters
        ----------
        pattern_based_embedding : PatternBasedEmbedding
            The pattern-based embedding used for training this segmentor.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : LogisticRegressionSegmentor
            Returns the instance itself
        """
        # Transpose the embedding matrix (because each column is a feature
        # vector, instead of each row)
        embedding_matrix = pattern_based_embedding.embedding_matrix.transpose()

        # Use K-means to cluster with varying number of clusters
        if self.n_jobs > 1 and len(self.n_segments) > 1:
            args = [(n_segments, embedding_matrix) for n_segments in self.n_segments]
            with multiprocessing.Pool(self.n_jobs) as pool:
                pool_results = pool.starmap(self._compute_kmeans_segmentation, args)
        else:
            pool_results = [
                self._compute_kmeans_segmentation(n_segments, embedding_matrix)
                for n_segments in self.n_segments
            ]

        # Identify the best cluster with maximum silhoutte score
        index_largest_silhouette_score = np.argmax([silhouette_avg for silhouette_avg, *_ in pool_results])
        best_kmeans_segmentation = pool_results[index_largest_silhouette_score][1]
        self.k_means_ = pool_results[index_largest_silhouette_score][2]

        # Fit the logistic regression model
        self.logistic_regression_ = LogisticRegression(
            C=self.regularization,
            multi_class='ovr',
            penalty='elasticnet',
            solver='saga',
            l1_ratio=0.5,
            random_state=0
        ).fit(embedding_matrix, best_kmeans_segmentation)

        return self

    def predict(self, pattern_based_embedding: PatternBasedEmbedding) -> np.ndarray:
        """
        Predicts the segment probabilities for the given pattern-based embedding.
        This is done by obtaining the predicted probabilities of the fitted
        logistic regression instance.

        Parameters
        ----------
        pattern_based_embedding : PatternBasedEmbedding
            The pattern-based embedding to predict the segmentation for.

        Returns
        -------
        segmentation : np.ndarray of shape (n_segments, n_samples)
            The segmentation based on the given pattern-based embedding, which consists
            of ``n_segments`` different semantic segments for a time series with ´`n_samples``
            observations. The value ``segmentation[s, t]`` equals the probability of being
            in semantic segment ``s`` at time step ``t``.
        """
        embedding_matrix = pattern_based_embedding.embedding_matrix.transpose()
        return self.logistic_regression_.predict_proba(embedding_matrix).transpose()

    def _compute_kmeans_segmentation(self, n_segments: int, embedding_matrix: np.ndarray):
        # Cluster the embedding
        k_means = KMeans(n_clusters=n_segments, **self.k_means_kwargs, random_state=0)
        segmentation = k_means.fit_predict(embedding_matrix)

        # Compute silhouette score
        if len(set(segmentation)) != n_segments:
            silhouette_avg = -1
        else:
            n = embedding_matrix.shape[0]
            sample_size = n if n < 2000 else 2000 + int(0.1 * (n - 2000))
            silhouette_avg = silhouette_score(embedding_matrix, segmentation, sample_size=sample_size)

        return silhouette_avg, segmentation, k_means
