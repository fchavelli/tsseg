# Taken from: https://github.com/davidhallac/TICC

import collections
import errno
import math
# import matplotlib
import os

# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.cluster import KMeans
import pandas as pd
from multiprocessing import Pool

import numpy as np


def getTrainTestSplit(m, num_blocks, num_stacked):
    """Compute the training indices for the TICC model.

    Parameters
    ----------
    m : int
        Number of observations.
    num_blocks : int
        ``window_size + 1``.
    num_stacked : int
        ``window_size``.

    Returns
    -------
    list[int]
        Sorted list of training indices.
    """
    # Now splitting up stuff
    # split1 : Training and Test
    # split2 : Training and Test - different clusters
    training_percent = 1
    # list of training indices
    training_idx = np.random.choice(
        m - num_blocks + 1, size=int((m - num_stacked) * training_percent), replace=False)
    # Ensure that the first and the last few points are in
    training_idx = list(training_idx)
    if 0 not in training_idx:
        training_idx.append(0)
    if m - num_stacked not in training_idx:
        training_idx.append(m - num_stacked)
    training_idx = np.array(training_idx)
    return sorted(training_idx)


def compute_confusion_matrix(num_clusters, clustered_points_algo, sorted_indices_algo):
    """
    computes a confusion matrix and returns it
    """
    seg_len = 400
    true_confusion_matrix = np.zeros([num_clusters, num_clusters])
    for point in range(len(clustered_points_algo)):
        cluster = clustered_points_algo[point]
        num = (int(sorted_indices_algo[point] / seg_len) % num_clusters)
        true_confusion_matrix[int(num), int(cluster)] += 1
    return true_confusion_matrix


def find_matching(confusion_matrix):
    """
    returns the perfect matching
    """
    _, n = confusion_matrix.shape
    path = []
    for i in range(n):
        max_val = -1e10
        max_ind = -1
        for j in range(n):
            if j in path:
                pass
            else:
                temp = confusion_matrix[i, j]
                if temp > max_val:
                    max_val = temp
                    max_ind = j
        path.append(max_ind)
    return path


def upperToFull(a, eps=0):
    ind = (a < eps) & (a > -eps)
    a[ind] = 0
    n = int((-1 + np.sqrt(1 + 8 * a.shape[0])) / 2)
    A = np.zeros([n, n])
    A[np.triu_indices(n)] = a
    temp = A.diagonal()
    A = np.asarray((A + A.T) - np.diag(temp))
    return A


def updateClusters(LLE_node_vals, switch_penalty=1):
    """
    Takes in LLE_node_vals matrix and computes the path that minimizes
    the total cost over the path
    Note the LLE's are negative of the true LLE's actually!!!!!

    Note: switch penalty > 0
    """
    (T, num_clusters) = LLE_node_vals.shape
    future_cost_vals = np.zeros(LLE_node_vals.shape)

    # compute future costs
    for i in range(T - 2, -1, -1):
        j = i + 1
        indicator = np.zeros(num_clusters)
        future_costs = future_cost_vals[j, :]
        lle_vals = LLE_node_vals[j, :]
        for cluster in range(num_clusters):
            total_vals = future_costs + lle_vals + switch_penalty
            total_vals[cluster] -= switch_penalty
            future_cost_vals[i, cluster] = np.min(total_vals)

    # compute the best path
    path = np.zeros(T)

    # the first location
    curr_location = np.argmin(future_cost_vals[0, :] + LLE_node_vals[0, :])
    path[0] = curr_location

    # compute the path
    for i in range(T - 1):
        j = i + 1
        future_costs = future_cost_vals[j, :]
        lle_vals = LLE_node_vals[j, :]
        total_vals = future_costs + lle_vals + switch_penalty
        total_vals[int(path[i])] -= switch_penalty

        path[i + 1] = np.argmin(total_vals)

    # return the computed path
    return path


def computeBIC(K, T, clustered_points, inverse_covariances, empirical_covariances):
    '''
    empirical covariance and inverse_covariance should be dicts
    K is num clusters
    T is num samples
    '''
    mod_lle = 0

    threshold = 2e-5
    clusterParams = {}
    for cluster, clusterInverse in inverse_covariances.items():
        mod_lle += np.log(np.linalg.det(clusterInverse)) - np.trace(
            np.dot(empirical_covariances[cluster], clusterInverse))
        clusterParams[cluster] = np.sum(np.abs(clusterInverse) > threshold)
    curr_val = -1
    non_zero_params = 0
    for val in clustered_points:
        if val != curr_val:
            non_zero_params += clusterParams[val]
            curr_val = val
    return non_zero_params * np.log(T) - 2 * mod_lle


class ADMMSolver:

    def __init__(self, lamb, num_stacked, size_blocks, rho, S, rho_update_func=None):
        self.lamb = lamb
        self.numBlocks = num_stacked
        self.sizeBlocks = size_blocks
        probSize = num_stacked * size_blocks
        self.length = int(probSize * (probSize + 1) / 2)
        self.x = np.zeros(self.length)
        self.z = np.zeros(self.length)
        self.u = np.zeros(self.length)
        self.rho = float(rho)
        self.S = S
        self.status = 'initialized'
        self.rho_update_func = rho_update_func

    def ij2symmetric(self, i, j, size):
        return (size * (size + 1)) / 2 - (size - i) * ((size - i + 1)) / 2 + j - i

    def upper2Full(self, a):
        n = int((-1 + np.sqrt(1 + 8 * a.shape[0])) / 2)
        A = np.zeros([n, n])
        A[np.triu_indices(n)] = a
        temp = A.diagonal()
        A = (A + A.T) - np.diag(temp)
        return A

    def Prox_logdet(self, S, A, eta):
        d, q = np.linalg.eigh(eta * A - S)
        q = np.matrix(q)
        X_var = (1 / (2 * float(eta))) * q * (
            np.diag(d + np.sqrt(np.square(d) + (4 * eta) * np.ones(d.shape)))) * q.T
        x_var = X_var[np.triu_indices(S.shape[1])]  # extract upper triangular part as update variable
        return np.matrix(x_var).T

    def ADMM_x(self):
        a = self.z - self.u
        A = self.upper2Full(a)
        eta = self.rho
        x_update = self.Prox_logdet(self.S, A, eta)
        self.x = np.array(x_update).T.reshape(-1)

    def ADMM_z(self, index_penalty=1):
        a = self.x + self.u
        probSize = self.numBlocks * self.sizeBlocks
        z_update = np.zeros(self.length)

        # TODO: can we parallelize these?
        for i in range(self.numBlocks):
            elems = self.numBlocks if i == 0 else (2 * self.numBlocks - 2 * i) / 2  # i=0 is diagonal
            for j in range(self.sizeBlocks):
                startPoint = j if i == 0 else 0
                for k in range(startPoint, self.sizeBlocks):
                    locList = [((l + i) * self.sizeBlocks + j, l * self.sizeBlocks + k) for l in range(int(elems))]
                    if i == 0:
                        lamSum = sum(self.lamb[loc1, loc2] for (loc1, loc2) in locList)
                        indices = [self.ij2symmetric(loc1, loc2, probSize) for (loc1, loc2) in locList]
                    else:
                        lamSum = sum(self.lamb[loc2, loc1] for (loc1, loc2) in locList)
                        indices = [self.ij2symmetric(loc2, loc1, probSize) for (loc1, loc2) in locList]
                    pointSum = sum(a[int(index)] for index in indices)
                    rhoPointSum = self.rho * pointSum

                    # Calculate soft threshold
                    ans = 0
                    # If answer is positive
                    if rhoPointSum > lamSum:
                        ans = max((rhoPointSum - lamSum) / (self.rho * elems), 0)
                    elif rhoPointSum < -1 * lamSum:
                        ans = min((rhoPointSum + lamSum) / (self.rho * elems), 0)

                    for index in indices:
                        z_update[int(index)] = ans
        self.z = z_update

    def ADMM_u(self):
        u_update = self.u + self.x - self.z
        self.u = u_update

    # Returns True if convergence criteria have been satisfied
    # eps_abs = eps_rel = 0.01
    # r = x - z
    # s = rho * (z - z_old)
    # e_pri = sqrt(length) * e_abs + e_rel * max(||x||, ||z||)
    # e_dual = sqrt(length) * e_abs + e_rel * ||rho * u||
    # Should stop if (||r|| <= e_pri) and (||s|| <= e_dual)
    # Returns (boolean shouldStop, primal residual value, primal threshold,
    #          dual residual value, dual threshold)
    def CheckConvergence(self, z_old, e_abs, e_rel, verbose):
        norm = np.linalg.norm
        r = self.x - self.z
        s = self.rho * (self.z - z_old)
        # Primal and dual thresholds. Add .0001 to prevent the case of 0.
        e_pri = math.sqrt(self.length) * e_abs + e_rel * max(norm(self.x), norm(self.z)) + .0001
        e_dual = math.sqrt(self.length) * e_abs + e_rel * norm(self.rho * self.u) + .0001
        # Primal and dual residuals
        res_pri = norm(r)
        res_dual = norm(s)
        if verbose:
            # Debugging information to print(convergence criteria values)
            print('  r:', res_pri)
            print('  e_pri:', e_pri)
            print('  s:', res_dual)
            print('  e_dual:', e_dual)
        stop = (res_pri <= e_pri) and (res_dual <= e_dual)
        return (stop, res_pri, e_pri, res_dual, e_dual)

    # solve
    def __call__(self, maxIters, eps_abs, eps_rel, verbose):
        num_iterations = 0
        self.status = 'Incomplete: max iterations reached'
        for i in range(maxIters):
            z_old = np.copy(self.z)
            self.ADMM_x()
            self.ADMM_z()
            self.ADMM_u()
            if i != 0:
                stop, res_pri, e_pri, res_dual, e_dual = self.CheckConvergence(z_old, eps_abs, eps_rel, verbose)
                if stop:
                    self.status = 'Optimal'
                    break
                new_rho = self.rho
                if self.rho_update_func:
                    new_rho = self.rho_update_func(self.rho, res_pri, e_pri, res_dual, e_dual)
                scale = self.rho / new_rho
                rho = new_rho
                self.u = scale * self.u
            if verbose:
                # Debugging information prints current iteration #
                print('Iteration %d' % i)
        return self.x


class TICC:
    def __init__(self, window_size=10, number_of_clusters=5, lambda_parameter=11e-2,
                 beta=400, maxIters=1000, threshold=2e-5, write_out_file=False,
                 prefix_string="", num_proc=1, compute_BIC=False, cluster_reassignment=20, biased=False):
        """
        Parameters:
            - window_size: size of the sliding window
            - number_of_clusters: number of clusters
            - lambda_parameter: sparsity parameter
            - switch_penalty: temporal consistency parameter
            - maxIters: number of iterations
            - threshold: convergence threshold
            - write_out_file: (bool) if true, prefix_string is output file dir
            - prefix_string: output directory if necessary
            - cluster_reassignment: number of points to reassign to a 0 cluster
            - biased: Using the biased or the unbiased covariance
        """
        self.window_size = window_size
        self.number_of_clusters = number_of_clusters
        self.lambda_parameter = lambda_parameter
        self.switch_penalty = beta
        self.maxIters = maxIters
        self.threshold = threshold
        self.write_out_file = write_out_file
        self.prefix_string = prefix_string
        self.num_proc = num_proc
        self.compute_BIC = compute_BIC
        self.cluster_reassignment = cluster_reassignment
        self.num_blocks = self.window_size + 1
        self.biased = biased
        pd.set_option('display.max_columns', 500)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        np.random.seed(102)

    def fit_transform(self, series):
        """
        Main method for TICC solver.
        Parameters:
            - input_file: location of the data file
        """
        if self.maxIters <= 0:
            raise ValueError("maxIters must be > 0, got {}".format(self.maxIters))
        self.log_parameters()

        # Get data into proper format
        times_series_arr = series
        time_series_rows_size, time_series_col_size = times_series_arr.shape

        ############
        # The basic folder to be created
        #str_NULL = self.prepare_out_directory()

        # Train test split
        training_indices = getTrainTestSplit(time_series_rows_size, self.num_blocks,
                                             self.window_size)  # indices of the training samples
        num_train_points = len(training_indices)

        # Stack the training data
        complete_D_train = self.stack_training_data(times_series_arr, time_series_col_size, num_train_points,
                                                    training_indices)

        # Initialization
        # Gaussian Mixture
        gmm = mixture.GaussianMixture(n_components=self.number_of_clusters, covariance_type="full")
        gmm.fit(complete_D_train)
        clustered_points = gmm.predict(complete_D_train)
        gmm_clustered_pts = clustered_points + 0
        # K-means
        kmeans = KMeans(n_clusters=self.number_of_clusters, random_state=0).fit(complete_D_train)
        clustered_points_kmeans = kmeans.labels_  # todo, is there a difference between these two?
        kmeans_clustered_pts = kmeans.labels_

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = Pool(processes=self.num_proc)  # multi-threading
        for iters in range(self.maxIters):
            # #print("\n\n\nITERATION ###", iters)
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

            # train_clusters holds the indices in complete_D_train
            # for each of the clusters
            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, complete_D_train,
                                          empirical_covariances, len_train_clusters, time_series_col_size, pool,
                                          train_clusters_arr)

            self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res,
                                   train_cluster_inverse)

            # update old computed covariance
            old_computed_covariance = computed_covariance

            # #print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                  'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'complete_D_train': complete_D_train,
                                  'time_series_col_size': time_series_col_size}
            clustered_points = self.predict_clusters()

            # recalculate lengths
            new_train_clusters = collections.defaultdict(list)  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            before_empty_cluster_assign = clustered_points.copy()

            if iters != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in
                                 range(self.number_of_clusters)]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.number_of_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        cluster_selected = valid_clusters[counter]  # a cluster that is not len 0
                        counter = (counter + 1) % len(valid_clusters)
                        # #print("cluster that is zero is:", cluster_num, "selected cluster instead is:", cluster_selected)
                        start_point = np.random.choice(
                            new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[
                                self.number_of_clusters, cluster_selected]
                            cluster_mean_stacked_info[self.number_of_clusters, cluster_num] = complete_D_train[
                                                                                              point_to_move, :]
                            cluster_mean_info[self.number_of_clusters, cluster_num] \
                                = complete_D_train[point_to_move, :][
                                  (self.window_size - 1) * time_series_col_size:self.window_size * time_series_col_size]

            for cluster_num in range(self.number_of_clusters):
                # print("length of cluster #", cluster_num, "-------->", sum([x == cluster_num for x in clustered_points]))
                pass

            #self.write_plot(clustered_points, str_NULL, training_indices)

            # TEST SETS STUFF
            # LLE + swtiching_penalty
            # Segment length
            # Create the F1 score from the graphs from k-means and GMM
            # Get the train and test points
            train_confusion_matrix_EM = compute_confusion_matrix(self.number_of_clusters, clustered_points,
                                                                 training_indices)
            train_confusion_matrix_GMM = compute_confusion_matrix(self.number_of_clusters, gmm_clustered_pts,
                                                                  training_indices)
            train_confusion_matrix_kmeans = compute_confusion_matrix(self.number_of_clusters, kmeans_clustered_pts,
                                                                     training_indices)
            ###compute the matchings
            matching_EM, matching_GMM, matching_Kmeans = self.compute_matches(train_confusion_matrix_EM,
                                                                              train_confusion_matrix_GMM,
                                                                              train_confusion_matrix_kmeans)

            # #print("\n\n\n")

            if np.array_equal(old_clustered_points, clustered_points):
                # print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training
        if pool is not None:
            pool.close()
            pool.join()
        train_confusion_matrix_EM = compute_confusion_matrix(self.number_of_clusters, clustered_points,
                                                             training_indices)
        train_confusion_matrix_GMM = compute_confusion_matrix(self.number_of_clusters, gmm_clustered_pts,
                                                              training_indices)
        train_confusion_matrix_kmeans = compute_confusion_matrix(self.number_of_clusters, clustered_points_kmeans,
                                                                 training_indices)

        self.compute_f_score(matching_EM, matching_GMM, matching_Kmeans, train_confusion_matrix_EM,
                             train_confusion_matrix_GMM, train_confusion_matrix_kmeans)

        if self.compute_BIC:
            bic = computeBIC(self.number_of_clusters, time_series_rows_size, clustered_points, train_cluster_inverse,
                             empirical_covariances)
            return clustered_points, train_cluster_inverse, bic

        return clustered_points, train_cluster_inverse

    def compute_f_score(self, matching_EM, matching_GMM, matching_Kmeans, train_confusion_matrix_EM,
                        train_confusion_matrix_GMM, train_confusion_matrix_kmeans):
        f1_EM_tr = -1  # computeF1_macro(train_confusion_matrix_EM,matching_EM,num_clusters)
        f1_GMM_tr = -1  # computeF1_macro(train_confusion_matrix_GMM,matching_GMM,num_clusters)
        f1_kmeans_tr = -1  # computeF1_macro(train_confusion_matrix_kmeans,matching_Kmeans,num_clusters)
        # print("\n\n")
        # print("TRAINING F1 score:", f1_EM_tr, f1_GMM_tr, f1_kmeans_tr)
        correct_e_m = 0
        correct_g_m_m = 0
        correct_k_means = 0
        for cluster in range(self.number_of_clusters):
            matched_cluster__e_m = matching_EM[cluster]
            matched_cluster__g_m_m = matching_GMM[cluster]
            matched_cluster__k_means = matching_Kmeans[cluster]

            correct_e_m += train_confusion_matrix_EM[cluster, matched_cluster__e_m]
            correct_g_m_m += train_confusion_matrix_GMM[cluster, matched_cluster__g_m_m]
            correct_k_means += train_confusion_matrix_kmeans[cluster, matched_cluster__k_means]

    def compute_matches(self, train_confusion_matrix_EM, train_confusion_matrix_GMM, train_confusion_matrix_kmeans):
        matching_Kmeans = find_matching(train_confusion_matrix_kmeans)
        matching_GMM = find_matching(train_confusion_matrix_GMM)
        matching_EM = find_matching(train_confusion_matrix_EM)
        correct_e_m = 0
        correct_g_m_m = 0
        correct_k_means = 0
        for cluster in range(self.number_of_clusters):
            matched_cluster_e_m = matching_EM[cluster]
            matched_cluster_g_m_m = matching_GMM[cluster]
            matched_cluster_k_means = matching_Kmeans[cluster]

            correct_e_m += train_confusion_matrix_EM[cluster, matched_cluster_e_m]
            correct_g_m_m += train_confusion_matrix_GMM[cluster, matched_cluster_g_m_m]
            correct_k_means += train_confusion_matrix_kmeans[cluster, matched_cluster_k_means]
        return matching_EM, matching_GMM, matching_Kmeans

    # def write_plot(self, clustered_points, str_NULL, training_indices):
    #     # Save a figure of segmentation
    #     plt.figure()
    #     plt.plot(training_indices[0:len(clustered_points)], clustered_points, color="r")  # ,marker = ".",s =100)
    #     plt.ylim((-0.5, self.number_of_clusters + 0.5))
    #     if self.write_out_file: plt.savefig(
    #         str_NULL + "TRAINING_EM_lam_sparse=" + str(self.lambda_parameter) + "switch_penalty = " + str(
    #             self.switch_penalty) + ".jpg")
    #     plt.close("all")
    #     # print("Done writing the figure")

    def smoothen_clusters(self, cluster_mean_info, computed_covariance,
                          cluster_mean_stacked_info, complete_D_train, n):
        clustered_points_len = len(complete_D_train)
        inv_cov_dict = {}  # cluster to inv_cov
        log_det_dict = {}  # cluster to log_det
        for cluster in range(self.number_of_clusters):
            cov_matrix = computed_covariance[self.number_of_clusters, cluster][0:(self.num_blocks - 1) * n,
                         0:(self.num_blocks - 1) * n]
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            log_det_cov = np.log(np.linalg.det(cov_matrix))  # log(det(sigma2|1))
            inv_cov_dict[cluster] = inv_cov_matrix
            log_det_dict[cluster] = log_det_cov
        # For each point compute the LLE
        # print("beginning the smoothening ALGORITHM")
        LLE_all_points_clusters = np.zeros([clustered_points_len, self.number_of_clusters])
        for point in range(clustered_points_len):
            if point + self.window_size - 1 < complete_D_train.shape[0]:
                for cluster in range(self.number_of_clusters):
                    cluster_mean = cluster_mean_info[self.number_of_clusters, cluster]
                    cluster_mean_stacked = cluster_mean_stacked_info[self.number_of_clusters, cluster]
                    x = complete_D_train[point, :] - cluster_mean_stacked[0:(self.num_blocks - 1) * n]
                    inv_cov_matrix = inv_cov_dict[cluster]
                    log_det_cov = log_det_dict[cluster]
                    lle = np.dot(x.reshape([1, (self.num_blocks - 1) * n]),
                                 np.dot(inv_cov_matrix, x.reshape([n * (self.num_blocks - 1), 1]))) + log_det_cov
                    LLE_all_points_clusters[point, cluster] = lle

        return LLE_all_points_clusters

    def optimize_clusters(self, computed_covariance, len_train_clusters, log_det_values, optRes, train_cluster_inverse):
        for cluster in range(self.number_of_clusters):
            if optRes[cluster] == None:
                continue
            val = optRes[cluster].get()
            # print("OPTIMIZATION for Cluster #", cluster, "DONE!!!")
            # THIS IS THE SOLUTION
            S_est = upperToFull(val, 0)
            X2 = S_est
            u, _ = np.linalg.eig(S_est)
            cov_out = np.linalg.inv(X2)

            # Store the log-det, covariance, inverse-covariance, cluster means, stacked means
            log_det_values[self.number_of_clusters, cluster] = np.log(np.linalg.det(cov_out))
            computed_covariance[self.number_of_clusters, cluster] = cov_out
            train_cluster_inverse[cluster] = X2
        for cluster in range(self.number_of_clusters):
            # print("length of the cluster ", cluster, "------>", len_train_clusters[cluster])
            pass

    def train_clusters(self, cluster_mean_info, cluster_mean_stacked_info, complete_D_train, empirical_covariances,
                       len_train_clusters, n, pool, train_clusters_arr):
        optRes = [None for i in range(self.number_of_clusters)]
        for cluster in range(self.number_of_clusters):
            cluster_length = len_train_clusters[cluster]
            # A covariance matrix cannot be computed for a single point.
            # Skip this cluster for this iteration if it has fewer than 2 points.
            if cluster_length < 2:
                continue
            
            if cluster_length != 0:
                size_blocks = n
                indices = train_clusters_arr[cluster]
                D_train = np.zeros([cluster_length, self.window_size * n])
                for i in range(cluster_length):
                    point = indices[i]
                    D_train[i, :] = complete_D_train[point, :]

                cluster_mean_info[self.number_of_clusters, cluster] = np.mean(D_train, axis=0)[
                                                                      (
                                                                              self.window_size - 1) * n:self.window_size * n].reshape(
                    [1, n])
                cluster_mean_stacked_info[self.number_of_clusters, cluster] = np.mean(D_train, axis=0)
                ##Fit a model - OPTIMIZATION
                probSize = self.window_size * size_blocks
                lamb = np.zeros((probSize, probSize)) + self.lambda_parameter
                S = np.cov(np.transpose(D_train), bias=self.biased)
                empirical_covariances[cluster] = S

                rho = 1
                solver = ADMMSolver(lamb, self.window_size, size_blocks, 1, S)
                # apply to process pool
                optRes[cluster] = pool.apply_async(solver, (1000, 1e-6, 1e-6, False,))
        return optRes

    def stack_training_data(self, Data, n, num_train_points, training_indices):
        complete_D_train = np.zeros([num_train_points, self.window_size * n])
        for i in range(num_train_points):
            for k in range(self.window_size):
                if i + k < num_train_points:
                    idx_k = training_indices[i + k]
                    complete_D_train[i][k * n:(k + 1) * n] = Data[idx_k][0:n]
        return complete_D_train

    def prepare_out_directory(self):
        str_NULL = self.prefix_string + "lam_sparse=" + str(self.lambda_parameter) + "maxClusters=" + str(
            self.number_of_clusters + 1) + "/"
        if not os.path.exists(os.path.dirname(str_NULL)):
            try:
                os.makedirs(os.path.dirname(str_NULL))
            except OSError as exc:  # Guard against race condition of path already existing
                if exc.errno != errno.EEXIST:
                    raise

        return str_NULL

    def load_data(self, input_file):
        Data = np.loadtxt(input_file, delimiter=",")
        (m, n) = Data.shape  # m: num of observations, n: size of observation vector
        # print("completed getting the data")
        return Data, m, n

    def log_parameters(self):
        # print("lam_sparse", self.lambda_parameter)
        # print("switch_penalty", self.switch_penalty)
        # print("num_cluster", self.number_of_clusters)
        # print("num stacked", self.window_size)
        return

    def predict_clusters(self, test_data=None):
        '''
        Given the current trained model, predict clusters.  If the cluster segmentation has not been optimized yet,
        than this will be part of the interative process.

        Args:
            numpy array of data for which to predict clusters.  Columns are dimensions of the data, each row is
            a different timestamp

        Returns:
            vector of predicted cluster for the points
        '''
        if test_data is not None:
            if not isinstance(test_data, np.ndarray):
                raise TypeError("input must be a numpy array!")
        else:
            test_data = self.trained_model['complete_D_train']

        # SMOOTHENING
        lle_all_points_clusters = self.smoothen_clusters(self.trained_model['cluster_mean_info'],
                                                         self.trained_model['computed_covariance'],
                                                         self.trained_model['cluster_mean_stacked_info'],
                                                         test_data,
                                                         self.trained_model['time_series_col_size'])

        # Update cluster points - using NEW smoothening
        clustered_points = updateClusters(lle_all_points_clusters, switch_penalty=self.switch_penalty)

        return (clustered_points)
