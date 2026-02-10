import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr
from sklearn import covariance
from time import time
import torch


def convertToTorch(data, req_grad=False, use_cuda=False):
    """Convert numpy arrays to torch tensors, preserving gradients if requested."""
    if not torch.is_tensor(data):
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        data = torch.from_numpy(np.asarray(data, dtype=np.float32)).type(dtype)
    data.requires_grad = req_grad
    return data

def _convert_numeric_column(series: pd.Series) -> pd.Series:
    """Best-effort numeric conversion matching the legacy errors='ignore' semantics."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    try:
        return pd.to_numeric(series)
    except (TypeError, ValueError):
        return series


def _ensure_finite_numeric(table: pd.DataFrame, msg: str) -> pd.DataFrame:
    """Ensure the table contains only finite numeric values."""
    table = table.replace([np.inf, -np.inf], np.nan)
    if table.isnull().all(axis=1).any():
        table = table.dropna(axis=0, how="all")
    if table.isnull().all(axis=0).any():
        table = table.dropna(axis=1, how="all")
    table = table.apply(_convert_numeric_column)
    table = table.fillna(table.mean())
    table = table.fillna(0.0)
    values = table.to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{msg}: Unable to remove inf or NaN values from input data.")
    return table


def eigVal_conditionNum(A):
    eig = [v.real for v in np.linalg.eigvals(A)]
    condition_number = max(np.abs(eig)) / max(min(np.abs(eig)), 1e-12)
    return eig, condition_number


def getCovariance(Xb, offset=0.1):
    """Empirical covariance for each batch element with eigenvalue regularisation."""
    Sb = []
    for X in Xb:
        S = covariance.empirical_covariance(X, assume_centered=False)
        eig, _ = eigVal_conditionNum(S)
        min_eig = min(eig)
        if min_eig <= 1e-6:
            S += np.eye(S.shape[-1]) * (offset - min_eig)
        Sb.append(S)
    return np.array(Sb)


def generateRandomGraph(num_nodes, sparsity, seed=None):
    min_s, max_s = sparsity
    rng = np.random.default_rng(seed)
    s = rng.uniform(min_s, max_s)
    G = nx.generators.random_graphs.gnp_random_graph(num_nodes, s, seed=seed, directed=False)
    edge_connections = nx.adjacency_matrix(G).todense()
    return edge_connections


def simulateGaussianSamples(num_nodes, edge_connections, num_samples, seed=None, u=0.1, w_min=0.5, w_max=1.0):
    mean_value = 0
    mean_normal = np.ones(num_nodes) * mean_value
    if seed is not None:
        np.random.seed(seed)
    U = np.random.random((num_nodes, num_nodes)) * (w_max - w_min) + w_min
    theta = np.multiply(edge_connections, U)
    theta = (theta + theta.T) / 2 + np.eye(num_nodes)
    smallest_eigval = np.min(np.linalg.eigvals(theta)).real
    precision_mat = theta + np.eye(num_nodes) * (u - smallest_eigval)
    cov = np.linalg.inv(precision_mat)
    if seed is not None:
        np.random.seed(seed)
    data = np.random.multivariate_normal(mean=mean_normal, cov=cov, size=num_samples)
    return data, precision_mat


def process_table(table, NORM="no", MIN_VARIANCE=0.0, msg="", COND_NUM=np.inf, eigval_th=1e-3, VERBOSE=True):
    start = time()
    if VERBOSE:
        print(f"{msg}: Processing the input table for basic compatibility check")
        print(f"{msg}: The input table has sample {table.shape[0]} and features {table.shape[1]}")

    total_samples = table.shape[0]
    initial_columns = list(table.columns)
    # Pandas 2.0+ removed the internal _convert helper; replicate the behaviour
    # by attempting a numeric conversion column-wise while leaving non-numeric
    # data untouched.
    table = table.apply(_convert_numeric_column)

    zero_row_mask = (table == 0).all(axis=1)
    zero_row_count = int(zero_row_mask.sum())
    dropped_zero_samples = 0
    if zero_row_count and zero_row_count < total_samples:
        table = table.loc[~zero_row_mask]
        dropped_zero_samples = zero_row_count
    if VERBOSE:
        print(f"{msg}: Total zero samples dropped {dropped_zero_samples}")

    table = table.fillna(table.mean())

    single_value_columns = [col for col in table.columns if len(table[col].unique()) == 1]
    drop_single_value = list(single_value_columns)
    if drop_single_value and len(drop_single_value) >= table.shape[1]:
        drop_single_value = drop_single_value[:-1]
    if drop_single_value:
        table.drop(drop_single_value, inplace=True, axis=1)
    if VERBOSE:
        print(
            f"{msg}: Single value columns dropped: total {len(drop_single_value)}, columns {drop_single_value}"
        )

    if table.empty or table.shape[1] == 0:
        raise ValueError(
            f"{msg}: No valid samples or features remain after preprocessing; cannot compute covariance."
        )

    table = _ensure_finite_numeric(table, f"{msg}: Pre-normalisation")
    table = normalize_table(table, NORM)
    table = _ensure_finite_numeric(table, f"{msg}: Post-normalisation")
    analyse_condition_number(table, "Input", VERBOSE)

    all_columns = table.columns
    table = table.T.drop_duplicates().T
    duplicate_columns = list(set(all_columns) - set(table.columns))
    if VERBOSE:
        print(f"{msg}: Duplicates dropped: total {len(duplicate_columns)}, columns {duplicate_columns}")

    table_var = table.var().sort_values(ascending=True)
    low_variance_columns = list(table_var[table_var < MIN_VARIANCE].index)
    table.drop(low_variance_columns, inplace=True, axis=1)
    if VERBOSE:
        print(
            f"{msg}: Low Variance columns dropped: min variance {MIN_VARIANCE}, total {len(low_variance_columns)}, columns {low_variance_columns}"
        )

    cov_table, eig, con = analyse_condition_number(table, "Processed", VERBOSE)

    itr = 1
    while con > COND_NUM:
        if VERBOSE:
            print(
                f"{msg}: {itr} Condition number is high {con}. Dropping the highly correlated features in the cov-table"
            )
        eig = np.array(sorted(eig))
        lb_ill_cond_features = len(eig[eig < eigval_th])
        if VERBOSE:
            print(f"Current lower bound on ill-conditioned features {lb_ill_cond_features}")
        if lb_ill_cond_features == 0:
            if VERBOSE:
                print(f"All the eig vals are > {eigval_th} and current cond num {con}")
            if con > COND_NUM:
                lb_ill_cond_features = 1
            else:
                break
        highly_correlated_features = get_highly_correlated_features(cov_table)
        highly_correlated_features = highly_correlated_features[: min(lb_ill_cond_features, len(highly_correlated_features))]
        highly_correlated_columns = table.columns[highly_correlated_features]
        if VERBOSE:
            print(
                f"{msg} {itr}: Highly Correlated features dropped {list(highly_correlated_columns)}, {len(highly_correlated_columns)}"
            )
        table.drop(highly_correlated_columns, inplace=True, axis=1)
        cov_table, eig, con = analyse_condition_number(table, f"{msg} {itr}: Corr features dropped", VERBOSE)
        itr += 1
    if VERBOSE:
        print(f"{msg}: The processed table has sample {table.shape[0]} and features {table.shape[1]}")
        print(f"{msg}: Total time to process the table {np.round(time()-start, 3)} secs")

    if initial_columns:
        table = table.reindex(columns=initial_columns, fill_value=0.0)
        table = _ensure_finite_numeric(table, f"{msg}: Final alignment")
    return table


def get_highly_correlated_features(input_cov):
    cov2 = covariance.empirical_covariance(input_cov)
    np.fill_diagonal(cov2, 0)
    cov_upper = upper_tri_indexing(np.abs(cov2))
    sorted_cov_upper = sorted(enumerate(cov_upper), key=lambda x: x[1], reverse=True)
    th = sorted_cov_upper[int(0.1 * len(sorted_cov_upper))][1]
    high_indices = np.transpose(np.nonzero(np.abs(cov2) >= th))
    high_indices_dict = {}
    for i in high_indices:
        high_indices_dict.setdefault(i[0], []).append(i[1])
    top_correlated_features = [[f, len(v)] for (f, v) in high_indices_dict.items()]
    top_correlated_features.sort(key=lambda x: x[1], reverse=True)
    top_correlated_features = np.array(top_correlated_features)
    features_to_drop = top_correlated_features[:, 0] if len(top_correlated_features) else np.array([], dtype=int)
    return features_to_drop


def upper_tri_indexing(A):
    m = A.shape[0]
    r, c = np.triu_indices(m, 1)
    return A[r, c]


def analyse_condition_number(table, MESSAGE="", VERBOSE=True):
    S = covariance.empirical_covariance(table, assume_centered=False)
    eig, con = eig_val_condition_num(S)
    if VERBOSE:
        print(
            f"{MESSAGE} covariance matrix: The condition number {con} and min eig {min(eig)} max eig {max(eig)}"
        )
    return S, eig, con


def eig_val_condition_num(A):
    eig = [v.real for v in np.linalg.eigvals(A)]
    condition_number = max(np.abs(eig)) / max(min(np.abs(eig)), 1e-12)
    return eig, condition_number


def normalize_table(df, typeN):
    if typeN == "min_max":
        minima = df.min()
        denom = df.max() - minima
        denom = denom.replace(0, np.nan)
        result = (df - minima) / denom
        return result.fillna(0.0)
    if typeN == "mean":
        mean = df.mean()
        std = df.std()
        std = std.replace(0, np.nan)
        result = (df - mean) / std
        return result.fillna(0.0)
    if typeN != "no":
        print(f"No Norm applied : Type entered {typeN}")
    return df


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n if n else 0.0
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return 0.0
    return np.sqrt(phi2corr / denom)


def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures) if len(cat_measures) else 0.0
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / max(np.sum(n_array), 1)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if denominator == 0:
        return 0.0
    return np.sqrt(numerator / denominator)


def pairwise_cov_matrix(df, dtype):
    features = df.columns
    D = len(features)
    cov = np.zeros((D, D))
    for i, fi in enumerate(features):
        for j, fj in enumerate(features):
            if j >= i:
                if dtype[fi] == "c" and dtype[fj] == "c":
                    cov[i, j] = cramers_v(df[fi], df[fj])
                elif dtype[fi] == "c" and dtype[fj] == "r":
                    cov[i, j] = correlation_ratio(df[fi], df[fj])
                elif dtype[fi] == "r" and dtype[fj] == "c":
                    cov[i, j] = correlation_ratio(df[fj], df[fi])
                else:
                    cov[i, j] = pearsonr(df[fi], df[fj])[0]
                cov[j, i] = cov[i, j]
    cov = pd.DataFrame(cov, index=features, columns=features)
    return cov
