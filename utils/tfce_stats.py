import numpy as np
from mne.stats.cluster_level import _find_clusters, _pval_from_histogram
import hcp_utils as hcp
from scipy import sparse
import time
from joblib import Parallel, delayed


def tfce_stats(
    scores, 
    permutation_scores, 
    adj=hcp.cortical_adjacency,
    tfce_step=0.005, 
    verbose=False
):
    '''Compute p-values of TFCE maps with permutation testing.
    Here we use functions for surface-based statistics from:
    https://github.com/mne-tools/mne-python/blob/main/mne/stats/cluster_level.py

    Parameters
    ----------
    scores : array-like of shape 1D
        original scores.

    permutation_scores : array-like of shape 1D
        scores based on permuted labels.

    adj : array-like of shape 2D
        adjacency matrix containing vertex neighborhoods.

    tfce_step : float
        step size for computing tfce values.

    verbose : boolean
        The verbosity level. Default is False

    Returns
    -------
    pvalues_tfce : array-like of the same shape as scores
        TFCE corrected pvalues.

    cluster_stats : array-like of the same shape as scores
        TFCE map of scores.
    '''

    # Convert to COO format for python MNEs _find_cluster() function.
    adj_coo = sparse.coo_matrix(adj)

    mne_threshold = dict(start=0, 
                         step=tfce_step)

    # Compute tfce map of original scores.
    _, cluster_stats = _find_clusters(scores, 
                                      adjacency=adj_coo, 
                                      threshold=mne_threshold)

    # Get tfce maps of permuted data.
    start_time = time.time()

    # Compute tfce map of scores based on permuted labels.
    _, perm_cluster_stats = zip(*Parallel(n_jobs=-1)(delayed(_find_clusters)
                                                     (single_perm_scores, 
                                                      adjacency=adj_coo, 
                                                      threshold=mne_threshold) 
                                                     for single_perm_scores 
                                                     in permutation_scores.T))
    end_time = time.time()

    if verbose:
        print('Finished after %i mins' % ((end_time - start_time)/60))

    # Build H0.
    H0 = []
    for stats in perm_cluster_stats:
        H0.append(stats.max())
    orig = cluster_stats.max()
    H0.insert(0, orig)
    H0 = np.stack(H0)

    # Compute p-values.
    pvalues_tfce = _pval_from_histogram(cluster_stats, 
                                        H0, 
                                        tail=0)

    return pvalues_tfce, cluster_stats