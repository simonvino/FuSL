"""The searchlight is a widely used approach for the study \
of the fine-grained patterns of information in fMRI analysis, \
in which multivariate statistical relationships are iteratively tested \
in the neighborhood of each location of a domain. This code expands \
classical searchlight analysis by adding permutation testing and 
feature interpretaion using SHAP."""

# Authors : Vincent Michel (vm.michel@gmail.com)
#           Alexandre Gramfort (alexandre.gramfort@inria.fr)
#           Philippe Gervais (philippe.gervais@inria.fr)
#           Simon Wein (simon.wein@ur.de)
#
# License: simplified BSD

import sys
import time
import warnings
import numpy as np
from joblib import Parallel, cpu_count, delayed
from sklearn import svm
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score, permutation_test_score
from nilearn._utils import fill_doc
from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_validate
import shap

ESTIMATOR_CATALOG = dict(svc=svm.LinearSVC, svr=svm.SVR)


@fill_doc
def fusion_search_light(
    X,
    y,
    estimator,
    A,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=-1,
    verbose=0,
    n_permutations=0,
    shap=True,
):
    """Compute a search_light. This function can 
    include permutation testing and compute SHAP values.

    Parameters
    ----------
    X : array-like of shape at least 2D
        data to fit.

    y : array-like
        target variable to predict.

    estimator : estimator object implementing 'fit'
        object to use to fit the data

    A : scipy sparse matrix.
        adjacency matrix. Defines for each feature the neigbhoring features
        following a given structure of the data.

    groups : array-like, optional, (default None)
        group label for each sample for cross validation.

        .. note::
            This will have no effect for scikit learn < 0.18

    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        for possible values.
        If callable, it takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.

    n_permutations : int, optional
        If not None performs permutation testing and returns 
        uncorrected p-values.

    shap : boolean
        If True, computes global (mean absolute) SHAP value for each feature.

    %(n_jobs_all)s
    %(verbose0)s

    Returns
    -------
    results : dictionary, with keys containing:
        * scores : numpy.ndarray of shape (n_vertices, n_folds)
            Classification scores for each vertex and each fold.
        * avg_scores : numpy.ndarray of shape (n_vertices) 
            Average scores over all folds.
        * perm_scores (optional) : of shape (n_vertices, n_permutations) 
            Scores of each permutation run.
        * pvals_uncor (optional) : numpy.ndarray of shape (n_vertices) 
            Uncorrected p-values of permutation testing.
        * shap_vals (optional) : numpy.ndarray of shape (n_features) 
            global SHAP value for each feature.
    """
    group_iter = GroupIterator(A.shape[0], n_jobs)
    with warnings.catch_warnings():  # might not converge
        warnings.simplefilter("ignore", ConvergenceWarning)
        results = merge_dicts(Parallel(n_jobs=n_jobs, verbose=verbose, prefer="threads")(
                                        delayed(_group_iter_search_light)(
                                        A.rows[list_i],
                                        estimator,
                                        X,
                                        y,
                                        groups,
                                        scoring,
                                        cv,
                                        thread_id + 1,
                                        A.shape[0],
                                        verbose,
                                        n_permutations,
                                        shap,
            )
            for thread_id, list_i in enumerate(group_iter)
        )
        )
    return results


@fill_doc
class GroupIterator:
    """Group iterator.

    Provides group of features for search_light loop
    that may be used with Parallel.

    Parameters
    ----------
    n_features : int
        Total number of features
    %(n_jobs)s

    """

    def __init__(self, n_features, n_jobs=1):
        self.n_features = n_features
        if n_jobs == -1:
            n_jobs = cpu_count()
        self.n_jobs = n_jobs

    def __iter__(self):
        yield from np.array_split(np.arange(self.n_features), self.n_jobs)


def _group_iter_search_light(
    list_rows,
    estimator,
    X,
    y,
    groups,
    scoring,
    cv,
    thread_id,
    total,
    verbose=0,
    n_permutations=0,
    shap=True,
):
    """Perform grouped iterations of search_light.

    Parameters
    ----------
    list_rows : array of arrays of int
        adjacency rows. For a voxel with index i in X, list_rows[i] is the list
        of neighboring voxels indices (in X).

    estimator : estimator object implementing 'fit'
        object to use to fit the data

    X : array-like of shape at least 2D
        data to fit.

    y : array-like
        target variable to predict.

    groups : array-like, optional
        group label for each sample for cross validation.

    scoring : string or callable, optional
        Scoring strategy to use. See the scikit-learn documentation.
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross validation is
        used or 3-fold stratified cross-validation when y is supplied.

    thread_id : int
        process id, used for display.

    total : int
        Total number of voxels, used for display

    verbose : int, optional
        The verbosity level. Default is 0

    n_permutations : int, optional
        If not None performs permutation testing and returns 
        uncorrected p-values.

    shap : boolean
        If True, computes global (mean absolute) SHAP value for each feature.

    Returns
    -------
    results : dictionary, with keys containing:
        * scores : numpy.ndarray of shape (n_vertices, n_folds)
            Classification scores for each vertex and each fold.
        * avg_scores : numpy.ndarray of shape (n_vertices) 
            Average scores over all folds.
        * perm_scores (optional) : of shape (n_vertices, n_permutations) 
            Scores of each permutation run.
        * pvals_uncor (optional) : numpy.ndarray of shape (n_vertices) 
            Uncorrected p-values of permutation testing.
        * shap_vals (optional) : numpy.ndarray of shape (n_features) 
            global SHAP value for each feature.
    """
    results = {}  # Initialize dict for results.

    results['scores'] = np.zeros((len(list_rows), cv.get_n_splits()))  # Scores of all CV splits. 
    results['avg_scores'] = np.zeros(len(list_rows))  # Averages over CV splits.
    if n_permutations > 0:
        results['perm_scores'] = np.ones((len(list_rows), n_permutations))  # Individual scores of permutations.
        results['pvals_uncor'] = np.ones(len(list_rows))  # Uncorrected p-values, computed by sklearn's permutation_test_score.
    if shap is True:
        results['shap_vals'] = []

    t0 = time.time()
    for i, row in enumerate(list_rows):
        kwargs = {"scoring": scoring, "groups": groups}

        # Do permutation testing.
        if n_permutations > 0:
            _, perm_scores, pvals_uncor = permutation_test_score(estimator, 
                                                                 X[:, row], 
                                                                 y, 
                                                                 cv=cv, 
                                                                 n_permutations=n_permutations, 
                                                                 n_jobs=1, 
                                                                 random_state=42, 
                                                                 verbose=0, 
                                                                 **kwargs
                                                                )
            results['perm_scores'][i] = perm_scores
            results['pvals_uncor'][i] = pvals_uncor

        # Compute unpermuted scores of each CV split.
        results['scores'][i, :] = cross_val_score(estimator, 
                                                  X[:, row], 
                                                  y, 
                                                  cv=cv, 
                                                  n_jobs=1, 
                                                  **kwargs
                                                 )
        results['avg_scores'][i] = np.mean(results['scores'][i, :])

        # Compute SHAP values.
        if shap is True:
            results['shap_vals'].append(cross_val_score_x(estimator, 
                                                          X[:, row], 
                                                          y, 
                                                          cv=cv, 
                                                          n_jobs=1, 
                                                          **kwargs
                                                         )
                                       )  

        if verbose > 0:
            # One can't print less than each 10 iterations
            step = 11 - min(verbose, 10)
            if i % step == 0:
                # If there is only one job, progress information is fixed
                crlf = "\r" if total == len(list_rows) else "\n"
                percent = float(i) / len(list_rows)
                percent = round(percent * 100, 2)
                dt = time.time() - t0
                # We use a max to avoid a division by zero
                remaining = (100.0 - percent) / max(0.01, percent) * dt
                sys.stderr.write(
                    f"Job #{thread_id}, processed {i}/{len(list_rows)} voxels "
                    f"({percent:0.2f}%, {remaining} seconds remaining){crlf}"
                )

    return results


def merge_dicts(dcts):
    '''Concatenate the content of multiple dictionaries.

    Parameters
    ----------
    dcts : list of dictionaries
        list of dictionaries with the same keys, 
        containing either numpy arrays or lists.

    Returns
    -------
    dct_concat : dict
        single dictionary, 
        with contents concatenated.
    '''
    dct_concat = {}
    for k in dcts[0].keys():
        if isinstance(dcts[0][k], np.ndarray):
            dct_concat[k] = np.concatenate(list(dct[k] for dct in dcts))
        if isinstance(dcts[0][k], list):
            dct_concat[k] = sum(list(dct[k] for dct in dcts), [])
    return dct_concat


def cross_val_score_x(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    error_score=np.nan,
):
    """Evaluate a score by cross-validation.
    Returns SHAP values.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only
        a single value.

        Similar to :func:`cross_validate`
        but only a single metric is permitted.

        If `None`, the estimator's default scorer (if available) is used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - `None`, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable that generates (train, test) splits as arrays of indices.

        For `int`/`None` inputs, if the estimator is a classifier and `y` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            `cv` default value if `None` changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        The verbosity level.

    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - ``None``, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

        .. versionadded:: 0.20

    Returns
    -------
    shap_results : array like
        Average absolute SHAP value for each feature.
    """
    # To ensure multisource format is not supported
    scorer = check_scoring(estimator, scoring=scoring)

    cv_results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        groups=groups,
        scoring={"score": scorer},
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        fit_params=fit_params,
        pre_dispatch=pre_dispatch,
        error_score=error_score,
        return_estimator=True,
        return_indices=True,
    )

    # Compute SHAP values.
    shap_results = explain_results(cv_results, X)

    return shap_results


def explain_results(cv_results, X, sel_idxs=None):
    '''Compute SHAP value for each feature.

    Parameters
    ----------
    cv_results : dict of float arrays of shape (n_splits,)
        Dictionary containing the estimator object
        and train/test indices
        for each run of the cross validation.

    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    sel_idxs : slice
        Indices of X for computing SHAP values.

    Returns
    -------
    shap_results : array-like of shape (n_features)
        Contains the mean absulte SHAP value for each feature.
    '''
    # N_estim = len(cv_results["estimator"])
    # N_samp = X.shape[0]
    N_feat = X.shape[1]  # Number of input features.
    if sel_idxs is None:
        sel_idxs = cv_results["indices"]["test"]  # Select only test indices to reduce computational load.
    shap_results = []
    for idx, estimator_cv in zip(sel_idxs, cv_results["estimator"]):  # Iterate over CV folds.
        explainer = shap.explainers.Permutation(estimator_cv.predict, 
                                                X[idx],
                                                max_evals=N_feat*2+1)  # Use model agnostic Permutation explainer.
        shap_values_cv = explainer(X[idx])
        shap_results.append(shap_values_cv.values)

    shap_results = np.concatenate(shap_results)  # Concatenate all CV folds.
    shap_results = np.mean(np.abs(shap_results), axis=0)  # Average over all samples and folds.

    return shap_results


def compute_importance_maps(
    shap_vals, 
    sources, 
    neigh_adj
):
    """Compute feature importance maps for each input modality.

    Parameters
    ----------
    shap_vals : list of length (n_vertices,)
        List containing arrays with global SHAP values.
        Each array has (n_sources) * (n_vertices within SL) values.

    sources : list of strings
        List with names of sources.

    neigh_adj : sparse lil_matrix of shape (n_vertices, n_vertices)
        Adjacency matrix containing SL neighborhoods.

    Returns
    -------
    shap_vals_avgs : dictionary
        Dictionary where keys are source names 
        and values are spatial maps of feature importance.
    """
    shap_vals_maps = {}
    axis_avg = 0  # Compute average along this axis, 0: Average over SLs overlapping in one vertex., 1: Average within SLs.
    n_sources = len(sources)  # Infer number of sources.
    for n_source, source in enumerate(sources):
        adj_shap = neigh_adj.copy().astype(float)
        for n_v, shap_vals_v in enumerate(shap_vals):
            shap_vals_source = np.split(shap_vals_v, n_sources)[n_source]  # Get shap values of one source.
            adj_shap[n_v, adj_shap.rows[n_v]] = shap_vals_source  # Sort shap values into right position.
        # Average over SLs or vertices.
        shap_vals_avg = adj_shap.sum(axis_avg) / (adj_shap != 0).sum(axis_avg)  
        shap_vals_maps.update({source: np.squeeze(np.array(shap_vals_avg))})
    return shap_vals_maps