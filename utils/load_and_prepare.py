import sys
from functools import reduce
import glob
import nibabel as nb
import pandas as pd
import numpy as np
from scipy import sparse
import re
from nilearn import masking, image
from .compute_neighborhoods import k_hop_adjacency_fslr, radial_adjacency_fslr, mask_adjacency_mni


def prepare_fusl_data(
    input_df, 
    groups, 
    sources, 
    neigh_adj, 
    labels=[1, -1],
    verbose=True,
    mask=slice(None)
):
    """This function prepartes the data in form of numpy array,
    including the labels and adjacency matrix for the searchlight.


    Parameters
    ----------
    input_df : pandas dataframe
        dataframe with subjects in rows and sources in columns.

    groups : list of strings
        names of groups.

    sources : list of strings
        names of sources used in the multi-modal searchlight.

    neigh_adj : array like of shape (n_vertices, n_vertices)
        adjacency matrix. Defines for each feature the neigbhoring features
        following a given structure of the data.

    labels : list of floats
        labels of the groups, used in the serachlight. Default is [1, -1]

    verbose : boolean
        The verbosity level. Default is True

    Returns
    -------
    X : numpy array of shape (n_samples, n_vertices * n_sources)
        data array for searchlight analysis, contining samples in rows and features in columns.

    y : numpy array of shape (n_samples)
        respective labels for serachlight analysis.

    neigh_adj_sources : array like of shape (n_vertices, n_vertices * n_sources)
        adjacency matrix matrix for multi-source searchlight analysis.    
        n_source-times horizontally concatenated adjacency matrices,
        contains spatial neighborhoods for all sources in one row simultaneously.
    """

    
    if verbose:
        print('Avgerage number of vertices within SLs: {:.1f}'.format(neigh_adj.sum(axis=0).mean()))

    # Generate training data.
    X, neigh_adjs = [], []
    for source in sources:
        X_metr, y = [], []
        for group, label in zip(groups, labels):
            grp_df = input_df.loc[input_df['group'].isin([group])]
            grp_np = np.stack(grp_df[source].to_numpy())
            grp_np = grp_np[:, mask]  # Mask vertices.
            grp_y = [label] * grp_np.shape[0]  # Create labels for each group.
            y.append(grp_y)
            X_metr.append(grp_np)  # Aggregate data for each group.

        X.append(np.concatenate(X_metr))
        y = np.concatenate(y)
        neigh_adj = neigh_adj[mask, mask]  # Mask adjacency matrix.
        neigh_adjs.append(neigh_adj)

    # Concatenate features horizontally.
    X = np.concatenate(X, axis=1)
    X = np.nan_to_num(X)
    
    # Concatenate neighborhood matrices horizontally.
    neigh_adj_sources = sparse.hstack(neigh_adjs) 
    neigh_adj_sources = sparse.lil_matrix(neigh_adj_sources)

    if verbose:
        print('X has shape: {}, FuSL adjacency matrix has shape: {}.'.format(X.shape, neigh_adj_sources.shape))

    return X, y, neigh_adj_sources
    

def load_fusl_cifti_data(
    data_dir, 
    groups, 
    sources, 
    mask=slice(None),
    radius=3,
    world_space=False,
    verbose=False,
):
    """This function loads your cifti data and stores it in
    a pandas dataframe. It additionally computes the neighborhood ajacency matrix.
    This function assumes data of each group is stored in a separate directory:
    /data_dir/group_dir/source_files
    For instance for group 1, subject 1, source 1:
    /my_artificial_data/StN-1.0_ampstd-1_nsub-30_fullovlp/grp-1/source-1_sub-01_ses-1_grp-1.dtseries.nii

    Parameters
    ----------
    data_dir : string
        path to data directory.

    groups : list of strings
        names of groups.

    sources : list of strings
        names of sources used in the analysis.

    verbose : boolean
        The verbosity level. Default is False.

    mask : slice
        Select vertices to include in analysis.

    radius : int
        Radius of searchlight.

    world_space : boolean
        Compute searchlight neighborhood in worldspace (spherical in mms) 
        or use k-hop neighbors.
        
    Returns
    -------
    input_df : pandas Dataframe
        dataframe with subjects in rows and specified sources in columns.

    neigh_adj : array like of shape (n_vertices, n_vertices)
        adjacency matrix. Defines for each feature the neigbhoring features
        following a given structure of the data.
    """
    input_dfs = []
    # Create separate dataframe for each source.
    for source in sources:
        source_dicts = []
        # Iterate over groups.
        for group in groups:
            # Find all source files of specific group and source.
            source_files = glob.glob('{}/{}/{}_*'.format(data_dir, 
                                                         group, 
                                                         source))
            source_files.sort()

            # Load data and save in dictionaries.
            for source_file in source_files:
                sub_id = get_id(source_file, 'sub')
                grp_id = get_id(source_file, 'grp')
                if verbose:
                    print('Load:', source_file)
                data_source = nb.load(source_file).get_fdata()
                data_source = data_source[:, mask]  # Mask data.
                source_dicts.append({'sub': sub_id,
                                     'group': grp_id,
                                     source: data_source})
        input_dfs.append(pd.DataFrame(source_dicts))

    # Concatenate dataframes.
    input_df = reduce(lambda x, y: pd.merge(x, y, on=['sub', 'group']), 
                      input_dfs)

    # Compute adjacency matrix.
    if world_space is True:
        radial_adjacency_fslr(radius)
    else:
        neigh_adj = k_hop_adjacency_fslr(k=radius)
        
    neigh_adj = neigh_adj[mask, mask]  # Mask adjacency matrix.

    return input_df, neigh_adj
    

def load_fusl_nifti_data(
    data_dir, 
    groups, 
    sources, 
    mask=None,
    radius=3,
    world_space=False,
    verbose=False, 
):
    """This function loads your nifti data and stores it in
    a pandas dataframe. It additionally computes the neighborhood ajacency matrix.
    This function assumes data of each group is stored in a separate directory:
    /data_dir/group_dir/source_files
    For instance for group 1, subject 1, source 1:
    /my_artificial_data/StN-1.0_ampstd-1_nsub-30_fullovlp/grp-1/source-1_sub-01_ses-1_grp-1.dtseries.nii

    Parameters
    ----------
    data_dir : string
        path to data directory.

    groups : list of strings
        names of groups.

    sources : list of strings
        names of sources used in the analysis.

    verbose : boolean
        The verbosity level. Default is False.

    mask : slice
        Select vertices to include in analysis.

    radius : int
        Radius of searchlight.

    world_space : boolean
        Compute searchlight neighborhood in worldspace (spherical in mms) 
        or use k-hop neighbors.
        
    Returns
    -------
    input_df : pandas Dataframe
        dataframe with subjects in rows and specified sources in columns.

    neigh_adj : array like of shape (n_vertices, n_vertices)
        adjacency matrix. Defines for each feature the neigbhoring features
        following a given structure of the data.
        
    mask : Nifti1Image
        3D mask image.
    """

    input_dfs = []
    # Create separate dataframe for each source.
    for source in sources:
        source_dicts = []
        # Iterate over groups.
        for group in groups:
            # Find all source files of specific group and source.
            source_files = glob.glob('{}/{}/{}_*'.format(data_dir, 
                                                         group, 
                                                         source))
            source_files.sort()

            # Load data and save in dictionaries.
            for source_file in source_files:
                sub_id = get_id(source_file, 'sub')
                grp_id = get_id(source_file, 'grp')
                if verbose:
                    print('Load:', source_file)
                # Modify this part if you want to load nifti data.
                source_nii = nb.load(source_file)

                # Load GM mask if none is provided.
                if mask is None:
                    mask = masking.compute_brain_mask(source_nii,
                                                      mask_type='gm',
                                                      threshold=0.15, 
                                                      connected=False)
        
                data_source = masking.apply_mask_fmri(source_nii, mask)
                source_dicts.append({'sub': sub_id,
                                     'group': grp_id,
                                     source: data_source})
        input_dfs.append(pd.DataFrame(source_dicts))

    # Concatenate dataframes.
    input_df = reduce(lambda x, y: pd.merge(x, y, on=['sub', 'group']), 
                      input_dfs)

    # Compute adjacency matrix.
    neigh_adj = mask_adjacency_mni(mask, 
                                   radius=radius, 
                                   world_space=world_space)

    return input_df, neigh_adj, mask


def reshape_3d_img(mask, array, output_nii=False): 
    """
    Reshapes data array back into 3d image, based on mask.

    Parameters
    ----------
    mask : Nifti1Image or numpy array
        Mask image.

    array : numpy array
        Data array.

    verbose : boolean
        Convert image to Nifti1Image or not. Default is False

    Returns
    -------
    array_3D : Nifti1Image or numpy array
        3d image, populated with values of array.
    """
    if type(mask) is nb.nifti1.Nifti1Image:
        mask_arr, _ = masking.load_mask_img(mask)
    else:
        mask_arr = mask
    array_3D = np.zeros(mask_arr.shape)
    array_3D[mask_arr] = array
    if output_nii is True and type(mask) is nb.nifti1.Nifti1Image:
        array_3D = image.new_img_like(mask, array_3D)
    else:
        print('Provided mask no Nifti1image.')
    return array_3D


def get_id(
    func_name, 
    keyword, 
    strip_keyword=False, 
    stop_at=['_', '.']
):
    """
    Returns for example subject, session or source ID from BIDS-like filename.
    
    Parameters
    ----------
    func_name : str
        Filename or full filepath to fmriprep output.
    keyword : str
        Keyword of ID, for example 'sub', 'ses' or 'source'.
        
    Examples
    --------
    sub_id = get_id('sub-02_ses-0_task-rest_bold.nii.gz', 'sub')
    
    """
    file_name = func_name.split('/')[-1]   # Remove path.
    stop_str = '|'.join(stop_at)
    res = re.findall(r"({}-.*?)[{}]".format(keyword, stop_str), file_name)
    if strip_keyword:
        res = re.findall(r"(?<=-).*", res[0])
    return res[0]