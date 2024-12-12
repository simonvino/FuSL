import numpy as np
import hcp_utils as hcp
from nilearn import surface, masking
from nilearn.image.resampling import coord_transform
from sklearn import neighbors
from scipy import sparse


def k_hop_adjacency_fslr(k, 
                         adj=hcp.cortical_adjacency
): 
    """Compute adjacency matrix inlcuding all neighboors
    of order k of all vertices in fsLR space.
    """
    # Compute adj matrix containing k-hop neighbors.
    neigh_adj = adj.copy()
    for order in range(k-1):
        neigh_adj = neigh_adj.dot(neigh_adj)
        neigh_adj[neigh_adj != 0] = 1  # Binarize adjacency matrix.

    return sparse.lil_matrix(neigh_adj)


def radial_adjacency_fslr(radius, 
                          mesh_left=hcp.mesh.midthickness_left,
                          mesh_right=hcp.mesh.midthickness_right,
                          add_subcortical=False
):
    """Compute adjacency matrix inlcuding all neighboors
    within the radius of all vertices in fsLR space.    
    """
    # Load vertex coordinates.
    coords_left, _ = surface.load_surf_mesh(mesh_left)  # Load mesh.
    coords_left = coords_left[hcp.vertex_info['grayl']]  # Vertices included in ciftis.
    coords_right, _ = surface.load_surf_mesh(mesh_right)
    coords_right = coords_right[hcp.vertex_info['grayr']]

    # Merge left and right vertex coordinates.
    coords = np.concatenate([coords_left, coords_right])

    # Add subcortical voxels. 
    if add_subcortical is True:
        coords_sub = np.load('./own_data/fsLR_coords_subcortical.npy')
        coords = np.concatenate([coords, coords_sub])

    # Get neighborhoods.
    nn = neighbors.NearestNeighbors(radius=radius)
    neigh_adj = nn.fit(coords).radius_neighbors_graph(coords)

    return neigh_adj.tolil()


def mask_adjacency_mni(mask, 
                       radius,
                       world_space=True
):
    """Compute adjacency matrix based on mask in mni space.    
    """
    mask_arr, mask_affine = masking.load_mask_img(mask)
    mask_arr_coords = np.where(mask_arr != 0)
    # Compute world coordinates of the seeds
    if world_space is True:
        mask_arr_coords = coord_transform(
            mask_arr_coords[0],
            mask_arr_coords[1],
            mask_arr_coords[2],
            mask_affine,
        )
    mask_arr_coords = np.asarray(mask_arr_coords).T
    
    nn = neighbors.NearestNeighbors(radius=radius)
    neigh_adj = nn.fit(mask_arr_coords).radius_neighbors_graph(mask_arr_coords)

    return neigh_adj.tolil()