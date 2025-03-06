import numpy as np
import scipy
import scipy.spatial
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics
import sklearn.neighbors
from sklearn.metrics.pairwise import cosine_similarity

import torch

def jaccard_index(set1, set2):
    """
    Calculate the Jaccard similarity between two sets.

    Parameters:
        set1 (set): First set of elements.
        set2 (set): Second set of elements.

    Returns:
        float: Jaccard similarity coefficient between set1 and set2.

    Example:
        score = jaccard(set([1,2,3]), set([2,3,4]))
    """

    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union




def neighbor_conservation(orig_data, latent_data, n_neighbors=10, exclude_top=0):
    """
    Calculate biological conservation by comparing neighborhoods in original and latent spaces using average Jaccard similarity.

    Parameters:
        orig_data (np.ndarray): Original feature matrix (samples × features).
        latent_data (np.ndarray): Embedded feature matrix (samples × latent features).
        n_neighbors (int, optional): Number of nearest neighbors to compare. Default is 10.
        exclude_top (int, optional): Number of closest neighbors to exclude (e.g., self). Default is 0.

    Returns:
        float: Mean Jaccard similarity across all samples.
    """
    knn_orig = NearestNeighbors(n_neighbors=n_neighbors + exclude_top)
    knn_latent = NearestNeighbors(n_neighbors=n_neighbors + exclude_top)
    
    knn_orig.fit(orig_data)
    knn_latent.fit(latent_data)
    
    orig_neighbors = knn_orig.kneighbors(orig_data, return_distance=False)
    latent_neighbors = knn_latent.kneighbors(latent_data, return_distance=False)
    
    orig_neighbors = orig_neighbors[:, exclude_top:]
    latent_neighbors = latent_neighbors[:, exclude_top:]
    
    jaccard_scores = []
    for i in range(orig_data.shape[0]):
        orig_set = set(orig_neighbors[i])
        latent_set = set(latent_neighbors[i])
        
        score = jaccard_index(orig_set, latent_set)
        jaccard_scores.append(score)
    
    return np.mean(jaccard_scores)






def foscttm(x, y, device='cpu'):
    """
    Calculate the Fraction of Samples Closer Than True Match (FOSCTTM) to quantify integration accuracy between two embedding spaces.

    Parameters:
        x (np.ndarray or torch.Tensor): First embedding matrix (samples × features).
        y (np.ndarray or torch.Tensor): Second embedding matrix (samples × features).

    Returns:
        tuple: Arrays (foscttm_x, foscttm_y) representing the fraction of samples closer than the true match in both embeddings.
    """

    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y)

    if x.shape != y.shape:
        raise ValueError("Shapes do not match!")

    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        x_sq = torch.sum(x**2, dim=1)  # [N]
        y_sq = torch.sum(y**2, dim=1)  # [N]

        dist_sq = x_sq.unsqueeze(1) + y_sq.unsqueeze(0) - 2 * (x @ y.t())
        d = torch.sqrt(torch.clamp(dist_sq, min=0.0)) 
        diag_d = torch.diag(d)
        foscttm_x = (d < diag_d.unsqueeze(1)).float().mean(dim=1)
        foscttm_y = (d < diag_d.unsqueeze(0)).float().mean(dim=0)

    del x, y, x_sq, y_sq, dist_sq, d, diag_d
    if device != 'cpu':
        torch.cuda.empty_cache()

    return foscttm_x.cpu().numpy(), foscttm_y.cpu().numpy()


def partially_matched_foscttm(
    x: np.ndarray,
    y: np.ndarray,
    x_barcodes: np.ndarray,
    y_barcodes: np.ndarray
):
    """
    Compute FOSCTTM scores for partially matched data embeddings, quantifying how well corresponding cells align across two modalities.

    Parameters:
        x (np.ndarray): Embedding coordinates from dataset X (shape: N_x × D).
        y (np.ndarray): Embedding coordinates from dataset Y (shape: N_y × D).
        x_barcodes (np.ndarray): Cell barcodes from dataset X.
        y_barcodes (np.ndarray): Cell barcodes from dataset Y.

    Returns:
        tuple: (foscttm_x, foscttm_y), arrays indicating the fraction of cells closer than their true matches for each dataset.
    """

    x_barcode_to_index = {barcode: idx for idx, barcode in enumerate(x_barcodes)}
    y_barcode_to_index = {barcode: idx for idx, barcode in enumerate(y_barcodes)}
    
    matched_barcodes = np.intersect1d(x_barcodes, y_barcodes)

    if matched_barcodes.size == 0:
        raise ValueError("No matching barcodes found between x_barcodes and y_barcodes.")

    matched_indices_x = np.array([x_barcode_to_index[barcode] for barcode in matched_barcodes])
    matched_indices_y = np.array([y_barcode_to_index[barcode] for barcode in matched_barcodes])
    
    x_matched = x[matched_indices_x]
    y_matched = y[matched_indices_y]
    
    D_x = scipy.spatial.distance_matrix(x_matched, y)  # shape: (num_matched, N_y)
    D_y = scipy.spatial.distance_matrix(y_matched, x)  # shape: (num_matched, N_x)
    print(f"D_x.shape:{D_x.shape}")
    print(f"D_y.shape:{D_y.shape}")

    d_x_diag = D_x[np.arange(len(matched_indices_x)), matched_indices_y]
    d_y_diag = D_y[np.arange(len(matched_indices_y)), matched_indices_x]

    foscttm_x = (D_x < d_x_diag[:, np.newaxis]).mean(axis=1)
    foscttm_y = (D_y < d_y_diag[:, np.newaxis]).mean(axis=1)
    
    return foscttm_x, foscttm_y





def batch_ASW(latent, modality, celltype, verbose=False, **kwargs):
    """
    Compute batch effect removal effectiveness using the Adjusted Silhouette Width (ASW) per cell type.

    Parameters:
        latent (np.ndarray): Embedded representation of cells (samples × latent features).
        modality (array-like): Batch labels indicating different modalities or batches.
        celltype (array-like): Cell type labels for each cell.
        verbose (bool, optional): Whether to print additional information. Default is False.

    Returns:
        float: Mean ASW score reflecting batch mixing quality.
    """
    s_per_ct = []
    for t in np.unique(celltype):
        mask = celltype == t
        try:
            s = sklearn.metrics.silhouette_samples(latent[mask], modality[mask], **kwargs)
        except ValueError:  # Too few samples
            s = 0
        s = (1 - np.fabs(s)).mean()
        s_per_ct.append(s)
        if verbose:
            print(f"Cell type: {t}, Adjusted silhouette width: {s}")
    return np.mean(s_per_ct).item()



def ct_ASW(latent, celltype, **kwargs):
    """
    Compute the Adjusted Silhouette Width (ASW) to evaluate how well cell types cluster in latent space.

    Parameters:
        latent (np.ndarray): Embedded feature matrix (samples × latent features).
        celltype (array-like): Cell type labels for each cell.

    Returns:
        float: Normalized cell-type ASW score between 0 and 1.
    """
    return (sklearn.metrics.silhouette_score(latent, celltype, **kwargs).item() + 1) / 2




def knn_matching(rna_latent, atac_latent, k=1):
    """
    Assess cross-modal matching accuracy between RNA and ATAC embeddings using k-Nearest Neighbors.

    Parameters:
        rna_latent (np.ndarray): RNA embedding matrix (cells × latent features).
        atac_latent (np.ndarray): ATAC embedding matrix (cells × latent features).
        k (int, optional): Number of neighbors to consider. Default is 1.

    Returns:
        tuple: Accuracy scores for RNA matching, ATAC matching, and overall accuracy.
    """
    similarity_matrix = cosine_similarity(rna_latent, atac_latent)

    # RNA -> ATAC
    rna_to_atac_neighbors = np.argsort(-similarity_matrix, axis=1)[:, :k]
    rna_matches = [i in rna_to_atac_neighbors[i] for i in range(len(rna_latent))]
    rna_correct_matches = np.sum(rna_matches)

    # ATAC -> RNA
    atac_to_rna_neighbors = np.argsort(-similarity_matrix.T, axis=1)[:, :k]
    atac_matches = [i in atac_to_rna_neighbors[i] for i in range(len(atac_latent))]
    atac_correct_matches = np.sum(atac_matches)

    rna_accuracy = rna_correct_matches / len(rna_latent)
    atac_accuracy = atac_correct_matches / len(atac_latent)
    overall_accuracy = (rna_correct_matches + atac_correct_matches) / (len(rna_latent) + len(atac_latent))

    return {
        "rna_to_atac_accuracy": np.round(rna_accuracy,5),
        "atac_to_rna_accuracy": np.round(atac_accuracy,5),
        "overall_accuracy": np.round(overall_accuracy,5)
    }




def cilisi(adata, batch_key, label_key, use_rep, k0=90,n_cores=10, scale=True, type_="embed", verbose=False):
    """
    Compute conditional iLISI (integration metric) per cell type, assessing integration quality conditioned on cell types.

    Parameters:
        adata (AnnData): AnnData object containing embeddings and metadata.
        batch_key (str): Key for modality labels in adata.obs (e.g., 'batch', 'modality').
        label_key (str): Key specifying cell type annotations.
        use_rep (str, optional): Embedding representation to use from adata.obsm. Default is 'X_pca'.
        k0 (int, optional): Number of nearest neighbors for ILISI calculation. Default is 90.
        n_cores (int, optional): Number of CPU cores for computation. Default is 10.
        scale (bool, optional): Whether to scale embeddings. Default is True.
        type_ (str, optional): Data type ('embed' or 'full'). Default is 'embed'.
        verbose (bool, optional): Print detailed output. Default is False.

    Returns:
        dict: ILISI scores per cell type indicating integration quality.
    """

    import scib

    cell_types = adata.obs[label_key].unique()
    ilisi_per_cell_type = []

    for cell_type in cell_types:
        subset_adata = adata[adata.obs[label_key] == cell_type]
        current_k0 = min(k0, int(subset_adata.shape[0]/2))
        if verbose:
            print(f"current_k0:{current_k0}")
        ilisi = scib.me.ilisi_graph(
            subset_adata, n_cores=n_cores, batch_key=batch_key, scale=scale, type_=type_, use_rep=use_rep,k0=current_k0, verbose=False
        )
        ilisi_per_cell_type.append(ilisi)
        if verbose:
            print(f"Cell type '{cell_type}': ilisi: {ilisi:.5f}")
    cilisi = np.nanmean(np.where(np.isfinite(ilisi_per_cell_type), ilisi_per_cell_type, np.nan))
    return cilisi




