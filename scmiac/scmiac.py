import numpy as np
from .integration_anchors import SeuratIntegration, get_integration_features

import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, WeightedRandomSampler
from .integration_dataset import MultiOmicDataset, AnchorCellsDataset
from .integration_model import VAE
from .integration_loss import NTXentLoss2, VAELoss
import torch.optim as optim
import torch.nn.functional as F





def find_anchors(
    adata_rna,
    adata_atac,
    all_nfeatures=3000,
    single_nfeatures=2000,
    k_anchor=5,
    n_components=30,
    ct_filter=True
):
    """
    Find anchors between RNA and ATAC data for integration.

    Parameters:
    - adata_rna: AnnData object containing RNA data.
    - adata_atac: AnnData object containing ATAC data.
    - all_nfeatures: Number of features to use for integration (default: 3000).
    - single_nfeatures: Number of features to use for individual datasets (default: 2000).
    - k_anchor: Number of neighbors to use when picking anchors (default: 5).
    - n_components: Number of components for dimensionality reduction (default: 30).
    - ct_filter: Whether to filter anchors by cell type (default: True).

    Returns:
    - adata_rna: Updated AnnData object with RNA embeddings.
    - adata_atac: Updated AnnData object with ATAC embeddings.
    - anchor_df: DataFrame containing anchor information.
    """
    # Step 1: Get integration features
    adata_list = get_integration_features(
        [adata_rna, adata_atac],
        all_nfeatures=all_nfeatures,
        single_nfeatures=single_nfeatures
    )
    
    # Step 2: Initialize SeuratIntegration
    integrator = SeuratIntegration()
    
    # Step 3: Find anchors
    integrator.find_anchor(
        adata_list,
        k_local=None,
        key_local="X_pca",
        k_anchor=k_anchor,
        key_anchor="X",
        dim_red="cca",
        max_cc_cells=50000,  
        k_score=30,          
        scale1=True,         
        scale2=True,         
        n_components=n_components,
        k_filter=None,       
        n_features=200,      
        alignments=[[[0], [1]]]  
    )
    
    # Step 4: Process anchor DataFrame
    anchor_df = integrator.anchor[(0, 1)]
    anchor_df['x1_ct'] = adata_rna.obs['cell_type'].values[anchor_df['x1'].values]
    anchor_df['x2_ct'] = adata_atac.obs['pred'].values[anchor_df['x2'].values]
    anchor_df['is_same'] = anchor_df['x1_ct'].values == anchor_df['x2_ct'].values
    anchor_df = anchor_df[anchor_df['score'] > 0.2].reset_index(drop=True)
    print(f"Number of anchor pairs: {anchor_df.shape[0]}")

    if ct_filter:
        anchor_df = anchor_df[anchor_df['is_same']]
        print(f"Number of anchors pairs after cell type filtering: {anchor_df.shape[0]}")    
    
    rna_is_anchor_num = len(np.unique(anchor_df['x1'].values))
    atac_is_anchor_num = len(np.unique(anchor_df['x2'].values))
    print(f"Number of RNA anchors: {rna_is_anchor_num}; Number of ATAC anchors: {atac_is_anchor_num}")
    
    # Step 5: Update embeddings in AnnData objects
    rna_embeddings = integrator.U
    atac_embeddings = integrator.V
    adata_rna.obsm['cca'] = rna_embeddings
    adata_atac.obsm['cca'] = atac_embeddings
    
    return anchor_df









def preprocess(adata_rna, 
               adata_atac,
               anchor_df,
               rna_latent_key = "X_pca",
               atac_latent_key = "lsi49",
               batch_size=1024, 
               hidden_dims=[128, 64], 
               latent_dim=30,
               balanced_sampler=True,
               device = 'cuda:0'):
    """
    Preprocess RNA and ATAC data for integration.
    
    Parameters:
    - adata_rna: AnnData object containing RNA data.
    - adata_atac: AnnData object containing ATAC data.
    - anchor_df: DataFrame containing anchor information.
    - batch_size: Batch size for DataLoader (default: 1024).
    - balanced_sampler: Whether to use a balanced sampler for anchor cells (default: True).

    Returns:
    - all_cells_loader: DataLoader for all cells.
    - anchor_cells_loader: DataLoader for anchor cells.
    """

    RNA_arr = adata_rna.obsm[rna_latent_key]
    ATAC_arr = adata_atac.obsm[atac_latent_key]

    # Create datasets
    all_cells_dataset = MultiOmicDataset(RNA_arr, ATAC_arr)
    anchor_cells_dataset = AnchorCellsDataset(RNA_arr, ATAC_arr, anchor_df)

    # DataLoader for all cells
    all_cells_loader = DataLoader(all_cells_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # DataLoader for anchor cells
    if balanced_sampler:
        cell_types = anchor_df['x1_ct']
        type_weights = 1.0 / (cell_types.value_counts())
        sample_weights = cell_types.map(type_weights)
        sampler = WeightedRandomSampler(weights=sample_weights.values, num_samples=len(sample_weights), replacement=True)
        anchor_cells_loader = DataLoader(anchor_cells_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, drop_last=True)
    else:
        anchor_cells_loader = DataLoader(anchor_cells_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    rna_vae = VAE(input_dim=RNA_arr.shape[1], hidden_dims=hidden_dims, latent_dim=latent_dim, dropout_rate=0.0).to(device)
    atac_vae = VAE(input_dim=ATAC_arr.shape[1], hidden_dims=hidden_dims, latent_dim=latent_dim, dropout_rate=0.0).to(device)

    return rna_vae, atac_vae, all_cells_loader, anchor_cells_loader






def train_model(rna_vae, 
                atac_vae, 
                all_cells_loader, 
                anchor_cells_loader, 
                device = "cuda:0", 
                num_epoches = 20000,
                lambda_rna_kl  = 1, 
                lambda_atac_kl = 1, 
                alpha_rna_rec  = 20, 
                alpha_atac_rec = 20, 
                lambda_contra  = 300, 
                temperature = 0.5, 
                lr = 1e-3,
                print_step = 10,
                save_model = False):
    """
    Train RNA and ATAC VAE models using contrastive learning.

    Parameters:
    - rna_vae: RNA VAE model.
    - atac_vae: ATAC VAE model.
    - all_cells_loader: DataLoader for all cells.
    - anchor_cells_loader: DataLoader for anchor cells.
    - device: Device to run the model on ('cuda' or 'cpu').
    - num_epoches: Number of training epochs.
    - lambda_rna_kl: Weight for RNA KL divergence loss.
    - lambda_atac_kl: Weight for ATAC KL divergence loss.
    - alpha_rna_rec: Weight for RNA reconstruction loss.
    - alpha_atac_rec: Weight for ATAC reconstruction loss.
    - lambda_contra: Weight for contrastive loss.
    - temperature: Temperature parameter for contrastive loss.
    - lr: Learning rate.
    - ft_rna_vae_file: Path to save the fine-tuned RNA VAE model.
    - ft_atac_vae_file: Path to save the fine-tuned ATAC VAE model.
    """

    # Loss functions and optimizer
    contr_loss_fn = NTXentLoss2(temperature=temperature)
    vae_loss_fn = VAELoss()
    optimizer = optim.Adam(list(rna_vae.parameters()) + list(atac_vae.parameters()), lr=lr)

    # Training loop
    for epoch in range(num_epoches):
        total_loss = 0
        total_rna_recon_loss = 0
        total_rna_kld_loss = 0
        total_atac_recon_loss = 0
        total_atac_kld_loss = 0
        total_contrastive_loss = 0

        rna_vae.train()
        atac_vae.train()

        for full_batch, anchor_batch in zip(all_cells_loader, anchor_cells_loader):
            rna_feat = full_batch["rna"].to(device)
            atac_feat = full_batch["atac"].to(device)
            rna_anchor = anchor_batch["rna_anchor"].to(device)
            atac_anchor = anchor_batch["atac_anchor"].to(device)

            # Forward pass through VAEs
            rna_recon, rna_mu, rna_logvar = rna_vae(rna_feat)
            atac_recon, atac_mu, atac_logvar = atac_vae(atac_feat)

            # Compute VAE losses (reconstruction + KL divergence)
            rna_recon_loss, rna_kld_loss = vae_loss_fn(rna_recon, rna_feat, rna_mu, rna_logvar)
            atac_recon_loss, atac_kld_loss = vae_loss_fn(atac_recon, atac_feat, atac_mu, atac_logvar)

            # Weighted losses
            rna_recon_loss *= alpha_rna_rec
            atac_recon_loss *= alpha_atac_rec
            rna_kld_loss *= lambda_rna_kl
            atac_kld_loss *= lambda_atac_kl

            # Contrastive loss (using anchor pairs)
            rna_mu_anchor, _ = rna_vae.encode(rna_anchor)
            atac_mu_anchor, _ = atac_vae.encode(atac_anchor)
            contrastive_loss = contr_loss_fn(rna_mu_anchor, atac_mu_anchor) * lambda_contra

            # Total loss
            loss = atac_recon_loss + atac_kld_loss + rna_recon_loss + rna_kld_loss + contrastive_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_rna_recon_loss += rna_recon_loss.item()
            total_rna_kld_loss += rna_kld_loss.item()
            total_atac_recon_loss += atac_recon_loss.item()
            total_atac_kld_loss += atac_kld_loss.item()
            total_contrastive_loss += contrastive_loss.item()

        # Print logs
        if epoch % print_step == 0:
            print(f"Epoch {epoch}, Total: {total_loss / len(all_cells_loader):.2f}, "
                  f"RNA Reco: {total_rna_recon_loss / len(all_cells_loader):.2f}, "
                  f"RNA KLD: {total_rna_kld_loss / len(all_cells_loader):.2f}, "
                  f"ATAC Reco: {total_atac_recon_loss / len(all_cells_loader):.2f}, "
                  f"ATAC KLD: {total_atac_kld_loss / len(all_cells_loader):.2f}, "
                  f"Contra: {total_contrastive_loss / len(all_cells_loader):.2f}")

    if save_model:
        torch.save(rna_vae.state_dict(), './model/rna_vae.pth')
        torch.save(atac_vae.state_dict(), './model/atac_vae.pth')

    return rna_vae, atac_vae








def model_inference(rna_vae, 
                    atac_vae, 
                    all_cells_loader, 
                    device = 'cpu'):
    """
    Generate embeddings for RNA and ATAC data using trained VAE models.

    Parameters:
    - rna_vae: Trained RNA VAE model.
    - atac_vae: Trained ATAC VAE model.
    - all_cells_dataset: Dataset containing RNA and ATAC data.
    - device: Device to run the model on ('cuda' or 'cpu').

    Returns:
    - rna_embeddings: Normalized RNA embeddings.
    - atac_embeddings: Normalized ATAC embeddings.
    """
    # Convert data to tensors
    scRNA_features = torch.tensor(all_cells_loader.dataset.rna_data, dtype=torch.float32)
    scATAC_features = torch.tensor(all_cells_loader.dataset.atac_data, dtype=torch.float32)

    # Generate embeddings
    rna_vae.to(device)
    atac_vae.to(device)
    rna_vae.eval()
    atac_vae.eval()
    with torch.no_grad():
        rna_embeddings, _ = rna_vae.encode(scRNA_features.to(device))
        atac_embeddings, _ = atac_vae.encode(scATAC_features.to(device))
        rna_embeddings = F.normalize(rna_embeddings, dim=1).cpu().numpy()
        atac_embeddings = F.normalize(atac_embeddings, dim=1).cpu().numpy()

    return rna_embeddings, atac_embeddings









