import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os

### NOTE: THIS IS CHAT-GPT GENERATED ORIGINALLY ###

# ============================================================
#  Dataset
# ============================================================

class LightcurveDataset(Dataset):
    """
    Loads:
      - data/generated_lightcurves/xy/xy_{i}.npy  (N_i, 2)
      - data/generated_lightcurves/params.csv      (2000 x 6)
    """
    def __init__(self, xy_dir, param_file, log_params=False, xy_mu=10.535774064510688, xy_sigma=10.542924743377538):
        self.xy_dir = xy_dir
        self.log_params = log_params
        
        self.xy_mu = np.array([30.26326826166035, 21.617800286628096], dtype=np.float32)
        self.xy_sigma = np.array([73.05752743732106, 0.6507757675785745], dtype=np.float32)
        
        # load all parameter rows
        self.params = np.loadtxt(param_file, delimiter=',')
        self.num_samples = len(self.params)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load curve
        xy = np.load(os.path.join(self.xy_dir, f"xy_{idx}.npy"))  # shape (N, 2)
        xy = xy[:,0] * xy[:,1] # shape (N)
        #xy = (xy - self.xy_mu) / self.xy_sigma

        # Load parameters
        params = self.params[idx].astype(np.float32)

        # log-transform (except alpha!)
        if self.log_params:
            # u0, tE, rho, q, s are positive
            params_log = params.copy()
            params_log[0] = np.log(params[0] + 1e-9)  # u0
            params_log[1] = np.log(params[1] + 1e-9)  # tE
            params_log[2] = np.log(params[2] + 1e-12) # rho
            params_log[3] = np.log(params[3] + 1e-12) # q
            params_log[4] = np.log(params[4] + 1e-9)  # s
            # alpha is angle, do NOT log
            params = params_log

        return torch.tensor(xy.astype(np.float32)), torch.tensor(params)


# ============================================================
#  Collate function for variable-length curves
# ============================================================

def collate_fn(batch):
    """
    batch: list of (seq, params)
       seq: (N_i, 2)
       params: (6,)
    Returns:
       padded: (B, Nmax, 2)
       mask:   (B, Nmax) with 0 for padded positions
       params: (B, 6)
    """
    seqs, params = zip(*batch)

    lengths = [s.shape[0] for s in seqs]

    # pad to (B, Nmax, 2)
    padded = pad_sequence(seqs, batch_first=True)  # (B, Nmax, 2)

    # attention mask: True = keep, False = ignore
    mask = torch.zeros(padded.shape[0], padded.shape[1], dtype=torch.bool)
    for i, L in enumerate(lengths):
        mask[i, :L] = True

    params = torch.stack(params)

    return padded, mask, params

# ============================================================
#  CNN + MLP
# ============================================================
import torch.nn as nn
import torch

class MLP_class(nn.Module):
    def __init__(self, hidden_mlp_dim=128, num_filters=64, kernel_size=3):
        super().__init__()
        point_features = 1
        sequence_length = 1000
        self.target_size = 6
        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=point_features, out_channels=num_filters, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
    
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), 

            nn.Conv1d(in_channels=num_filters * 2, out_channels=num_filters * 4, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            # No MaxPool here, or use AdaptiveAvgPool1d later
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, point_features, sequence_length)
            cnn_output_shape = self.cnn_feature_extractor(dummy_input).shape
            self.flattened_cnn_features = cnn_output_shape[1] * cnn_output_shape[2]
        
        self.mlp_head = nn.Sequential(
            nn.Linear(self.flattened_cnn_features, hidden_mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.2), # Regularization
            nn.Linear(hidden_mlp_dim, hidden_mlp_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_mlp_dim // 2, self.target_size),
            nn.ReLU()
        )

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        cnn_features = self.cnn_feature_extractor(x_permuted)
        flattened_features = cnn_features.view(cnn_features.size(0), -1)
        predictions = self.mlp_head(flattened_features)
        
        return predictions

# ============================================================
#  Transformer Model
# ============================================================

class CurveTransformer(nn.Module):
    """
    Transformer Encoder → pooled embedding → 6 outputs (log-space params)
    """
    def __init__(self, 
                 d_model=128, 
                 n_heads=4, 
                 num_layers=4, 
                 dim_feedforward=256, 
                 dropout=0.1):
        super().__init__()

        # (x,y) → d_model
        self.input_proj = nn.Linear(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # pooled embedding → hidden → 6 outputs
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        self.softplus = nn.Softplus()

    def forward(self, x, mask):
        """
        x:    (B, N, 2)
        mask: (B, N)   True = keep; False = pad
        """
        h = self.input_proj(x)  # (B, N, d)

        # Transformer expects mask with True = pad.
        # Our mask is True = real → invert it.
        src_key_padding_mask = ~mask

        h = self.encoder(
            h,
            src_key_padding_mask=src_key_padding_mask
        )  # (B, N, d)

        # masked mean pooling
        mask_float = mask.unsqueeze(-1).float()  # (B, N, 1)
        h_sum = (h * mask_float).sum(dim=1)
        lengths = mask_float.sum(dim=1)
        h_pooled = h_sum / (lengths + 1e-9)
        raw_out = self.mlp(h_pooled)
        positive_out = self.softplus(raw_out)
        return positive_out


# ============================================================
#  Convenience: load everything
# ============================================================

def get_dataloader(xy_dir, param_file, batch_size=16, shuffle=True):
    dataset = LightcurveDataset(xy_dir, param_file, log_params=False)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


# If running directly (debug)
if __name__ == "__main__":
    dl = get_dataloader(
        xy_dir="data/generated_lightcurves/xy",
        param_file="data/generated_lightcurves/params.csv"
    )
    batch = next(iter(dl))
    seqs, mask, params = batch
    print(seqs.shape, mask.shape, params.shape)

    model = CurveTransformer()
    out = model(seqs, mask)
    print(out.shape)
