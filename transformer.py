import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os

### NOTE: THIS IS DERIVED FROM A SCRIPT FROM GENERATIVE AI ###

# ============================================================
#  Dataset
# ============================================================

class LightcurveDataset(Dataset):
    """
    Loads:
      - data/generated_lightcurves/xy/xy_{i}.npy  (N_i, 2)
      - data/generated_lightcurves/params.csv      (2000 x 6)
    """
    def __init__(self, xy_dir, param_file, log_params=True):
        self.xy_dir = xy_dir
        self.log_params = log_params

        # load all parameter rows
        self.params = np.loadtxt(param_file, delimiter=',')
        self.num_samples = len(self.params)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load curve
        xy = np.load(os.path.join(self.xy_dir, f"xy_{idx}.npy"))  # shape (N, 2)

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

        return self.mlp(h_pooled)


# ============================================================
#  Convenience: load everything
# ============================================================

def get_dataloader(xy_dir, param_file, batch_size=16, shuffle=True):
    dataset = LightcurveDataset(xy_dir, param_file, log_params=True)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
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
