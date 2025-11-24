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
      - data/generated_lightcurves/params.csv      (1000 x 6)
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

        # Normalize x and y for stability
        x = xy[:, 0]
        y = xy[:, 1]

        # Normalize x to [0,1]
        x = (x - x.min()) / (x.max() - x.min() + 1e-9)

        # Standardize y
        y_mean = y.mean()
        y_std = y.std() + 1e-9
        y = (y - y_mean) / y_std

        xy_norm = np.stack([x, y], axis=-1).astype(np.float32)

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

        return torch.tensor(xy_norm), torch.tensor(params)


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
