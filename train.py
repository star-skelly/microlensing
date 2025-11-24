from transformer import CurveTransformer, get_dataloader
import torch
import torch.nn as nn

dl = get_dataloader(
    xy_dir="data/generated_lightcurves/xy",
    param_file="data/generated_lightcurves/params.csv",
    batch_size=32
)

model = CurveTransformer().cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

for step, (seqs, mask, params) in enumerate(dl):
    seqs = seqs.cuda()
    mask = mask.cuda()
    params = params.cuda()

    pred = model(seqs, mask)

    loss = nn.MSELoss()(pred, params)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.5f}")
