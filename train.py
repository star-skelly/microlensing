from transformer import CurveTransformer, get_dataloader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

### NOTE: THIS WAS WRITTEN WITH THE HELP OF GENERATIVE AI ###

dl = get_dataloader(
    xy_dir="data/generated_lightcurves/xy",
    param_file="data/generated_lightcurves/params.csv",
    batch_size=32
)

model = CurveTransformer().cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 20
loss_hist = []
for epoch in range(num_epochs):
    opt.zero_grad()
    print(f"epoch {epoch}")
    for step, (seqs, mask, params) in enumerate(dl):
        seqs = seqs.cuda()
        mask = mask.cuda()
        params = params.cuda()

        pred = model(seqs, mask)

        loss = nn.SmoothL1Loss()(pred, params)

        loss.backward()
        opt.step()

        if step % 10 == 0:
            print(f"step {step} loss {loss.item():.5f}")
    loss_hist.append(loss.item())

print("Training complete!")
torch.save(model, "chkpt.pth")

plt.plot(range(num_epochs), loss_hist)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig('figures/loss.png')