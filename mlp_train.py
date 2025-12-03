from models import MLP_class, get_dataloader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np

dl = get_dataloader(
    xy_dir="data/generated_lightcurves/xy",
    param_file="data/generated_lightcurves/params.csv",
    batch_size=32
)

model = MLP_class().cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(opt, step_size=5, gamma=0.5)

num_epochs = 100
loss_hist = []
print(f"Training Beginning for {num_epochs} epochs")
for epoch in range(num_epochs):
    opt.zero_grad()
    for step, (seqs, mask, params) in enumerate(dl):
        seqs = seqs.cuda()
        params = params.cuda()

        pred = model(seqs)

        loss = nn.SmoothL1Loss()(pred, params)

        loss.backward()
        opt.step()
        
    scheduler.step()

    print(f"epoch {epoch} | loss {loss.item():.5f}")
    loss_hist.append(loss.item())

print("Training complete!")
torch.save(model, "mlp_chkpt.pth")

plt.plot(range(num_epochs), loss_hist)
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.savefig(f'figures/loss_{np.mean(loss_hist)}.png')