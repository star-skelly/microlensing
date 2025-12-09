from models import MLP_class, get_dataloader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np
from test import get_curve, calculate_rmse

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
criterion = nn.SmoothL1Loss(reduction='none')
for epoch in range(num_epochs):
    opt.zero_grad()
    for step, (seqs, mask, params) in enumerate(dl):
        seqs = seqs.to('cuda')
        params = params.to('cuda')
        per_item_loss = []
        pred = model(seqs)
        for i, pr in enumerate(pred):
            try:
                pred_x, pred_y = get_curve(pr.tolist()) # in very rare cases, the model fails to solve this
            except:
                pred_x = pred_y = torch.zeros(1000)
            
            pred_curve = torch.tensor(pred_x[:999] * pred_y[:999], requires_grad=True).to('cuda')
        
            true_x, true_y = get_curve(params[i].tolist())
            true_curve = torch.tensor(true_x[:999] * true_y[:999], requires_grad=True).to('cuda')
            per_item_loss.append(criterion(pred_curve, true_curve))
        
        per_item_loss = torch.stack(per_item_loss)

        loss = per_item_loss.mean()
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