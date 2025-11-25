from transformer import CurveTransformer, get_dataloader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import MulensModel as mm

dl = get_dataloader(
    xy_dir="data/generated_lightcurves/xy",
    param_file="data/generated_lightcurves/params.csv",
    batch_size=32
)

def get_curve(args):
    my_1S2L_model = mm.Model({'t_0': 0, 'u_0': args[0],
                            't_E': args[1], 'rho': args[2], 'q': args[3], 's': args[4],
                            'alpha': args[5]})
    times = my_1S2L_model.set_times()
    lc = my_1S2L_model.get_lc(source_flux=1)
    return np.array(lc)

def get_curves(args):
    return np.array([get_curve(arg) for arg in args])

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

        y_pred = get_curves(pred.tolist())

        y_observed = seqs[:,1]
        loss = nn.SmoothL1Loss()(y_pred, y_observed)

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