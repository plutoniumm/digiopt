from model import Autoencoder, nn, init
from utils import Imageset, nam
import matplotlib.pyplot as plt
import torch, random
import numpy as np

inploader = Imageset("./inputs/*")
outloader = Imageset("./outputs/*")

inlen = [nam(p) for p in inploader.paths]
outlen = [nam(p) for p in outloader.paths]
all_paths = list(set(inlen) & set(outlen))

inlen = len(inlen)
outlen = len(outlen)
totlen = len(all_paths)

model = Autoencoder()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
model.apply(init)

# Phase 1
for epoch in range(2):
    for i in range(len(inploader)):
        img = inploader[i].unsqueeze(0)
        img += torch.randn_like(img) * 0.01

        out = model(img)
        loss = loss_fn(out, img)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"epoch [{epoch+1}/2], loss: {loss.item():.4f}")

opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)
for epoch in range(5):
    i_off, o_off = 0, 0
    i = 0
    loss = 0

    while i in range(totlen):
        path = all_paths[i]
        name = path.split("/")[-1].split(".")[0]
        iidx = inploader.open(name)
        oidx = outloader.open(name)
        if iidx is None or oidx is None:
            i += 1
            continue

        img = iidx.unsqueeze(0)
        oth = oidx.unsqueeze(0)
        if img.shape[2] != oth.shape[2]:
            i += 1
            continue

        img += torch.randn_like(img) * 0.001
        oth += torch.randn_like(oth) * 0.001

        out = model(img)
        loss = loss_fn(out, oth)

        opt.zero_grad()
        loss.backward()
        opt.step()
        i += 1
    # endfor
    print(f"epoch [{epoch+1}/5], loss: {loss.item():.4f}")


def show_tensor(t):
    t = t.squeeze().detach().cpu()
    t = t * 0.3081 + 0.1307
    return t.numpy()


for i in range(5):
    idx = random.randint(0, len(inploader) - 1)
    x = inploader[idx].unsqueeze(0)
    gt = outloader[idx].unsqueeze(0)
    pred = model(x)

    x_np = show_tensor(x)
    pred_np = show_tensor(pred)
    gt_np = show_tensor(gt)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Input", "Prediction", "Ground Truth"]
    images = [x_np, pred_np, gt_np]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"./test/combined_{i}.png")
    plt.close()
