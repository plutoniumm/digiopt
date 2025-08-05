from model import Autoencoder, nn, init
import matplotlib.pyplot as plt
from utils import Imageset
import torch, random
import numpy as np

device = "mps"

inploader, inpset = Imageset("./inputs/*")
outloader, outset = Imageset("./outputs/*")

model = Autoencoder().to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
model.apply(init)

# Phase 1
for epoch in range(2):
    for i in range(len(inploader)):
        img = inploader.dataset[i].unsqueeze(0).to(device)
        img += torch.randn_like(img) * 0.01

        out = model(img)
        loss = loss_fn(out, img)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"epoch [{epoch+1}/2], loss: {loss.item():.4f}")

opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)
for epoch in range(5):
    for i in range(len(inploader)):
        img = inploader.dataset[i].unsqueeze(0).to(device)
        oth = outloader.dataset[i].unsqueeze(0).to(device)
        img += torch.randn_like(img) * 0.001
        oth += torch.randn_like(oth) * 0.001

        out = model(img)
        loss = loss_fn(out, oth)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"epoch [{epoch+1}/5], loss: {loss.item():.4f}")


def show_tensor(t):
    t = t.squeeze().detach().cpu()
    t = t * 0.3081 + 0.1307
    return t.numpy()

for i in range(5):
    idx = random.randint(0, len(inpset) - 1)
    x = inploader.dataset[idx].unsqueeze(0).to(device)
    gt = outloader.dataset[idx].unsqueeze(0).to(device)
    pred = model(x)

    x_np = show_tensor(x)
    pred_np = show_tensor(pred)
    gt_np = show_tensor(gt)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Input', 'Prediction', 'Ground Truth']
    images = [x_np, pred_np, gt_np]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'./test/combined_{i}.png')
    plt.close()
