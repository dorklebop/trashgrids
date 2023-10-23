import wandb
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True

params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern Serif"]}
plt.rcParams.update(params)

colors = ["forestgreen", "limegreen"]
fontsize = 20
lw = 3.0

fig, ax = plt.subplots(figsize=(14,10))

ax.hlines(y=0.0, xmin=0.0, xmax=2*np.pi, lw=lw, color="red")

api = wandb.Api()

# invariant model
run_invariant = api.run("ck-experimental/dense_point_clouds/hrm3nj4i")


history = run_invariant.scan_history()

for item in history:
    if item["epoch"] is not None:
        epoch = item["epoch"]
    if item["invariance_loss"] is not None:
        losses = item["invariance_loss"]
        angles = item["angles"]

#         print(item["global_step"])

# invariance_loss, angles = history[-1]["invariance_loss"], history[-1]["angles"]

ax.plot(angles, losses, c=colors[0], lw=lw, label=r"$SO(3)$ Invariant")

# non-equivariant model
run = api.run("ck-experimental/dense_point_clouds/7jrby70e")

history = run.scan_history()

for item in history:
    if hasattr(item, "epoch"):
        if item["epoch"] is not None:
            epoch = item["epoch"]
            if epoch == "26" or epoch == 26:
                break
    if item["invariance_loss"] is not None:
        losses = item["invariance_loss"]
        angles = item["angles"]

ax.plot(angles, losses, c=colors[1], lw=lw, label=r"Non-equivariant")

ax.set_xticks([0.0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi], labels=[r"$0$", r"$\frac{1}{2}\pi$", r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"], fontsize=fontsize)
plt.grid()
plt.yticks(fontsize=fontsize)
ax.legend(fontsize=fontsize)
plt.xlim(0.0, 2*np.pi)


plt.show()

# circle = np.linspace(0, 2*np.pi, 100)
#
# fig = plt.figure()
# plt.plot(np.cos(circle), np.sin(circle), c="red")
#
# start_x = np.cos(0.0)
# start_y = np.sin(0.0)
#
# for loss, angle in zip(losses, angles):
#
#     x = np.cos(angle) * (1+loss)
#     y = np.sin(angle) * (1+loss)
#
#     plt.plot([start_x, x], [start_y, y], c="blue")
#
#     start_x = x
#     start_y = y
#
# plt.show()
# print(losses)

# print(epoch)
# plt.plot(angles, invariance_loss)
# plt.show()

