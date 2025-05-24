import numpy as np
import matplotlib.pyplot as plt


exclude = [
    (0, 0), (8, 8), (0, 8), (8, 0),  # corners
    (1, 1), (1, 7), (7, 1), (7, 7),
    (0, 1), (1, 0), (0, 7), (7, 0), (0, 2), (2, 0), (0, 6), (6, 0),  # edges
    (8, 1), (1, 8), (8, 7), (7, 8), (8, 2), (2, 8), (8, 6), (6, 8),  # edges
    (5, 8), (8, 5), (5, 0), (0, 5),
    (0, 3), (3, 0), (3, 8), (8, 3),
    (2, 5), (3, 2), (3, 6), (4, 4), (4, 5), (5, 4), (5, 6)  # manual
]
xy = np.array([(i, j) for i in range(9) for j in range(9) if (i, j) not in exclude])
x, y = xy[:, 0], xy[:, 1]
c = plt.get_cmap("viridis")((x + y) / 13.5)
facecolor = "#0A0E17"
text = dict(
    color='white',
    s=r"Heatmap$\text{TS}$",
    fontweight='light',
    # s=r"Heatmap$\bf{TS}$",
    # fontweight='light',
    # s=r"Heatmap$\bf{TS}$",
    # fontweight='normal',
)


def logo_square():
    shift_up = 3.5
    pad = 3

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor(facecolor)
    ax.axis('off')

    ax.scatter(x, y + shift_up, c=c, s=100)

    ax.text(8 / 2, 0, **text, fontsize=26, ha='center', va='bottom')
    ax.set_xlim(-pad - shift_up / 2, 8 + pad + shift_up / 2)
    ax.set_ylim(-pad, 8 + pad + shift_up)

    fig.savefig('docs/logo.png', dpi=300, bbox_inches='tight', pad_inches=0.5)


def logo_wide():
    pad = 1
    fig, ax = plt.subplots(figsize=(2.4, 8), subplot_kw={'aspect': 'equal'})
    fig.patch.set_facecolor(facecolor)
    ax.axis('off')

    ax.scatter(x, y, c=c, s=100)

    ax.text(8 + 1.5, 8 / 2, **text, fontsize=32, ha='left', va='center')
    ax.text(8 + 1.5 + 14, 8 / 2, r" ", fontsize=32, alpha=0)
    ax.set_xlim(-pad / 2 - 0.8, 8 + pad / 2)
    ax.set_ylim(-pad, 8 + pad)

    fig.savefig('docs/logo_wide.png', dpi=300, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    logo_square()
    logo_wide()
