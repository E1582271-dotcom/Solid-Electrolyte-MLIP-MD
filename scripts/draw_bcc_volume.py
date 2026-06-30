"""画 bcc 晶胞，直观解释 Fig 3 横坐标「每个 S 的体积」= 骨架松紧度。
并排两个同比例尺的 bcc 盒子：挤(小) vs 松(大)。输出 PNG 到 Obsidian AI4SSB 文件夹。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product, combinations

# Windows 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LIM = 5.6  # 两个子图统一坐标范围，这样小盒子看着真的小、大盒子真的大

def draw_bcc(ax, a, title):
    corners = np.array(list(product([0, a], repeat=3)))     # 8 个角
    center = np.array([a/2, a/2, a/2])                       # 1 个体心
    # 立方体 12 条棱：角之间只差一个坐标的连起来
    for p, q in combinations(corners, 2):
        if np.sum(~np.isclose(p, q)) == 1:
            ax.plot(*zip(p, q), color="0.4", lw=1.2)
    # 8 个角 S（黄）+ 体心 S（橙），marker 大小相同 = 原子物理尺寸一样
    ax.scatter(*corners.T, s=420, c="gold", edgecolors="k", linewidths=0.8, depthshade=True, label="角上的 S (×8, 各算1/8)")
    ax.scatter(*center, s=420, c="orange", edgecolors="k", linewidths=0.8, depthshade=True, label="体心的 S (×1)")
    ax.text(a/2, a/2, a/2+0.45, "中心 S", color="darkred", ha="center", fontsize=9)
    ax.set_title(title, fontsize=12, pad=2)
    ax.set_xlim(-0.3, LIM); ax.set_ylim(-0.3, LIM); ax.set_zlim(-0.3, LIM)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=18, azim=-58)

fig = plt.figure(figsize=(12, 6.2))

a1 = 4.0  # 挤
ax1 = fig.add_subplot(121, projection="3d")
draw_bcc(ax1, a1, f"挤  (Fig3 横轴左边)\n边长 a={a1}Å → 盒子={a1**3:.0f} ų\n每个 S = {a1**3:.0f} ÷ 2 = {a1**3/2:.0f} ų")

a2 = 5.0  # 松
ax2 = fig.add_subplot(122, projection="3d")
draw_bcc(ax2, a2, f"松  (Fig3 横轴右边)\n边长 a={a2}Å → 盒子={a2**3:.0f} ų\n每个 S = {a2**3:.0f} ÷ 2 = {a2**3/2:.1f} ų")

ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
fig.suptitle("bcc 晶胞 = 8个角 + 1个体心 = 共 2 个 S    │    每个 S 的体积 = 盒子体积 ÷ 2  （这就是 Fig3 横坐标）",
             fontsize=13, y=0.97)
fig.text(0.5, 0.03,
         "同一比例尺看：盒子撑大 → 每个 S 占的地盘变大（32→62.5 ų）→ S 之间缝隙变大 → Li 钻过的窗口变大 → 势垒变低（Fig3 曲线往右往下）",
         ha="center", fontsize=10, color="navy")

out = r"/Users/jiajieao/Migrated/obsidian/Notes/AI4SSB/fig_bcc_volume_per_S.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print("已保存:", out)
