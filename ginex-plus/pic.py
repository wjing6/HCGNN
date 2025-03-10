import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

 # 调整字体大小，保证图表清晰
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
mpl.rcParams['font.size'] = 20 
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据设置
users = ['缓存配置 1', '缓存配置 2', '缓存配置 3', '缓存配置 4', '缓存配置 5']
data = {
    '过期阈值: 5': [46.52, 46.48, 46.32, 46.25, 46.28],
    '过期阈值: 10': [46.15, 46.21, 46.18, 46.21, 45.92],
}
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 绘制柱状图
x = np.arange(len(users))
bar_width = 0.3  # 增大柱子宽度
opacity = 0.8     # 增加柱子的透明度

fig, ax = plt.subplots(figsize=(12, 8))

hatch_list = ['', '.', '/']
for i, (label, values) in enumerate(data.items()):
    ax.bar(x + i * bar_width, values, bar_width, alpha=opacity, label=label, color=colors[i], edgecolor='black', hatch=hatch_list[i], zorder=1)

# 添加水平虚线
ax.axhline(y=46.4, color='red', alpha=0.4, linestyle='--', zorder=0)
ax.text(0.5, 46.4, '基线测试准确率', color='red', fontsize=12, verticalalignment='bottom', horizontalalignment='center')

# 设置标签和标题
ax.set_ylabel('测试准确率 (%)', fontsize=16, weight='bold')

# 设置 x 轴刻度和标签
ax.set_xticks(x + bar_width * (len(data) - 1) / 2)
ax.set_xticklabels(users, fontsize=14)

# 优化图例位置
ax.legend(loc='upper right', fontsize=12)

# 设置纵坐标范围
plt.ylim(40, 50)

# 显示网格线
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

# 自动调整布局
plt.tight_layout()

# 保存图片
plt.savefig('papers100M测试.pdf', dpi=300, bbox_inches='tight')  # 高清保存
plt.show()