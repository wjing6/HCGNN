import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline

# 读取CSV文件并提取Step和Value数据
def read_csv(file_path, time_path, sep):
    df = pd.read_csv(file_path, sep=sep, header=None)
    df2 = pd.read_csv(time_path, sep=sep, header=None)
    loss = df.iloc[0].tolist()
    cur_time = df2.iloc[0].tolist()
    return cur_time, loss

# 平滑曲线
def exponential_moving_average(data, alpha):
    ema = [data[0]]  # 第一个值保持不变
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return ema

# 读取四个CSV文件
file_paths = ['ogbn-papers100M_loss.csv', 'ogbn-papers100M_loss_Ginex.csv']
time_paths = ['ogbn-papers100M_time.csv', 'ogbn-papers100M_time_Ginex.csv']
label = ['CacheGNN', 'Ginex']
smooth_factor = 0.3  # 平滑程度，可根据需要调节
linewidth = 4
mpl.rcParams['font.size'] = 20
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

fig, ax = plt.subplots(figsize=(10, 6))

for file_path, time_path, l in zip(file_paths, time_paths, label):
    step, loss = read_csv(file_path, time_path, ',')
    step = step[:1000]
    loss = loss[:1000]
    ema_value = exponential_moving_average(loss, smooth_factor)
    ax.plot(step, ema_value, linewidth=linewidth, label=l)

ax.set_xlabel('训练时间（单位：秒）')
ax.set_ylabel('Loss')
ax.set_xlim(left=60)
ax.set_ylim(bottom=1)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig('papers100M_loss.pdf', dpi=100)  # 保存高清晰度图片
plt.show()