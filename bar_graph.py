# -*- coding: utf-8 -*-
"""
使用 Plotly 从 /mnt/data/irreps_frequency.csv 读取数据，绘制蓝色柱状图，
在每个柱子上方标注 count，并保存为 PDF（/mnt/data/irreps_frequency_barplot.pdf）。
- 若 CSV 只有类别列：自动做频数统计
- 若存在数值列：按类别对该数值列求和
- 若存在名为 count/frequency/freq/n 的数值列：优先使用
"""

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.kaleido.scope.mathjax = None
# -------- 参数 --------
csv_path = Path("/home/mila/w/weis/DL/group-training-refactored/coset_freq.csv")
out_pdf = Path("/home/mila/w/weis/DL/group-training-refactored/coset_freq_barplot.pdf")

# -------- 读取数据 --------
df = pd.read_csv(csv_path)

# -------- 自动识别列 --------
num_cols = df.select_dtypes(include="number").columns.tolist()
preferred_num = [c for c in df.columns if c.lower() in {"count", "frequency", "freq", "n"} and pd.api.types.is_numeric_dtype(df[c])]
cat_candidates = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

if preferred_num:
    count_col = preferred_num[0]
    # 类别列：优先第一个非数值列；若没有，就取除 count_col 外的第一列
    if cat_candidates:
        cat_col = cat_candidates[0]
    else:
        cat_col = [c for c in df.columns if c != count_col][0]
    df["_order"] = pd.factorize(df[cat_col], sort=False)[0]
    agg = (
        df.groupby([cat_col, "_order"], sort=False)[count_col]
        .sum()
        .reset_index()
        .sort_values("_order")
        .rename(columns={cat_col: "category", count_col: "count"})
    )
elif num_cols and cat_candidates:
    count_col = num_cols[0]
    cat_col = cat_candidates[0]
    df["_order"] = pd.factorize(df[cat_col], sort=False)[0]
    agg = (
        df.groupby([cat_col, "_order"], sort=False)[count_col]
        .sum()
        .reset_index()
        .sort_values("_order")
        .rename(columns={cat_col: "category", count_col: "count"})
    )
elif not num_cols:
    # 只有类别列：对第一列做频数统计，保持首次出现顺序
    cat_col = df.columns[0]
    order = pd.unique(df[cat_col])
    vc = df[cat_col].value_counts(sort=False)
    agg = pd.DataFrame({"category": order.astype(str), "count": [int(vc.get(x, 0)) for x in order]})
else:
    # 兜底：第一列当类别、第一数值列当 count
    cat_col = df.columns[0]
    count_col = num_cols[0]
    df["_order"] = pd.factorize(df[cat_col], sort=False)[0]
    agg = (
        df.groupby([cat_col, "_order"], sort=False)[count_col]
        .sum()
        .reset_index()
        .sort_values("_order")
        .rename(columns={cat_col: "category", count_col: "count"})
    )

# 转字符串，避免类别为数值时被当作数值轴处理
agg["category"] = agg["category"].astype(str)

# -------- 计算画布尺寸与文本格式 --------
n_cat = len(agg)
fig_width = min(600, 15 * n_cat + 100)  # 随类别数自适应宽度（像素）
fig_height = 300

all_int = np.allclose(agg["count"].values, np.round(agg["count"].values))
text_template = "%{y:.0f}" if all_int else "%{y:.2f}"

# 为了给顶部文本留空间，适度增加 y 轴上限
ymax = float(agg["count"].max()) if n_cat > 0 else 0.0
yrange = [0, ymax * (1.15 if ymax > 0 else 1.0)]

# -------- 绘图（蓝色柱 + 顶部标注）--------
fig = px.bar(
    agg,
    x="category",
    y="count",
    text="count",
    title="Counts of the cayley graph learned by neurons over 1000 seeds; p=18",
    template="plotly_white",
)

fig.update_traces(
    marker_color="lightblue",              # 蓝色柱
    texttemplate=text_template,       # 顶部标注格式
    textposition="outside",
    cliponaxis=False,                 # 避免标注被坐标轴裁剪
)

# 轴与布局美化
fig.update_layout(
    xaxis_title="Cayley graph learned by a neuron",
    yaxis_title="Count",
    width=fig_width,
    height=fig_height,
    margin=dict(l=60, r=30, t=60, b=80),
    bargap=0,
)
# if n_cat > 8:
#     fig.update_layout(xaxis=dict(tickangle=-45))
if ymax > 0:
    fig.update_yaxes(range=yrange)

fig.update_layout(font=dict(size=12))

# 图标题字号
fig.update_layout(title=dict(font=dict(size=14)))  # 比如 14

# 坐标轴标题与刻度字号
fig.update_xaxes(title_font=dict(size=12), tickfont=dict(size=12))
fig.update_yaxes(title_font=dict(size=12), tickfont=dict(size=12))

# 柱顶文本（count 标签）字号
fig.update_traces(textfont=dict(size=8))

# -------- 保存为 PDF --------
try:
    fig.write_image(str(out_pdf), format="pdf")
except ValueError as e:
    # 若缺少 kaleido，自动安装后再尝试一次
    import sys, subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaleido"])
        fig.write_image(str(out_pdf), format="pdf")
    except Exception as ee:
        raise RuntimeError(
            f"导出 PDF 失败：{e}\n尝试自动安装 kaleido 也失败：{ee}\n"
            "请在环境中手动安装 kaleido 后重试：pip install kaleido"
        )

print(f"Saved to: {out_pdf}")
