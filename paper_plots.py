import io, json, numpy as np, plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from PyPDF2 import PdfReader
import plotly.io as pio
pio.kaleido.scope.mathjax = None
import report
import pca_diffusion_plots_w_helpers
import color_rules
# ---------- 用 JSON 重建同一页图，并可直接导出 PDF ----------
## preacts GFT
def render_neuron_page_from_json(
    page: dict,
    to_pdf_path: str | None = None
) -> go.Figure:
    # 拿出字段
    names = page["heatmaps"]["a_dft"].get("x", [])
    layout = page["layout"]
    dft_cax = page["coloraxis"]["dft"]
    pre_cax = page["coloraxis"]["preact"]

    style = page.get("style", {})
    font_base   = style.get("font_size_base", 18)
    title_fs    = style.get("title_font_size", 18)
    tick_fs     = style.get("tick_font_size", 16)
    marker_sz   = style.get("marker_size", 6)
    dft_angle   = style.get("dft_tick_angle", 45)
    dtick = style.get("dtick", 1)

    pad = layout.get("padding", {})
    title_yshift = pad.get("title_yshift", 6)
    title_xshift = pad.get("title_xshift", 0)
    axis_standoff= pad.get("axis_title_standoff", 2)

    # 建子图
    fig = make_subplots(
        rows=2, cols=4,
        specs=[[{"type":"xy"}, {"type":"xy"}, {"type":"heatmap"}, {"type":"heatmap"}],
               [{"type":"xy"}, {"type":"xy"}, {"type":"heatmap"}, {"type":"heatmap"}]],
        horizontal_spacing=0.01, #layout["hspace"]
        vertical_spacing=layout["vspace"],
        subplot_titles=[
            page["titles"]["a"], page["titles"]["b"],
            page["titles"]["a_dft"], page["titles"]["b_dft"],
            page["titles"]["a_remap"], page["titles"]["b_remap"],
            page["titles"]["preact2d"], page["titles"]["full_dft"],
        ],
    )
    

    fig.update_layout(
        font=dict(size=font_base),
    )

    def _bump_title(a):
        xref = getattr(a, "xref", None)
        if isinstance(xref, str) and xref.endswith(" domain"):
            a.update(font=dict(size=title_fs), yshift=title_yshift, xshift=title_xshift)

    fig.for_each_annotation(_bump_title)
    fig.update_layout(annotations=list(fig.layout.annotations))
    fig.update_xaxes(title_standoff=axis_standoff, title_font=dict(size=title_fs), tickfont=dict(size=tick_fs))
    fig.update_yaxes(title_standoff=axis_standoff, title_font=dict(size=title_fs), tickfont=dict(size=tick_fs))

    
    # —— 折线：黑线 + marker size=6；marker 颜色挂 preact 的 coloraxis2 —— 
    def _add_line(pack, row, col):
        x1,y1 = np.array(pack["x1"]), np.array(pack["y1"], dtype=float)
        x2,y2 = np.array(pack["x2"]), np.array(pack["y2"], dtype=float)
        fig.add_trace(go.Scatter(
            x=x1, y=y1, mode="markers+lines",
            line=dict(width=1.5, color="black"),
            marker=dict(size=marker_sz, color=y1, coloraxis="coloraxis2"),
            connectgaps=True, showlegend=False
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=x2, y=y2, mode="markers+lines",
            line=dict(width=1.5, color="black"),
            marker=dict(size=marker_sz, color=y2, coloraxis="coloraxis2"),
            connectgaps=True, showlegend=False
        ), row=row, col=col)
        # 关键：直接使用构建时写入的 y_range，保证重建与原图一致
        fig.update_yaxes(range=pack["y_range"], row=row, col=col)
        fig.update_xaxes(showgrid=True, row=row, col=col)
        fig.update_yaxes(showgrid=True, row=row, col=col)

    _add_line(page["lines"]["a"],       1, 1)
    _add_line(page["lines"]["b"],       1, 2)
    _add_line(page["lines"]["a_remap"], 2, 1)
    _add_line(page["lines"]["b_remap"], 2, 2)

    # —— Heatmaps —— 
    fig.add_trace(go.Heatmap(z=page["heatmaps"]["a_dft"]["z"], x=names, y=names,
                             showscale=False, coloraxis="coloraxis1"), row=1, col=3)
    fig.add_trace(go.Heatmap(z=page["heatmaps"]["b_dft"]["z"], x=names, y=names,
                             showscale=False, coloraxis="coloraxis1"), row=1, col=4)
    fig.add_trace(go.Heatmap(z=page["heatmaps"]["preact2d"]["z"],
                             showscale=False, coloraxis="coloraxis2"), row=2, col=3)
    fig.add_trace(go.Heatmap(z=page["heatmaps"]["full_dft"]["z"], x=names, y=names,
                             showscale=False, coloraxis="coloraxis1"), row=2, col=4)

    # a, b, full DFT
    fig.update_xaxes(tickangle=dft_angle, dtick=dtick, row=1, col=3)
    fig.update_xaxes(tickangle=dft_angle, dtick=dtick, row=1, col=4)
    fig.update_xaxes(tickangle=dft_angle, dtick=dtick, row=2, col=4)
    fig.update_yaxes(dtick=dtick, row=1, col=3)
    fig.update_yaxes(dtick=dtick, row=1, col=4)
    fig.update_yaxes(dtick=dtick, row=2, col=4)

    # —— 颜色轴（关键：在 coloraxis 上声明 Viridis） —— 
    fig.update_layout(
        coloraxis1=dict(
            colorscale=dft_cax.get("colorscale", "Inferno"),
            colorbar=dict(
            title=dict(text="GFT", side="right"),  # ← 标题竖排在侧边
            orientation="v",                       # ← 垂直 colorbar（显式声明）
            len=0.36, y=0.80, yanchor="middle",
            x=1.02, xanchor="left",
            thickness=12
            )
        ),
        coloraxis2=dict(
            colorscale=pre_cax.get("colorscale", "Viridis"),
            cmin=pre_cax.get("cmin", None),
            cmax=pre_cax.get("cmax", None),
            colorbar=dict(
            title=dict(text="Preactivation", side="right"),
            orientation="v",
            len=0.36, y=0.20, yanchor="middle",
            x=1.02, xanchor="left",
            thickness=12
            )
        ),
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white"
    )

    # 只对数值-数值的 preact2d 强制像素方形
    axis_name = f"x{(2-1)*4 + 3}"  # row=2, col=3 → x7
    fig.update_xaxes(constrain="domain", row=2, col=3)
    fig.update_yaxes(constrain="domain", row=2, col=3)
    fig.update_yaxes(scaleanchor=axis_name, scaleratio=1, row=2, col=3)

    axis_name = f"x{(1-1)*4 + 3}"  
    fig.update_xaxes(constrain="domain", row=1, col=3)
    fig.update_yaxes(constrain="domain", row=1, col=3)
    fig.update_yaxes(scaleanchor=axis_name, scaleratio=1, row=1, col=3)

    axis_name = f"x{(1-1)*4 + 4}"  
    fig.update_xaxes(constrain="domain", row=1, col=4)
    fig.update_yaxes(constrain="domain", row=1, col=4)
    fig.update_yaxes(scaleanchor=axis_name, scaleratio=1, row=1, col=4)

    axis_name = f"x{(2-1)*4 + 4}"  
    fig.update_xaxes(constrain="domain", row=2, col=4)
    fig.update_yaxes(constrain="domain", row=2, col=4)
    fig.update_yaxes(scaleanchor=axis_name, scaleratio=1, row=2, col=4)

    # 统一网格开关
    for (r,c) in [(1,3),(1,4),(2,3),(2,4)]:
        fig.update_xaxes(showgrid=False, row=r, col=c)
        fig.update_yaxes(showgrid=False, row=r, col=c)

    # 收紧第1/2列 & 拉近色条 & 缩窄右边距
    report._tighten_cols_and_colorbars(fig, want_gap12=1, cb_dx=0.03, right_margin=50)

    # 计算宽高，确保近似方格
    tight_margins = dict(l=28, r=51, t=28, b=24)
    W, H, margins = _compute_fig_size_for_square_cells(
        rows=2, cols=4,
        hspace=layout["hspace"] if "layout" in locals() else 0.05,
        vspace=layout["vspace"] if "layout" in locals() else 0.14,
        cell_px=layout["cell_px"] if "layout" in locals() else 250,
        margins=tight_margins
    )
    fig.update_layout(width=W, height=H, margin=margins, showlegend=False)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    if to_pdf_path is not None:
        pdf_bytes = fig.to_image(format="pdf", engine="kaleido",
                         width=W, height=H, scale=1)
        assert len(PdfReader(io.BytesIO(pdf_bytes)).pages) == 1, "kaleido multiple pages"
        with open(to_pdf_path, "wb") as f:
            f.write(pdf_bytes)

    return fig

def _compute_fig_size_for_square_cells(rows, cols, hspace, vspace, cell_px=240,
                                       margins=dict(l=28, r=80, t=28, b=24)):
    frac_w = (1.0 - (cols - 1) * hspace) / cols
    frac_h = (1.0 - (rows - 1) * vspace) / rows
    H = margins["t"] + margins["b"] + cell_px / frac_h
    W = margins["l"] + margins["r"] + (cell_px / frac_w)
    return int(round(W)), int(round(H)), margins

def export_sign_irreps_strip(
    json_paths,                # 长度=4 的 json 路径列表
    out_pdf_path,              # 导出的横向 PDF 路径
    *,
    titles=None,               # 每个面板上方的标题（默认用文件名 stem）
    cell_px=240,               # 每个子图“绘图区”的目标像素边长
    hspace=0.03,               # 子图间水平留白（domain 比例）
    margins=dict(l=28, r=88, t=28, b=24),  # 紧边距，右边给 colorbar
    font_size=16, title_fs=16, tick_fs=12,  # 字体
    show_axes=False            # 是否显示坐标轴刻度（默认关）
):
    assert len(json_paths) == 4, "需要正好四个 JSON 文件"

    # 1) 读 preact2d 矩阵，并做全局 cmin/cmax（统一色条）
    Zs = []
    names = None
    for pth in map(Path, json_paths):
        page = json.loads(Path(pth).read_text(encoding="utf-8"))
        Z = np.array(page["heatmaps"]["preact2d"]["z"], dtype=float)
        Zs.append(Z)
        # 如需刻度名，这里也可从 JSON 取（但 preact2d 没有 x/y names）
        if names is None and "x" in page["heatmaps"].get("a_dft", {}):
            names = page["heatmaps"]["a_dft"]["x"]

    gmin = min(float(np.nanmin(Z)) for Z in Zs)
    gmax = max(float(np.nanmax(Z)) for Z in Zs)
    if gmin == gmax:
        gmin -= 1e-9; gmax += 1e-9

    # 2) figure 布局（1 行 4 列）
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type":"heatmap"} for _ in range(4)]],
        horizontal_spacing=hspace,
        vertical_spacing=0.0,
        subplot_titles=titles or [Path(p).stem for p in json_paths],
    )

    # 3) 添加 4 张热力图（共用一个 coloraxis）
    for i, Z in enumerate(Zs, start=1):
        fig.add_trace(go.Heatmap(
            z=Z, showscale=False, coloraxis="coloraxis"
        ), row=1, col=i)
        # 保证每一块是“像素等比 + 不挤压 domain”（看起来一样大&正方）
        ax = "x" if i == 1 else f"x{i}"
        fig.update_xaxes(constrain="domain", row=1, col=i)
        fig.update_yaxes(constrain="domain", row=1, col=i)
        fig.update_yaxes(scaleanchor=ax, scaleratio=1, row=1, col=i)

    # 4) 主题、字体、色条（竖向在右侧）
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=font_size),
        coloraxis=dict(
            colorscale="Viridis",
            cmin=gmin, cmax=gmax,
            colorbar=dict(
                title=dict(text="Preactivation", side="right"),
                orientation="v",
                len=0.80, y=0.50, yanchor="middle",
                x=1.0, xanchor="left",
                thickness=14
            )
        )
    )
    # 标题字号微调
    for a in fig.layout.annotations:
        if getattr(a, "xref", "").endswith(" domain"):
            a.font.size = title_fs
            a.yshift = 6

    # 轴是否可见
    if not show_axes:
        fig.update_xaxes(showticklabels=True, ticks="", showgrid=False)
        fig.update_yaxes(showticklabels=True, ticks="", showgrid=False)
    else:
        fig.update_xaxes(tickfont=dict(size=tick_fs))
        fig.update_yaxes(tickfont=dict(size=tick_fs))

    # 5) 计算宽高并更新；强制单页导出
    W, H, margins = _compute_fig_size_for_square_cells(
        rows=1, cols=4, hspace=hspace, vspace=0.0,
        cell_px=cell_px, margins=margins
    )
    margins=dict(l=28, r=80, t=28, b=24)
    fig.update_layout(width=W, height=H, margin=margins, showlegend=False)

    pdf_bytes = fig.to_image(format="pdf", engine="kaleido", width=W, height=H, scale=1)
    assert len(PdfReader(io.BytesIO(pdf_bytes)).pages) == 1, "Kaleido 导出了多页 PDF？"
    Path(out_pdf_path).write_bytes(pdf_bytes)
    print(f"✅ saved: {out_pdf_path}  (size: {W}×{H})")

import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import plotly.express as px
#### freq stats
# ------------ config ------------
OUTPUT_PDF    = "irreps_frequency.pdf"
OUTPUT_CSV    = "irreps_frequency.csv"
MISMATCH_CSV  = "names_mismatch.csv"

# 计数模式：
#   "occurrence": 每出现一次就+1（同一个 seed 内重复出现会累计多次）
#   "per_seed"  : 每个 seed 对某 irreps 最多计 1 次（统计覆盖率）
COUNT_MODE = "occurrence"  # 可改为 "per_seed"

# ------------ 解析函数 ------------
def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def extract_names_and_labels(data: dict):
    """
    返回 (names_list, labels_in_items_list)
    names_list: JSON 顶层的 'names'（若无则空列表）
    labels_in_items_list: 从 items.*[].label 收集到的 label 列表（可能有重复）
    """
    names = data.get("names", []) or []
    items = data.get("items", {}) or {}

    labels = []
    for key in ("approx_coset", "coset_2d", "coset_1d", "others"):
        arr = items.get(key, [])
        if isinstance(arr, list):
            for obj in arr:
                if isinstance(obj, dict):
                    lab = obj.get("label")
                    if isinstance(lab, str):
                        labels.append(lab)
    return names, labels

# ------------ 主流程 ------------
def main(root_dir: str):
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"根目录不存在：{root}")

    # 1) 收集 JSON
    root = Path(root_dir).expanduser().resolve()
    json_files = list(root.rglob("approx*.json"))
    print(f"发现 JSON 文件 {len(json_files)} 个")
    if not json_files:
        print("未找到任何 JSON，退出。")
        return

    # 2) 提取每个 seed 的 names、labels
    names_lists = []
    file_to_names = {}
    file_to_labels = {}

    for p in json_files:
        try:
            data = load_json(p)
        except Exception as e:
            print(f"[WARN] 读取失败：{p} ({e})")
            continue
        names, labels = extract_names_and_labels(data)
        names_lists.append(tuple(names))
        file_to_names[p] = names
        file_to_labels[p] = labels

    # 3) 检查 names 是否完全一致（含顺序）
    unique_names_sets = set(names_lists)
    if len(unique_names_sets) != 1:
        print("❌ 所有 JSON 的 `names` 不一致，导出详单并退出。")
        # 导出每个文件的 names（按列展开）
        rows = []
        max_len = max((len(v) for v in file_to_names.values()), default=0)
        for p, names in file_to_names.items():
            row = {"file": str(p)}
            for i in range(max_len):
                row[f"name_{i}"] = names[i] if i < len(names) else ""
            rows.append(row)
        pd.DataFrame(rows).to_csv(MISMATCH_CSV, index=False, encoding="utf-8")
        print(f"已写出不一致报告：{MISMATCH_CSV}")
        return

    # 一致 → 取统一 names 顺序
    NAMES_ORDER = list(unique_names_sets.pop())
    print(f"✅ `names` 一致，共 {len(NAMES_ORDER)} 个 irreps。")

    # 4) 统计
    # 初始化为 0，保证顺序完整（即使某个从未出现，也会显示 0）
    counts = {name: 0 for name in NAMES_ORDER}
    extras_counter = Counter()  # 记录 items 中出现但不在 names 里的 label（若有）

    for p, labels in file_to_labels.items():
        if COUNT_MODE == "per_seed":
            labels_iter = set(labels)  # 每个 seed 内去重
        else:
            labels_iter = labels       # occurrence 模式：不去重

        for lab in labels_iter:
            if lab in counts:
                counts[lab] += 1
            else:
                extras_counter[lab] += 1   # 记录“越界”标签，帮助排查

    # 5) 整表 & 导出
    freq_df = pd.DataFrame({
        "irreps": NAMES_ORDER,
        "count": [counts[name] for name in NAMES_ORDER]
    })
    freq_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"频次表已保存：{OUTPUT_CSV}")

    if extras_counter:
        print("⚠️ 发现 items 中存在不在 names 列表内的 label（已忽略计数）：")
        for lab, c in extras_counter.most_common(10):
            print(f"  {lab}: {c}")
        # 如需记录全量，可另存：
        pd.DataFrame(extras_counter.items(), columns=["label_not_in_names", "count"])\
          .sort_values("count", ascending=False)\
          .to_csv("labels_not_in_names.csv", index=False, encoding="utf-8")
        print("已写出 labels_not_in_names.csv")

    # 6) 画图（Plotly，按 NAMES_ORDER 顺序）
    fig = px.bar(
        freq_df,
        x="irreps",
        y="count",
        title=f"Irreps Frequency ({COUNT_MODE}) over {len(json_files)} seeds",
        text="count"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-45, margin=dict(l=40, r=20, t=60, b=120))
    # 保存为 PDF（需要 kaleido）
    fig.write_image(OUTPUT_PDF)
    print(f"柱状图已保存：{OUTPUT_PDF}")

    # 可选：同时保存交互式 HTML
    # fig.write_html("irreps_frequency.html")

import json, math
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _pad_range(arr, frac=0.06):
    arr = np.asarray(arr)
    lo = float(np.nanmin(arr)); hi = float(np.nanmax(arr))
    pad = (hi - lo) * frac or 1.0  # 退化时给个最小 padding
    return [lo - pad, hi + pad]


def make_three_panel_pca_pdf(
    bundle_json_path: str,
    contrib_json_path: str,
    out_pdf: str,
    *,
    f: int = 6,
    marker_size: int = 3,
    line_width: float = 1.2,
) -> str:
    """
    生成一个单页 PDF，三张并排的 3D 图（仅 PC0, PC1, PC2）：
      1) bundle: a mod g + 对应连线
      2) bundle: b mod g + 对应连线
      3) contrib: c mod g + 对应连线
    颜色条标题固定为 'a mod g' / 'b mod g' / 'c mod g'。
    """
    # ---- 载入两个 bundle（你已有的工具） ----
    B = pca_diffusion_plots_w_helpers.load_embedding_bundle_json(bundle_json_path)
    C = pca_diffusion_plots_w_helpers.load_embedding_bundle_json(contrib_json_path)

    # ---- 取 PCA 的前三维 ----
    pca_B = B["_PCS"]; pca_C = C["_PCS"]
    if pca_B.shape[1] < 3 or pca_C.shape[1] < 3:
        raise ValueError("PCA 维度不足 3，无法绘制 3D。")

    XB, YB, ZB = pca_B[:, 0], pca_B[:, 1], pca_B[:, 2]
    XC, YC, ZC = pca_C[:, 0], pca_C[:, 1], pca_C[:, 2]

    # ---- 取 p, tag_q 与 a,b（若缺失则尝试按 N/p/tag_q 重构；再不行就给 0）----
    p_B = int(B["meta"].get("p", 0));  tag_q_B = B["meta"].get("tag_q", "") or ""
    p_C = int(C["meta"].get("p", 0));  tag_q_C = C["meta"].get("tag_q", "") or ""

    A_B = B.get("_A", None);  BB = B.get("_B", None)
    A_C = C.get("_A", None);  BC = C.get("_B", None)

    if A_B is None or BB is None:
        N = pca_B.shape[0]
        if tag_q_B == "full" and N == (2 * p_B) * (2 * p_B):
            side = 2 * p_B; idx = np.arange(N); A_B = idx // side; BB = idx % side
        elif N == p_B * p_B:
            idx = np.arange(N); A_B = idx // p_B; BB = idx % p_B
        else:
            A_B = np.zeros(N, dtype=int); BB = np.zeros(N, dtype=int)

    if A_C is None or BC is None:
        N = pca_C.shape[0]
        if tag_q_C == "full" and N == (2 * p_C) * (2 * p_C):
            side = 2 * p_C; idx = np.arange(N); A_C = idx // side; BC = idx % side
        elif N == p_C * p_C:
            idx = np.arange(N); A_C = idx // p_C; BC = idx % p_C
        else:
            A_C = np.zeros(N, dtype=int); BC = np.zeros(N, dtype=int)

    # ---- 颜色向量（a/b/c）----
    col_a, _cap_a, pbar_a, cs_a = color_rules.colour_quad_a_only(A_B, BB, p_B, f, tag_q_B)
    col_b, _cap_b, pbar_b, cs_b = color_rules.colour_quad_b_only(A_B, BB, p_B, f, tag_q_B)
    col_c, _cap_c, pbar_c, cs_c = color_rules.colour_quad_mod_g(A_C, BC, p_C, f, tag_q_C)

    # ---- 连线索引集合 ----
    g_B = p_B // math.gcd(p_B, abs(int(f))) or p_B
    g_C = p_C // math.gcd(p_C, abs(int(f))) or p_C
    h_pairs = color_rules.lines_a_mod_g(A_B, BB, p_B, g_B)  # for a mod g
    v_pairs = color_rules.lines_b_mod_g(A_B, BB, p_B, g_B)  # for b mod g
    c_pairs = color_rules.lines_c_mod_g(A_C, BC, p_C, g_C)  # for c mod g

    # ---- Figure（1×3 并排）----
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type':'scene'}, {'type':'scene'}, {'type':'scene'}]],
        horizontal_spacing=0.0,
        subplot_titles=["(preactivation) colored by a mod 3", "(preactivation) colored by b mod 3", "(logits) colored by ans mod 3"]
    )

    # hover: 显示 (a,b)
    hover_B = pca_diffusion_plots_w_helpers._make_hover(A_B, BB)
    hover_C = pca_diffusion_plots_w_helpers._make_hover(A_C, BC)

    def _add_points(col_idx, X, Y, Z, colour, pbar, colorscale, cbar_title, hover_kw):
        fig.add_trace(
            go.Scatter3d(
                x=X, y=Y, z=Z,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=colour,
                    colorscale=colorscale,
                    cmin=0,
                    cmax=int(pbar) - 1,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text=cbar_title, side="right",font=dict(size=20)),
                        tickfont=dict(size=18),
                        len=0.70,
                        x=0.98,
                        xpad = 0
                    )
                ),
                showlegend=False,
                **hover_kw
            ),
            row=1, col=col_idx
        )
        scene_id = f"scene{col_idx if col_idx>1 else ''}"
        fig.layout[scene_id].xaxis.title.text = "PC0"
        fig.layout[scene_id].yaxis.title.text = "PC1"
        fig.layout[scene_id].zaxis.title.text = "PC2"

    def _add_lines_for_a(col_idx):
        # a-lines：按 A_B 排序；>=3 点闭合
        for idx_arr, dash, color, gid in h_pairs:
            idx_sorted = idx_arr[np.argsort(A_B[idx_arr])]
            idx_plot = np.concatenate([idx_sorted, idx_sorted[:1]]) if idx_sorted.size > 2 else idx_sorted
            fig.add_trace(
                go.Scatter3d(
                    x=pca_B[idx_plot, 0], y=pca_B[idx_plot, 1], z=pca_B[idx_plot, 2],
                    mode="lines",
                    line=dict(color=color, dash=dash, width=line_width),
                    hoverinfo="skip",
                    showlegend=False
                ),
                row=1, col=col_idx
            )

    def _add_lines_for_b(col_idx):
        # b-lines：按 BB 排序
        for idx_arr, dash, color, gid in v_pairs:
            idx_sorted = idx_arr[np.argsort(BB[idx_arr])]
            idx_plot = np.concatenate([idx_sorted, idx_sorted[:1]]) if idx_sorted.size > 2 else idx_sorted
            fig.add_trace(
                go.Scatter3d(
                    x=pca_B[idx_plot, 0], y=pca_B[idx_plot, 1], z=pca_B[idx_plot, 2],
                    mode="lines",
                    line=dict(color=color, dash=dash, width=line_width),
                    hoverinfo="skip",
                    showlegend=False
                ),
                row=1, col=col_idx
            )

    def _add_lines_for_c(col_idx):
        # c-lines：按 (A_C, B_C) 词典序
        for idx_arr, dash, color, gid in c_pairs:
            a_sub = A_C[idx_arr]; b_sub = BC[idx_arr]
            order = np.lexsort((b_sub, a_sub))
            idx_sorted = idx_arr[order]
            idx_plot = np.concatenate([idx_sorted, idx_sorted[:1]]) if idx_sorted.size > 2 else idx_sorted
            fig.add_trace(
                go.Scatter3d(
                    x=pca_C[idx_plot, 0], y=pca_C[idx_plot, 1], z=pca_C[idx_plot, 2],
                    mode="lines",
                    line=dict(color=color, dash=dash, width=line_width),
                    hoverinfo="skip",
                    showlegend=False
                ),
                row=1, col=col_idx
            )

    # ---- 三个面板：点 + 线 ----
    _add_points(1, XB, YB, ZB, col_a, pbar_a, cs_a, "", hover_B)
    _add_lines_for_a(1)

    _add_points(2, XB, YB, ZB, col_b, pbar_b, cs_b, "", hover_B)
    _add_lines_for_b(2)

    _add_points(3, XC, YC, ZC, col_c, pbar_c, cs_c, "a/b/ans mod g", hover_C)
    _add_lines_for_c(3)

    fig.update_layout(
        # 2) 每个 scene 用数据范围 + padding，并设 aspectmode="data"
        scene=dict(
            xaxis=dict(range=_pad_range(XB)),
            yaxis=dict(range=_pad_range(YB)),
            zaxis=dict(range=_pad_range(ZB)),
            aspectmode="data",
        ),
        scene2=dict(
            xaxis=dict(range=_pad_range(XB)),
            yaxis=dict(range=_pad_range(YB)),
            zaxis=dict(range=_pad_range(ZB)),
            aspectmode="data",
        ),
        scene3=dict(
            xaxis=dict(range=_pad_range(XC)),
            yaxis=dict(range=_pad_range(YC)),
            zaxis=dict(range=_pad_range(ZC)),
            aspectmode="data",
        ),

    )
    fig.update_layout(
        scene=dict(domain=dict(x=[0.0, 0.33])),
        scene2=dict(domain=dict(x=[0.33, 0.66])),
        scene3=dict(domain=dict(x=[0.66, 1.0])),
    )
    fig.update_layout(
        scene2_camera=dict(
            eye=dict(x=1.25, y=-1.25, z=1.25),  # 把默认 (1.25, 1.25, 1.25) 的 y 取反
            up=dict(x=0, y=0, z=1)
        )
    )
    fig.update_layout(
        width=1700, height=600,
        margin=dict(l=5, r=5, t=60, b=50),
        title=dict(
            text=f"PC0 vs PC1 vs PC2 with graph lines",
            font=dict(size=24)
        ),
        showlegend=False
    )
    fig.update_annotations(font=dict(size=24))

    fig.update_scenes(
    xaxis=dict(title=dict(font=dict(size=24)), tickfont=dict(size=16)),
    yaxis=dict(title=dict(font=dict(size=24)), tickfont=dict(size=16)),
    zaxis=dict(title=dict(font=dict(size=24)), tickfont=dict(size=16)),
    )

    for ann in fig.layout.annotations:
        ann.update(yshift=-20)
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(out_pdf, format="pdf")   # needs kaleido
    return out_pdf

def make_four_panel_pca_pdf(
    bundle_json_path: str,
    contrib_json_path: str,
    out_pdf: str,
    *,
    f: int = 6,
    marker_size: int = 3,
    line_width: float = 1.2
) -> str:
    """
    生成一个单页 PDF，四张并排的 3D 图（PC0, PC1, PC2）：
      1) preactivations(B): a mod g + 对应连线
      2) preactivations(B): b mod g + 对应连线
      3) preactivations(B): C mod g + 对应连线
      4) contrib/logits(C):  C mod g + 对应连线
    Colorbar 放置与 three-panel 一致：每列各自一根，统一 x=0.98；
    仅第 4 列显示合并标题 "a/b/C mod g"。
    """
    # ---- 载入 ----
    B = pca_diffusion_plots_w_helpers.load_embedding_bundle_json(bundle_json_path)
    C = pca_diffusion_plots_w_helpers.load_embedding_bundle_json(contrib_json_path)

    # ---- PCA 前三维 ----
    pca_B = B["_PCS"]; pca_C = C["_PCS"]
    if pca_B.shape[1] < 3 or pca_C.shape[1] < 3:
        raise ValueError("PCA 维度不足 3，无法绘制 3D。")

    XB, YB, ZB = pca_B[:, 0], pca_B[:, 1], pca_B[:, 2]
    XC, YC, ZC = pca_C[:, 0], pca_C[:, 1], pca_C[:, 2]

    # ---- 元信息 ----
    p_B = int(B["meta"].get("p", 0));  tag_q_B = B["meta"].get("tag_q", "") or ""
    p_C = int(C["meta"].get("p", 0));  tag_q_C = C["meta"].get("tag_q", "") or ""

    A_B = B.get("_A", None);  BB = B.get("_B", None)
    A_C = C.get("_A", None);  BC = C.get("_B", None)

    # 若缺失 A/B 索引则尝试重构
    if A_B is None or BB is None:
        N = pca_B.shape[0]
        if tag_q_B == "full" and N == (2 * p_B) * (2 * p_B):
            side = 2 * p_B; idx = np.arange(N); A_B = idx // side; BB = idx % side
        elif N == p_B * p_B:
            idx = np.arange(N); A_B = idx // p_B; BB = idx % p_B
        else:
            A_B = np.zeros(N, dtype=int); BB = np.zeros(N, dtype=int)

    if A_C is None or BC is None:
        N = pca_C.shape[0]
        if tag_q_C == "full" and N == (2 * p_C) * (2 * p_C):
            side = 2 * p_C; idx = np.arange(N); A_C = idx // side; BC = idx % side
        elif N == p_C * p_C:
            idx = np.arange(N); A_C = idx // p_C; BC = idx % p_C
        else:
            A_C = np.zeros(N, dtype=int); BC = np.zeros(N, dtype=int)

    # ---- 颜色（B: a/b/C；C: C）----
    col_a, _, pbar_a, cs_a = color_rules.colour_quad_a_only(A_B, BB, p_B, f, tag_q_B)
    col_b, _, pbar_b, cs_b = color_rules.colour_quad_b_only(A_B, BB, p_B, f, tag_q_B)
    col_cB, _, pbar_cB, cs_cB = color_rules.colour_quad_mod_g(A_B, BB, p_B, f, tag_q_B)  # preacts C
    col_cC, _, pbar_cC, cs_cC = color_rules.colour_quad_mod_g(A_C, BC, p_C, f, tag_q_C)  # logits C

    # ---- 连线索引 ----
    g_B = p_B // math.gcd(p_B, abs(int(f))) or p_B
    g_C = p_C // math.gcd(p_C, abs(int(f))) or p_C
    h_pairs = color_rules.lines_a_mod_g(A_B, BB, p_B, g_B)   # for a mod g (B)
    v_pairs = color_rules.lines_b_mod_g(A_B, BB, p_B, g_B)   # for b mod g (B)
    c_pairs_B = color_rules.lines_c_mod_g(A_B, BB, p_B, g_B) # for C mod g (B)
    c_pairs_C = color_rules.lines_c_mod_g(A_C, BC, p_C, g_C) # for C mod g (C)

    # ---- 子图（1×4）----
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{'type':'scene'}, {'type':'scene'}, {'type':'scene'}, {'type':'scene'}]],
        horizontal_spacing=0.00,
        subplot_titles=[
            "(preactivation) colored by a mod g",
            "(preactivation) colored by b mod g",
            "(preactivation) colored by C mod g",
            "(logits) colored by C mod g",
        ]
    )

    # hover 信息
    hover_B = pca_diffusion_plots_w_helpers._make_hover(A_B, BB)
    hover_C = pca_diffusion_plots_w_helpers._make_hover(A_C, BC)

    # ---- 与 three-panel 相同的 colorbar 放置方式（固定在右侧 x=0.98）----
    def _add_points(col_idx, X, Y, Z, colour, pbar, colorscale, cbar_title: str, hover_kw):
        fig.add_trace(
            go.Scatter3d(
                x=X, y=Y, z=Z,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=colour,
                    colorscale=colorscale,
                    cmin=0,
                    cmax=int(pbar) - 1,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text=cbar_title, side="right", font=dict(size=20)),
                        tickfont=dict(size=18),
                        len=0.70,
                        x=0.98,
                        xpad=0
                    )
                ),
                showlegend=False,
                **hover_kw
            ),
            row=1, col=col_idx
        )
        scene_id = f"scene{col_idx if col_idx>1 else ''}"
        fig.layout[scene_id].xaxis.title.text = "PC0"
        fig.layout[scene_id].yaxis.title.text = "PC1"
        fig.layout[scene_id].zaxis.title.text = "PC2"

    # ---- 通用折线添加器 ----
    def _add_polyline(col_idx, X, Y, Z, dash, color):
        fig.add_trace(
            go.Scatter3d(
                x=X, y=Y, z=Z,
                mode="lines",
                line=dict(color=color, dash=dash, width=line_width),
                hoverinfo="skip",
                showlegend=False
            ),
            row=1, col=col_idx
        )

    # ---- 加线 ----
    def _add_lines_for_a(col_idx, X, Y, Z):
        for idx_arr, dash, color, gid in h_pairs:
            idx_sorted = idx_arr[np.argsort(A_B[idx_arr])]
            idx_plot = np.concatenate([idx_sorted, idx_sorted[:1]]) if idx_sorted.size > 2 else idx_sorted
            _add_polyline(col_idx, X[idx_plot], Y[idx_plot], Z[idx_plot], dash, color)

    def _add_lines_for_b(col_idx, X, Y, Z):
        for idx_arr, dash, color, gid in v_pairs:
            idx_sorted = idx_arr[np.argsort(BB[idx_arr])]
            idx_plot = np.concatenate([idx_sorted, idx_sorted[:1]]) if idx_sorted.size > 2 else idx_sorted
            _add_polyline(col_idx, X[idx_plot], Y[idx_plot], Z[idx_plot], dash, color)

    def _add_lines_for_c_B(col_idx, X, Y, Z):
        for idx_arr, dash, color, gid in c_pairs_B:
            a_sub = A_B[idx_arr]; b_sub = BB[idx_arr]
            order = np.lexsort((b_sub, a_sub))
            idx_sorted = idx_arr[order]
            idx_plot = np.concatenate([idx_sorted, idx_sorted[:1]]) if idx_sorted.size > 2 else idx_sorted
            _add_polyline(col_idx, X[idx_plot], Y[idx_plot], Z[idx_plot], dash, color)

    def _add_lines_for_c_C(col_idx, X, Y, Z):
        for idx_arr, dash, color, gid in c_pairs_C:
            a_sub = A_C[idx_arr]; b_sub = BC[idx_arr]
            order = np.lexsort((b_sub, a_sub))
            idx_sorted = idx_arr[order]
            idx_plot = np.concatenate([idx_sorted, idx_sorted[:1]]) if idx_sorted.size > 2 else idx_sorted
            _add_polyline(col_idx, X[idx_plot], Y[idx_plot], Z[idx_plot], dash, color)

    # ---- 面板 1：preacts a mod g ----
    _add_points(1, XB, YB, ZB, col_a, pbar_a, cs_a, "", hover_B)
    _add_lines_for_a(1, XB, YB, ZB)

    # ---- 面板 2：preacts b mod g ----
    _add_points(2, XB, YB, ZB, col_b, pbar_b, cs_b, "", hover_B)
    _add_lines_for_b(2, XB, YB, ZB)

    # ---- 面板 3：preacts C mod g（来自 B）----
    _add_points(3, XB, YB, ZB, col_cB, pbar_cB, cs_cB, "", hover_B)
    _add_lines_for_c_B(3, XB, YB, ZB)

    # ---- 面板 4：contrib/logits C mod g（来自 C；仅这里给合并标题）----
    _add_points(4, XC, YC, ZC, col_cC, pbar_cC, cs_cC, "a/b/C mod g", hover_C)
    _add_lines_for_c_C(4, XC, YC, ZC)

    def rotate_scene_camera_cw(fig, scene_key: str, deg_cw: float, base_eye=(1.25, 1.25, 1.25)):
        import numpy as np
        x0, y0, z0 = base_eye
        rho = (x0**2 + y0**2) ** 0.5
        phi0 = np.arctan2(y0, x0)          # 默认是 45° = π/4
        phi  = phi0 - np.deg2rad(deg_cw)   # 顺时针 => 角度减
        x = float(rho * np.cos(phi))
        y = float(rho * np.sin(phi))
        z = z0
        fig.update_layout(**{
            f"{scene_key}_camera": dict(eye=dict(x=x, y=y, z=z), up=dict(x=0, y=0, z=1))
        })

    # 用法：把 scene2 顺时针旋转 90°
    rotate_scene_camera_cw(fig, "scene2", 110.0)
    # ---- 轴范围与比例 ----
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=_pad_range(XB)),
            yaxis=dict(range=_pad_range(YB)),
            zaxis=dict(range=_pad_range(ZB)),
            aspectmode="data",
        ),
        scene2=dict(
            xaxis=dict(range=_pad_range(XB)),
            yaxis=dict(range=_pad_range(YB)),
            zaxis=dict(range=_pad_range(ZB)),
            aspectmode="data",
        ),
        scene3=dict(
            xaxis=dict(range=_pad_range(XB)),
            yaxis=dict(range=_pad_range(YB)),
            zaxis=dict(range=_pad_range(ZB)),
            aspectmode="data",
        ),
        scene4=dict(
            xaxis=dict(range=_pad_range(XC)),
            yaxis=dict(range=_pad_range(YC)),
            zaxis=dict(range=_pad_range(ZC)),
            aspectmode="data",
        ),
    )

    # ---- 四列布局域 ----
    fig.update_layout(
        scene=dict(domain=dict(x=[0.00, 0.25])),
        scene2=dict(domain=dict(x=[0.25, 0.50])),
        scene3=dict(domain=dict(x=[0.50, 0.75])),
        scene4=dict(domain=dict(x=[0.75, 1.00])),
    )

    # ---- 画布外观 ----
    fig.update_layout(
        width=2400, height=630,
        margin=dict(l=5, r=5, t=50, b=50),
        title=dict(text="Projecting the data onto the first three principal components reveals networks learn cosets as clean geometric structure", font=dict(size=24)),
        showlegend=False
    )
    fig.update_annotations(font=dict(size=22))
    fig.update_scenes(
        xaxis=dict(title=dict(font=dict(size=22)), tickfont=dict(size=16)),
        yaxis=dict(title=dict(font=dict(size=22)), tickfont=dict(size=16)),
        zaxis=dict(title=dict(font=dict(size=22)), tickfont=dict(size=16)),
    )
    for ann in fig.layout.annotations:
        ann.update(yshift=-18)

    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(out_pdf, format="pdf")   # 需要 kaleido
    return out_pdf

import numpy as np
import itertools
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def make_pca_4x3_from_bundle(
    bundle_json_path: str,
    out_pdf: str,
    *,
    f: int,
    marker_size: int = 3,
    title: str | None = None,
) -> str:
    """
    用 bundle JSON 重建一个 3×4 的 3D 面板（仅 PCA）：
      行1：颜色索引“上半段”（~ p..2p-1）
      行2：颜色索引“下半段”（~ 0..p-1）
      行3：全部（0..2p-1）
    列为四个三维组合：(0,1,2), (0,1,3), (0,2,3), (1,2,3)
    色条（colorbar）统一使用整体范围 [0, p_cbar-1]，每行只在最后一列显示一个色条。
    """
    # ---- 载入 bundle & 取 PCA 坐标 ----
    B = pca_diffusion_plots_w_helpers.load_embedding_bundle_json(bundle_json_path)
    meta = B.get("meta", {})
    p = int(meta.get("p", 0))
    tag_q = meta.get("tag_q", "") or ""
    seed = meta.get("seed", "")
    PCS = np.asarray(B["_PCS"])
    if PCS.shape[1] < 4:
        raise ValueError("PCA 维度不足 4，无法生成 4 组 3D 组合。")

    # ---- 颜色向量（根据 bundle 内记录的 colour_rule 与 f）----
    colour, caption, p_cbar, colorscale = pca_diffusion_plots_w_helpers._get_colour_for_f(B, f)
    colour = np.asarray(colour, int)
    N = PCS.shape[0]

    # ---- 三行的掩码（上半 / 下半 / 全部）----
    half = int(p_cbar // 2)
    mask_upper = (colour >= half)          # ~ p..2p-1（若 p_cbar=2p）
    mask_lower = (colour <  half)          # ~ 0..p-1
    mask_all   = np.ones(N, dtype=bool)

    row_defs = [
        ("upper half", mask_upper),
        ("lower half", mask_lower),
        ("all points", mask_all),
    ]

    # ---- 四列：前四主成分的 3D 组合 ----
    triplets = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
    titles = [f"PC{i} vs PC{j} vs PC{k}" for (i,j,k) in triplets]

    # ---- 画布 ----
    fig = make_subplots(
        rows=3, cols=4,
        specs=[[{'type': 'scene'}]*4 for _ in range(3)],
        horizontal_spacing=0.0,
        vertical_spacing=0.0,
        subplot_titles=titles
    )
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=22)      # 改字体大小
        ann['yshift'] = -20               # 调整与图的垂直距离（正值往上挪，负值往下靠近图）
        ann['xshift'] = 0                # 如果想水平移动也可以
    # helper：往 (row,col) 放一个 3D 散点
    def _add_panel(row_idx: int, col_idx: int, m: np.ndarray, i: int, j: int, k: int, showscale: bool):
        idx = np.where(m)[0]
        if idx.size == 0:
            return
        fig.add_trace(
            go.Scatter3d(
                x=PCS[idx, i], y=PCS[idx, j], z=PCS[idx, k],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=colour[idx],
                    colorscale=colorscale,
                    cmin=0,
                    cmax=int(p_cbar) - 1,
                    showscale=showscale,
                    colorbar=dict(
                        title=dict(text=f"C = 0…{p_cbar-1}", 
                                   side="right", 
                                   font=dict(size=24)),
                        tickfont=dict(size=20),
                        len=0.82,
                        x=1.02  # 把每行的色条放到最右边
                    )
                ),
                showlegend=False,
                hovertemplate="index=%{pointNumber}<extra></extra>"
            ),
            row=row_idx, col=col_idx
        )
        scene_id = f"scene{(row_idx-1)*4 + col_idx if (row_idx, col_idx)!=(1,1) else ''}"
        scene = fig.layout[scene_id]
        scene.xaxis.title.text = f"PC{i}"
        scene.yaxis.title.text = f"PC{j}"
        scene.zaxis.title.text = f"PC{k}"
        scene.xaxis.title.font = dict(size=22)
        scene.yaxis.title.font = dict(size=22)
        scene.zaxis.title.font = dict(size=22)
        scene.xaxis.tickfont = dict(size=16)
        scene.yaxis.tickfont = dict(size=16)
        scene.zaxis.tickfont = dict(size=16)

    # ---- 填充 3×4 面板（每行只在第4列显示色条）----
    for r, (rname, rmask) in enumerate(row_defs, start=1):
        for c, (i,j,k) in enumerate(triplets, start=1):
            _add_panel(r, c, rmask, i, j, k, showscale=(c==4))

    # 行标签（放在最左侧）
    fig.add_annotation(text=f"C, sign -1",
                       xref="paper", yref="paper", x=-0.015, y=0.86,
                       showarrow=False, font=dict(size=22), textangle=270)
    fig.add_annotation(text=f"C, sign +1",
                       xref="paper", yref="paper", x=-0.015, y=0.49,
                       showarrow=False, font=dict(size=22), textangle=270)
    fig.add_annotation(text=f"all Cs",
                       xref="paper", yref="paper", x=-0.015, y=0.12,
                       showarrow=False, font=dict(size=22), textangle=270)

    # 标题 & 尺寸 & 外边距
    fig.update_layout(
        width=2200, height=1400,
        margin=dict(l=30, r=80, t=70, b=40),
        title=dict(
            text=title or f"PCAs D_18 approximate cosets colored by C",
            font=dict(size=22)
        ),
        showlegend=False
    )

    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(out_pdf, format="pdf")  # 需要 kaleido
    return out_pdf


# =========================
# Pair-of-quadrants (merge two bundles into one colorbar 0..2g-1)
# =========================
import re
from typing import Literal

Mode = Literal["a", "b", "c"]

def _infer_quad(bundle: dict, fallback_path: str | None = None) -> str:
    """从 bundle 的 meta 或路径名里推断象限（BL/BR/TL/TR）"""
    meta = bundle.get("meta", {})
    for key in ("tag_q", "tag", "class_string"):
        v = (meta.get(key) or "").upper()
        if v in ("BL", "BR", "TL", "TR"):
            return v
    if fallback_path:
        m = re.search(r"\b(BL|BR|TL|TR)\b", fallback_path.upper())
        if m:
            return m.group(1)
    raise ValueError("无法从 bundle 中推断象限（需要 BL/BR/TL/TR）")

def _compute_g(p: int, f: int) -> int:
    import math
    g = p // math.gcd(p, f) if math.gcd(p, f) != 0 else p
    return g or p

def _base_by_mode_for_quad(a: np.ndarray, b: np.ndarray, quad: str, g: int, mode: Mode) -> np.ndarray:
    """返回该象限在所选模式下的局部 base 值（范围 0..g-1）"""
    quad = quad.upper()
    if mode == "a":
        return (a % g)
    if mode == "b":
        return (b % g)
    if mode == "c":
        # 右半平面（BR/TR）：b-a；左半平面（BL/TL）：a+b
        if quad in ("BR", "TR"):
            return (b - a) % g
        else:
            return (a + b) % g
    raise ValueError("mode must be 'a' | 'b' | 'c'")

def _load_bundle_with_path(path_or_obj):
    """和 load_embedding_bundle_json 类似，但也返回原始路径（便于从文件名推断象限）"""
    if isinstance(path_or_obj, (str, Path)):
        with open(path_or_obj, "r") as fh:
            bundle = json.load(fh)
        src_path = str(path_or_obj)
    else:
        bundle = path_or_obj
        src_path = None
    b = pca_diffusion_plots_w_helpers.load_embedding_bundle_json(bundle)  
    return b, src_path

from typing import Iterable, Optional

def build_pair_colour_from_bundles(
    bundle_a: str | dict,
    bundle_b: str | dict,
    *,
    kind: str = "pca",          
    mode: Mode = "c",           
    f: int | None = None,       
    low_half_first: bool = True, 
    c_keep: Optional[Iterable[int]] = None
):
    """
    读入两个“单象限”的 bundle，合成为同一图的一个颜色向量：
      - 第一个象限映射到 [0..g-1]（或高半区，取决于 low_half_first）
      - 第二个象限映射到 [g..2g-1]
      - 返回：coords, colour, caption, p, p_cbar(=2g), colorscale, A_concat, B_concat, quadrants_str
    """
    b1, p1 = _load_bundle_with_path(bundle_a)
    b2, p2 = _load_bundle_with_path(bundle_b)

    meta1 = b1["meta"]; meta2 = b2["meta"]
    p = int(meta1["p"])
    if int(meta2["p"]) != p:
        raise ValueError(f"两个 bundle 的 p 不一致：{p} vs {meta2['p']}")

    if f is None:
        fl = meta1.get("freq_list") or []
        f = int(fl[0]) if fl else 1
    g = _compute_g(p, abs(int(f)))

    coords1 = b1["_PCS"] if kind == "pca" else b1["_DMAP"]
    coords2 = b2["_PCS"] if kind == "pca" else b2["_DMAP"]

    A1, B1 = b1.get("_A", None), b1.get("_B", None)
    A2, B2 = b2.get("_A", None), b2.get("_B", None)
    if A1 is None or B1 is None or A2 is None or B2 is None:
        raise ValueError("这两个 bundle 不包含 a_vals/b_vals（或无法重构）；无法着色。")

    quad1 = _infer_quad(b1, p1); quad2 = _infer_quad(b2, p2)
    if quad1 == "BL":
        A1 = A1
    elif quad1 == "TL":
        A1 = A1 + p
    elif quad1 == "BR":
        B1 = B1 + p
    else:
        A1 = A1 + p
        B1 = B1 + p

    if quad2 == "BL":
        A2 = A2
    elif quad2 == "TL":
        A2 = A2 + p
    elif quad2 == "BR":
        B2 = B2 + p
    else:
        A2 = A2 + p
        B2 = B2 + p

    base1 = _base_by_mode_for_quad(A1, B1, quad1, g, mode)
    base2 = _base_by_mode_for_quad(A2, B2, quad2, g, mode)

    if mode == "c" and c_keep is not None:
        keep_vals = np.unique(np.mod(np.fromiter(c_keep, dtype=int), g))
        m1 = np.isin(base1, keep_vals)
        m2 = np.isin(base2, keep_vals)

        coords1 = coords1[m1]
        A1, B1 = A1[m1], B1[m1]
        base1 = base1[m1]

        coords2 = coords2[m2]
        A2, B2 = A2[m2], B2[m2]
        base2 = base2[m2]

    if low_half_first:
        col1 = base1                # 0..g-1
        col2 = g + base2            # g..2g-1
    else:
        col1 = g + base1
        col2 = base2

    coords = np.vstack([coords1, coords2])
    colour = np.concatenate([col1, col2])
    A_cat = np.concatenate([A1, A2])
    B_cat = np.concatenate([B1, B2])

    p_cbar = 2 * g
    colorscale = color_rules.build_split_scale_red_orange(g)  
    caption = f"{mode} mod {g} — {quad1}→{('0..'+str(g-1)) if low_half_first else (str(g)+'..'+str(2*g-1))}, " \
              f"{quad2}→{(str(g)+'..'+str(2*g-1)) if low_half_first else ('0..'+str(g-1))}"
    quads = f"{quad1}+{quad2}"
    return coords, colour, caption, p, p_cbar, colorscale, A_cat, B_cat, quads, g, f

def rebuild_pair_html_from_bundles(
    bundle_a: str | dict,
    bundle_b: str | dict,
    *,
    kind: str = "pca",
    mode: Mode = "c",
    f: int | None = None,
    low_half_first: bool = True,
    out_html: str | None = None,
    c_keep: Optional[Iterable[int]] = None
):
    
    coords, colour, caption, p, p_cbar, colorscale, A_cat, B_cat, quads, g, f_used = \
        build_pair_colour_from_bundles(bundle_a, bundle_b, kind=kind, mode=mode, f=f, low_half_first=low_half_first, c_keep=c_keep)

    meta = (pca_diffusion_plots_w_helpers.load_embedding_bundle_json(bundle_a) if isinstance(bundle_a, (str, Path)) else bundle_a).get("meta", {})
    seed = meta.get("seed", "")
    class_string = meta.get("class_string", "")
    label = "PC" if kind == "pca" else "DM"

    fig = pca_diffusion_plots_w_helpers._make_multiplot_3d_figure_html_only(
        coords=coords,
        colour=colour,
        caption=caption,
        p=p,
        p_cbar=p_cbar,
        colorscale=colorscale,
        seed=seed,
        label=label,
        title_tag=f"{quads} & {class_string}",
        f=f_used,
        mult=False,                
        a_vals=A_cat,
        b_vals=B_cat,
        tag_q="full"
    )
    if out_html:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_html, include_plotlyjs="cdn")
    return fig

import math
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio

def make_single_pc012_3d_from_bundles(
    bundle_a,
    bundle_b,
    *,
    kind: str = "pca",       
    mode: str = "c",         
    f: int | None = None,
    low_half_first: bool = True,
    c_keep=None,            
    azimuth_deg: float = 45,
    elevation_deg: float = 0,
    distance: float = 2.6,
    marker_size: int = 3,
    colorscale_override=None,  
    # colorbar
    cbar_title: str | None = None,
    cbar_len: float = 0.8,
    cbar_thickness: int = 18,
    cbar_title_size: int = 14,
    cbar_tick_size: int = 10,
    width: int = 400,
    height: int = 300,
    margin_left: int = 10,
    margin_right: int = 10,   
    margin_top: int = 0,
    margin_bottom: int = 0,
    out_pdf: str | None = None,
    out_html: str | None = None,
):
    
    
    coords, colour, caption, p, p_cbar, colorscale, A_cat, B_cat, quads, g, f_used = \
        build_pair_colour_from_bundles(
            bundle_a, bundle_b,
            kind=kind, mode=mode, f=f,
            low_half_first=low_half_first, c_keep=c_keep
        )

    
    if coords.shape[1] < 3:
        raise ValueError("not enough dimensions.")
    X, Y, Z = coords[:, 0], coords[:, 1], coords[:, 2]

    # view point
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    eye = dict(
        x=distance * math.cos(el) * math.cos(az),
        y=distance * math.cos(el) * math.sin(az),
        z=distance * math.sin(el),
    )

    
    cb_title = cbar_title if cbar_title is not None else caption

    fig = go.Figure(
        go.Scatter3d(
            x=X, y=Y, z=Z,
            mode="markers",
            marker=dict(
                size=marker_size,
                color=colour,
                colorscale= (colorscale_override if colorscale_override is not None else colorscale),
                cmin=0, cmax=int(p_cbar) - 1,
                showscale=True,
                colorbar=dict(
                    title=dict(text=cb_title, side="right", font=dict(size=cbar_title_size)),
                    tickfont=dict(size=cbar_tick_size),
                    len=cbar_len,
                    tickvals=c_keep,
                    thickness=cbar_thickness,
                    x=0.8,  
                ),
            ),
            hovertemplate="PC0=%{x:.3f}<br>PC1=%{y:.3f}<br>PC2=%{z:.3f}<extra></extra>",
        )
    )

    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=dict(text="PC0",font=dict(size=16)), tickfont=dict(size=12)),
            yaxis=dict(title=dict(text="PC1",font=dict(size=16)), tickfont=dict(size=12)),
            zaxis=dict(title=dict(text="PC2",font=dict(size=16)), tickfont=dict(size=12)),
            aspectmode="data",
            camera=dict(eye=eye),
        ),
        width=width, height=height,
        margin=dict(l=margin_left, r=margin_right, t=margin_top, b=margin_bottom),
        title=dict(text="Quadrants affect representations",font=dict(size=16),
                   y=0.93,                
                   x=0.5,
                pad=dict(t=0, b=10)),
        showlegend=False,
    )

    # 导出
    if out_pdf:
        Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
        pio.write_image(fig, out_pdf, format="pdf")  
    if out_html:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_html, include_plotlyjs="cdn")
    return fig

def render_neuron_page_from_json_v(
    page: dict,
    to_pdf_path: str | None = None
) -> go.Figure:
    # 拿出字段
    names = page["heatmaps"]["a_dft"].get("x", [])
    layout = page["layout"]
    dft_cax = page["coloraxis"]["dft"]
    pre_cax = page["coloraxis"]["preact"]

    style = page.get("style", {})
    font_base   = style.get("font_size_base", 18)
    title_fs    = style.get("title_font_size", 18)
    tick_fs     = 14
    marker_sz   = style.get("marker_size", 6)
    dft_angle   = style.get("dft_tick_angle", 45)
    dtick       = 1

    pad = layout.get("padding", {})
    title_yshift = pad.get("title_yshift", 6)
    title_xshift = pad.get("title_xshift", 0)
    axis_standoff= pad.get("axis_title_standoff", 2)

    ROWS, COLS = 4, 2

    # 工具：计算第 r 行（1-based）的中心 y（用于把 colorbar 对齐某一行）
    def _row_center(rows, vspace, r):
        frac_h = (1.0 - (rows - 1) * vspace) / rows
        # 从上往下第 r 行的中心（plotly y 轴 0..1，自下向上）
        # 顶部行的中心是：1 - frac_h/2
        return 1.0 - ((r - 1) * (frac_h + vspace) + frac_h / 2.0), frac_h

    # 建 4×2 子图： 
    #   第1行：a, b（折线）
    #   第2行：a_remap, b_remap（折线）
    #   第3行：a_dft, b_dft（heatmap）
    #   第4行：preact2d, full_dft（heatmap）
    HSPACE = 0.08
    VSPACE = 0.00
    fig = make_subplots(
        rows=ROWS, cols=COLS,
        specs=[
            [{"type":"xy"}, {"type":"xy"}],          # row 1
            [{"type":"xy"}, {"type":"xy"}],          # row 2
            [{"type":"heatmap"}, {"type":"heatmap"}],# row 3
            [{"type":"heatmap"}, {"type":"heatmap"}] # row 4
        ],
        horizontal_spacing=HSPACE,
        vertical_spacing=HSPACE,
        subplot_titles=[
            page["titles"]["a"],         page["titles"]["b"],
            page["titles"]["a_remap"],   page["titles"]["b_remap"],
            page["titles"]["a_dft"],     page["titles"]["b_dft"],
            page["titles"]["preact2d"],  page["titles"]["full_dft"],
        ],
    )

    fig.update_layout(font=dict(size=font_base))

    def _bump_title(a):
        xref = getattr(a, "xref", None)
        if isinstance(xref, str) and xref.endswith(" domain"):
            a.update(font=dict(size=title_fs), yshift=title_yshift, xshift=title_xshift)
    fig.for_each_annotation(_bump_title)
    fig.update_layout(annotations=list(fig.layout.annotations))
    fig.update_xaxes(title_standoff=axis_standoff, title_font=dict(size=title_fs), tickfont=dict(size=tick_fs))
    fig.update_yaxes(title_standoff=axis_standoff, title_font=dict(size=title_fs), tickfont=dict(size=tick_fs))

    # —— 折线：黑线 + marker 颜色挂 preact 的 coloraxis2 —— 
    def _add_line(pack, row, col):
        x1,y1 = np.array(pack["x1"]), np.array(pack["y1"], dtype=float)
        x2,y2 = np.array(pack["x2"]), np.array(pack["y2"], dtype=float)
        fig.add_trace(go.Scatter(
            x=x1, y=y1, mode="markers+lines",
            line=dict(width=1.5, color="black"),
            marker=dict(size=marker_sz, color=y1, coloraxis="coloraxis2"),
            connectgaps=True, showlegend=False
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=x2, y=y2, mode="markers+lines",
            line=dict(width=1.5, color="black"),
            marker=dict(size=marker_sz, color=y2, coloraxis="coloraxis2"),
            connectgaps=True, showlegend=False
        ), row=row, col=col)
        fig.update_yaxes(range=pack["y_range"], row=row, col=col)
        fig.update_xaxes(showgrid=True, row=row, col=col)
        fig.update_yaxes(showgrid=True, row=row, col=col)

    # 折线位置（前两行）
    _add_line(page["lines"]["a"],         1, 1)
    _add_line(page["lines"]["b"],         1, 2)
    _add_line(page["lines"]["a_remap"],   2, 1)
    _add_line(page["lines"]["b_remap"],   2, 2)

    # —— Heatmaps（后两行）—— 
    fig.add_trace(go.Heatmap(z=page["heatmaps"]["a_dft"]["z"], x=names, y=names,
                             showscale=False, coloraxis="coloraxis1"), row=3, col=1)
    fig.add_trace(go.Heatmap(z=page["heatmaps"]["b_dft"]["z"], x=names, y=names,
                             showscale=False, coloraxis="coloraxis1"), row=3, col=2)
    fig.add_trace(go.Heatmap(z=page["heatmaps"]["preact2d"]["z"],
                             showscale=False, coloraxis="coloraxis2"), row=4, col=1)
    fig.add_trace(go.Heatmap(z=page["heatmaps"]["full_dft"]["z"], x=names, y=names,
                             showscale=False, coloraxis="coloraxis1"), row=4, col=2)

    # DFT 的轴刻度（第 3 行 + 第 4 行右侧）
    fig.update_xaxes(tickangle=dft_angle, dtick=2, row=3, col=1)
    fig.update_xaxes(tickangle=dft_angle, dtick=2, row=3, col=2)
    fig.update_xaxes(tickangle=dft_angle, dtick=2, row=4, col=2)
    fig.update_yaxes(dtick=dtick, row=3, col=1)
    fig.update_yaxes(dtick=dtick, row=3, col=2)
    fig.update_yaxes(dtick=dtick, row=4, col=2)

    # —— 颜色轴：两个 colorbar 贴第2/3行 —— 
    y2, frac_h = _row_center(ROWS, layout["vspace"], 2)  # preact colorbar 对齐第2行
    y3, _       = _row_center(ROWS, layout["vspace"], 3)  # dft   colorbar 对齐第3行
    cb_len = max(0.90 * frac_h, 0.2)                     # 让色条长度略短于该行高度

    fig.update_layout(
        coloraxis1=dict(
            colorscale=dft_cax.get("colorscale", "Inferno"),
            colorbar=dict(
                title=dict(text="GFT", side="right"),
                orientation="v",
                len=cb_len, y=y3+0.05, yanchor="middle",
                x=1.02, xanchor="left",
                thickness=12
            )
        ),
        coloraxis2=dict(
            colorscale=pre_cax.get("colorscale", "Viridis"),
            cmin=pre_cax.get("cmin", None),
            cmax=pre_cax.get("cmax", None),
            colorbar=dict(
                title=dict(text="Preactivation", side="right"),
                orientation="v",
                len=cb_len, y=y2, yanchor="middle",
                x=1.02, xanchor="left",
                thickness=12
            )
        ),
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white"
    )

    # —— 四张热图强制像素方形 —— 
    def _anchor_square(row, col):
        idx = (row - 1) * COLS + col
        axk = "x" if idx == 1 else f"x{idx}"
        fig.update_xaxes(constrain="domain", row=row, col=col)
        fig.update_yaxes(constrain="domain", row=row, col=col)
        fig.update_yaxes(scaleanchor=axk, scaleratio=1, row=row, col=col)

    for r,c in [(3,1),(3,2),(4,1),(4,2)]:
        _anchor_square(r,c)

    # def _anchor_square_xy(row, col):
    #     idx = (row - 1) * COLS + col
    #     axk = "x" if idx == 1 else f"x{idx}"
    #     fig.update_xaxes(constrain="domain", row=row, col=col)
    #     fig.update_yaxes(constrain="domain", row=row, col=col)
    #     fig.update_yaxes(scaleanchor=axk, scaleratio=1, row=row, col=col)

    # for r, c in [(1,1),(1,2),(2,1),(2,2)]:
    #     _anchor_square_xy(r, c)
    # 统一网格开关（热图关闭网格）
    for (r,c) in [(3,1),(3,2),(4,1),(4,2)]:
        fig.update_xaxes(showgrid=False, row=r, col=c)
        fig.update_yaxes(showgrid=False, row=r, col=c)

    # 收紧列与色条 & 右边距
    report._tighten_cols_and_colorbars(fig, want_gap12=1, cb_dx=0.03, right_margin=50)

    # 计算宽高（4×2）
    tight_margins = dict(l=28, r=51, t=30, b=30)
    W, H, margins = _compute_fig_size_for_square_cells(
        rows=ROWS, cols=COLS,
        hspace=HSPACE,
        vspace=VSPACE,
        cell_px=180,
        margins=tight_margins
    )
    fig.update_layout(width=W, height=H, margin=margins, showlegend=False)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    

    if to_pdf_path is not None:
        pdf_bytes = fig.to_image(format="pdf", engine="kaleido",
                                 width=W, height=H, scale=1)
        #assert len(PdfReader(io.BytesIO(pdf_bytes)).pages) == 1, "kaleido multiple pages"
        with open(to_pdf_path, "wb") as f:
            f.write(pdf_bytes)

    return fig

def export_sign_irreps_strip_v(
    json_paths,
    out_pdf_path,
    *,
    titles=None,
    cell_px=150,
    hspace=0.03,
    margins=dict(l=28, r=88, t=28, b=24),
    font_size=16, title_fs=16, tick_fs=12,
    show_axes=False,
    orientation: str = "col"  # ← 新增："row"=1×4，"col"=4×1
):
    assert len(json_paths) == 4, "需要正好四个 JSON 文件"

    # 读 preact2d 并做全局 cmin/cmax
    Zs = []
    names = None
    for pth in map(Path, json_paths):
        page = json.loads(Path(pth).read_text(encoding="utf-8"))
        Z = np.array(page["heatmaps"]["preact2d"]["z"], dtype=float)
        Zs.append(Z)
        if names is None and "x" in page["heatmaps"].get("a_dft", {}):
            names = page["heatmaps"]["a_dft"]["x"]

    gmin = min(float(np.nanmin(Z)) for Z in Zs)
    gmax = max(float(np.nanmax(Z)) for Z in Zs)
    if gmin == gmax:
        gmin -= 1e-9; gmax += 1e-9

    # —— 排布：1×4（横排）或 4×1（竖排）——
    if orientation.lower().startswith("col"):
        rows, cols = 4, 1
        specs = [[{"type":"heatmap"}] for _ in range(4)]
        hspace_use = hspace
        vspace_use = 0.05  # 用 hspace 作为竖向间距
        subplot_titles = (titles or [Path(p).stem for p in json_paths])
    else:
        rows, cols = 1, 4
        specs = [[{"type":"heatmap"} for _ in range(4)]]
        hspace_use = hspace
        vspace_use = 0.0
        subplot_titles = (titles or [Path(p).stem for p in json_paths])

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        horizontal_spacing=hspace_use,
        vertical_spacing=vspace_use,
        subplot_titles=subplot_titles,
    )

    # 添加 4 张热力图（共用一个 coloraxis）
    for i, Z in enumerate(Zs, start=1):
        r = i if rows == 4 else 1
        c = 1 if cols == 1 else i
        fig.add_trace(go.Heatmap(
            z=Z, showscale=False, coloraxis="coloraxis"
        ), row=r, col=c)
        ax = "x" if (r-1)*cols + c == 1 else f"x{(r-1)*cols + c}"
        fig.update_xaxes(constrain="domain", row=r, col=c)
        fig.update_yaxes(constrain="domain", row=r, col=c)
        fig.update_yaxes(scaleanchor=ax, scaleratio=1, row=r, col=c)

    # 主题、字体、色条（竖向在右侧）
    # 如果是纵排，让色条尽量占据总高度的 80%；横排则占 80% 宽高中的高度
    if rows == 4:
        frac_h = (1.0 - 3*vspace_use) / 4.0
        cbar_len = max(0.70 * frac_h, 0.50)  # 纵排给更长一些
    else:
        cbar_len = 0.3

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=font_size),
        coloraxis=dict(
            colorscale="Viridis",
            cmin=gmin, cmax=gmax,
            colorbar=dict(
                title=dict(text="Preactivation", side="right"),
                orientation="v",
                len=cbar_len,
                y=0.5, yanchor="middle",
                x=1.0, xanchor="left",
                thickness=14
            )
        )
    )
    # 标题字号微调
    for a in fig.layout.annotations:
        if getattr(a, "xref", "").endswith(" domain"):
            a.font.size = title_fs
            a.yshift = 6

    # 轴显隐
    if not show_axes:
        fig.update_xaxes(showticklabels=True, ticks="", showgrid=False)
        fig.update_yaxes(showticklabels=True, ticks="", showgrid=False)
    else:
        fig.update_xaxes(tickfont=dict(size=tick_fs))
        fig.update_yaxes(tickfont=dict(size=tick_fs))

    # 尺寸（按排布自动计算）
    W, H, _ = _compute_fig_size_for_square_cells(
        rows=rows, cols=cols,
        hspace=hspace_use, vspace=vspace_use,
        cell_px=cell_px, margins=margins
    )
    # 稍缩右边距，给色条靠右
    fig.update_layout(width=W, height=H, margin=dict(l=margins["l"], r=80, t=margins["t"], b=margins["b"]), showlegend=False)

    # fig.update_layout(
    #     width=300,
    #     height=1200
    # )
    pdf_bytes = fig.to_image(format="pdf", engine="kaleido", width=W, height=H, scale=1)
    # assert len(PdfReader(io.BytesIO(pdf_bytes)).pages) == 1, "Kaleido 导出了多页 PDF？"
    Path(out_pdf_path).write_bytes(pdf_bytes)
    #
    # print(f"✅ saved: {out_pdf_path}  (size: {W}×{H})")

import io, json, numpy as np, plotly.graph_objects as go
from plotly.subplots import make_subplots
from PyPDF2 import PdfReader
from pathlib import Path

def render_sign_neuron_stacked(
    page: dict,
    to_pdf_path: str | None = None,
    *,
    cell_px: int = 200,          # 每个子图绘图区像素边长
    vspace: float | None = None  # 行间距（默认取 page["layout"]["vspace"] 或 0.03）
) -> go.Figure:
    names    = page["heatmaps"]["a_dft"].get("x", [])
    layout   = page["layout"]
    dft_cax  = page["coloraxis"]["dft"]
    pre_cax  = page["coloraxis"]["preact"]

    style       = page.get("style", {})
    font_base   = style.get("font_size_base", 18)
    title_fs    = style.get("title_font_size", 18)
    tick_fs     = style.get("tick_font_size", 14)
    marker_sz   = style.get("marker_size", 6)
    dft_angle   = style.get("dft_tick_angle", 45)
    dtick       = style.get("dtick", 1)

    pad = layout.get("padding", {})
    title_yshift   = pad.get("title_yshift", 6)
    title_xshift   = pad.get("title_xshift", 0)
    axis_standoff  = pad.get("axis_title_standoff", 2)

    ROWS, COLS = 4, 1
    VSPACE = float(layout.get("vspace", 0.03) if vspace is None else vspace)

    def _row_center(rows, vspace, r):
        frac_h = (1.0 - (rows - 1) * vspace) / rows
        y_center = 1.0 - ((r - 1) * (frac_h + vspace) + frac_h / 2.0)
        return y_center, frac_h

    fig = make_subplots(
        rows=ROWS, cols=COLS,
        specs=[[{"type":"xy"}],
               [{"type":"xy"}],
               [{"type":"heatmap"}],
               [{"type":"heatmap"}]],
        vertical_spacing=VSPACE,
        horizontal_spacing=0.00,
        subplot_titles=[
            page["titles"]["a"],
            page["titles"]["b"],
            page["titles"]["preact2d"],
            page["titles"]["full_dft"],
        ],
    )
    fig.update_layout(font=dict(size=font_base))
     # 计算尺寸：保证每个子图“画布像素方形”
    tight_margins = dict(l=60, r=30, t=30, b=30)
    W, H, margins = _compute_fig_size_for_square_cells(
        rows=ROWS, cols=COLS,
        hspace=0.0, vspace=VSPACE,
        cell_px=cell_px,
        margins=tight_margins
    )
    fig.update_layout(width=W, height=H, margin=margins, showlegend=False)
    fig.update_xaxes(automargin=False)
    fig.update_yaxes(automargin=False)

    def _bump_title(a):
        if isinstance(getattr(a, "xref", None), str) and a.xref.endswith(" domain"):
            a.update(font=dict(size=title_fs), yshift=title_yshift, xshift=title_xshift)
    fig.for_each_annotation(_bump_title)
    fig.update_xaxes(title_standoff=axis_standoff, title_font=dict(size=title_fs), tickfont=dict(size=tick_fs))
    fig.update_yaxes(title_standoff=axis_standoff, title_font=dict(size=title_fs), tickfont=dict(size=tick_fs))

    # —— 折线：像素方形由整体尺寸保证；不设 scaleanchor，避免把 y 拉成 0..35 —— 
    def _add_line(pack, row):
        x1,y1 = np.array(pack["x1"]), np.array(pack["y1"], dtype=float)
        x2,y2 = np.array(pack["x2"]), np.array(pack["y2"], dtype=float)
        fig.add_trace(go.Scatter(
            x=x1, y=y1, mode="markers+lines",
            line=dict(width=1.5, color="black"),
            marker=dict(size=marker_sz, color=y1, coloraxis="coloraxis2"),
            connectgaps=True, showlegend=False
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=x2, y=y2, mode="markers+lines",
            line=dict(width=1.5, color="black"),
            marker=dict(size=marker_sz, color=y2, coloraxis="coloraxis2"),
            connectgaps=True, showlegend=False
        ), row=row, col=1)
        fig.update_yaxes(range=pack["y_range"], showgrid=True, row=row, col=1)
        fig.update_xaxes(showgrid=True, row=row, col=1)

    _add_line(page["lines"]["a"], 1)
    _add_line(page["lines"]["b"], 2)

    # —— 热图 —— 
    fig.add_trace(go.Heatmap(
        z=page["heatmaps"]["preact2d"]["z"],
        showscale=True, coloraxis="coloraxis2"
    ), row=3, col=1)
    fig.add_trace(go.Heatmap(
        z=page["heatmaps"]["full_dft"]["z"], x=names, y=names,
        showscale=True, coloraxis="coloraxis1"
    ), row=4, col=1)

    fig.update_xaxes(tickangle=dft_angle, dtick=2, row=4, col=1)
    fig.update_yaxes(dtick=dtick, row=4, col=1)

    # —— 色条对齐到第2/3行 —— 
    y2, frac_h = _row_center(ROWS, VSPACE, 2)
    y3, _       = _row_center(ROWS, VSPACE, 3)
    cb_len = min(0.90 * frac_h, 0.22)  # 相比你原来更合理的长度

    fig.update_layout(
        coloraxis1=dict(  # DFT
            colorscale=dft_cax.get("colorscale", "Inferno"),
            colorbar=dict(
                title=dict(text="GFT", side="right"),
                orientation="v", len=cb_len, y=y3, yanchor="middle",
                x=1.02, xanchor="left", thickness=12
            )
        ),
        coloraxis2=dict(  # Preactivation
            colorscale=pre_cax.get("colorscale", "Viridis"),
            cmin=pre_cax.get("cmin", None), cmax=pre_cax.get("cmax", None),
            colorbar=dict(
                title=dict(text="Preactivation", side="right"),
                orientation="v", len=cb_len, y=y2, yanchor="middle",
                x=1.02, xanchor="left", thickness=12
            )
        ),
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white"
    )

    # 热图像素正方（只对第3/4行）
    for r in (3,4):
        idx = (r-1)*COLS + 1
        ax = "x" if idx == 1 else f"x{idx}"
        fig.update_xaxes(constrain="domain", row=r, col=1)
        fig.update_yaxes(constrain="domain", row=r, col=1)
        fig.update_yaxes(scaleanchor=ax, scaleratio=1, row=r, col=1)
        fig.update_xaxes(showgrid=False, row=r, col=1)
        fig.update_yaxes(showgrid=False, row=r, col=1)


    if to_pdf_path:
        pdf_bytes = fig.to_image(format="pdf", engine="kaleido", width=W, height=H, scale=1)
        with open(to_pdf_path, "wb") as f:
            f.write(pdf_bytes)

    return fig



if __name__ == "__main__":

    # main("/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_512_features_128_k_59")
    path = '/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2500_training_set_size=1062/graphs_seed_1/cluster_2D_6_2D_6_embeds/bundles/full/bundle_seed_2D_6_2D_6_full.json'
    path1 = '/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2500_training_set_size=1062/graphs_seed_1/pdf_plots/seed_1/bundles/cluster_contributions_to_logits_freq=6/bundle_seed_1_cluster_contributions_to_logits_freq=6.json'
    
    # pca_diffusion_plots_w_helpers.rebuild_embedding_html_from_bundle(path,
    #                                                                  kind="pca",
    #                                                                  f=6,
    #                                                                  out_html="reconstructed_pca_f6_c_new.html")
    # make_three_panel_pca_pdf(
    #     path,
    #     path1,
    #     "three_panel_pc012_0_vi_new.pdf",
    #     f=6
    # )
    # make_four_panel_pca_pdf(
    #     path,
    #     path1,
    #     "4_panel_pc012_0_vi_new-t.pdf",
    #     f=6
    # )

    logits_f1='/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2500_training_set_size=1062/graphs_seed_0_c/pdf_plots/seed_0/bundles/cluster_contributions_to_logits_freq=1/bundle_seed_0_cluster_contributions_to_logits_freq=1.json'
    make_pca_4x3_from_bundle(
        logits_f1,
        "f1_logits_vi.pdf",
        f=1,
    )
    # sign_path =[
    #         "/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2000_training_set_size=1062/p_graphs_seed_43_8.14/cluster_sign_sign/127/page.json",
    #         "/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2000_training_set_size=1062/p_graphs_seed_43_8.14/cluster_sign_sign/100/page.json",
    #         "/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2000_training_set_size=1062/p_graphs_seed_43_8.14/cluster_sign_sign/22/page.json",
    #         "/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2000_training_set_size=1062/p_graphs_seed_43_8.14/cluster_sign_sign/6/page.json",
    #     ]
    with open("/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2000_training_set_size=1062/p_graphs_seed_43_8.14/cluster_sign_sign/127/page.json","r",encoding="utf-8") as f:
        page = json.load(f)

    render_neuron_page_from_json_v(page, "2-sign_neuron_stack.pdf")

    # # export_sign_irreps_strip(
    # #     sign_path,
    # #     out_pdf_path="sign_strip.pdf",
    # #     titles=["sign, +1", "sign, +1", "sign, -1", "sign, -1"],  
    # #     cell_px=230,  
    # #     hspace=0.03,  
    # #     margins=dict(l=28, r=92, t=24, b=20),  
    # #     show_axes=False
    # # )
    # export_sign_irreps_strip_v(
    #     sign_path,
    #     out_pdf_path="2sign_strip_v.pdf",
    #     titles=["sign, +1", "sign, +1", "sign, -1", "sign, -1"],  
    #     cell_px=180,  
    #     hspace=0.0,  
    #     margins=dict(l=28, r=92, t=24, b=20),  
    #     show_axes=False
    # )


    # allowed = [0,  15, 18,  33]
    # path_bl='/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2500_training_set_size=1062/graphs_seed_0/cluster_2D_2_2D_2_embeds/bundles/BL/bundle_seed_2D_2_2D_2_BL.json'
    # path_tr='/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2500_training_set_size=1062/graphs_seed_0/cluster_2D_2_2D_2_embeds/bundles/TR/bundle_seed_2D_2_2D_2_TR.json'
    # fig = rebuild_pair_html_from_bundles(
    #     path_bl,
    #     path_tr,
    #     kind="pca",         
    #     mode="c",           
    #     f=None,             
    #     low_half_first=True,
    #     out_html="bltr_pair.html",
    #     c_keep=allowed
    # )

    # path_bl="/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2500_training_set_size=1062/graphs_seed_0/cluster_2D_1_2D_1_embeds/bundles/BL/bundle_seed_2D_1_2D_1_BL.json"
    # path_tr="/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2500_training_set_size=1062/graphs_seed_0/cluster_2D_1_2D_1_embeds/bundles/TR/bundle_seed_2D_1_2D_1_TR.json"
    # c_keep=[0, 15, 18, 33]
    # # c_keep=[0, 5, 8, 13]
    # # path_bl='/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2500_training_set_size=1062/graphs_seed_0/cluster_2D_2_2D_2_embeds/bundles/BL/bundle_seed_2D_2_2D_2_BL.json'
    # # path_tr='/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2500_training_set_size=1062/graphs_seed_0/cluster_2D_2_2D_2_embeds/bundles/TR/bundle_seed_2D_2_2D_2_TR.json'
    # fig = make_single_pc012_3d_from_bundles(
    #     path_bl,
    #     path_tr,
    #     kind="pca",
    #     mode="c",            
    #     f=None,               
    #     low_half_first=True,  
    #     c_keep=c_keep,  
    #     azimuth_deg=45, elevation_deg=5, distance=2.6,
    #     cbar_title="C mod 18",   
    #     out_pdf="pc012_single_with_cbar_vi.pdf",
    #     out_html="bltr_pair_5_vi.html"
    # )
    # import plotly.io as pio
    # pio.kaleido.scope.mathjax = None
    # page1 = "/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2000_training_set_size=1062/p_graphs_seed_43_8.14/cluster_2D_5_2D_5/3/page.json"
    
    # page2 = "/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2000_training_set_size=1062/p_graphs_seed_43_8.14/cluster_2D_6_2D_6/2/page.json"
    # with open(page1, "r", encoding="utf-8") as f:
    #     page_dict = json.load(f)
    # render_neuron_page_from_json_v(page_dict, "2d5_mod_v.pdf")

    # with open(page2, "r", encoding="utf-8") as f:
    #     page_dict = json.load(f)
    # render_neuron_page_from_json_v(page_dict, "2d6_mod_v.pdf")


    
# if __name__ == "__main__":
#     export_sign_irreps_strip(
#         [
#             "/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2000_training_set_size=1062/p_graphs_seed_43_8.14/cluster_sign_sign/127/page.json",
#             "/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2000_training_set_size=1062/p_graphs_seed_43_8.14/cluster_sign_sign/100/page.json",
#             "/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2000_training_set_size=1062/p_graphs_seed_43_8.14/cluster_sign_sign/22/page.json",
#             "/home/mila/w/weis/scratch/DL/qualitative_18_two_embed_1_128_features_128_k_59/18_models_embed_128p=18_bs=18_nn=128_wd=0.0001_epochs=2000_training_set_size=1062/p_graphs_seed_43_8.14/cluster_sign_sign/6/page.json",
#         ],
#         out_pdf_path="sign_strip.pdf",
#         titles=["sign-0", "sign-1", "sign-2", "sign-3"],  
#         cell_px=230,  
#         hspace=0.03,  
#         margins=dict(l=28, r=92, t=24, b=20),  
#         show_axes=False
#     )