import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 我们仍然用它来设置整体风格
import argparse
import os
from pathlib import Path
import matplotlib.font_manager as fm
import math

# --- [!!! 关键设置 !!!] ---
# (您的设置是正确的，请保留)
CHINESE_FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
# ------------------------

# --- 检查路径 ---
my_font = None
if os.path.exists(CHINESE_FONT_PATH):
    my_font = fm.FontProperties(fname=CHINESE_FONT_PATH)
    print(f"--- 成功加载字体文件: {CHINESE_FONT_PATH} ---")
else:
    print(f"!!! 错误: 找不到字体文件: {CHINESE_FONT_PATH}")
    
# 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False 
# --------------------------------


def simplify_nvtx_text(text):
    """
    将 NVTX 范围名称简化为 A-E 几个主要组件。
    """
    if text.startswith('A_'): return 'A_Prep_CPU_Decode'
    if text.startswith('B_'): return 'B_DataCopy_HtoD'
    if text.startswith('C_'): return 'C_Prep_GPU_Transform'
    if text.startswith('D_'): return 'D_Inference'
    if text.startswith('E_'): return 'E_Sync'
    return 'Other'

def load_data_from_sqlite(sqlite_path):
    """
    从 Nsight Systems 导出的 SQLite 文件中查询 NVTX 数据。
    """
    print(f"--- 正在连接到 SQLite 数据库: {sqlite_path} ---")
    
    if not os.path.exists(sqlite_path):
        print(f"!!! 错误: 找不到 SQLite 文件: {sqlite_path}")
        return None

    QUERY = """
    SELECT
        text,
        (end - start) AS duration_ns
    FROM
        NVTX_EVENTS
    WHERE
        text LIKE 'A_%' OR
        text LIKE 'B_%' OR
        text LIKE 'C_%' OR
        text LIKE 'D_%' OR
        text LIKE 'E_%'
    ORDER BY
        start;
    """
    
    try:
        with sqlite3.connect(f'file:{sqlite_path}?mode=ro', uri=True) as conn:
            df = pd.read_sql_query(QUERY, conn)
            
        print(f"--- 成功加载了 {len(df)} 条 NVTX 事件 ---")
        return df
        
    except Exception as e:
        print(f"!!! 加载数据时发生未知错误: {e}")
        return None

def process_data(df):
    """
    处理原始 DataFrame，添加 'step', 'component', 'duration_ms' 列。
    """
    if df is None or df.empty:
        print("--- DataFrame 为空, 跳过处理 ---")
        return None
        
    print("--- 正在处理数据 (计算毫秒, 简化名称, 标记 Step)... ---")
    
    df['duration_ms'] = df['duration_ns'] / 1_000_000
    df['component'] = df['text'].apply(simplify_nvtx_text)
    df['step'] = (df['component'] == 'A_Prep_CPU_Decode').cumsum() - 1
    final_df = df[['step', 'component', 'duration_ms']]
    
    if not final_df.empty:
        step_counts = final_df['step'].value_counts()
        if 0 not in step_counts:
             print("!!! 警告: 在数据中未找到 'step 0'。数据可能不完整。 !!!")
             return None
        first_step_count = step_counts.loc[0]
        valid_steps = step_counts[step_counts == first_step_count].index
        final_df = final_df[final_df['step'].isin(valid_steps)]
    
    print("--- 数据处理完成 ---")
    return final_df

# [!!! 函数已重写 !!!]
def plot_data(df, output_image_path, title_suffix):
    """
    使用处理后的 DataFrame 绘制两个图。
    (已重写为纯 Matplotlib，以强制应用字体)
    """
    if df is None or df.empty:
        print("--- 无数据可供绘图 ---")
        return
        
    if my_font is None:
        print("!!! 字体未加载, 无法绘图。请检查脚本顶部的 CHINESE_FONT_PATH。 !!!")
        return
        
    print(f"--- 正在生成图表 (纯 Matplotlib 模式)... ---")
    
    component_order = sorted(df['component'].unique())
    
    # 使用 seaborn 设置一个好看的网格背景
    sns.set_style("whitegrid")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # 设置总标题
    fig.suptitle(
        f"流水线性能分析\n({title_suffix})", 
        fontsize=18, 
        y=1.02, 
        fontproperties=my_font
    )
    
    # --- 图 1: 折线图 (使用 ax1.plot 替换 sns.lineplot) ---
    for component in component_order:
        component_data = df[df['component'] == component]
        ax1.plot(
            component_data['step'], 
            component_data['duration_ms'], 
            marker='o', 
            label=component
        )
    
    ax1.set_title("图 1: 各阶段耗时 (用于观察 JIT 预热/稳定性)", fontproperties=my_font)
    ax1.set_ylabel("耗时 (毫秒)", fontproperties=my_font)
    ax1.set_xlabel("Step 编号", fontproperties=my_font)
    # 为图例设置字体
    legend = ax1.legend(title='阶段', prop=my_font)
    plt.setp(legend.get_title(), fontproperties=my_font) # 图例标题也需要设置

    
    # --- 图 2: 堆叠条形图 (使用 ax2.bar 替换 df_pivot.plot) ---
    # 1. 数据透视
    df_pivot = df.pivot_table(
        index='step', 
        columns='component', 
        values='duration_ms',
        fill_value=0
    )
    df_pivot = df_pivot.reindex(columns=component_order, fill_value=0)
    
    # 2. 循环绘制堆叠条形图
    bottom = pd.Series(0, index=df_pivot.index, dtype=float)
    
    for component in component_order:
        values = df_pivot[component]
        ax2.bar(
            df_pivot.index, 
            values, 
            width=0.8, 
            label=component, 
            bottom=bottom
        )
        # 更新堆叠的底部
        bottom += values
        
    ax2.set_title("图 2: 每个 Step 耗时堆叠图 (用于观察瓶颈)", fontproperties=my_font)
    ax2.set_ylabel("总耗时 (毫秒)", fontproperties=my_font)
    ax2.set_xlabel("Step 编号", fontproperties=my_font)
    
    # 设置 X 轴刻度 (pandas.plot 会自动做这个, 我们需要手动做)
    ax2.set_xticks(df_pivot.index)
    ax2.set_xticklabels(df_pivot.index)

    # --- 为所有刻度标签应用字体 ---
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontproperties(my_font)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontproperties(my_font)
    
    # 调整布局并保存
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    print(f"--- 正在保存图表到: {output_image_path} ---")
    plt.savefig(output_image_path)
    print("--- 图表已保存 ---")



def print_stable_averages(df, warmup_steps=10, iqr_multiplier=1.5):
    """
    计算并打印稳定状态下的平均耗时，同时筛除预热阶段和统计异常值。

    :param df: process_data 返回的 DataFrame
    :param warmup_steps: 要跳过的初始 'step' 数量 (用于 JIT 预热)
    :param iqr_multiplier: IQR 乘数, 用于定义异常值 (1.5 是标准值)
    """
    if df is None or df.empty:
        print("--- 无数据可供计算平均值 ---")
        return

    print("\n" + "---" * 10)
    print(f"--- 正在计算稳定状态平均值 (跳过前 {warmup_steps} steps) ---")

    # 1. 移除预热阶段 (JIT, 缓存等)
    stable_df = df[df['step'] >= warmup_steps].copy()

    if stable_df.empty:
        print(f"--- 警告: 移除预热 (前 {warmup_steps} steps) 后, 没有剩余数据。---")
        print("---" * 10)
        return

    print(f"--- 预热数据已移除, 剩余 {len(stable_df)} 条记录 ---")
    print(f"--- 正在使用 {iqr_multiplier} * IQR 方法筛除统计异常值... ---")

    # 2. 筛除统计异常值 (IQR 方法)
    # 我们需要按 'component' 分组计算各自的 IQR
    
    Q1 = stable_df.groupby('component')['duration_ms'].transform('quantile', 0.25)
    Q3 = stable_df.groupby('component')['duration_ms'].transform('quantile', 0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR

    # 使用 .between() 进行高效筛选
    is_not_outlier = stable_df['duration_ms'].between(lower_bound, upper_bound)
    filtered_df = stable_df[is_not_outlier]

    original_count = len(stable_df)
    filtered_count = len(filtered_df)
    
    if original_count > 0:
        outlier_count = original_count - filtered_count
        outlier_percent = (outlier_count / original_count) * 100
        print(f"--- 已筛除 {outlier_count} 个统计异常值 ({outlier_percent:.2f}%) ---")
    else:
        print("--- 无需筛除统计异常值 ---")


    # 3. 计算最终平均值
    if filtered_df.empty:
        print(f"--- 警告: 筛除异常值后, 没有剩余数据。---")
        print("---" * 10)
        return
        
    final_averages = filtered_df.groupby('component')['duration_ms'].mean()

    print("\n--- [稳定状态平均耗时 (ms)] ---")
    
    # 4. 格式化输出
    # 计算总时间以便计算百分比
    total_avg = final_averages.sum()
    
    if total_avg == 0:
        print("--- 总平均耗时为 0, 无法计算百分比。 ---")
        print(final_averages)
        print("---" * 10)
        return

    # 按组件名称 (A, B, C...) 排序
    final_averages = final_averages.sort_index()

    for component, avg_ms in final_averages.items():
        percentage = (avg_ms / total_avg) * 100
        # 格式化: 
        # {<20} = 左对齐, 宽度20
        # {:>8.4f} = 右对齐, 宽度8, 4位小数
        # {:5.2f} = 宽度5, 2位小数
        print(f"  {component:<20}: {avg_ms:>8.4f} ms ({percentage:5.2f} %)")
        
    print("-" * 34)
    print(f"  {'Total Pipeline':<20}: {total_avg:>8.4f} ms (100.00 %)")
    print("---" * 10)





def main():
    parser = argparse.ArgumentParser(description="从 Nsight Systems SQLite 报告中分析并绘制 NVTX 性能。")
    parser.add_argument(
        '-i', '--input', 
        type=str, 
        required=True, 
        help="输入的 .sqlite 文件路径 (由 nsys export 生成)。"
    )
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        default=None, 
        help="输出的 .png 图像文件路径。如果未提供, 将基于输入文件名自动生成。"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"plot_{input_path.stem}.png")
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    raw_df = load_data_from_sqlite(input_path)
    processed_df = process_data(raw_df)
    plot_data(processed_df, output_path, title_suffix=input_path.name)

    # 在这里调用新函数，计算并打印平均值
    # 您可以调整 warmup_steps 参数，例如改为 5 或 20
    print_stable_averages(processed_df, warmup_steps=5)

if __name__ == "__main__":
    main()