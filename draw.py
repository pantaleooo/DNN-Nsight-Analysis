import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 我们仍然用它来设置整体风格
import argparse
import os
import re  # 导入正则表达式库
from pathlib import Path
import matplotlib.font_manager as fm


# --- [!!! 关键设置 !!!] ---
# (请确保您的系统中存在此字体路径)
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


def simplify_component_name(text):

    """
    [!!! 已修改 (2025-11-13) !!!]
    将 NVTX 标签简化为用于绘图的类别名称。

    - [!!! 核心修改 !!!] 
    - 新增对 'A_Load_Decode_Batch' 的支持。
    - 'A_' 开头的检查必须更具体，以区分 Load 和 Preprocessing。
    - 'B_' 开头的检查也必须更具体，以区分 (Raw) 和 (Processed) HtoD。
    """
    if text is None:
        return 'Other'

    # --- [A] 阶段 (必须按特异性排序) ---
    if text.startswith('A_Load_Decode_Batch'):
        return 'A_Load_Decode' # 重命名

    if text.startswith('A_Pre_ToTensor_Stack'): # [!!! 新增 !!!]
        return 'A_Pre_ToTensor_Stack'

    if text.startswith('A_Pre_Label_Processing'): # [!!! 新增 !!!]
        return 'A_Pre_Label_Processing'

    if text.startswith('A_Preprocessing_Batch'): # 旧的组件
        return 'A_Pre_Transforms (CPU)'

    # --- [B] 阶段 (必须按特异性排序) ---
    if 'B_data_copy_HtoD (Processed Batch)' in text:
        return 'B_HtoD (Processed)' # CPU Prep 路径

    if 'B_data_copy_HtoD (RAW Frame Batch)' in text:
        return 'B_HtoD (Raw Frame)' # GPU Prep 路径

    # --- [C] 阶段 ---
    if text.startswith('C_Preprocessing_GPU'): 
        return 'C_Pre_Transforms (GPU)'

    # --- [D] 阶段 ---
    if text.startswith('D_retrain'): 
        return 'D_Retrain'
    if text.startswith('D_inference'): 
        return 'D_Inference'

    # --- [E] 阶段 (!!! 新增类别 !!!) ---
    if text.startswith('E_Sync_Wait'): # 必须在 'E_Sync_and_Log' 之前
        return 'E_Sync_Wait (GPU)'
    if text.startswith('E_Sync_and_Log'):
        return 'E_Sync_Log'

    # --- [Step] 标记 ---
    if text.startswith('step_batch_'):
        return '_StepMarker'
    
    # --- 后备 (Fallback) ---
    # 移除模糊的 'startswith' 检查
    print(f"--- 警告: 发现未知的 NVTX 标签: {text}. 将其归类为 'Other'。")
    return 'Other'





def load_data_from_sqlite(sqlite_path):
    """
    [!!! 已修改 (2025-11-13) !!!] 
    - 鉴于 'nsys stats' 显示数据存在，但 'REGEXP' 查询失败，
      我们移除 'REGEXP'。
    - 使用 'WHERE text LIKE "M_%"' 来加载所有相关的 NVTX 事件。
    - 完整的 regex 过滤将推迟到 pandas 的 process_data 函数中完成。
    """
    print(f"--- 正在连接到 SQLite 数据库: {sqlite_path} ---")

    if not os.path.exists(sqlite_path):
        print(f"!!! 错误: 找不到 SQLite 文件: {sqlite_path}")
        return None

    # [!!! 核心修改 !!!]
    # 不再使用 REGEXP。使用一个更简单、更通用的查询。
    # 我们将在 pandas 中进行 regex 匹配，而不是在 SQL 中。
    QUERY = """
    SELECT
        text,
        (end - start) AS duration_ns
    FROM
        NVTX_EVENTS
    WHERE
        text LIKE 'M_%'
    ORDER BY
        start;
    """
    
    # [!!! 核心修改 !!!] 'regexp_func' 和 'create_function' 已被移除

    try:
        # [!!! 核心修改 !!!] 
        # 使用一个简单的 connect 即可
        with sqlite3.connect(sqlite_path) as conn:
            df = pd.read_sql_query(QUERY, conn)

        # [!!! 核心修改 !!!] 
        # 更新了日志消息以反映新的查询
        print(f"--- 成功加载了 {len(df)} 条 'M_%' NVTX 事件 (等待 pandas 过滤) ---")

        if len(df) == 0:
            print("!!! 警告: 未查询到任何匹配 'M_%' 格式的 NVTX 事件。")
            print("!!!   (已使用 SQL 搜索: 'WHERE text LIKE 'M_%'')")
            print("!!! 请检查: 1. 'nsys stats' 的输出是否确实显示了 'M_...' 事件？")
            print("!!!         2. test.py 是否正确添加了 NVTX_PREFIX？")
            return None

        return df

    except Exception as e:
        print(f"!!! 加载数据时发生未知错误: {e}")
        return None




def process_data(df):
    """
    [!!! 已修改 (2025-11-13) !!!]
    
    - [核心修改] 修复了 'ValueError: Can only compare identically-labeled Series objects'
      通过为 'components_df' 和 'steps_df' 分别计算 'transform' 来实现。
    """
    if df is None or df.empty:
        print("--- DataFrame 为空, 跳过处理 ---")
        return None, None

    print("--- 正在处理数据 (解析 Worker ID, 简化名称, 按 _StepMarker 标记 Step)... ---")

    # 1. 计算毫秒
    df['duration_ms'] = df['duration_ns'] / 1_000_000

    # 2. 解析 worker_id 和 component_full
    pattern = re.compile(r"^(?P<worker_id>M_.+_P_\d+)_(?P<component_full>step_batch_\d+|[A-E]_.+)")
    extracted = df['text'].str.extract(pattern)
    df = df.join(extracted)

    # 3. 丢弃无法解析的行
    df = df.dropna(subset=['worker_id', 'component_full']).copy()
    if df.empty:
        print("!!! 错误: Regex 未能从 NVTX 标签中解析出任何 'M_..._P_..._A_...' 或 '..._step_batch_...' 模式。")
        print("!!!   (Regex: M_.+_P_\\d+_(step_batch_\\d+|[A-E]_.+))")
        return None, None

    # 4. 简化组件名称
    df['component'] = df['component_full'].apply(simplify_component_name)

    # 5. 按 worker_id 分组计算 step
    print("--- 正在按 (component == '_StepMarker') 逻辑标记 Step... ---")
    df['is_step_marker'] = (df['component'] == '_StepMarker')
    df['step'] = df.groupby('worker_id')['is_step_marker'].cumsum()
    
    # 将数据分为 _StepMarker 和 A,B,C,D,E 组件
    step_marker_df = df[df['component'] == '_StepMarker'].copy()
    # [!!!] 'components_df' 现在包含 A, B, C, D 和 E
    components_df = df[df['component'] != '_StepMarker'].copy()
    
    components_df = components_df.drop(columns=['is_step_marker'])
    step_marker_df = step_marker_df.drop(columns=['is_step_marker'])

    # 6. 过滤掉不完整的 steps (新逻辑)
    # 我们不再假设 'expected_components = 4'。
    # 我们只保留那些 *至少有一个* A..E 组件的 step。
    
    # 找出哪些 (worker_id, step) 组合 *至少有一个* A..E 组件
    valid_steps = components_df[['worker_id', 'step']].drop_duplicates()

    if components_df.empty or valid_steps.empty:
        print("!!! 错误: 找到了 Step 标记 (step_batch_...), 但未找到任何 A/B/C/D/E 组件。")
        return None, None

    # 'components_df' 已经是 'final' 的了，因为它只包含 A..E
    final_components_df = components_df.copy()
    
    # 使用 'inner' merge 来过滤 step_marker_df，只保留匹配的 steps
    final_steps_df = step_marker_df.merge(valid_steps, on=['worker_id', 'step'], how='inner')

    print(f"--- 数据处理完成 (保留了 {len(valid_steps)} 个有效的 [worker, step] 组合) ---")


    # 7. 过滤每个 worker 的首个 step (JIT 预热)
    if not final_components_df.empty:
        
        # --- [!!! 核心修复 1/2 !!!] ---
        per_worker_min_step_COMPONENTS = final_components_df.groupby('worker_id')['step'].transform('min')
        unique_min_steps = sorted(per_worker_min_step_COMPONENTS.unique())
        print(f"--- Z_g_guolv_mei_ge_worker_de_zui_zao_step_ {unique_min_steps} (jia_ding_wei_JIT/_yu_re)... ---")
        
        final_components_df = final_components_df[final_components_df['step'] > per_worker_min_step_COMPONENTS].copy()

        # --- 步骤 7b: 过滤 final_steps_df (使用 *它自己* 的 transform) ---
        if not final_steps_df.empty:
            per_worker_min_step_STEPS = final_steps_df.groupby('worker_id')['step'].transform('min')
            final_steps_df = final_steps_df[final_steps_df['step'] > per_worker_min_step_STEPS].copy()

        if final_components_df.empty:
            print("--- 警告: 过滤预热后, 没有剩余数据。---")
            return None, None

        # --- [!!! 核心修复 2/2 !!!] ---
        
        # --- 步骤 7c: 重新编号 final_components_df ---
        per_worker_min_step_after_COMPONENTS = final_components_df.groupby('worker_id')['step'].transform('min')
        final_components_df['step'] = (final_components_df['step'] - per_worker_min_step_after_COMPONENTS).astype(int)
        
        # --- 步骤 7d: 重新编号 final_steps_df ---
        if not final_steps_df.empty:
            per_worker_min_step_after_STEPS = final_steps_df.groupby('worker_id')['step'].transform('min')
            final_steps_df['step'] = (final_steps_df['step'] - per_worker_min_step_after_STEPS).astype(int)

    # 返回两个已清理的 DataFrame
    return final_components_df[['worker_id', 'step', 'component', 'duration_ms']], \
           final_steps_df[['worker_id', 'step', 'component', 'duration_ms']]


# [!!! 函数无需修改 !!!]
def plot_data(df, output_image_path, title_suffix):
    """
    使用处理后的 DataFrame 绘制两个图。
    [!!! 无需修改 !!!] 
    绘图逻辑 (聚合、matplotlib) 仍然正确。
    它已足够健壮，可以处理 'D_Retrain' 组件的出现。
    """
    if df is None or df.empty:
        print("--- 无数据可供绘图 ---")
        return

    if my_font is None:
        print("!!! 字体未加载, 无法绘图。请检查脚本顶部的 CHINESE_FONT_PATH。 !!!")
        return

    print(f"--- 正在生成图表 (纯 Matplotlib 模式)... ---")

    try:
        # [!!!] 这里的 'step' 列现在是由新的 process_data 逻辑生成的
        df_agg = df.groupby(['step', 'component'])['duration_ms'].mean().reset_index()
        num_workers = df['worker_id'].nunique()
        print(f"--- 绘图数据已聚合 (平均 {num_workers} 个 worker) ---")
    except Exception as e:
        print(f"!!! 聚合数据时出错: {e}")
        print("--- DataFrame 详情 ---")
        print(df.info())
        return

    # [!!!] 这一行现在会自动包含 D_Retrain (如果存在)，并且不会包含 E_Sync
    component_order = sorted(df_agg['component'].unique())

    sns.set_style("whitegrid")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    fig.suptitle(
        f"流水线性能分析 ({num_workers} 个 Worker 平均值)\n({title_suffix})",
        fontsize=18,
        y=1.02,
        fontproperties=my_font
    )

    # --- 图 1: 折线图 (绘制聚合后的 df_agg) ---
    for component in component_order:
        component_data = df_agg[df_agg['component'] == component]
        ax1.plot(
            component_data['step'],
            component_data['duration_ms'],
            marker='o',
            label=component
        )

    ax1.set_title("图 1: 各阶段平均耗时 (用于观察稳定性)", fontproperties=my_font)
    ax1.set_ylabel("平均耗时 (毫秒)", fontproperties=my_font)
    ax1.set_xlabel("Step 编号 (已跳过预热)", fontproperties=my_font)
    legend = ax1.legend(title='阶段', prop=my_font)
    plt.setp(legend.get_title(), fontproperties=my_font)  # 图例标题也需要设置

    # --- 图 2: 堆叠条形图 (绘制聚合后的 df_agg) ---
    df_pivot = df_agg.pivot_table(
        index='step',
        columns='component',
        values='duration_ms',
        fill_value=0
    )
    # [!!!] 这一行现在会自动包含 D_Retrain
    df_pivot = df_pivot.reindex(columns=component_order, fill_value=0)

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
        bottom += values

    ax2.set_title(f"图 2: 每个 Step 平均耗时堆叠图 ({num_workers} 个 Worker 平均值)", fontproperties=my_font)
    ax2.set_ylabel("平均总耗时 (毫秒)", fontproperties=my_font)
    ax2.set_xlabel("Step 编号 (已跳过预热)", fontproperties=my_font)

    max_step = df_pivot.index.max()
    if max_step > 20:
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    else:
        # 确保 xticks 是整数
        if not df_pivot.empty:
            int_steps = df_pivot.index.astype(int)
            ax2.set_xticks(int_steps)
            ax2.set_xticklabels(int_steps)

    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontproperties(my_font)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontproperties(my_font)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    print(f"--- 正在保存图表到: {output_image_path} ---")
    plt.savefig(output_image_path)
    print("--- 图表已保存 ---")


# [!!! 函数无需修改 !!!]
def print_stable_averages(df, steps_df, warmup_steps=10, iqr_multiplier=1.5):
    """
    [!!! 已修改 (2025-11-13) !!!]
    
    - [核心修改] 函数签名现在接受 'df' (A,B,C,D组件) 和 'steps_df' (_StepMarker)。
    - 对两个 DataFrame 分别应用 warmup 过滤和 IQR 异常值筛除。
    - 在最终摘要中同时打印 'Total Pipeline (Sum)' 和 '_StepMarker (Total)'。
    """
    
    step_average = None
    final_averages = pd.Series(dtype=float)
    num_workers = 0

    # --- 1. 处理 A, B, C, D, E 组件 ---
    if df is None or df.empty:
        print("--- [Components] 无组件数据可供计算平均值 ---")
    else:
        # ( ... 内部逻辑 (过滤 warmup, IQR 筛除) ... )
        # ( ... 这部分逻辑是健壮的，无需修改 ... )
        # [--- 省略内部过滤代码 ---]
        
        # 假设过滤已完成:
        print("\n" + "---" * 10)
        print(f"--- [Components] 正在计算组件 (A,B,C,D,E) 稳定状态平均值 (再跳过前 {warmup_steps} steps) ---")
        stable_df = df[df['step'] >= warmup_steps].copy()
        if stable_df.empty:
            print(f"--- [Components] 警告: 移除预热后, 没有剩余数据。---")
        else:
            num_workers = df['worker_id'].nunique()
            print(f"--- [Components] 预热数据已移除, 剩余 {len(stable_df)} 条记录 (来自 {num_workers} 个 worker) ---")
            print(f"--- [Components] 正在使用 {iqr_multiplier} * IQR 方法筛除统计异常值... ---")
            
            Q1 = stable_df.groupby('component')['duration_ms'].transform('quantile', 0.25)
            Q3 = stable_df.groupby('component')['duration_ms'].transform('quantile', 0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            is_not_outlier = stable_df['duration_ms'].between(lower_bound, upper_bound)
            filtered_df = stable_df[is_not_outlier]
            
            original_count = len(stable_df)
            filtered_count = len(filtered_df)
            if original_count > 0:
                 outlier_count = original_count - filtered_count
                 outlier_percent = (outlier_count / original_count) * 100
                 print(f"--- [Components] 已筛除 {outlier_count} 个统计异常值 ({outlier_percent:.2f}%) ---")
            
            if filtered_df.empty:
                 print(f"--- [Components] 警告: 筛除异常值后, 没有剩余数据。---")
            else:
                 final_averages = filtered_df.groupby('component')['duration_ms'].mean()


    # --- 2. [!!!] 处理 _StepMarker 数据 (此逻辑无需修改) ---
    if steps_df is None or steps_df.empty:
        print("--- [_StepMarker] 无 Step 数据可供计算平均值 ---")
    else:
        # ( ... 内部逻辑 (过滤 warmup, IQR 筛除) ... )
        # ( ... 这部分逻辑是健壮的，无需修改 ... )
        # [--- 省略内部过滤代码 ---]
        
        print(f"--- [_StepMarker] 正在计算 _StepMarker 稳定状态平均值 (再跳过前 {warmup_steps} steps) ---")
        stable_steps_df = steps_df[steps_df['step'] >= warmup_steps].copy()
        if stable_steps_df.empty:
            print(f"--- [_StepMarker] 警告: 移除预热后, 没有剩余数据。---")
        else:
            if num_workers == 0:
                num_workers = steps_df['worker_id'].nunique()
            
            print(f"--- [_StepMarker] 预热数据已移除, 剩余 {len(stable_steps_df)} 条记录 (来自 {num_workers} 个 worker) ---")
            print(f"--- [_StepMarker] 正在使用 {iqr_multiplier} * IQR 方法筛除统计异常值... ---")

            Q1 = stable_steps_df['duration_ms'].quantile(0.25)
            Q3 = stable_steps_df['duration_ms'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            is_not_outlier = stable_steps_df['duration_ms'].between(lower_bound, upper_bound)
            filtered_steps_df = stable_steps_df[is_not_outlier]
            
            original_count = len(stable_steps_df)
            filtered_count = len(filtered_steps_df)
            if original_count > 0:
                 outlier_count = original_count - filtered_count
                 outlier_percent = (outlier_count / original_count) * 100
                 print(f"--- [_StepMarker] 已筛除 {outlier_count} 个统计异常值 ({outlier_percent:.2f}%) ---")
            
            if filtered_steps_df.empty:
                 print(f"--- [_StepMarker] 警告: 筛除异常值后, 没有剩余数据。---")
            else:
                 step_average = filtered_steps_df['duration_ms'].mean()


    # --- 3. [!!! 核心修改 !!!] 打印组合摘要 ---
    print(f"\n--- [稳定状态平均耗时 (ms) - {num_workers} 个 Worker 平均值] ---")

    if final_averages.empty and step_average is None:
        print("--- 所有计算均未产生有效平均值。 ---")
        print("---" * 10)
        return

    total_avg = final_averages.sum()

    if total_avg == 0:
        print("--- 组件 (A..E) 总和为 0, 无法计算百分比。 ---")
    else:
        final_averages = final_averages.sort_index()
        for component, avg_ms in final_averages.items():
            percentage = (avg_ms / total_avg) * 100
            print(f"   {component:<35}: {avg_ms:>8.4f} ms ({percentage:5.2f} %)")

    print("-" * 52) 
    # [!!! 核心修改 !!!] 更新了标签
    print(f"   {'Total Pipeline (Sum A..E)':<35}: {total_avg:>8.4f} ms (100.00 %)")

    # [!!!] 这部分现在变得至关重要
    if step_average is not None:
        print(f"   {'_StepMarker (Total E2E)':<35}: {step_average:>8.4f} ms")
        if total_avg > 0:
            diff = step_average - total_avg
            diff_percent = (diff / total_avg) * 100
            print(f"   {'  (Gap / Overhead)':<33}: {diff:>+8.4f} ms ({diff_percent:>+5.2f} %)")

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
    parser.add_argument(
        '-w', '--warmup',
        type=int,
        default=5,
        help="在平均值计算中要跳过的预热 'step' 数量。(注意: process_data 已自动跳过第一个 step) (默认: 5)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"plot_{input_path.stem}.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_df = load_data_from_sqlite(input_path)
    processed_df, steps_df = process_data(raw_df)
    
    # 绘图和打印平均值现在使用已处理和过滤的数据
    plot_data(processed_df, output_path, title_suffix=input_path.name)
    print_stable_averages(processed_df, steps_df, warmup_steps=args.warmup)

if __name__ == "__main__":
    main()