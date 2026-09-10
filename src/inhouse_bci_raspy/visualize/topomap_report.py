"""Explain the two channel maps in a Chinese report with links to auditable arrays."""

from pathlib import Path
import numpy as np


def write_report(destination: Path, metadata: dict, figures: list) -> None:
    """Write interpretation, baseline checks, channel rankings and comparison caveats."""
    m = metadata
    arrays = np.load(destination/'summary_arrays.npz')
    channels = m['channel_names']
    lines = ['# 通道 topomap：消融 ΔR² 与 kernel 权重分布','',
        '只分析当前五折的固定 checkpoint；没有重新训练或修改训练协议。输入空间为 CSD、训练折 RMS 归一化后的 16 个通道。',
        '所有图使用 standard_1020 模板坐标、线性插值和电极名称；同指标图使用统一色标。头部轮廓使用统一 0.11 m 显示球半径以容纳额部电极，不代表实测头围。插值仅用于显示，不增加空间测量信息。','']
    if 'trial_mean_delta_r2' in arrays:
        lines += ['## 消融 ΔR²','',
            '逐个把模型输入的一个通道全部置零，保持其余数据和模型不变。正值表示消融降低拟合质量，负值表示消融改善拟合质量。所有正负结果均保留。',
            '四维 one-hot 为目标；R² 使用 softmax 前的原始输出，不使用类别编号，也不使用 softmax。主结果先在 trial 内平均 31 个窗口的原始输出，补充结果在 window 层计算。',
            '各类别图在全部验证 trial 上计算该输出的 one-vs-rest R²，不是只取该类别的 trial。总体为四个输出 R² 的等权平均。这里使用未加权 R²，不复用训练损失的样本权重或 gamma。',
            '跨折均值按验证 trial 数加权，标准差也是加权总体标准差，不是标准误、置信区间或 pooled R²。',
            '准确率另外按照原规则计算：窗口取 softmax argmax，trial 对窗口 softmax 概率平均后取 argmax。它与 R² 的汇总目的不同。','',
            '![Trial ablation](ablation_trial_overall.png)','',
            '| Fold | Trials | 基线 trial R² | 基线 window R² | Trial 准确率 | 最大概率复现误差 |',
            '|---|---:|---:|---:|---:|---:|']
        for f in m['folds']:
            check=f['baseline_verification']
            lines.append(f"| {f['fold']} | {f['validation_trials']} | {f['baseline_trial_r2']:.5f} | {f['baseline_window_r2']:.5f} | {check['trial_accuracy_percent']:.2f}% | {check['max_probability_error']:.3g} |")
        lines += ['', '## 通道数值','', '| Channel | Trial ΔR² | 折间 SD |'+(' Kernel 能量占比 | 折间 SD |' if 'kernel_mean_fraction' in arrays else ''),
                  '|---|---:|---:|'+('---:|---:|' if 'kernel_mean_fraction' in arrays else '')]
        for i in np.argsort(-arrays['trial_mean_delta_r2'][:,0]):
            row=f"| {channels[i]} | {arrays['trial_mean_delta_r2'][i,0]:.5f} | {arrays['trial_std_delta_r2'][i,0]:.5f} |"
            if 'kernel_mean_fraction' in arrays:
                row+=f" {100*arrays['kernel_mean_fraction'][i]:.2f}% | {100*arrays['kernel_std_fraction'][i]:.2f} pp |"
            lines.append(row)
    if 'kernel_mean_fraction' in arrays:
        lines += ['', '## Kernel 权重分布','',
            '每折分别提取 8 个时间 kernel 和 16 个空间 kernel，空间权重经过原前向实现的 max-norm 约束。逐 kernel 图显示 W_effective × BN1 slope × BN2 slope，有符号，编号仅在该折内部有效。',
            '总体图计算 H[j,c,t] = BN2[j] × W_effective[j,c] × BN1[parent(j)] × T[parent(j),t]，对 filter/time 平方求和后，归一化成各通道占总能量的比例。BN slope = gamma / sqrt(running_var + eps)，忽略偏置。',
            '总体统计是第一层 ELU 之前的系数能量，不是数据激活能量，也不包含后续分类器贡献。它不生成类别归因图。不同折先各自归一化再等权平均；标准差为总体标准差。',
            '![Kernel energy](kernel_energy.png)','']
        if 'trial_mean_delta_r2' not in arrays:
            lines += ['| Channel | 能量占比 | 折间 SD |','|---|---:|---:|']
            for i in np.argsort(-arrays['kernel_mean_fraction']):
                lines.append(f"| {channels[i]} | {100*arrays['kernel_mean_fraction'][i]:.2f}% | {100*arrays['kernel_std_fraction'][i]:.2f} pp |")
    if 'comparison' in m:
        overall=m['comparison']['overall']
        lines += ['', '## 两种方法的比较','',
            f"汇总通道排名的 Spearman 相关系数：{overall['spearman_rho']}。这只是 16 个通道上的描述性一致性，不计算独立样本显著性。",'',
            '| Fold | Spearman rho |','|---|---:|']
        for f in m['comparison']['by_fold']:
            lines.append(f"| {f['fold']} | {f['spearman_rho']} |")
        lines += ['', '两者一致表示系数分布与消融拟合影响在这些通道上相符；不一致可能与输入分布、通道冗余、后续非线性及消融扰动有关。本分析不据此确定具体原因，也不把两种数值合成一个重要性分数。']
    lines += ['', '## 核验与解释范围','',
        '- 消融前逐窗口核验了标签、trial/time 对应关系、概率和原有两种准确率；未通过时停止分析。' if 'trial_mean_delta_r2' in arrays else '- 本次只分析 kernel 系数，不计算消融指标。',
        '- 分折清单覆盖全部 trial，各折训练/验证 trial 无交集；推理中模型参数与 BatchNorm buffer 未变化，checkpoint 文件哈希未变化。' if 'trial_mean_delta_r2' in arrays else '- 已核验 checkpoint 元数据与模型状态；kernel-only 不运行验证预测复现。',
        '- 验证折参与过 checkpoint 选择；三个 session 混合分折，预处理先于分折。本图不代表独立测试或跨 session 泛化。',
        '- 通道置零是特定输入扰动，不等同于移除原始电极、重新计算参考/CSD 或重新训练。相关通道可以彼此补偿。',
        '- Kernel 大系数不能直接解释成最终预测贡献或脑活动源；头皮插值图不是脑区定位。', '',
        '## 文件与复现','',
        '- `channel_summary.csv`：通道均值、标准差；`summary_arrays.npz`：五折与汇总原始数值。',
        '- `fold_*/ablation_scores.csv`：消融前后 trial/window、总体/四类 R²，及准确率变化。',
        '- `fold_*/trial_outputs.csv`：基线与每个被消融通道的 trial 原始输出及平均概率。',
        '- `fold_*/kernels.npz`：时间、空间、合成 kernel 与能量，可独立复核。',
        '- `metadata.json`：公式、加权方式、软件版本、输入文件 SHA256、核验结果及通道排名。',
        '- `figure_manifest.json`：全部 PNG/PDF 文件和每组色标范围。','',
        '```powershell',"$env:PYTHONPATH = (Join-Path (Get-Location) 'src')",
        'python -m inhouse_bci_raspy.visualize.topomap --method both',
        '# 已有计算结果时，仅重新绘图：', 'python -m inhouse_bci_raspy.visualize.topomap --plot-only','```','',
        '重新计算已有目录需显式传入 `--overwrite`；仅覆盖本分析输出。运行中可查看终端或 `status.json`。','',
        '## 图表索引','']
    for file in figures:
        if file.endswith('.png'):
            lines.append(f'- [{file}]({file}) · [PDF]({file[:-4]}.pdf)')
    lines += ['', '方法参考：[R²](https://scikit-learn.org/1.6/modules/generated/sklearn.metrics.r2_score.html)、[EEGNet](https://arxiv.org/abs/1611.08024)、[权重解释](https://pubmed.ncbi.nlm.nih.gov/24239590/)。','']
    arrays.close()
    (destination/'report.md').write_text('\n'.join(lines),encoding='utf-8')
