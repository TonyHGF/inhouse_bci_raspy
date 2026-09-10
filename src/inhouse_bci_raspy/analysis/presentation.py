"""Render descriptive aggregation diagnostics without altering experiment metrics."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def write_outputs(destination: Path, summary: dict, rows: list) -> None:
    """Save an English scientific figure and a Chinese report with metric definitions."""
    s, outcomes = summary, summary['trial_outcomes']
    pooled = s['pooled']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    curve = s['random_subset']['curve']
    axes[0,0].plot([r['windows'] for r in curve], [r['mean_percent'] for r in curve], 'o-')
    axes[0,0].set(xlabel='Windows averaged per trial', ylabel='Accuracy (%)',
                  title='A. Random subsets of saved predictions', ylim=(20, 100))
    axes[0,0].axhline(25, color='gray', linestyle=':', label='Four-class uniform chance')
    axes[0,0].legend()
    bins = np.linspace(0,100,11)
    for status, label, color in [(True,'Trial prediction correct','#238b45'), (False,'Trial prediction wrong','#cb4b43')]:
        axes[0,1].hist([r['window_accuracy_percent'] for r in rows if r['soft_correct']==status],
                       bins=bins, alpha=.6, label=label, color=color)
    axes[0,1].axvline(50, color='gray', linestyle=':')
    axes[0,1].set(xlabel='Correct windows within a trial (%)', ylabel='Number of trials',
                  title='B. Trial outcome versus individual windows')
    axes[0,1].legend(fontsize=9)
    axes[1,0].plot(s['temporal']['window_start_s'], s['temporal']['accuracy_percent'], 'o-', markersize=3)
    axes[1,0].axhline(pooled['soft_percent'], label='31-window trial accuracy', color='#238b45', linestyle='--')
    axes[1,0].set(xlabel='Start time of 1-second window (s)', ylabel='Accuracy (%)',
                  title='C. Accuracy at each fixed time', ylim=(20,100))
    axes[1,0].legend(fontsize=9)
    x = np.arange(5)
    for offset,key,label in [(-.25,'window_percent','Window'),(0,'hard_percent','Hard vote'),(.25,'soft_percent','Mean probability')]:
        axes[1,1].bar(x+offset, [f[key] for f in s['by_fold']], width=.25, label=label)
    axes[1,1].set(xticks=x, xlabel='Validation fold', ylabel='Accuracy (%)',
                  title='D. Same checkpoint, different aggregation', ylim=(0,100))
    axes[1,1].legend(fontsize=9)
    fig.suptitle('Four-class motor imagery | 235 trials, 31 overlapping windows per trial', fontsize=14)
    fig.tight_layout()
    fig.savefig(destination / 'aggregation_diagnostics.png', dpi=170)
    plt.close(fig)

    lines = ['# 四分类运动想象：窗口与 trial 结果差异分析', '',
        '本报告只读取五折已保存的验证预测，不重新训练、不修改模型或 protocol。', '',
        '## 指标与核验', '',
        '- 四类：左手、右手、双手、双脚运动想象（marker 5–8）。',
        '- Window：每个 1 秒窗口分别 argmax。Trial：同一 trial 的 31 个窗口先平均四类概率，再 argmax。',
        '- 窗口起点为 1.0–4.0 秒，间隔 0.1 秒，覆盖 1.0–5.0 秒；相邻窗口重叠 90%。',
        '- 已核验 235 个 trial 各进入一个验证折，每个恰好 31 个窗口；每折训练/验证 trial 无交集；标签与 prepared HDF5 一致；五折原有指标全部复现。',
        '- 以下总体指标按全部验证样本合并计算，区别于各折等权平均。因为每个 trial 的窗口数相同，差异不是样本权重不同造成的。', '',
        '| Fold | Trials | Window | 硬投票 | 概率平均 |', '|---|---:|---:|---:|---:|']
    for f in s['by_fold']:
        lines.append(f"| {f['fold']} | {f['trials']} | {f['window_percent']:.2f}% | {f['hard_percent']:.2f}% | {f['soft_percent']:.2f}% |")
    lines += [f"| 合并 | {pooled['trials']} | {pooled['window_percent']:.2f}% | {pooled['hard_percent']:.2f}% | {pooled['soft_percent']:.2f}% |", '',
        '## 差异来自哪里', '',
        f"1. 概率平均最终判对 {outcomes['soft_correct']} 个 trial，判错 {outcomes['soft_wrong']} 个。判对的 trial 内，窗口平均正确率只有 {outcomes['correct_trial_mean_window_percent']:.2f}%；判错的 trial 内仍有 {outcomes['wrong_trial_mean_window_percent']:.2f}% 的窗口判对。", '',
        f"2. 有 {outcomes['soft_correct_below_half_window_correct']} 个最终判对的 trial，其正确窗口不到一半。四分类只需正确类的平均分高于每个其他类；不要求超过 50%。同时有 {outcomes['soft_wrong_above_half_window_correct']} 个 trial 虽然多数窗口判对，概率平均却判错，说明汇总也可能损失正确判断。", '',
        f"3. 硬投票是 {pooled['hard_percent']:.2f}%，概率平均是 {pooled['soft_percent']:.2f}%。软投票相对硬投票纠正 {outcomes['soft_correct_hard_wrong']} 个、损失 {outcomes['soft_wrong_hard_correct']} 个。硬投票有 {pooled['hard_ties']} 个并列第一，本报告按最小类别索引打破平票。", '',
        f"4. 差值可严格分解为：判对 trial 中原先判错的 {outcomes['recovered_wrong_windows_on_correct_trials']} 个窗口所对应的 +{outcomes['positive_gain_pp']:.2f} 个百分点，减去判错 trial 中原先判对的 {outcomes['lost_correct_windows_on_wrong_trials']} 个窗口所对应的 {outcomes['negative_gain_pp']:.2f} 个百分点，净增 {pooled['soft_percent']-pooled['window_percent']:.2f} 个百分点。这只是统一分母下的数学分解，未改变窗口预测。", '',
        '## 窗口数量和时间对照', '',
        f"每个 trial 随机选 k 个不同窗口平均概率，重复 {s['random_subset']['repeats']:,} 次；k=1 用精确期望，k=31 用全部窗口。以下不是重新训练，也不是统计置信区间。", '',
        '| 窗口数 | 平均准确率 |', '|---:|---:|']
    for r in curve:
        lines.append(f"| {r['windows']} | {r['mean_percent']:.2f}% |")
    temporal = s['temporal']
    lines += ['', f"固定使用起点 1、2、3、4 秒的四个互不重叠窗口：{temporal['nonoverlap_1_2_3_4_s']['soft_percent']:.2f}%。这与 31 个重叠窗口的结果可比较，但二者也同时改变窗口位置及数量，不能独立证明重叠的因果效应。", '',
        f"最前面连续四个窗口（起点 1.0–1.3 秒）：{temporal['first_1_0_to_1_3_s']['soft_percent']:.2f}%；最后连续四个窗口（3.7–4.0 秒）：{temporal['last_3_7_to_4_0_s']['soft_percent']:.2f}%。", '',
        f"遍历全部 28 种连续四窗口位置，平均准确率 {temporal['consecutive_four']['mean_percent']:.2f}%，范围 {temporal['consecutive_four']['min_percent']:.2f}%–{temporal['consecutive_four']['max_percent']:.2f}%。因此上面的局部窗口结果并非仅由挑选最早位置造成。跨时间均匀取四个窗口达到 68.51%，支持时间覆盖而非单纯增加相邻窗口数量的重要性。", '',
        f"相邻窗口预测类别一致率 {temporal['prediction_agreement_by_lag']['1']:.2f}%，相隔 1 秒为 {temporal['prediction_agreement_by_lag']['10']:.2f}%。这描述预测相似程度，不等同于独立样本数。", '',
        '## 类别和 session', '', '| 类别 | Trials | Window | Trial |', '|---|---:|---:|---:|']
    names = ['左手', '右手', '双手', '双脚']
    for r in s['by_class']:
        lines.append(f"| {names[r['label']]} | {r['trials']} | {r['window_percent']:.2f}% | {r['soft_percent']:.2f}% |")
    lines += ['', '| Session | Trials | Window | Trial |', '|---|---:|---:|---:|']
    for r in s['by_session']:
        lines.append(f"| {r['session']} | {r['trials']} | {r['window_percent']:.2f}% | {r['soft_percent']:.2f}% |")
    lines += ['', '## 具体 trial', '', '索引为 prepared 数据的零基全局 trial_index；原始 trial_id 与 recording_id 见 CSV。票数和概率均按左手、右手、双手、双脚排列。', '']
    examples = [r for r in rows if r['soft_correct'] and r['window_correct'] < 16]
    plurality_example = min((r for r in examples if r['hard_correct']), key=lambda r:r['window_correct'])
    selected_examples = [plurality_example] + sorted(examples, key=lambda r:r['window_correct'])[:3]
    for r in selected_examples:
        vote = [r[f'votes_{c}'] for c in range(4)]
        prob = [round(r[f'mean_p_{c}'],4) for c in range(4)]
        lines.append(f"- Trial {r['trial_index']}（{r['recording_id']}，fold {r['fold']}，真实{names[r['true_label']]}）：{r['window_correct']}/31 窗口正确；票数 {vote}；平均概率 {prob}；最终{names[r['soft_prediction']]}。")
    lines += ['', '## Topomap 与解释范围', '',
        '在本地 gitclone_bci_raspy 中检索 topomap/topoplot、saliency、attribution、Grad-CAM、importance、plot_patterns/plot_filters 等，并检查模型及预处理实现，未找到模型通道重要性 topomap 的现成方法。因此没有新增归因算法或生成各折通道重要性图。MNE 预处理中的头皮图不能当作模型通道重要性。', '',
        '本分析支持“跨时间汇总能稳定部分窗口级错误、且不同错误类别可以分散”这一解释；无法仅凭已保存概率把误差归因于具体生理波动、伪迹或某个通道。', '',
        '当前训练为 OneHotMSE 对 logits 的监督，后处理 softmax 数值不能直接当成经过校准的可信概率。平均最大 softmax 值为 '
        f"{s['confidence']['mean_window_max_probability']:.3f}，本报告未用它声称模型置信度已校准。", '',
        '这些结果使用参与早停/选 checkpoint 的验证折，不是独立测试集；三个 session 混合分折，session 表也不代表跨 session 泛化。ICA 等清理是在 session 级别、分折之前完成；本次完整性核验不等于证明全预处理严格隔离。', '',
        '## 输出与复现', '',
        '- `summary.json`：完整统计、随机子集设置及预测文件 SHA256。',
        '- `trial_diagnostics.csv`：全部 235 个 trial 的票数、平均概率及软/硬投票结果。',
        '- `aggregation_diagnostics.png`：汇总图。',
        '- 从 inhouse_bci_raspy 目录执行 PowerShell：', '',
        '```powershell', "$env:PYTHONPATH = (Join-Path (Get-Location) 'src')", 'python -m inhouse_bci_raspy.analysis.aggregation', '```', '']
    (destination / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
