# Inhouse：统一划分、80%最终训练与预处理对照

2026-09-11新增。入口：`bash run_inhouse_comparison.sh local|server [config.json]`；默认配置为 `config/inhouse_comparison.json`。仅运行Inhouse，不更改BCI及已经完成的夜间40项实验。此前64%配对结果保留为历史记录，本方案是其80%最终训练后续实验。

## 三组共享的协议

先对237个原始trial，按固定seed42做类别分层五折，固定每个trial的session、序号、起始时间和标签。不是在两种清洗后各自重新分折，也不沿用旧235个clean trials或夜间实验各自的折成员。每折外层约80%用于最终训练、20%用于最终评分；预处理剔除后实际数量另行保存。

| 组 | 预处理 | 选模型方式 | 最终训练 |
|---|---|---|---|
| old_strict | 当前环境重新运行原MNE-BIDS-Pipeline＋CSD | 内部验证选择轮数及学习率序列 | 重新初始化，用完整外层80%重训；无验证/测试回调 |
| new_strict | 当前新流程：逐trial滤波、训练侧ICA＋CSD | 同上 | 同上 |
| old_test_selected | 与old_strict相同 | 外层测试accuracy调学习率、loss选checkpoint和早停 | 直接在相同外层80%训练 |

两个strict组在内部选轮阶段仍划成约64%训练、16%验证、20%测试；这只是选轮阶段。选轮结束后，内部验证数据并回最终训练池，预处理的训练侧拟合及RMS也按新的80%训练池重新拟合。最终训练重放内部选出的学习率序列，不读取测试集；选轮和最终模型初始化seed均为42+fold。

三组均使用原Raspy EEGNet、1秒31窗、100Hz、32个打散窗口/batch、原窗＋4个固定噪声副本、OneHotMSE、Adam学习率0.001。最大500轮，选择阶段10轮无改善早停。使用同一训练入口、模型初始化与独立sampler RNG；预处理两组各自选出的轮数允许不同。

**old_strict与new_strict比较整套预处理流程。old_strict与old_test_selected比较两种模型选择协议。后者包含选轮后重训与直接选checkpoint的差异，不再是原64%实验中仅切换选择集来源的对照，不能称为纯粹的“偷看test acc”单因素估计。**

## 预处理与环境

两套预处理均使用已有 `eeg-preprocessing` 环境（MNE1.11.0、NumPy2.2.6，含MNE-BIDS-Pipeline1.10.1），重新生成旧流程产物，不直接使用历史MNE1.8清洗文件。训练和报告使用现有 `mirepnet`。完整包版本写入manifest；原始缓存和生成文件均保存校验和。

旧流程仍按连续session、分折前清洗，保留其拟合范围；新流程分别在内部训练/外层训练部分拟合。比较包含滤波边界、ICA拟合范围、检测输入、分段端点与剔除等整体差异，不进一步拆解各环节贡献。旧流程潜在的预处理信息泄漏仍是被比较的流程特点，并未被“严格模型选择”消除。

主要指标为共同保留外层测试trial上的窗口概率平均准确率，另存各组全部保留trial的成绩、F1、kappa、混淆矩阵、窗口准确率、预测概率和硬投票预测。剔除与保留身份均保存；若训练保留集不同，差异也属于流程结果，不能仅归因于信号数值转换。

## 执行、保护与验证

CPU准备 → 单GPU两轮验证 → 单GPU正式训练 → CPU汇总。每张GPU最多2个折进程并发，各折内三组顺序执行。正式共15个最终模型＋10次内部选轮。GPU申请8CPU、24GB系统内存、12小时；CPU准备4CPU、8GB、4小时。

输出根目录：`/public/home/hugf2022/inhouse_bci_raspy/inhouse-comparison-v1/`。

- `manifest.json`：原始trial身份、五折及内部划分、版本、原始缓存哈希。
- `old_pipeline/`：当前环境生成的BIDS、MNE清洗日志及产物。
- `prepared/fold_N/{old,new}/{selection,final}/`：输入窗口与拟合/保留记录。
- `fold_N/{old_strict,new_strict,old_test_selected}/`：选轮/重训历史、模型、测试预测。
- `fold_N/pair_verified.json`：两组旧预处理的最终训练输入、初始化、第一轮loss配对验证。
- `smoke-validation/`：独立的两轮验证结果，不混入正式成绩。
- `reports/时间戳/`：JSON、CSV、Markdown、PNG/PDF比较图；未完成五折标记provisional。

配置和源码身份冻结。准备和GPU队列使用独占锁；重复执行跳过已完成项目，失败训练从初始化重跑，不提供epoch级恢复。不可在作业运行时修改相关Python源码或覆盖旧结果；新科学配置使用新的输出根目录。

新增轻量测试覆盖80%训练池是否恰好等于非测试数据、RMS不依赖测试信号、refit拒绝验证输入、固定序列重训与直接训练首轮一致。连同原队列及隔离测试共8项通过。GPU验证需另以Slurm实际状态为准。

## 提交记录

| 作业ID | 名称 | 分区 | 脚本 | 日志前缀 |
|---|---|---|---|---|
| 4107543 | comparison-prepare | bme_cpu | slurm/comparison_cpu.slurm | logs/comparison-prepare_4107543 |
| 4107544 | comparison-smoke | bme_gpu | slurm/comparison_gpu.slurm | logs/comparison-smoke_4107544 |
| 4107545 | comparison-train | bme_gpu | slurm/comparison_gpu.slurm | logs/comparison-train_4107545 |
| 4107546 | comparison-report | bme_cpu | slurm/comparison_cpu.slurm | logs/comparison-report_4107546 |

提交不代表运行完成，实时状态以 `sacct`、日志和结果文件为准。
