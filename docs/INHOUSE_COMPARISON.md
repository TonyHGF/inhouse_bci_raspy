# Inhouse：直接80/20训练与预处理对照

2026-09-11修订。用户明确取消内部验证，因此**没有64%＋16%的内部划分、内部选轮或二次重训**。每折从头到尾用外层80%训练、20%测试。

入口：`bash run_inhouse_comparison.sh local|server [config.json]`；默认配置 `config/inhouse_comparison.json`。仅运行Inhouse。此前夜间实验和64%模型选择配对结果保留为历史记录，不能混作当前协议的结果。

## 当前三组

| 组 | 预处理 | 训练方式 | 评分模型 |
|---|---|---|---|
| old_fixed | 当前环境重新生成的原MNE-BIDS-Pipeline＋CSD | 80%直接训练，固定60轮、Adam学习率0.001 | 第60轮模型 |
| new_fixed | 逐trial滤波、训练侧ICA＋CSD | 与old_fixed完全相同 | 第60轮模型 |
| old_test_selected | 与old_fixed相同 | 相同80%训练，最多60轮；外层测试accuracy调学习率、loss控制checkpoint和10轮无改善早停 | 测试选择的checkpoint |

**前两组比较预处理流程，测试集不参与调参、调学习率、早停或选模型。第三组保留原先“测试集参与选择”的诊断用途；它与第一组的差别包含整个测试反馈机制，不是只改最终的准确率汇总方式。**

三组统一原始trial五折、模型、窗口、batch、增强及初始化；最大训练预算均为60轮。固定组的学习率恒为0.001，不随任何验证指标变化；测试选择组沿用ReduceLROnPlateau。所有组都不对测试数据反向传播。

## 划分与相同条件

先对237个原始trial按seed42做类别分层五折，记录session、原始trial序号、起始时间和标签。每折训练池恰为测试折的补集，代码和配置均禁止内部验证比例。清洗不会触发重新分折。

目前两种预处理的保留身份已核对完全一致：

| 折 | 保留训练trial | 保留测试trial |
|---|---:|---:|
| 0 | 187 | 48 |
| 1 | 187 | 48 |
| 2 | 189 | 46 |
| 3 | 189 | 46 |
| 4 | 188 | 47 |

这些是原始80/20划分经过清洗后的数量，共235个保留trial。不能把清洗剔除误读为额外留出内部验证数据。

使用原Raspy EEGNet、1秒31窗、100Hz、32个打散窗口/batch、原窗＋4个固定噪声副本、OneHotMSE、Adam。训练seed为42+fold，独立sampler RNG；每组重置为相同初始化。两组旧预处理还核对训练数组、初始化及第一轮loss完全匹配。

## 预处理、复用与解释范围

两套预处理在已有 `eeg-preprocessing` 环境生成：MNE1.11.0、NumPy2.2.6、MNE-BIDS-Pipeline1.10.1。训练和报告使用现有 `mirepnet`。原始缓存、处理结果与软件版本记录在manifest和校验和中。

旧流程仍在分折前对连续session清洗；新流程仅用对应外层80%拟合ICA与RMS。旧流程RMS也只由该折训练数据拟合。比较包含滤波边界、ICA范围与检测输入、分段端点等整体差异，不将差值进一步拆成各环节贡献。旧session级流程潜在的预处理信息泄漏并未因取消测试选模型而自动消失。

此次直接复用已经生成的**外层80%**数据，不再运行内部阶段：旧MNE清洗来自 `inhouse-comparison-v1/old_pipeline/`，两组80%窗口来自 `inhouse-comparison-v2/prepared/fold_N/{old,new}/final/`。复用前核验原始数据、折成员、拟合身份、保留身份、增强参数/seed、软件版本及NPZ哈希；不读取v2的selection窗口，也不加载其训练模型。

## 作业与输出

当前输出根目录：`/public/home/hugf2022/inhouse_bci_raspy/inhouse-comparison-v3/`。

CPU核验复用 → GPU两轮验证 → GPU正式五折 → CPU报告。正式共15次模型训练，没有额外内部选轮。单GPU最多两个折进程并发，各折内三组顺序执行。GPU申请8CPU、24GB系统内存、12小时；CPU申请4CPU、8GB。

| 作业ID | 名称 | 分区 | 脚本 | 日志前缀 |
|---|---|---|---|---|
| 4107553 | comparison-prepare | bme_cpu | slurm/comparison_cpu.slurm | logs/comparison-prepare_4107553 |
| 4107554 | comparison-smoke | bme_gpu | slurm/comparison_gpu.slurm | logs/comparison-smoke_4107554 |
| 4107555 | comparison-train | bme_gpu | slurm/comparison_gpu.slurm | logs/comparison-train_4107555 |
| 4107556 | comparison-report | bme_cpu | slurm/comparison_cpu.slurm | logs/comparison-report_4107556 |

提交不等于完成，实时状态以sacct、日志和status.json为准。两轮验证位于独立smoke-validation目录，不混入正式成绩。

结果包含训练历史、checkpoint、逐窗口/逐trial预测，及共同保留trial和各组保留trial两种口径的准确率、F1、kappa、混淆矩阵；保存CSV、JSON、Markdown及PNG/PDF对比图。未完成五折标记provisional。

配置与源码身份冻结，准备和GPU队列使用独占锁。重复执行跳过完成项；失败训练从初始化重新开始，不提供epoch级恢复。运行期间不修改相关Python源码，新科学设置使用新目录。

## 验证与纠错记录

8项轻量测试通过，包括：80%训练池严格等于测试补集且没有内部划分字段、RMS不使用测试信号、固定训练禁止传入选择集、相同训练输入首轮一致，以及原队列保护/隔离测试。

首轮CPU作业4107543完成三段MNE清洗后，因新增衔接代码误将3个misc辅助通道计入EEG而失败；已改为与原脚本一致，只选16个EEG/CSD通道。后续CPU作业4107548成功生成五折数据，耗时2分26秒。用户明确取消内部选轮后，停止GPU验证4107549并取消尚未启动的4107550、4107551；该旧协议没有启动正式五折。当前v3仅使用其已完成的80%预处理数据。

当前v3接入验证已通过：CPU4107553完成（3秒），GPU4107554完成（43秒）。两轮验证中三组均实际使用187个训练trial，两个固定组各训练2轮，固定训练标记为true；旧流程两组的训练数组、初始化与第一轮loss一致。正式4107555已成功完成，耗时38分55秒；报告4107556成功完成，耗时4秒。15个模型全部完成，五折配对检查通过，provisional=false。

## 正式结果

共同保留的测试trial共235个，三组保留身份一致。主指标为31个窗口先softmax、再平均概率所得的trial准确率；两个固定组均实际训练60轮，无内部验证。

| 组 | Trial准确率 | 判对数 | Macro-F1 | Kappa |
|---|---:|---:|---:|---:|
| 旧MNE-BIDS＋固定60轮 | 68.51% | 161/235 | 0.6877 | 0.5800 |
| 新预处理＋固定60轮 | 62.98% | 148/235 | 0.6315 | 0.5063 |
| 旧MNE-BIDS＋测试集参与选择 | 70.21% | 165/235 | 0.7038 | 0.6026 |

| 折 | 旧流程固定60轮 | 新流程固定60轮 | 旧流程测试选择 |
|---|---:|---:|---:|
| 0 | 85.42% | 70.83% | 85.42% |
| 1 | 70.83% | 62.50% | 68.75% |
| 2 | 58.70% | 58.70% | 60.87% |
| 3 | 69.57% | 67.39% | 71.74% |
| 4 | 57.45% | 55.32% | 63.83% |

- 预处理：旧流程比新流程高5.53个百分点，多判对13个trial；五折4胜1平。训练/测试身份、模型、batch、增强规则、初始化、轮数和固定学习率均对齐。该差异比较整套预处理，不拆解滤波、ICA或拟合范围的单独贡献。
- 模型选择：旧流程测试选择比旧流程固定60轮高1.70个百分点，多判对4个trial；五折3胜1平1负。测试选择的最佳轮数为37、8、19、33、7，包含学习率、早停和checkpoint选择反馈。它与此前64%配对的+3.40个百分点不是同一训练协议，不相互替代。
- 单被试、单训练种子方案的五折观察，不宣称总体显著性。当前旧流程仍包含分折前session级预处理，不能把68.51%视为完全排除了预处理信息泄漏的估计。

Git中的 [完整摘要](results/2026-09-11/inhouse-comparison-v3/summary.json)、[逐折CSV](results/2026-09-11/inhouse-comparison-v3/folds.csv)、[PNG图](results/2026-09-11/inhouse-comparison-v3/comparison.png)、[PDF图](results/2026-09-11/inhouse-comparison-v3/comparison.pdf)；逐trial预测、完成度和来源校验也在同目录。未入Git的模型等输出见 [下载清单](OUTPUT_TRANSFER.md)。
