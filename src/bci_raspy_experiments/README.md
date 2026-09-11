# EEG ablation experiments

独立新增的实验框架。已有 `run.py`、训练器、配置和结果保持不变。
本框架不自动安装依赖、上传数据、创建环境或提交集群作业。

## 快速使用

BME 服务器使用新增的 `run_experiments.sh` 和 `profiles/server-bme.json`；完整提交、恢复和结果路径见 [服务器运行手册](../../docs/SERVER_EXPERIMENTS.md)。下文 `server.json` 仍为通用空模板。

从 `inhouse_bci_raspy` 项目目录执行，使用满足项目根目录 pyproject.toml 依赖的环境（Python 3.10/3.11）。

Windows PowerShell：

```powershell
$env:PYTHONPATH = "$PWD/src"
python -B -m bci_raspy_experiments --help
python -B -m bci_raspy_experiments inspect --dataset all
python -B -m bci_raspy_experiments prepare --dataset all
python -B -m bci_raspy_experiments manifest --dataset all --suite all
python -B -m bci_raspy_experiments run --dataset inhouse --manifest-file outputs/bci_raspy_experiments/manifests/all-all.json --smoke --smoke-preprocessing full
```

Linux / server：

```bash
export PYTHONPATH="$PWD/src"
python -B -m bci_raspy_experiments prepare --profile server --profile-file /absolute/path/server.json
python -B -m bci_raspy_experiments manifest --profile server --profile-file /absolute/path/server.json --suite all
python -B -m bci_raspy_experiments run --profile server --profile-file /absolute/path/server.json --suite protocol --manifest-file /absolute/output/manifests/all-all.json
```

`profiles/server.json` 是空路径模板。复制后填写 `inhouse`、`bci2a`、`physionet`、`cache`、`output`；
`device` 为 `cpu` 或 `cuda:0` 等，`threads` 为正整数，`workers` 为非负整数。
路径可用服务器绝对路径；相对路径统一相对项目目录，不相对 profile 文件。
server 必填路径为空时直接报错，无本地路径回退。Windows/Linux 共享同一套算法。

正式 `run` 拒绝 local profile，防止误启动完整扫描。服务器填写配置后显式启动。
`--smoke` 只执行一个任务，将外层集合缩小为每类最多 8 个训练 trial、2 个测试 trial，
内部改用分层 trial 验证，最多两轮；这只是接线检查，不代表完整 subject 协议。
可指定 `--smoke-objective window|mil|whole` 和 `--smoke-preprocessing filter_rms|full`。
smoke 独立存储，不进入正式报告。

## 分阶段任务和恢复

`manifest` 枚举任务，不训练。预定义 suite：`protocol`、`single`、`interaction`、`extension`、`all`。
训练种子为 42/43/44，划分种子固定为 42。三个数据集现有完整清单共 7,176 个去重任务；
严格任务包括内部选轮数和完整重训两阶段，因此这个数字不是单次模型拟合数。
应先按 suite 运行；显存、时间取决于服务器和配置，没有把本地两轮耗时当成正式估计。

manifest v2 的 `splits` 保存一次划分；每个 task 用 `config.split` 引用，避免跨被试清单重复保存大量 ID。
`config` 给出模型、训练目标、采样、增强、预处理和选择协议；`suites` 记录任务归属。
模型与实验配置集中于新增 `presets.py`，运行参数集中于新增 profiles，旧配置没有被借用为可写状态。
如需改科学配置，修改新增 presets、使用新的结果目录并重新生成 manifest；不要编辑已有结果对应任务。

```bash
# RUN_ID 从 manifest 的 tasks[].id 复制
python -B -m bci_raspy_experiments run --profile server --profile-file /absolute/path/server.json --manifest-file /absolute/output/manifests/all-all.json --run-id RUN_ID
# 恢复相同任务
python -B -m bci_raspy_experiments run --profile server --profile-file /absolute/path/server.json --manifest-file /absolute/output/manifests/all-all.json --run-id RUN_ID --resume
# 四个进程分别使用 shard-index 0/1/2/3；按硬件为各进程提供独立 device profile
python -B -m bci_raspy_experiments run --profile server --profile-file /absolute/path/server.json --manifest-file /absolute/output/manifests/all-all.json --suite single --shards 4 --shard-index 0 --resume
```

分片应使用同一 manifest 和相同过滤条件。不同 suite 有共享任务，应顺序执行并用 `--resume` 跳过完成项，
不要让重叠 suite 同时写同一个 run。任务锁及共享缓存锁记录 hostname/PID；进程被强制终止后，
先核实已退出再人工清理对应 `.lock`，不按时间自动删除锁。

恢复发生在 epoch 边界；保存模型、最佳模型、优化器、调度器、Python/NumPy/torch/CUDA RNG、
下一 epoch 的确定性 sampler 信息。中途未完成 epoch 会从上一完整 epoch 重算。
修改源码、配置、划分、软件版本或设备/worker/线程配置后拒绝恢复；CPU/GPU 之间不承诺逐位一致。

## 数据、协议和泄漏边界

输入契约：

| 数据 | 输入/类别 | 原始采样与分析片段 |
|---|---|---|
| Inhouse | BrainVision；仅16个 EEG，排除 x_dir/y_dir/z_dir；marker 5/6/7/8 对应左手/右手/双手/双脚 | 250 Hz，MI onset 后 [1,5) 秒 |
| BCI-IV2a | MAT `data` runs；`X` 单位 µV，25列中前22列 EEG、末3列 EOG；`trial` 为 MATLAB 1-based fixation onset，cue 晚2秒；类别左手/右手/双脚/舌头 | 250 Hz，cue 后 [0,4) 秒 |
| PhysioNet MI | EDF；runs 4/8/12 的 T1/T2 为左手/右手；6/10/14 为双手/双脚；不读 baseline/ME 作为分类 trial | 160 Hz，MI onset 后 [0,4) 秒 |

仅支持该明确 MAT 导出结构，不猜测其他 MAT 的单位或时间原点。
PhysioNet 现有105个被试，缺少 S088/S092/S100/S104；不自动下载或静默替换缺失记录。
原始数据不修改。`prepare` 检查缺失任务文件、标签映射、统一通道/采样率、有限数据并保存源文件哈希。
原始 HDF5 另有完整校验和，运行时每进程核验一次，文件变化后重新核验。

所有外层划分先按原始 trial 身份确定，随后处理信号/排除 trial；剔除不重算比例或移动 session 边界。
Inhouse 237 个原始 MI trial，与旧流程235个 clean trial 不同。
BCI-IV2a 为5,184个 trial，PhysioNet MI 为9,450个 trial；实际保留量见各 transform 的排除清单。

- Inhouse：时序 50/50、60/40、70/30、80/20；随机分层 trial 五折。
- BCI-IV2a：每人上述协议；50/50 特指完整 T→E，即使某一导出样本数不等也不移动边界。
  其余比例按 T 后接 E 切分，因此部分 E 可进入训练；另有留一被试，测试人的两个 session 都留出。
- PhysioNet：按完整 subject 随机分组五折。内部验证也按 subject，最终测试人不进入训练。

`strict`：只在外层训练部分内部留20%选轮数和学习率日程，然后重新初始化，在完整外层训练部分
重新拟合预处理并按固定日程训练。最终 test 在训练结束后才做信号转换与推理。
`test_selected`：完整外层训练集训练，外层 test 每轮驱动调度、早停、checkpoint 选择；仅作诊断。
两者均不对 test 反向传播。报告始终标识 `test_used_for_selection`。

ICA、ICA 成分选择和 RMS 只拟合当前训练部分。ICA 使用全部通过拟合前幅值规则的训练 epochs，
Picard、10分量、decim=2、seed=42；BCI 使用 EOG，其余使用 Fp1 识别眼动相关成分。
ICA 不在测试人的数据上重新拟合，应用同一训练变换。ICA 拟合临时需要训练 epochs 的内存，
PhysioNet 全量任务需为数 GB 的输入及 MNE/Picard 工作副本预留 RAM；增强窗口不全量展开。

固定处理为每个 trial 的独立带上下文片段滤波1–40 Hz。保留至少2秒不与前一任务重叠的 baseline context；
不足者记录排除原因，不跨任务或填充虚假分析数据。基线使用 [-2,0)，分析区间末端为开区间。
平均参考、ICA、基线、幅值拒绝、CSD、RMS 按预设开关控制。500 µV 拒绝作用于 CSD 前 EEG；
`no_reject` 同时关闭 ICA 拟合前和最终 epoch 的幅值拒绝。BCI 原始 artifact 标志保留溯源，但不额外当作隐藏剔除规则。
RMS 使用训练片段全部样本；EOG 从不进入分类器。

训练划分相同且预处理配置相同的任务共享缓存，包括不同训练种子（预处理自身 seed 固定42）。
缓存含 source/配置/训练身份哈希，允许单写多读。旧 session 级全数据 ICA 缓存没有被用作严格评测输入。

## 模型、训练目标和增强

Raspy 基准直接调用冻结的旧 EEGNet 类，已做同权重输出精确一致验证。新增常规 EEGNet 使用
same-padded separable 卷积、4/8池化、可独立设置 F2；BatchNorm 使用 PyTorch 的默认参数。
这是明确的 PyTorch EEGNet 基准，不声称与 Keras 实现逐数值一致。
额外1/2/3个 separable blocks 在最后池化前加入，使用 same padding，无额外下采样。
所有配置有层尺寸与参数量记录；完整4秒 trial 模式对应400个100 Hz输入点。

`window` 对各窗分别计算损失；`mil` 先在 trial 内平均 logits 再算损失；`whole` 输入完整分析片段。
window 与 mil 使用相同8个 trial副本/batch及相同前向展开。MSE 为100倍加权 one-hot MSE，
权重只来自训练 trial；CE 为未加权交叉熵。Adam lr=0.001，无隐藏 weight decay。
选轮数阶段最多500轮；scheduler 监测选择集 window accuracy，patience=10、factor=0.1；
按选择集 loss 保存最佳模型，连续10轮不改善且 epoch index>5 时早停。
复现保留的是旧模型和选择规则，新增按 bag 组 batch、严格预处理等使新基准不等于旧实验重跑。

上游增强出处：<https://github.com/kaolab-research/bci_raspy/blob/main/Offline_EEGNet/shared_utils/dataset.py>。
每个窗口每个元素采样独立 U(-0.5,0.5)，乘整个窗口的 `max`（不是绝对值最大值或逐通道 RMS），
加在重采样前信号上。原版及四份固定噪声副本可复用；seed由训练种子/trial/window/副本决定，不随 epoch 刷新。

`matched` 固定每个 trial 每epoch五个呈现位置，均匀选择原版/可用副本，关闭增强时重复原窗。
`expanded` 为原版加所有噪声副本，包含0副本及4副本参照。后者改变数据呈现量，报告记录实例数和优化步数。
不同窗口采样都保持相同 trial batch/更新次数，但窗口实例数不同，此差异明确保留。

## 评测、报告和解释

```bash
python -B -m bci_raspy_experiments evaluate --profile server --profile-file /absolute/path/server.json --manifest-file /absolute/output/manifests/all-all.json
python -B -m bci_raspy_experiments evaluate --profile server --profile-file /absolute/path/server.json --manifest-file /absolute/output/manifests/all-all.json --explain
python -B -m bci_raspy_experiments report --profile server --profile-file /absolute/path/server.json
```

正式输出位于 `output/runs/RUN_ID`，小样本检查位于 `output/smoke`。
`evaluate` 从保存的 NPZ 重算指标；`--explain` 对固定 checkpoint 额外做通道扰动。
未指定 run-id 时，只解释三个核心架构的严格、逐窗、dense、完整预处理、默认增强、MSE 基准；
指定 run-id 可明确分析其他模型。已有 explanation 默认拒绝覆盖。

每个 run 保存：task/环境/源码哈希、外层/内层划分、拟合身份及参数、缓存引用、训练history与LR日程、
模型与恢复checkpoint、未增强train/test的窗口和trial CSV/NPZ、指标、聚合诊断、运行状态和资源量。
恢复训练/重新解释需要保留 `cache` 中引用的数据；仅重算指标/生成报告不需要重新读取原始信号。

trial 主准确率使用平均 softmax 概率；平均 logits 和硬投票另外报告。MSE 后 softmax 不声称已校准。
窗口模型固定评测31个窗口，即使训练只用center/sparse/medium；whole模型评测一个4秒片段，不把它误当作31窗。
随机1/2/4/8/16/31窗汇总、四个不重叠窗、逐位置、全部28种连续四窗位置均预先声明；
随机抽样默认1,000次，描述稳定性而非独立样本置信区间。不使用这些结果挑选测试时“最佳窗口”。

报告包括自身有效trial成绩、保留量、相同split/seed下共同test trial交集成绩和配对差值。
共同交集会随已完成配置变化，因此 `cohorts.json` 保存具体成员，清单未完成前报告是阶段性结果。
按 subject 合并其 out-of-fold trial，先在 subject 内平均训练种子，再等权平均 subject；
随机种子、窗口、fold 不当作独立被试。时序比例使用不同测试集合，不能仅以跨比例差异断言训练规模的因果作用。
test-selected 分支只用于协议诊断，不用于选择最终算法。

通道图输出 trial/window ΔR²、类别 ΔR²、trial准确率和有效第一层kernel能量；保存CSV/NPZ及PNG/PDF。
通道置零发生于处理后模型输入，不能解释为移除原始电极或脑源定位。前向前后核验模型/BN状态不变。

## 测试

```bash
python -B -m unittest discover -s src/bci_raspy_experiments/tests -v
python -B -m unittest discover -s tests -v
```

测试覆盖身份隔离、时序/T→E边界、manifest稳定性、server空配置、16/22/64通道与额外0–3块、
MIL梯度/置换不变性、固定噪声/预算、test标签/信号扰动、选择集驱动checkpoint变化、精确epoch恢复、
真实样例接入、预测往返及共同trial报告。真实数据路径不在本机时，相应样例测试会明确 skip。
本地验证日志与 smoke 产物在 `outputs/bci_raspy_experiments/verification` 和 `outputs/bci_raspy_experiments/smoke`；
旧文件完整哈希核验另存验证目录。上述为早期验收记录；截至2026-09-11，服务器精简正式实验已完成，见 [结果汇总](../../docs/RESULTS_2026-09-11.md)。

## 当前代码位置

唯一维护位置为 `inhouse_bci_raspy/src/bci_raspy_experiments`，与 `src/inhouse_bci_raspy` 是并列 Python 包，不是其子模块。新入口为根目录 `run_experiments.py`；原 `run.py` 不变。新缓存和结果在 `outputs/bci_raspy_experiments`。完整验收命令为 `python -B verify_experiments.py`。
