# inhouse_bci_raspy

服务器实验已完成，见 [2026-09-11 结果汇总](docs/RESULTS_2026-09-11.md)（Inhouse 五折、BCI-IV2a 六配置、模型选择配对）。运行入口见 [夜间实验](docs/OVERNIGHT.md) 和 [配对实验](docs/SELECTION_BIAS.md)。

BrainVision → MNE 清洗 → CSD → GitClone trial 分层五折 → EEGNet。
只处理运动想象四类，不再使用此前固定的 80/20 train/test NPZ。

当前后续实验：[Inhouse直接80/20训练与预处理对照](docs/INHOUSE_COMPARISON.md)。

## 环境与运行

请自行创建并激活新的 conda 环境，Python 使用 **3.10 或 3.11**。本项目不创建环境。
激活后在本目录执行：

```powershell
python -m pip install -r requirements.txt
python run.py all
```

`requirements.txt` 包含预处理和训练所需的直接依赖。PyTorch 允许 2.5～2.x，
CPU/CUDA 构建请按机器自行选择；`--device auto` 默认在 CUDA 可用时用 GPU。
如果先安装了满足版本范围的 CUDA PyTorch，pip 会保留它。
运行记录保存实际使用的版本；依赖文件不是新环境的完整 lockfile，未在新环境中安装验证。

也可按阶段执行：

```powershell
python run.py preprocess
python run.py prepare
python run.py smoke --device cpu
python run.py train --device auto
```

已经有本批运动想象的 clean epochs 时，可复用，路径指向含 `sub-01` 的 derivatives 根目录：

```powershell
python run.py prepare --clean-root outputs/bids/derivatives/mne-bids-pipeline
python run.py train
```

`--clean-root` 相对执行命令时的目录解析；配置文件内路径相对本项目目录解析。
默认读取本目录 `data/hyz_1`、`data/hyz_2`、`data/hyz_3`，不复制、不修改原始数据。
如果移动项目，修改 `config/experiment.yaml` 的 `source_vhdr`。

## 数据规则

| 原始 marker | 条件 | 模型标签 |
| --- | --- | ---: |
| 5 | 左手运动想象 | 0 |
| 6 | 右手运动想象 | 1 |
| 7 | 双手运动想象 | 2 |
| 8 | 双脚运动想象 | 3 |

marker 1～4 不选作任务 epoch；每个起始 marker 后首个 99 是结束，重复 99 忽略。
BIDS duration 使用实际配对时长；不足 5 秒时报错。

主项目的预处理参数保持：16 EEG 通道、250 Hz 原始采样、standard_1020、1～40 Hz、
平均参考、−2～5 秒 epoch、−2～0 秒基线、500 µV 阈值、Picard ICA 10 分量。
ICA 按记录在切分前拟合；自动成分识别仍需结合质控报告理解。
原转换脚本仅保留选中任务 annotations，未额外添加 LostSamples 坏段剔除。

## 五折与训练协议

1. 合并三条记录的 clean trials，通过 `(recording_id, trial_id)` 保留身份。
2. 用 GitClone `partition_data()` 的同一组操作：每类内部 `np.random.shuffle`，
   `np.array_split(..., 5)`，再按类别拼接各折。固定 seed=42。
3. 每轮留一折验证，其余四折训练；重新初始化一个 EEGNet，共训练五个模型。
   每个 trial 在五轮中训练四次、验证一次；所有窗口跟随所属 trial。
4. 保留主项目的数据变换：CSD 后，用当轮四个训练折的完整 epochs 计算逐通道 RMS，
   应用于训练和验证。不会沿用之前 80/20 划分的尺度。
5. 从 1.0 到 4.0 秒每 0.1 秒取一个 1 秒窗，共 31 窗；FFT 重采样至 100 Hz。
   训练每窗原版＋4 份 `max(window) * U(-0.5,0.5)` 噪声增强，验证只用原版。
6. 复用 GitClone 的模型和训练算法，分别位于 `src/inhouse_bci_raspy/models/eegnet.py`
   和 `src/inhouse_bci_raspy/training/engine.py`；损失函数独立在 `training/losses.py`。输入 `[N,16,100,1]`，输出四类 logits。
   默认配置来自其 `Offline_EEGNet/config.yaml`：Adam，学习率 0.001，
   OneHotMSE（gamma=10）及基于训练标签计算的 balanced 权重，最多 500 epochs。
   学习率依据验证准确率调节，依据验证损失保存最佳 epoch。
   原训练函数实际早停为连续 10 轮验证损失未改善且 epoch>5；
   YAML 中的 `patience=100`、`weight_decay` 等部分字段并未被原函数使用，此处没有改写其行为。

**这是一套混合流程：前处理和窗口参数来自主项目，五折和模型训练来自 GitClone。**
GitClone 原始连续数据归一化、64 通道空间处理和基于 1000 Hz 的窗口采样索引不复用。
这里保留 0.1 秒步长，未照抄其配置中的 99 个原始采样点步长。
为了支持每轮独立 RMS，先存 trial 级 HDF5，再生成该轮训练/验证数组供原训练函数使用；
不要求沿用原 `EEGData` 的全局固定 HDF5 窗口布局。

没有额外独立测试集，五折指标均为**参与早停和模型选择的验证指标**。
输出额外提供 trial 平均概率分类准确率，用于观察窗口相关性；原窗口准确率仍保留。
不训练 Kalman 方向控制器，不把双手/双脚类别解释为光标上下方向。

## 文件与输出

```text
run.py                         仅启动 src 包，保留原有命令
config/                        只存实验 YAML 和数据 JSON
src/inhouse_bci_raspy/
  cli.py                       参数解析和阶段调度
  settings.py                  项目路径及配置读取
  runtime.py                   缓存、设备、随机种子
  io.py                        通用 JSON 写入
  preprocessing/
    brainvision.py             marker 解析与 BIDS 转换
    pipeline.py                逐记录执行转换和官方预处理
    mne_config.py              MNE-BIDS-Pipeline Python 参数
  data/
    preparation.py             clean epochs → CSD → trial HDF5
    partition.py               GitClone trial 分层五折
    folds.py                   当轮训练 RMS 与数据组装
    windows.py                 滑窗、加噪、重采样
    loaders.py                 DataLoader 与短程样本选择
  models/eegnet.py              EEGNet 网络层与前向计算
  models/LICENSE               GitClone 许可证
  training/
    cross_validation.py        五折轮换及覆盖检查
    fold.py                    单折模型训练流程
    engine.py                  原优化循环、早停、逐 epoch 验证
    losses.py                  原损失函数和 criterion 选择
    checkpoints.py             权重及推理元数据保存
    logging.py                 TensorBoard/短程日志接口
  evaluation.py                概率推理、窗口/trial 指标、五折汇总
  reporting.py                 CSV、JSON、曲线和混淆矩阵输出
tests/test_protocol.py         协议检查
tests/test_entrypoints.py      启动入口与子进程路径检查
requirements.txt               环境依赖（不自动创建环境）
```

所有生成结果在本目录 `outputs/`：

- `bids/`：BIDS、clean epochs 和 HTML 质控报告。
- `preprocessing/`：逐记录配置和日志。
- `prepared/clean_trials.h5`：CSD 后尚未归一化的 trials、标签、溯源信息和五折索引。
- `prepared/folds.json`：各折类别数、验证 trial 清单、输入文件哈希。
- `cv/fold_0`～`fold_4`：`best.pt`、`split.json`（含 RMS 尺度）、`history.json/png`、
  `metrics.json`、`validation_predictions.csv`、`confusion_matrix.json/png`、TensorBoard 日志。
- `cv/summary.json`：五折平均/标准差及合并验证混淆矩阵。
- `smoke/`：仅用于接线检查的短程结果，不能当作实验结果。

已有正式 checkpoint 时，`train` 会拒绝覆盖；开始新实验前请另行归档原 `outputs/cv`。
模型 checkpoint 内包含网络配置、标签、通道顺序及该折 RMS；推理数据也应经过相同的
MNE/CSD、窗口处理与对应 RMS，不能将原始电压数组直接传入模型。

## 来源与兼容修正

源码来源和原文件 SHA256 见 `PROVENANCE.json`；GitClone 的许可证保存在 `src/inhouse_bci_raspy/models/LICENSE`。
原项目目录和 GitClone 目录不作修改，运行时只依赖本目录代码及配置指定的数据路径。

- `np.Inf` 改为 `np.inf`，兼容 NumPy 2。
- 移除调度器已不支持的 `verbose` 参数，不改变调度规则。
- 原 `BatchNorm2d(self.F1, False)` 实际将 `eps` 设为 0，现代 PyTorch 训练报错。
  此处显式设置 `eps=1e-5`，保留原本启用的 affine。这是明确的数值稳定性修正，
  因而不声称网络与含该问题的旧实现逐数值等价。
- 删除未使用的 torchvision crop 导入，因此不需额外安装 torchvision。
- 独立入口不使用原脚本的 Git/SQL 记录、自动删除中间数据和吞掉异常的包装层。

协议检查：

```powershell
python -m unittest discover -s tests -v
```

检查五折完备/互斥/类别均衡、训练 RMS 不使用验证数据、验证不增强。
逐索引核对归档前由原 GitClone 函数生成的参考快照；快照包含原文件 SHA256，测试不再依赖外部 checkout。
`smoke` 在第一折取每类少量窗口训练两轮，验证加载、前后向、原训练函数、权重保存和评估输出；
它不需要 TensorBoard，但正式 `train` 需要 requirements 中的 TensorBoard。

## 模块接口与调用边界

`run.py → cli.main()` 只负责启动和选择阶段。预处理、数据准备和训练是独立阶段，
命令仍为 `python run.py preprocess|prepare|train|all|smoke`，无需安装本项目包。

- `preprocess(config, dataset, output)` 负责子进程与日志，不实现滤波/ICA；实际参数在 `mne_config.py`。
- `prepare(config, dataset, clean_root, output)` 读取 epochs 并生成缓存与 fold 清单，不拟合分类器。
- `build_fold(cache, config, validation_fold)` 返回训练块、验证块和 RMS/索引元数据，不写结果文件。
- `make_windows(...)` 返回 `X[N,C,T,1]`、`y[N]` 及等长的 trial、窗口、增强字段。
- `train_folds(...)` 只控制五折顺序、整体随机种子与覆盖检查；`train_one_fold(...)` 管理一个新模型。
- `engine.train(...)` 保留 GitClone 优化与早停规则；`losses.py` 只处理损失函数。
- `predict_probabilities(...)` 按原始窗口顺序返回 `[N,4]` 概率，确保与 trial 元数据对应。
- `reporting.py` 只保存和绘图，`checkpoints.py` 只序列化已选定的最佳模型，不参与模型选择。

各模块有职责说明，主要函数的 docstring 写明参数、返回值、形状和副作用。
移动前后的训练协议与计算保持一致；已有数据路径配置不会因代码位置变化而被改写。

## 已有五折结果的汇总诊断

在本目录执行以下命令，只读取已有预测，不重新训练：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m inhouse_bci_raspy.analysis.aggregation
```

`src/inhouse_bci_raspy/analysis/aggregation.py` 负责 trial/窗口完整性核验、软/硬投票对照、
窗口子集与时间位置分析；`analysis/presentation.py` 负责报告和绘图。
结果位于 `outputs/analysis/aggregation/`：`report.md`、`summary.json`、
`trial_diagnostics.csv` 和 `aggregation_diagnostics.png`。
这些是原验证折的事后诊断，不是新测试集结果或新训练 protocol。

## 模型通道 topomap

在本目录、已有 `raspy` 环境下执行：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -u -m inhouse_bci_raspy.visualize.topomap --method both
```

默认读取 `outputs/cv` 中的五折 checkpoint、冻结的 `experiment.json` 和各折验证数据，
不读取当前 YAML 重新决定模型或窗口设置，不生成训练增强、不重新训练。
`--method ablation|kernel|both` 选择分析；默认 CPU、batch size 32，支持 `--device`、
`--batch-size` 和 `--output-root`。已有结果时拒绝覆盖，需显式使用 `--overwrite`。
仅重新绘图可用 `--plot-only`，此时沿用保存的分析方法，忽略 `--method`。

- `analysis/channel_inputs.py`：重载 checkpoint、重建验证窗口、核验原预测。
- `analysis/channel_ablation.py`：逐通道置零，计算 one-hot/原始输出的 ΔR² 及原概率平均准确率。
- `analysis/channel_kernels.py`：提取包含 max-norm 和 BatchNorm 缩放的第一层时空 kernel。
- `analysis/channel_pipeline.py`：组织五折、保存数值和溯源信息、计算两种方法的排名相关性。
- `visualize/topomap.py` 和 `topomap_report.py`：分别负责 CLI/绘图与说明报告。

结果在 `outputs/visualize/topomap/`。`report.md` 汇总每折、各类别、跨折均值和标准差，
并链接 PNG/PDF；CSV/NPZ 保留数值、基线与消融 trial 输出。终端逐通道显示进度，
`status.json` 记录运行/完成/失败状态。
ΔR² 主图使用 trial 平均 logits，类别图对全部验证 trial 计算 one-vs-rest R²；
准确率仍用平均 softmax 概率。Kernel 图是系数能量占比，不等同于最终决策贡献或脑区定位。
