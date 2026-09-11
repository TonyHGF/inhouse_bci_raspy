# 哪些结果在Git中，哪些需要另行传输

更新于2026-09-11。Git存放代码、配置、说明及轻量结果；仓库根目录的 `outputs/`、`data/`、`logs/` 被忽略，`/public/home/hugf2022/...` 本身也在仓库之外。

## 一次下载所有实验结果

已新增统一导出目录，包含所有正式实验、本地历史结果和诊断记录。此前只有最新Inhouse对照提供了压缩包，现在其余实验也统一打包。

在Mac上执行，把 `YOUR_SERVER` 替换成SSH地址或别名：

```bash
mkdir -p ~/Downloads/inhouse-results
scp -r hugf2022@YOUR_SERVER:/public/home/hugf2022/inhouse_bci_raspy/exports/all-experiments-20260911 ~/Downloads/inhouse-results/
cd ~/Downloads/inhouse-results/all-experiments-20260911
shasum -a 256 -c SHA256SUMS
```

按需解压某个包，例如 `tar -xzf overnight-bci2a.tar.gz`。目录中还有MANIFEST.json（来源、排除项、体积、文件数、校验和及符号链接列表）和README.txt。

| 压缩包 | 覆盖内容 |
|---|---|
| `inhouse-comparison-v3-results-20260911.tar.gz` | 最新直接80/20三组五折，15个模型 |
| `overnight-inhouse.tar.gz` | Inhouse夜间8配置×5折、报告与审计 |
| `overnight-bci2a.tar.gz` | BCI-IV2a六配置×九被试 |
| `selection-bias-64pct.tar.gz` | 先前64%训练的两组五折配对 |
| `legacy-local-outputs.tar.gz` | 最早本地五折、历史ablation、analysis和visualize等输出 |
| `server-validation.tar.gz` | 服务器初始正式接入检查、smoke及失败诊断 |
| `earlier-validation.tar.gz` | 夜间/配对小轮数验证及已停止的v1/v2对照记录 |
| `cluster-logs.tar.gz` | 集群日志快照 |

包内保留源目录中现有的权重、预测、训练历史、解释图、报告和配置。诊断包单独命名，不将smoke/失败记录混为正式结果。原始记录和主要预处理缓存不在统一结果包中，详细排除项见MANIFEST.json；离线重训的数据需求见下文。元数据内的服务器路径不会自动改写。

全部8个包已生成并完整读取校验通过，合计约1.65GB（十进制），CPU作业耗时4分28秒。Git中的 [导出清单](results/2026-09-11/exports/MANIFEST.json) 和 [校验和](results/2026-09-11/exports/SHA256SUMS) 可用于核对下载。正式实验包没有符号链接；earlier-validation诊断包保留一条历史smoke预处理链接，目标见清单，不影响正式结果查看。

打包入口：`sbatch slurm/export_results.slurm`（CPU节点；脚本拒绝覆盖既有导出目录）。本次作业4108723，日志为 `logs/export-results_4108723.out/.err`。下载包本身在服务器，不纳入Git；脚本、清单和下载说明纳入Git。

## 当前直接80/20实验

已入Git：`docs/INHOUSE_COMPARISON.md`、更新的总汇总文档，以及 `docs/results/2026-09-11/inhouse-comparison-v3/`。快照包括汇总JSON、逐折CSV、逐trial概率/预测JSON、混淆矩阵、PNG/PDF对比图、五折manifest、配置、完成度与来源校验。因此仅查看结果、做表或引用现有图，用 `git pull` 即可。

未入Git：15个模型的 `final/model.pt`、完整 `final/history.json`、`final/training.json`、逐窗口 `test_predictions.csv/.npz`、worker日志，以及大体积预处理输入。这些仍在服务器：

`/public/home/hugf2022/inhouse_bci_raspy/inhouse-comparison-v3/`

已打包可独立查看的结果与模型（约1.5MB，精确1,490,248字节）：

`/public/home/hugf2022/inhouse_bci_raspy/exports/inhouse-comparison-v3-results-20260911.tar.gz`

包内含五折三组模型、历史、窗口预测、报告、配置、manifest和worker日志；不含预处理窗口、原始数据或两轮smoke。它可用于离线结果分析，不能单凭这个包重新训练。部分元数据保留原服务器绝对路径，不自动改写。

在Mac终端执行，把 `YOUR_SERVER` 替换成平时SSH登录的地址或已配置的SSH别名：

```bash
mkdir -p ~/Downloads/inhouse-results
scp hugf2022@YOUR_SERVER:/public/home/hugf2022/inhouse_bci_raspy/exports/inhouse-comparison-v3-results-20260911.tar.gz ~/Downloads/inhouse-results/
scp hugf2022@YOUR_SERVER:/public/home/hugf2022/inhouse_bci_raspy/exports/inhouse-comparison-v3-results-20260911.tar.gz.sha256 ~/Downloads/inhouse-results/
cd ~/Downloads/inhouse-results
shasum -a 256 -c inhouse-comparison-v3-results-20260911.tar.gz.sha256
tar -xzf inhouse-comparison-v3-results-20260911.tar.gz
```

若SSH别名已经指定用户名，可用 `YOUR_SERVER:` 替代 `hugf2022@YOUR_SERVER:`。SSH端口不是22时，给scp添加平时使用的 `-P 端口`。

## 只有需要重训或检查清洗时才传

以下服务器路径共用根目录 `/public/home/hugf2022/inhouse_bci_raspy/`：

| 内容 | 相对路径 | 大小 | 用途 |
|---|---|---:|---|
| 当前五折80%输入窗口 | `inhouse-comparison-v2/prepared/fold_*/{old,new}/final/` | NPZ合计约1.9GB | 使用已生成窗口重训；需传对应complete.json等元数据 |
| 两阶段完整旧缓存 | `inhouse-comparison-v2/prepared/` | 约3.3GB | 同时含废弃的内部阶段；本轮不需要selection目录 |
| 当前环境MNE清洗产物及质控 | `inhouse-comparison-v1/old_pipeline/` | 约381MB | 检查清洗FIF、日志、HTML报告、BIDS转换数据 |
| 原始数据缓存 | `server-v1/cache/inhouse/` | 按需 | 从原始trial重新执行新预处理 |

**v3的prepared目录是指向v2的绝对符号链接，不能把单独复制v3当成完整可移植训练环境。** 当前结果包已排除这些链接；若传预处理数据，应复制上表中的实际源目录，迁移后重新配置本地路径。原BrainVision记录位于仓库 `data/`，也不在Git中；如果Mac已有同一份原始数据，不必重复传。

## 之前几轮实验的完整输出（可选）

| 内容 | 服务器目录 | 大小 |
|---|---|---:|
| Inhouse夜间8配置×5折 | `/public/home/hugf2022/inhouse_bci_raspy/overnight-v1/inhouse/` | 约62MB |
| BCI-IV2a六配置×九被试 | `/public/home/hugf2022/inhouse_bci_raspy/overnight-v1/bci2a/` | 约135MB |
| 先前64%训练的模型选择配对 | `/public/home/hugf2022/inhouse_bci_raspy/selection-bias-v1/` | 约51MB |
| 最早本地五折产物 | `/home_data/home/hugf2022/code/inhouse_bci_raspy/outputs/` | 按需 |

之前几轮的轻量结果快照已经在 `docs/results/2026-09-11/` 中。只有需要完整模型、预测、解释图等产物时才另传这些目录，例如：

```bash
scp -r hugf2022@YOUR_SERVER:/public/home/hugf2022/inhouse_bci_raspy/overnight-v1/inhouse ~/Downloads/inhouse-results/
scp -r hugf2022@YOUR_SERVER:/public/home/hugf2022/inhouse_bci_raspy/overnight-v1/bci2a ~/Downloads/inhouse-results/
```

无需为当前结论下载早期smoke、失败作业或全量server-v1验证结果。集群标准输出/错误日志在仓库 `logs/` 下；本次训练worker日志已在结果包中，排查Slurm资源或环境问题时可再取 `logs/comparison-{prepare,smoke,train,report}_作业ID.{out,err}`。
