# 哪些结果在Git中，哪些需要另行传输

更新于2026-09-11。Git存放代码、配置、说明及轻量结果；仓库根目录的 `outputs/`、`data/`、`logs/` 被忽略，`/public/home/hugf2022/...` 本身也在仓库之外。

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
