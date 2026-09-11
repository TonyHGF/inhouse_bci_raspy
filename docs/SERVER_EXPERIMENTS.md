# BME 服务器实验运行

当前两卡精简实验请使用 [OVERNIGHT.md](OVERNIGHT.md) 中的 `run_overnight.sh`：Inhouse完整五折40项，BCI-IV2a九被试54项，每卡默认2进程。下文的旧 `pipeline` 是7176项全量入口，不是当前夜间方案。

保留现有 `run_experiments.py`、`presets.py`、协议和目录结构。新增入口 `run_experiments.sh`，复用原有训练、预测、报告和解释实现。`scripts/server_experiments.py` 只连接服务器阶段；`report()` 新增可选输出目录，避免覆盖旧报表。原空模板 `profiles/server.json` 保留，实际配置为 `src/bci_raspy_experiments/profiles/server-bme.json`。

## 环境与存储

默认使用已有 `mirepnet` conda 环境，不安装任何包。它与原本本地版本不同：CUDA PyTorch 2.6.0+cu124、MNE 1.11.0、NumPy 2.2.6；训练记录实际软件版本。不使用本地旧 checkpoint 续训。

- Inhouse：项目内 `data/`。
- BCI-IV2a：`/public/home/hugf2022/motor/eeg-sim/data/raw/bci_competition_iv_2a`。
- PhysioNet：同级 `physionet_mi`。
- 缓存：`/public/home/hugf2022/inhouse_bci_raspy/server-v1/cache`。
- 结果：`/public/home/hugf2022/inhouse_bci_raspy/server-v1/results`。

`CONDA_ENV` 可指定其他已有环境。新实验需要复制 profile，并修改 cache/output 到新的绝对路径；同一次扫描必须使用同一份 profile、源码和 manifest。不要在任务运行期间修改这些文件或环境。提交前在项目根目录执行命令。

## 运行

```bash
# 登录节点仅核对路径、profile 和依赖版本，不运行训练
bash run_experiments.sh local check

# 一条命令提交完整依赖链，默认64片、同时最多1块GPU
# verify → prepare+manifest → 三数据集×三目标smoke → run → evaluate/explain → report
bash run_experiments.sh server pipeline

# 例如允许同时2块GPU，仍每个进程只使用cuda:0（Slurm映射到分配的GPU）
SHARDS=64 MAX_PARALLEL=2 bash run_experiments.sh server pipeline

# 先只跑协议诊断；训练、指标和解释使用这个筛选，报告包含该目录所有完成任务
SHARDS=16 bash run_experiments.sh server pipeline \
  src/bci_raspy_experiments/profiles/server-bme.json --suite protocol
```

完整清单预计约7176个去重任务，严格协议每个任务包含内部选轮数和全训练集重训。分片只是分配任务，不降低轮数、种子数、模型数或评估次数。64片不是64个GPU并发；默认并发为1。正式耗时需参考完成任务的 `metrics.json`，不以两轮 smoke 推断完整扫描耗时。默认每个GPU作业24小时，可用 `TIME_LIMIT=5-00:00:00` 改到分区上限，或增加分片数。

分阶段使用（每次提交输出 Job ID；将示例数字替换为实际ID）：

```bash
bash run_experiments.sh server verify
bash run_experiments.sh server prepare
DEPENDENCY=12345 bash run_experiments.sh server smoke
SHARDS=64 MAX_PARALLEL=1 DEPENDENCY=12346 bash run_experiments.sh server run \
  src/bci_raspy_experiments/profiles/server-bme.json --resume
DEPENDENCY=12347 bash run_experiments.sh server evaluate \
  src/bci_raspy_experiments/profiles/server-bme.json --resume
SHARDS=64 DEPENDENCY=12347 bash run_experiments.sh server explain \
  src/bci_raspy_experiments/profiles/server-bme.json --resume
DEPENDENCY=12348:12349 bash run_experiments.sh server report
```

`prepare` 总是准备三个数据集，生成同一 `manifests/all-all.json`。`smoke` 使用完整预处理，三个数据集分别运行 window/MIL/whole，每个最多两轮且训练/测试样本明确限量；另外生成通道解释图。`verify` 在CPU节点跑实验包原有测试。`run`/`evaluate`/`explain` 支持 `--dataset`、`--suite`、`--run-id`；run/explain 可分片。普通 `evaluate` 仅从预测文件计算指标，使用CPU。`explain` 使用GPU，对默认核心严格基准生成通道图，指定 `--run-id` 可分析其他正式任务。

默认CPU作业4核8GB，GPU作业1GPU、4核32GB。`MEMORY=...`、`TIME_LIMIT=...` 可覆盖Slurm资源，`MAX_PARALLEL` 控制数组并发。无需修改slurm文件中的实验参数。

## 恢复与结果

`--resume` 跳过已完成训练、已重算指标和已完成解释；未完成训练从完整epoch边界恢复。相同输出目录下，不要同时启动有重叠任务的suite或两套pipeline。训练失败会阻止依赖作业启动；修复后重新提交失败阶段与下游依赖。超时/强杀可能留下 `.lock`，必须核实锁中hostname/PID已退出后才清理，脚本不自动删锁。改变分片数可以重新分配未完成任务，但必须确认原数组已结束。

每个正式任务在 `results/runs/RUN_ID/` 保存：

- `metrics.json`、`recomputed_metrics.json`：window/trial准确率、macro-F1、kappa、混淆矩阵等。
- `train_predictions.*`、`test_predictions.*`：CSV/NPZ预测。
- `aggregation.json`：概率/logits/投票聚合，随机窗口数量、窗口位置和连续四窗分析。
- `training.json`、`inner/`、`final/`：训练曲线、选择日程、checkpoint、拟合身份和预处理记录。
- `explanation/`：通道扰动ΔR²、各类别ΔR²、kernel能量；CSV/NPZ和PNG/PDF。

`results/reports/时间戳-作业ID/` 每次新建报表快照，包括 `report.md`、`runs.csv`、`common_trials.csv`、`paired_differences.csv`、`subject_seed_results.csv`、`subject_summary.csv`、`cohorts.json`、`window_trial.png`。`manifest_completion.json` 给出选定任务的完成数和缺失任务；没有正式任务时明确报错，部分完成时标记provisional。Smoke结果独立存储，不能作为正式结果。

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS
# 数组的日志使用每个子任务的实际Slurm Job ID，目录内查找即可
tail -n 80 logs/bci-run_JOBID.out logs/bci-run_JOBID.err
```
