# 两张卡分数据集、单卡并发训练

入口为 `run_overnight.sh`，配置为 `config/overnight.json`。继续使用既有模型、数据缓存、严格划分、训练器和报告。PhysioNet不参与。

| GPU作业 | 划分 | 配置 | 总任务 |
|---|---|---|---:|
| Inhouse | 完整5折trial分层验证 | Raspy、常规EEGNet、加深EEGNet、Raspy-MIL、Raspy-whole、关闭增强、关闭ICA、关闭CSD | 40 |
| BCI-IV2a | 9人，每人自己的T训练、E测试 | Raspy和常规EEGNet分别做window、MIL、whole | 54 |

固定seed42。选轮阶段最多60轮，patience=10；严格内部验证选轮后重新初始化、重新拟合训练预处理并在完整外层训练集重训。测试集不参与选轮。配置的名称在manifest的label中，训练参数本身不附加无关标签。

## 使用

```bash
# 只读缓存元数据，冻结精简manifest和profile，不启动计算
bash run_overnight.sh local

# 提交两个单GPU作业，以及各自的CPU报告作业
bash run_overnight.sh server
```

每个GPU作业申请1GPU、8CPU、32GB、12小时，默认2个训练进程共享该GPU，每个进程4线程、DataLoader workers=0。所有子进程继承Slurm设置的CUDA_VISIBLE_DEVICES，并在进程内部使用cuda:0；不能给第二个进程写cuda:1。两张GPU作业可能分到相同节点或不同节点，均由Slurm分配。

并发不是分布式训练。每个进程负责一个独立任务，完成后队列补下一个，每个任务有独立日志、checkpoint和任务锁。相同拟合数据共享预处理缓存，其写入由现有缓存锁串行保护。单卡并发是否加速取决于GPU、CPU及共享存储瓶颈，不能把并发数当作速度倍率。

`config/overnight.json` 可调整 `parallel_per_gpu`，wrapper自动申请对应的CPU总核数；默认2，需实测后再提高。显存请求仍为32GB系统内存，GPU显存由所分配显卡决定；不要仅按模型参数量推断显存够用。

训练队列最多运行配置的 `training_hours`（默认10小时）。达到时间预算时，对活动worker发送中断信号，保留上一完整epoch checkpoint，正常释放锁；随后CPU节点为已完成任务生成阶段性报表。Slurm还在12小时时限前120秒发送清理信号。进程卡死或SIGKILL可能留下锁，须核实记录的节点/PID和作业确已退出后再清理，不自动抢锁。

## 节点访问

并发进程已经由sbatch脚本自动启动，无需SSH后手动开多个训练。需要进入已经申请的节点时使用：

```bash
bash attach_overnight.sh GPU_JOB_ID
```

这通过 `srun --jobid` 进入现有allocation，保持Slurm资源身份，不额外申请GPU。集群为Slurm19.05，不使用该版本没有的 `--overlap`。不要在这个shell里再次启动同一队列；队列锁会拒绝第二个管理进程。SSH本身不是GPU资源分配机制，也不用于后台启动脱离作业生命周期的训练。

## 输出与恢复

独立结果位于：

```text
/public/home/hugf2022/inhouse_bci_raspy/overnight-v1/
  config.json
  inhouse/ 或 bci2a/
    profile.json
    manifest.json
    worker_logs/任务ID-本次时间戳.log
    queue-作业ID-时间戳.json
    runs/任务ID/...
    reports/时间戳/...
```

沿用 `server-v1/cache` 中已准备好的原始缓存和符合身份的转换缓存。不会覆盖旧server-v1结果。config/profile/manifest一旦冻结，修改语义或并发配置需要选择新output_root；相同配置重新执行server入口会恢复现有训练，并跳过已完成训练及已完成解释。不要在旧的同数据集队列仍运行时重复提交。

每个run包含窗口/trial预测CSV和NPZ、准确率/F1/kappa/混淆矩阵、聚合窗口分析、训练历史、checkpoint等。Raspy-window和standard-window生成通道扰动及kernel PNG/PDF，其他配置保留完整预测和指标。

CPU报告通过 `afterany` 依赖启动：即使GPU队列部分失败/超时，也会汇总成功项。报表每次写到独立时间戳目录。`completion.json`列出缺失任务、缺少解释的任务、达到选轮上限的任务、配置名称映射和是否provisional；不会把不完整五折或不完整九被试结果标为完整结果。

```bash
squeue -u "$USER"
tail -n 80 logs/night-inhouse_JOBID.out
# 看单项训练轮数时，读取输出目录下对应的worker_logs文件
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

完整运行能否一晚结束需看实测吞吐，特别是BCI从18项扩展到了54项。报告会保留完成度，不预设两进程有两倍提速。

## 本次提交与验证

2026-09-10已提交：Inhouse GPU作业4106955、CPU报告4106956；BCI GPU作业4106957、CPU报告4106958。两个正式GPU作业依赖并发验证的CPU报告4106950成功后启动，不与验证GPU作业4106944叠加占用第三张卡。截至2026-09-11，四个正式训练/报告作业均成功完成：Inhouse 40/40项，GPU耗时3小时43分19秒；BCI 54/54项，GPU耗时8小时48分27秒。预定解释结果无缺失。详细成绩与解释见 [结果汇总](RESULTS_2026-09-11.md)。

实际验证使用独立 `overnight-validation-v1`，每阶段最多2轮、队列预算12分钟，不混入正式结果。已确认同一张V100S上两个Python进程同时运行（各约578MiB显存）；首批Raspy-window、standard-window均exit0，包含训练、预测及解释，各约250秒；队列已自动补入MIL任务。队列的并发上限、失败后继续、独占锁和冻结配置拒绝修改均通过轻量测试。该250秒包含小轮数验证的预处理与评估，不可直接换算为60轮正式任务耗时。
