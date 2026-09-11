# 服务器接入验证（2026-09-10）

本次保持实验配置、数据划分和模型不变。实际使用已有 `mirepnet` 环境，GPU为Tesla V100-SXM2-32GB。依赖版本与本地固定版本不同，不能据此声称与旧环境逐位复现。

已完成：

- Shell语法和 `git diff --check` 通过。
- 用模拟sbatch验证pipeline七阶段依赖、数组分片、默认并发1和resume参数；模拟不提交作业。
- CPU准备作业 **4106899** 成功：Inhouse 237、BCI-IV2a 5184、PhysioNet 9450个原始trial；manifest共7176任务（Inhouse 1023、BCI-IV2a 5238、PhysioNet 915）。
- 最终CPU回归作业 **4106915** 成功：15项测试，14通过、1跳过。跳过项使用旧本地公开数据路径，真实服务器读取由prepare和smoke覆盖。
- GPU smoke作业 **4106914** 成功：三个数据集×window/MIL/whole共9项，均使用full预处理、内部选择和最终重训，每阶段最多两轮。所有status为complete，均生成预测、指标、聚合分析和解释summary；共63张PNG、63张PDF，已抽查通道图。
- 修复了GPU首次调用 `reset_peak_memory_stats` 前未初始化CUDA的问题；初始化及状态写入移进try/finally，初始化错误也能释放任务锁。
- conda激活时临时关闭nounset，兼容已有编译器激活钩子。失败的早期作业已结束，首次GPU失败产物保留于 `results/smoke-failed-4106901`，没有覆盖。

正式验证链（截至2026-09-11均已成功完成）：

| 作业 | 分区 | 内容 |
|---|---|---|
| 4106916（数组任务0） | bme_gpu | Inhouse严格时序50/50，Raspy、seed42、默认增强、完整预处理；使用正式最大500轮/早停配置 |
| 4106918 | bme_cpu | 从正式预测文件重算评估指标，依赖训练成功 |
| 4106919（数组任务0） | bme_gpu | 冻结模型通道扰动和PNG/PDF，依赖训练成功 |
| 4106920 | bme_cpu | 独立时间戳报表，依赖指标和解释都成功 |

正式run ID：`inhouse-8fd7ddca8cb2fe44df2e`。训练记录耗时531.99秒。此链只验证一个正式任务，不等于完成7176项扫描。完整扫描尚未提交。实时状态以 `sacct` 和结果中的 `status.json` 为准。

日志分别位于项目 `logs/bci-prepare_4106899.*`、`logs/bci-verify_4106915.*`、`logs/bci-smoke_4106914.*` 和对应正式作业文件。结果位于 `/public/home/hugf2022/inhouse_bci_raspy/server-v1/results/`。完整运行方法见 [SERVER_EXPERIMENTS.md](SERVER_EXPERIMENTS.md)。

后续精简实验与模型选择配对均已完成，见 [2026-09-11结果汇总](RESULTS_2026-09-11.md)。
