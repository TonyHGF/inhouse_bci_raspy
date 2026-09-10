# 整合验证记录

## 五折通道解释（2026-09-10）

- `raspy` CPU 环境完成五折、每折 16 通道的消融和 kernel 分析；没有新增依赖或重新训练。
- 五折全部基线概率与原保存 CSV 的最大绝对误差为 0，两种准确率复现。
- 模型参数、BatchNorm 状态、checkpoint 文件哈希保持不变。
- 15 项 unittest 通过，包括 9 项新增检查：Windows 控制台 CLI 编码、R² 与 sklearn 一致、常量目标报错、
  有效/无关/有害通道、概率平均与 logits 平均区别、重复窗口、基线篡改、加权标准差、
  合成时空 kernel 与实际线性层一致。
- 结果和输入哈希见 `outputs/visualize/topomap/metadata.json`，全部图表入口为该目录 `report.md`。

## 当前 data 的端到端检查与正式提交（2026-09-10）

- 使用用户创建的 `raspy` 环境（Python 3.10.20、PyTorch 2.14.0 CPU），未安装或更新依赖。
- 已从本目录 `data/` 完整执行 BIDS 转换、MNE 清洗及 HTML 报告生成，保留 77/79/79 个 trial。
- 重新准备五折索引 48/48/48/46/45；六项检查通过。
- 两轮短程训练、checkpoint 重载、PNG/CSV 和 TensorBoard 写入均通过。
- 删除了一条未使用的 torchvision crop 导入，以适配不需要 torchvision 的运行环境。
- 正式 `python -u -B run.py all --device cpu` 已后台提交。进程与日志位置见
  `outputs/full_pipeline_job.json`；正式结果尚未完成，完成后见 `outputs/cv/summary.json`。
- `outputs/smoke/` 和 `outputs/refactor_baseline/` 已清理，仅保留可复用的检查入口。

下文为此前整合/重构阶段的验证记录，所列“未执行”仅描述当时状态。

未创建 conda 环境，未安装或更新任何包。使用已有 `research-general` 做代码核验：
Python 3.10.20、PyTorch 2.13.0、NumPy 2.0.2、MNE 1.8.0。
新 requirements 的完整安装解析尚未在新环境执行。

## 已通过

- 从已有四类运动想象 clean epochs 生成新的 trial 级 HDF5，235 个 trial，16 个 EEG 通道。
- 各类 trial 数：58、58、60、59；五折大小：48、48、48、46、45。
- 相同 seed=42 下，五折索引逐项等于 GitClone 原 `partition_data()` 输出。
- 四项 unittest：原函数一致性、分折完备与分层均衡、验证数据不参与 RMS/不增强、真实数据溯源。
- CPU 短程训练：第一折每类取 8 个训练窗口和 8 个验证窗口，运行原训练函数两轮，损失有限。
- 短程流程完成最佳权重、曲线、预测 CSV、混淆矩阵及指标文件保存。
- 从保存的 checkpoint 重新构造模型并加载权重，前向输出形状 `[2,4]` 且数值有限。
- 来源哈希核验通过，主项目和 GitClone 源文件未因本次整合而修改。

## 未执行

- 新 conda 环境的创建与依赖安装。
- 新目录下从原始 BrainVision 数据完整重跑预处理；复制的转换/预处理代码来自之前已跑通流程，
  本次适配验证复用了其运动想象 clean epochs。
- 正式五折训练。`outputs/smoke/` 只代表运行检查，不能作为分类效果结论。

环境准备完成后，可执行 `python run.py all` 完整运行；
也可先 `python run.py prepare --clean-root ../bids_dataset_imagery/derivatives/mne-bids-pipeline`，
再 `python run.py train` 复用已清洗数据。

## src 模块化重构回归

- 算法函数去掉说明文字后，计算 AST 与重构前一致。
- 四项协议测试和两项入口/子进程路径检查通过（共六项）；CPU 两轮短程训练通过。
- 重构前后的 history、metrics、checkpoint 元数据和每个权重 tensor 逐项完全一致。
- run.py 仅保留启动，配置内容、数据和既有正式训练协议未改变。
