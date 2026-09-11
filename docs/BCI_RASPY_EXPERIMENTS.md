# EEG 消融实验

BME 服务器已补充数据路径配置和 CPU/GPU Slurm 入口，使用方式见 [服务器运行手册](SERVER_EXPERIMENTS.md)。

唯一维护位置为 `inhouse_bci_raspy/src/bci_raspy_experiments`，独立包名 `bci_raspy_experiments`。

从 `inhouse_bci_raspy` 目录执行：

```powershell
python -B run_experiments.py --help
python -B run_experiments.py inspect --dataset all
python -B run_experiments.py prepare --dataset all
python -B run_experiments.py manifest --dataset all --suite all
python -B run_experiments.py run --dataset inhouse --manifest-file outputs/bci_raspy_experiments/manifests/all-all.json --smoke --smoke-preprocessing full
python -B verify_experiments.py
```

也可设置 `PYTHONPATH=src` 后使用 `python -m bci_raspy_experiments`。新增 `pyproject.toml` 仅打包本实验包，支持 `python -m pip install -e .`；迁移未修改 Python 环境。

配置在 `src/bci_raspy_experiments/profiles`。local 相对路径以 `inhouse_bci_raspy` 项目根目录为基准。server 保留空模板，填写服务器数据、缓存和输出绝对路径后使用 `--profile server --profile-file ...`。缓存与新结果在 `outputs/bci_raspy_experiments`，不覆盖原 `outputs/ablation`。

[详细实验手册](../src/bci_raspy_experiments/README.md) 包含协议、消融、恢复、报告和解释分析。

历史 smoke 和验证文件已保留，冻结元数据中的旧绝对路径不改写，不用于跨位置强制续训。新验收使用新的输出目录。截至2026-09-11，服务器准备及精简正式实验已完成，见 [结果汇总](RESULTS_2026-09-11.md)。全量7,176项扫描未提交；迁移服务器时仍需重新 prepare，不上传含本地路径的缓存。原项目模型、数据、配置及原 run.py 保持不变。
