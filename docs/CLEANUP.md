# 外部旧代码清理记录

日期：2026-09-10。

## 范围与归档

当前开发和实验入口统一为 `inhouse_bci_raspy/`。根目录以下旧内容已在归档验证后移除：

- `gitclone_bci_raspy/`（包括其 Git 历史与自带示例文件）
- `scripts/`、`config/`
- `CodeReview.md`、`dataset_config.json`
- `environment.yml`、`requirements.txt`、`run_pipeline.ps1`
- 原 `README.md`（已替换为当前项目导航）

归档：`../../_archive/legacy_code_20260910_180927/legacy_code.zip`。
清单：同目录 `manifest.json`。

归档包含 535 个文件，源文件约 947.97 MiB，压缩后约 788.82 MiB。每个归档条目均经解压读取并校验 SHA-256；删除前再次核对源文件数量、大小和 SHA-256。递归删除前核验目标位于项目根目录内，且无符号链接或其他重解析点。

如需恢复，请将 ZIP 解压到单独的恢复目录，先查阅旧文件，避免直接覆盖当前入口。

## 独立性调整

原 `test_partition_matches_gitclone` 会读取外部 GitClone 源码。现改用 `tests/fixtures/gitclone_partition_reference.json`，其中保存删除前运行原始 `partition_data` 函数得到的五折索引，以及原源码 SHA-256、随机种子和类别数量。测试仍逐折比较完整索引，不再依赖外部 checkout，也不会因其不存在而跳过。

README 中预处理示例路径改为当前项目内部输出目录。训练、切分和模型逻辑未修改。

## 保留内容

根目录原始数据、历史 BIDS 数据、训练/测试数据及结果目录、`.venv-mne/`、`.vscode/` 和 `.gitignore` 保留。三组原始 BrainVision 数据共 9 个文件与当前项目 `data/` 副本逐文件 SHA-256 一致。本次仅整理旧代码及其配置，不删除历史数据或环境。

## 验证

外部源码移除后，在已有 `raspy` 环境运行 `python -m unittest discover -s tests -v`：15 项全部通过，无跳过。覆盖命令行入口、预处理路径解析、五折参考索引、trial 独立性与归一化/增强边界，以及 topomap 消融和 kernel 数值验证。本次未重新训练模型。
