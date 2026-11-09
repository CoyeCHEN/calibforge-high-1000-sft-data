# Open-Harness SFT

将 Open-Harness 教师轨迹转换成保留 Thinking、正文、工具交互和真实上下文的训练样本，通过 LLaMA-Factory 原生 Trainer / DeepSpeed 运行 Qwen3.5-4B 两阶段 SFT。数据准备、远端作业、保存、归档与恢复都从 `python -m sft` 进入。

本仓库不包含 Agent Loop、Harbor Runner、模型服务、分布式训练循环或 GPU 调度器。Harbor 负责教师任务执行和官方 Verifier；LLaMA-Factory / Transformers / DeepSpeed 负责训练生命周期。systemd 用户服务负责脱离 SSH 运行和进程组清理。

仓库同时保留 [TMax-2B 32K 历史研究](studies/tmax32k/README.md)：它使用固定原生 Trainer、独立依赖和实验身份，不能安装到 Qwen 的 LF 环境。两条训练路径只共享文件级 Recorder/canonical 合同；TMax 的历史评测同题适配数据不得进入 Qwen 正式 M2。集成范围、部署验证与合并顺序见[双 PR 集成说明](docs/integration.md)。

| 阶段 | 输入与监督 | 固定设置 |
|---|---|---|
| M0 | `Qwen/Qwen3.5-4B` | revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；官方 Tokenizer / Chat Template |
| M1 | `NiuNiu0110/TMax-Agent-SFT-terminus@7d74fd38a3bc437bbd2c377bd8a1effba64d79ba`；5320 train / 191 validation；完整多轮 Assistant 监督 | 65536 Tokens；LR `2e-6` |
| M2 | DeepSeek + 固定版本 Open-Harness + Harbor 成功证据；每次实际 Assistant 决策一个样本 | 优先 131072；经真实失败审查才回退 98304；LR `5e-7` |

共同配置：BF16 mixed precision、全语言参数、冻结视觉模块、AdamW、4 GPU ZeRO-3、Activation Checkpointing、microbatch 1、梯度累积 4、首轮 1 Epoch、packing / neat_packing 关闭。学习率是已冻结的候选值，不代表已证明最优。

模型、权重 SHA、依赖、LF 及上游 Commit 见 [`sft/versions.json`](sft/versions.json)、[`sft/model-sha256.json`](sft/model-sha256.json) 和 [`requirements/`](requirements/)。LF 固定为 `c683e59edd8f893abc5e1403184a1d0644817d2a`，实际运行版本 `0.9.6.dev0`；Transformers 5.8.0 / PyTorch 2.8.0 CUDA 12.8 / DeepSpeed 0.19.8。模型加载后还检查语言可训练参数为 4,205,751,296、视觉可训练参数为 0。

正式 M2 合格数据目前为 **0**。TB2.1 官方任务和轨迹、改名的评测任务、五个历史 smoke fixture 均不得作为正式 M2。`fixtures/` 与测试生成的 Recorder / Checkpoint 只用于测试。

## 开始使用

安装不含 ML 依赖的操作入口：

```bash
python -m pip install .
python -m sft --help
python -m unittest discover -s tests -v
```

GPU 环境需要单独按 [部署与运行](docs/deployment.md) 准备。运行目录位于源码之外，所有路径、主机身份和授权 GPU UUID 都通过初始化参数给出：

```bash
python -m sft --workspace "$SFT_WORKSPACE" init \
  --model "$QWEN_MODEL" --hot "$SFT_HOT" --archive "$SFT_ARCHIVE" \
  --gpu-uuids "$AUTHORIZED_GPU_UUIDS"
```

初始化不启动 GPU 或服务，正式训练授权默认为 false。不要给两个独立工程复用同一 workspace。

已有获准数据时直接复用，避免复制或重算：

```bash
python -m sft --workspace "$SFT_WORKSPACE" adopt-m1 /path/to/frozen/m1-64k.json
python -m sft --workspace "$SFT_WORKSPACE" verify-model
python -m sft --workspace "$SFT_WORKSPACE" evidence --source /path/to/existing/preflight
python -m sft --workspace "$SFT_WORKSPACE" check
```

新服务器从官方固定源重新准备 M1，或生成独立 M2 的入口、Manifest 字段和完整命令见 [数据合同](docs/data.md)。所有样本按完整样本保留或拒绝，不截断历史、Observation 或目标回答。

## 连续训练与恢复

正式作业是一段连续的原生 Trainer 训练。正常步骤、Evaluation、周期完整 Checkpoint 均不退出 Rank；保存后继续同一 DeepSpeed Engine。完整阶段结束或真实失败后，由原生 torchrun 和所属 systemd cgroup 清理资源。

检查点先同步保存到本地 SSD；全部 Rank / native callbacks 完成并 fsync 后标记完整。CPU 归档固定已关闭文件的硬链接，独立复制到共享存储的 `.partial`，核对全文件 SHA 后原子发布。这样可以与后续训练计算重叠，并避免 `save_total_limit` 删除尚未复制完的源文件。没有替换 DeepSpeed 格式或使用 DCP async_save。

M1 → M2 加载 **M1 最终 HF 导出权重**，新建 M2 优化器和 Scheduler；同阶段 `--resume` 才恢复完整模型、优化器、Scheduler、RNG、数据进度和 Trainer 状态。没有 `Restart=always`，没有从失联 PID 自动重新派发，也不自动选择检查点。

## 验证边界

既有 A800 的短样本、M1 实际长样本、128K 容量两次更新与恢复证据可以只读导入。它们不等于迁移后的所有生产场景均已验证。当前仍未完成的 `verify-continuous` 和 128K 长监督目标恢复沿用原后台队列，本次迁移不会再派一份 GPU 测试。

[兼容矩阵](docs/compatibility.md) 记录旧功能去留及上游合同；[可靠性审计](docs/reliability.md) 区分已执行的 CPU / 原生源码证据与未验证的 GPU 故障、长期训练场景。CPU CI 不调用教师 API、不加载模型权重、不执行 A800 训练。

本轮实际运行的测试数量、原生数据加载和故障注入结果见[验证记录](docs/validation.md)。

历史 TMax / RLBridge reader、canonical schema、转换器和非敏感测试仍可用于诊断。旧 TMax 训练配置、硬编码外部 Trainer 的根目录 `train.py` 和过时 `uv.lock` 已移除；它们不再是训练入口。历史报告保持原样，不能作为当前 Readiness。

## 新公开任务池语料

新采集的 CalibForge、TMax 语料通过 `sft.public_corpus.build_corpus` 分别构建。输入必须是 Harbor 审计得到的首条合格成功索引、固定任务级 split，以及实际 Provider Payload / 原始 SSE 已核对的 canonical 导出；不读取旧 TMax 教师数据或 TB2.1 轨迹。`sft.teacher_segments` 使用固定官方 Qwen 模板，按实际可见历史生成连续兼容分段，每个新 Assistant 目标只监督一次，重叠历史不计 loss。

构建同时保存 128K / 256K 的原生 LF `tokenized_path` 缓存、Task/Trial/Call 映射、文件 SHA、逐任务 Token 占比和收尾覆盖。长度包含输入、目标、Processor 增补及 padding；超长目标单独隔离，不截断。无法保真的整条轨迹不会贡献一个看似可用的文本前缀。`sft.public_corpus_audit.audit_corpus` 逐行读取两种长度及 train/validation 全部缓存，通过正式 Loader/Collator 检查每个标签与新增位置 mask，再执行明确标注为小型 CPU 模型的手工 loss/梯度参考；该参考不代表 Qwen 4B 的 GPU Kernel 验证。

构建输出始终标记为 staging，独立的原生审计保留输入 Manifest SHA。发布仍需本轮完整采集索引和最终 DATA_AUDIT，不因 canary 缓存通过就宣称两个 500 题任务池已完成，也不自动启动 GPU 训练。
