# 验证记录与尚未覆盖的边界

本文件区分数值/工程检查与完整训练质量。CPU 测试通过不表示论文指标已复现；合成输入的 GPU 短测也不能评价补全效果。以下保留各次验证的日期和源码边界；最新记录见下一节。

## 2026-11-01 暂停与发布整理

当前 `sd_partial` 实验在 **120,994步**完成更新后响应 SIGTERM，原子保存完整断点并退出。
保留完整训练权重和冻结源码，另提供独立生成器导出。当前配置、数据来源和使用方式见
[README](../README.md)、[训练说明](training.md)和[checkpoint](checkpoint.md)。

本次本地 PyTorch2.6.0 / torchvision0.21.0 CPU检查为 **350 passed，5 CUDA skipped**。
新增检查覆盖新旧骨干的导出前后逐位相同、生成器校验和损坏拒绝、禁止用推理包恢复训练、
以及 LPIPS-only准备不下载Real-ESRGAN。未改变 encoder、decoder、PCConv和loss的前向数学。
生产checkpoint的远程导出一致性及来源见[release.json](releases/step120994/release.json)。


## 2026-10-27 原始 GAN 与损失，大 batch 新实验

`configs/released-large-batch.yaml` 恢复发布代码的 L1=1、VGG 感知=0.2、style=250、
结构/纹理辅助各1、GAN=0.2；使用两个随机初始化的 SN Patch D 和 RaLSGAN，
LPIPS=0，无 Real-ESRGAN 初始化和 GAN 预热。从 seed 2019 重新初始化，
不接续旧实验的模型或 Adam 状态。现有随机 mask、1,197,875 张混合训练数据和
60 epoch 预算保留，因此不是官方完整数据协议的重放。

本地 PyTorch 2.14.1：**303 passed、5 CUDA skipped**；服务器 PyTorch 2.6.0+cu124 / H100：
**308 passed、无跳过**。新增独立对照覆盖原 Encoder 保存特征的前向/梯度、
现代参数化 SN 的初始化与相位更新、发布版 VGG 层及输入语义、激活重算和分块插值。
Ruff、格式检查与 diff 空白检查通过。

真实数据 **batch=128、256²、BF16** 已完成两次全量 D/G 更新以及断点恢复后的下一步对照：
恢复瞬间全状态相同，下一步 **618/618 状态张量逐位相同**，loss 相同，最大绝对误差0。
使用真实缓存 VGG 权重，未下载数据。分配峰值 **23.737 GiB**，allocator 保留峰值
**27.740 GiB**，低于38 GiB预算；首步耗时6.451秒，不代表长期数据加载吞吐。
诊断权重未用于正式训练。证据及源码哈希见
[released-launch-20261027.json](evidence/released-launch-20261027.json)。

首次未优化的 batch128 在确定性 bilinear 插值中超过38 GiB预算。修复采用 mutual
模块非重入激活重算，并将独立图像的 bilinear resize 按最多8张分块后拼接；
未改变插值模式、align_corners 或 resize 后的卷积顺序。前向/输入梯度与直接插值一致，
保留原全 batch GAN 均值。正式包装器和测试均使用 cudnn.benchmark=False、
cudnn.deterministic=True、use_deterministic_algorithms(True, warn_only=True)，
以及 CUBLAS_WORKSPACE_CONFIG=:4096:8。reflection padding 的 CUDA 警告仍保留，
不能将本次逐位一致推广为所有设备/输入的确定性保证。

正式运行独立保存于 `runs/unsplash-placeshalf-originalgan-b128-20261027`，
冻结源码与配置；每100步覆盖latest.pt并验证固定32张。旧 Real-ESRGAN MEDFE
实验在3310步停止并保留，CSA继续运行。数值正确性与容量验证不等于图像质量已改善。

## 已完成的 CPU 检查

主集成环境为 macOS CPU，PyTorch **2.14.1**、torchvision **0.29.1**、SciPy **1.18.1**。执行：

```bash
python -m pytest tests -q
```

结果：**198 passed，5 skipped**；跳过项需要 CUDA。远程独立环境 PyTorch **2.6.0+cu124**、torchvision **0.21.0+cu124**、NumPy **1.26.4**、SciPy **1.15.3**、H100 全套结果为 **203 passed，无跳过**。这两次均包括官方判别器权重校验和官方结构对照。Ruff、格式检查及 `git diff --check` 通过；可复查[机器可读记录和源码 SHA256](evidence/validation-20261026.json)。

| 检查对象 | 使用的证据与目的 |
| --- | --- |
| Gaussian BPA | 与独立 dense 参考比较前向和梯度，覆盖通道/空间布局 |
| 局部 BPA | 独立邻域参考，覆盖边界、softmax 与梯度 |
| Partial convolution | 独立计数/归一化参考，固定 buffer、全洞、不同有效窗口 |
| 批量与监督 | 批量对逐样本结果，GT 不被原地修改，已知像素保留 |
| RaLSGAN | 全 batch 公式、两相对项梯度与分块行为 |
| 辅助监督 | 结构/纹理各计一次，由统一 G backward 更新 |
| Real-ESRGAN D | 原官方源码结构与相同权重逐位比较；官方文件 SHA256 严格加载 |
| SN 与 checkpoint 重算 | 阶段内固定 u/v，重算不重复更新谱归一化状态 |
| LPIPS / VGG | 输入范围、冻结参数、输入梯度、FP32 与加权分块 |
| 数据与 mask | 同步几何、内容 hash、固定验证、worker 独立与 persistent worker epoch |
| RTV | 独立 dense 系统、常量、纹理/边缘、非方形、范围与可重复性 |
| 缓存与 manifest | 参数/内容变化失效、损坏重建、原子输出、并行准备 |
| 训练与恢复 | 完整状态恢复、拒绝不兼容语义/数据/源码，失败更新不保存健康断点 |

测试目录的独立参考目标是修正后的数学定义，不是保留已确认的旧代码错误。RTV dense 对照验证线性系统移植，尚不能据此宣称与 MATLAB PCG 输出逐像素一致。

官方 D 架构参考代码保存在 `tests/references/`，包含原来源与 BSD 许可；随机同权重结构对照无需下载。官方文件加载测试可用 `MEDFE_REALESRGAN_D` 指向已存在权重。没有真实预训练权重的 fixture 测试只证明接口与数值路径，不应标为真实预训练链路验证。

## 有界 CUDA 短测

`scripts/smoke_cuda.py` 已在 PyTorch 2.6.0+cu124 / H100 上通过两个配置的完整检查。两者都执行完整 D/G 更新、保存/恢复及下一步对照，使用 full-width 64 的 103,118,630 参数生成器。命令需要已安装项目、可用 CUDA、已准备权重、足够空闲显存，以及尚不存在的输出目录：

```bash
export TORCH_HOME="$PWD/weights/torch"
python scripts/smoke_cuda.py \
  --config configs/realesrgan-lpips.yaml \
  --weights weights/RealESRGAN_x4plus_netD.pth \
  --output outputs/cuda-smoke --batch-size 2 --memory-gib 16
```

输入是合成 RGB 及实际 RTV 结构图，固定 batch=2、256×256、BF16。当前实验使用真实官方全图 D 和 LPIPS；论文配置使用两个 Patch D 和真实预训练 VGG。结果如下：

| 配置 | 恢复瞬间 | 继续一步的状态张量 | 分配显存峰值 | allocator 保留峰值 |
| --- | --- | --- | --- | --- |
| Real-ESRGAN + LPIPS | 全状态逐位相同 | 642/642 逐位相同；所有 loss 相同 | 2.272 GiB | 2.764 GiB |
| 修正论文损失 | 全状态逐位相同 | 618/618 逐位相同；所有 loss 相同 | 2.287 GiB | 2.809 GiB |

两者模型与优化器最大绝对偏差均为 0。事先规定的数值检查阈值为模型 `rtol=1e-4, atol=1e-6`，Adam moments `rtol=1e-3, atol=1e-5`；计步、RNG、冻结特征网络等要求严格相同。本次实际结果比容差要求更强，但不能据此保证所有硬件/输入都逐位确定。小 batch 短测显存不代表大 batch 容量或长期吞吐。

检查中发现原生 BF16 reflection-padding backward 的原子低精度累积使同值前向产生不同梯度：修正前 53 个状态张量超出预设容差。现在只将 FP16/BF16 反射填充的反向累计提升到 FP32，前向复制值逐位不变，FP32/FP64 路径不变；独立参考和 CUDA 测试均覆盖。保持原容差重跑后得到上表结果。CUDA 仍不提供该算子的严格确定性实现，因此脚本使用 `warn_only=True` 并保留警告，而非宣称消除了 CUDA 非确定性。

脚本将 PyTorch allocator 预算限制为 16 GiB，并检查运行前剩余显存；这不是全进程 CUDA/驱动开销的绝对上限。结果写入 `report.json`，记录软件、GPU、参数量、峰值显存、损失、恢复比较与权重 provenance。只运行短测，不自动开始长期训练。

精确恢复只对该次通过验证的同一软件/硬件与确定性配置成立。允许将完整 checkpoint 迁到另一台机器，但跨 PyTorch、CUDA 或 GPU 型号不承诺逐位一致。

## 固定样本推理与质量验证

实际训练的视觉比较应固定 manifest、样本顺序、mask seed、预处理及输出分辨率，并记录 checkpoint 的步数和 SHA256：

```bash
medfe evaluate --checkpoint runs/medfe-realesrgan-lpips/latest.pt \
  --manifest data/validation.jsonl --output outputs/validation \
  --limit 64 --device cuda:0
```

比较时保留原图、masked 输入、原始预测与完成图，查看边缘连续性、结构合理性和纹理质量。数字损失、通过测试或已生成拼图本身不能证明补全质量提高。当前未完成论文数据集全量训练及论文表格指标复现。

## 集成记录与边界

| 项目 | 当前状态 |
| --- | --- |
| macOS CPU 全套测试 | 198 passed，5 CUDA 项跳过 |
| 本地官方 D 源码/权重对照 | 已包含在上述通过结果中 |
| H100 / torch 2.6.0+cu124 全套测试 | 203 passed，无跳过 |
| CUDA BF16 两配置完整更新/恢复 | 通过；本次下一步参数、优化器、冻结网络、RNG 与 loss 逐位相同 |
| 当前硬件可用 batch / 吞吐 | 验证 batch=2；最大容量和长期吞吐未测量 |
| 长期训练收敛与图像质量 | 独立实验，尚未完成 |

远程 MEDFE 使用独立 `.venv`，复用现有 CUDA/PyTorch，不修改已有 CSA 环境。旧共享 NumPy 1.23.5/OpenBLAS 0.3.20 在该 CPU 上的稠密求解出现大残差，导致独立 RTV 参考测试失败；生产 SciPy sparse LU 正确。参考测试改用独立 SciPy Cholesky，并新增方程残差检查，保留 `2e-12` 精度要求；项目环境采用 NumPy 1.26.4。没有通过放宽误差阈值处理该问题。

原发布实现的 29 个源文件在 `legacy/` 中逐字节保留。以上短测 checkpoint 只用于数值验证，不用于正式初始化，未据此声称论文表格或真实图像效果复现。

## 2026-10-26 正式训练接入

在用户确认启动后，新增 `indexed_rtv` 数据后端，直接读取现有 TFRecord/TAR 索引：1,197,875 张训练图（296,145 Unsplash + 901,730 Places365 Standard），2,000 张固定 Unsplash 验证图。先将完整 RGB 规范化到 256²，再计算真实 RTV，结构缓存按需生成；不是在被 mask 的图上生成目标，也不替换为普通模糊。

这一启动版本的本地测试为 **231 passed、5 CUDA skipped**，远程 H100 为 **236 passed**。新增测试覆盖两种分片格式、受限解码、源文件/缓存变化、跨 worker/epoch 一致性、显存预算及独立验证 batch。与前述数值检查的源码版本差异和容量报告见 [launch-20261026.json](evidence/launch-20261026.json)。

正式配置为 batch 64、8 workers、BF16、32 GiB allocator 上限；LPIPS=1、L1=0.1、结构/纹理 aux 各1、GAN=0.2。容量测试两次完整更新及 FP32 验证通过，分配峰值13.799 GiB、保留峰值18.625 GiB。真实数据先完成两步并保存完整 checkpoint，再从 step 2 接续持久训练；源码和全量 manifest 均冻结在运行目录。每100步覆盖latest.pt、每500步验证32张（验证batch4）。首次生成RTV缓存受CPU求解限制，容量测试时间不能代表数据加载吞吐。

## 2026-10-26 暂停与对抗失衡诊断

上述首轮运行经用户要求在 **968 步**正常停止，保存全部模型、优化器和游标；
checkpoint SHA256 为 `3cbce67a6cbb079d6f0e053d56b8223dc920caadd9253260bfb67cf7ab8e1a11`。
900 步的固定验证和原有 500 步内置验证均有黑块、重复纹理。检查原始预测后发现，
主解码器在已知区域也退化；完成图复制已知像素，掩盖了这一问题。

同一 968 步权重和 4 张固定图的 FP32 诊断中，加权 GAN 对最后解码层的梯度范数为
24.478，非 GAN 目标合计为 0.1974，比例约 **124**。这是特定层/批次的梯度，
不是全模型范数或所有训练时刻的比例。该 probe 不执行先行 D optimizer update，
区别于正式 BF16 D/G 更新，但以下受控干预进一步支持对抗失衡是主要问题。
同权重 train/eval 的 FP32 raw 输出最大差为 1.79e-7，BF16 train/eval 完全一致；
FP32/BF16 平均绝对差为 0.00141（[-1,1] 范围），不支持模式切换是黑块的主要原因。

所有分支采用同一组 16 张训练图、固定 mask、batch4，验证为独立的固定 32 张图。
训练/验证 sample ID 无交集。目标为诊断训练机制，不是全量训练质量或泛化结论。

| 对照 | GAN=0.2 | GAN=0.002 | GAN=0（D 完全冻结） |
| --- | ---: | ---: | ---: |
| 同一 968 步 G/D/Adam，再更新 100 步：验证 hole PSNR | 8.38 | 10.43 | 10.60 |
| 相同随机初始化，再更新 300 步：验证 hole PSNR | 8.13 | 10.77 | 10.66 |
| 相同随机初始化，再更新 300 步：训练小集 hole PSNR | 8.89 | 16.43 | 17.08 |

旧断点续接前的验证 hole PSNR 为 8.20 dB，随机初始化为 9.31 dB。
降低/关闭 GAN 后开始随输入重建内容，但仍有模糊和网纹，不能称图像质量已经修好。
原 Adam 二阶矩也保留了强 GAN 历史，100 步救援不等于从头训练的对照。
静态检查未发现新的确定性 mask、数据归一化、梯度断开或张量布局错误；这不证明所有实现均无问题。

修复增加显式对抗预热/线性渐增以及真正禁用 D 的路径。论文配置默认 0/0 不变；
Real-ESRGAN 变体候选为 GAN=0.002、预热/渐增各2000步，LPIPS=1、L1=0.1 不变。
该2000/2000时长尚未做全量训练验证，正式运行保持暂停。

本次机制修复的本地全套结果为 **250 passed、5 CUDA skipped**。远程 H100 回归初次
254 项通过，1 项因诊断包漏传 legacy 对照文件而失败；补齐原文件后该项通过，
无遗留测试失败。Ruff、格式检查与 diff 空白检查通过。

此外，以实际完整 G、官方 D/LPIPS、64 条不同真实图像做 BF16 有界测试，将预热和渐增
暂设为 2/2 以跨过关键边界。四次更新的实际 GAN 权重为 `0, 0, 0.001, 0.002`；
前两次 D 参数、SN 和 Adam 状态不变，第三次 D 开始更新。保存于预热结束点后恢复，
首个 GAN 更新的 **642/642 个状态张量逐位一致**。峰值分配23.50 GiB、保留27.75 GiB，
在32 GiB allocator预算内。该测试进程已退出，没有恢复正式长跑。
结果和源码哈希见 [debug-20261026.json](evidence/debug-20261026.json)。
