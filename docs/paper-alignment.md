# MEDFE 论文、原代码与现代实现的对应关系

当前发布准备版本使用 `sd_partial` 骨干、L1/结构/纹理/LPIPS 各1、原始双 PatchGAN 渐增到0.002。
它是保留 MEDFE 融合模块的现代实验，训练在120,994步暂停。下文的 `released`、论文系数和
Real-ESRGAN 对照属于独立历史配置；当前版本见 [训练说明](training.md) 和 [新骨干](sd-partial-backbone.md)。

本文依据 [ECCV 2020 正文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470715.pdf)、[官方补充材料](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470715-supp.pdf)，以及 `legacy/` 保留的上游提交 `3c5e8e8bd4d6e0c1df22fee4e67f5ae8c645c7fa`。页码指论文印刷页码。目标是保留 MEDFE 方法、修复可确认的实现错误，并明确记录原文歧义与实验变体；不是逐位复刻旧实现，也不是已完成论文指标复现。

## 联合训练与生成器结构

2026-10-27 原 GAN 基线使用 `configs/released-large-batch.yaml`：遵循发布版的 L1=1、感知=0.2、style=250、GAN=0.2，辅助损失各1；LPIPS=0，无 GAN 预热，两个 D 随机初始化。已有数据、在线随机 mask、BF16、大 batch 和 60 epoch 预算继续保留，因此不是完整数据协议及优化轨迹的逐位复现。论文感知系数0.1仍由 `paper.yaml` 单独表达。

本次同权重前向及输入/参数梯度测试确认了 Encoder 保存特征的旧行为：后续层的原地 LeakyReLU 会激活前五层保存值。v2 用显式非原地张量保留该语义，再保留 PCconv 内的后续激活；不会让现代编码风格暗中改变 skip 的数值。Patch D 改用参数化谱归一化，并在注册前采用原版正态 std0.02、零 bias。参数化 API 自带15次初始化 power iteration，与旧 API 初始化轨迹不同；之后每个 D/G 阶段更新一次、阶段内冻结，支持完整 batch 的分块/重算。

大 batch 配置启用 mutual 模块的非重入激活重算。双线性插值按最多8张图分块，
限制确定性 CUDA 实现中的 FP32 临时张量；插值公式、顺序和全 batch GAN 均值不变。
两项均有前向及梯度对照，并已在真实 batch128 上通过完整更新/断点恢复检查。

补充材料第 1 页 Algorithm 1 每批先更新全图/局部 D，再更新整个 G。结构、纹理两路属于同一个 G，没有 CSA 式分阶段训练。

| 论文机制 | 现代实现 | 保留的语义 |
| --- | --- | --- |
| 正文第 4–5 页的编码器和残差块 | `model.py` | 六级编码、四个残差块；浅层纹理，深层结构 |
| 两路特征聚合 | `model.py` | 对齐空间尺度后分别形成纹理/结构特征 |
| 多尺度部分卷积 | `blocks.py` | 3/5/7 三种核，每路五层 PConv |
| 纹理/结构辅助监督 | `model.py`, `trainer.py` | RGB GT / RTV GT，双线性缩放到辅助头分辨率，各计算一次 |
| Feature equalization | `blocks.py` | SE 通道重标定、全局 Gaussian 与局部传播，拼接后 1×1 融合 |
| 解码与跳接 | `model.py` | 均衡特征进入解码，各尺度跳接重建 RGB |

原代码让两路复用 PConv streams、降维层和辅助 RGB 头；现代实现保留这些共享权重。论文关于“两路结构相同”的文字没有完整限定参数共享，因此此处明确采用代码拓扑。论文文字称五级 decoder，发布代码含最终 RGB 输出级共六次上采样；本实现保留后者，不擅自删去一级。

公开输入统一为 RGB `[-1,1]` 与单通道 bool mask，`True=hole`。进入网络时按原代码使用 RGB 均值填洞，并拼接三通道 known mask。补充材料“零填充”的文字与发布代码不完全一致；此处跟随发布代码。输出同时提供原始预测与完成图，完成图通过 `where(mask, prediction, image)` 保留已知像素。

## Partial convolution 与 BPA 修复

| 原发布代码问题 | 修正与界限 |
| --- | --- |
| Gaussian/local 传播得到 `[position, channel]` 后直接 `view(B,C,H,W)` | 显式还原轴顺序；传播本身逐通道做空间加权，不把通道与位置混排 |
| 部分路径只处理 batch 第一个样本 | 每个样本独立传播，批量计算与逐样本参考一致 |
| 平均后的洞 mask 使用 `>1` 阈值，结果恒为零 | 使用 known-mask 约定传播有效性；感受野有已知点才有效，全洞仍无效 |
| mask 卷积的全 1 核被通用初始化重新随机化 | 计数核为固定 buffer，不参加参数初始化或优化 |
| 原 PConv 归一化未一致处理窗口面积和通道重复计数 | 使用标准空间计数 `kernel_area / valid_count`，零有效点输出零；这是明确的数值修正 |
| 全局 Gaussian 构造巨大广播张量 | 用可分离、全支持的归一化矩阵乘法，保留修正后数学目标 |
| 固定 32×32、硬编码 `.cuda()` | 从输入推导尺寸、继承 device/dtype，支持 batch 和合法非方形尺寸 |

正文第 6 页式 (2) 的 Gaussian 是全局空间加权。现代实现对每个 query 的所有位置归一化，使用全高、全宽支持，不用截断的小卷积核替代。实现与独立 dense 参考比较数值和梯度，而不是要求与含轴混排错误的旧输出一致。

局部分支依据第 6–7 页式 (3)–(5) 与原代码：sigmoid 后的 query/key 计算通道平均乘积，针对 3×3 九个位置 softmax，加权原始 value；边界采用零填充。正文第 6 页把归一化常数写为 `C=N`，第 7 页又提到 softmax，数学表述存在歧义。这里明确选择发布代码的九邻域 softmax 语义。

低精度训练时，InstanceNorm 统计量、PConv 计数与必要的归一化计算使用 FP32。反射 padding 对 FP16/BF16 输入先提升到 FP32、复制边界后转回原 dtype，前向取值逐位不变，反向重复位置的梯度在 FP32 累计后只转换一次，避免低精度原子累计误差。FP32/FP64 padding 保留原路径。这些措施不承诺恢复旧代码全部浮点顺序，也不保证 CUDA 算子严格确定。

## 损失与梯度路径

原代码在输入已位于目标 GPU 时存在输入/GT 别名与原地填洞污染风险；正常 CPU DataLoader 经两次 `.to(cuda)` 的路径并未证明发生该污染。现代实现不原地修改 GT。原辅助模块内部 backward 一次，外层又将返回 loss 与保存的 loss 相加再 backward，实际计入三份梯度；现代结构/纹理损失各计一次，由统一 G loss 做一次 backward。

正文第 9 页总目标权重如下；GAN 权重作用于全图与局部目标之和：

| 目标 | 修正论文配置 | 历史 LPIPS / Real-ESRGAN 配置 |
| --- | --- | --- |
| 全图原始预测 L1 | 1 | 0.1 |
| LPIPS | 0 | 1 |
| VGG 感知 | 0.1 | 0 |
| VGG 风格 | 250 | 0 |
| 结构辅助 L1 | 1 | 1 |
| 纹理辅助 L1 | 1 | 1 |
| GAN 全图 + 局部 | 0.2 | 目标 0.002，先预热再渐增 |

旧训练选项默认感知系数 0.2，与论文 0.1 不同；`configs/paper.yaml` 采用论文 0.1。LPIPS/Real-ESRGAN 配置是用户选定的新实验，不能标为原论文等价结果。

2026-10-26 的首轮 LPIPS / Real-ESRGAN 实验沿用了 GAN=0.2，并在 968 步暂停。
同断点梯度抽查中，GAN 对最后解码层的加权梯度范数为其余目标合计的约 124 倍；
固定小集对照也显示 0.2 会导致重复纹理和重建退化。因此该变体的候选配置改为
GAN=0.002，前 2000 次 G 更新完全禁用对抗路径，之后 2000 次更新线性增加至目标权重。
这两个时长尚未经过全量长跑验证；论文配置的 0.2 和从第一步联合训练保持原样。
预热期间全图/局部 D 的参数、谱归一化缓冲区和 Adam 状态均不更新，不只是把 G 的 GAN loss 乘零。
该调度以 checkpoint 的已完成 `global_step` 计算；改变调度或源码需要显式新实验，不能静默续接旧运行。

### RaLSGAN 与分块

令 `r=D(real)`、`f=D(fake)`，均值覆盖整个 batch 和 logits 的空间维度：

```text
L_D = 0.5 * mean((r - mean(f) - 1)^2)
    + 0.5 * mean((f - mean(r) + 1)^2)
L_G = 0.5 * mean((r - mean(f) + 1)^2)
    + 0.5 * mean((f - mean(r) - 1)^2)
```

正文第 9 页称 relativistic average least-squares，但式 (9) 写成 log/sigmoid 形式。现代实现采用发布代码中明确的 RaLSGAN 平方目标，并记录该歧义。D 更新时 fake detach；G 更新时冻结 D 参数，但保留输入梯度，两个相对项均参与 G 梯度。

显存 chunk 只控制 D 前向的组织；所有 logits 拼接后再求均值。因此它不把目标变成每个 chunk 独立的 RaLSGAN。局部裁剪针对每个样本独立生成，同一更新内 real/fake 以及 D/G 阶段共享对应位置。

采用谱归一化的 D 在每个 D 阶段前做一次 FP32 power iteration，D optimizer step 后、G 阶段前再做一次；阶段内固定 `u/v`。这样 activation checkpoint 重算和 chunk 数量不改变目标。它与旧代码每次 forward 都更新 SN buffer 有明确工程语义差异，不能称逐位等价。

### 感知监督与 Real-ESRGAN D

旧 VGG `relu5_2` / `relu5_3` 错接为 `relu5_1` 的重复调用；现代实现使用正确共享 VGG16 网络和 ImageNet 输入归一化。正文第 8 页式 (7) 后指定感知层为 `relu1_1, relu2_1, relu3_1, relu4_1, relu5_1`，式 (8) 后明确要求风格与感知采用相同激活层。因此现代论文配置的两种损失均使用这五层。旧风格损失选择 `relu2_2, relu3_3, relu4_3, relu5_2`，这是发布代码对论文的偏离，本次予以修正。旧代码也缺少相同输入归一化，因此这里不追求旧损失数值复刻。

历史 Real-ESRGAN 变体只将全图 D 换为官方预训练 Real-ESRGAN U-Net D；局部 Patch D 保留，全图 D 继续训练。全图 D 输入为 `[0,1]`；局部 D 使用 `[-1,1]`；二者均判断原始预测。LPIPS 使用合成完成图和 GT，输入为 `[-1,1]`，冻结预训练网络、FP32 计算、保留对输入的梯度。L1 使用全图原始预测，包含已知区域；结构和纹理辅助监督仍各为 1。

## RTV 结构监督与数据协议

RTV 源于 `legacy/data/Matlab/tsmooth.m`。移植保留零填充 Gaussian、前向差分及末端零梯度、RGB 聚合权重、`lambda/2`、每轮 sigma 减半且下限 0.5，以及每轮使用原始 RGB 作为线性系统右端。默认参数为 `(0.015, 3, 0.001, 3)`，先对原图求结构，再与 RGB 同步增强。

现代实现用 SciPy sparse LU 求解相同加权网格系统；MATLAB 发布实现在存在 ichol 时使用预条件 PCG（容差 0.1、最多 100 步），否则走直接求解分支。LU 与 PCG 求解器误差不同，现代缓存也从量化图像改成 float32 NPY，所以不保证与既有 MATLAB 结构图逐像素一致。独立 dense 线性系统测试验证移植公式，不代替真实 MATLAB 对照。

Manifest 严格配对 RGB/结构图，检查唯一 ID、尺寸、图像合法性及可选内容 hash。自动准备的文件含 hash；手写 manifest 若省略 hash，断点只能约束 manifest 内容，无法据此保证底层文件从未改变。共享随机变换按 seed/epoch/index 决定，与 worker 调度无关。

正文第 10 页实验使用 256×256、中心 128×128 洞及 PConv 不规则 mask 分组。现代默认在线矩形/笔刷/混合 mask，覆盖 10%–40%；它是明确的数据策略变化。需要原文对照时应准备对应文件 mask，不能把在线随机分布称作完全相同的评测协议。

## 训练状态与兼容边界

新的发布版损失 profile 为 `loss.vgg_profile=released`：按原代码直接将 RGB [-1,1] 输入 VGG，感知层为 relu1_1/2_1/3_1/4_1/5_1，风格层为 relu2_2/3_3/4_3/5_2。只修正原 relu5_2/5_3 的误连，不把这个 profile 静默改为 ImageNet 归一化或论文层选择。`paper` profile 保留前述论文式归一化及层选择。两个 profile 都共享一个冻结 backbone，在 FP32 中计算、分块后按实际 batch 大小加权，且其 profile/归一化/层选择写入断点 provenance。

Encoder 执行语义以及 Patch D 参数键已经改变，新 checkpoint 语义标记为 `medfe_joint_released_encoder_corrected_pconv_bpa_aux_v2`。旧实验继续使用自己的冻结源码，不能将旧参数在形状一致时静默解释成新生成器。新实验从头初始化，不续接存在格纹的3310步断点。

正文第 10 页采用 Adam、学习率 0.0002；CelebA / Paris / Places2 分别训练 6 / 30 / 60 个 epoch。当前两个示例均默认 60 个 epoch，必须根据数据与实验目标设置，而不是假定所有数据集原文均为 60。

现代断点有独立格式和训练语义标识，保存全部模型、冻结感知权重、优化器、调度器、scaler、RNG、数据游标、权重 provenance 及源码 hash。恢复检查源码一致；运行目录保留 `source/medfe/` 快照。旧 checkpoint 或测试 fixture 不会静默进入生产训练，暂不提供自动转换旧 optimizer/语义的承诺。

完整断点自包含预训练权重，恢复不依赖原下载文件。允许迁移设备与路径，但精确逐位恢复只对已验证的相同软件、设备和确定性设置成立；跨版本或硬件不承诺逐位一致。推理校验模型结构/语义/权重，不要求当前源码 hash 一致。数值测试和短测的边界见 [validation.md](validation.md)。
