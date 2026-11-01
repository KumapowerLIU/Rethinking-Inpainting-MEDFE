# SD 风格 PartialConv 骨干

`model.backbone=sd_partial` 是保留 MEDFE 特征融合模块的新架构实验。
它借鉴 [Stable Diffusion 的残差块与上采样结构](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py)，
在 RGB 空间一次前向完成修复；不使用扩散时间步、文本条件、attention、VAE 或 SD 预训练权重。
它不属于 ECCV 2020 原网络的等价重构，也不承诺单靠更换骨干消除网格。

## 网络与接口

| 部分 | `released` | `sd_partial` |
| --- | --- | --- |
| 编码器 | 4×4 stride-2 普通卷积 | 3×3 stride-2 PartialConv，每尺度两个 PartialConv 残差块 |
| 最深层 | 四个膨胀卷积残差块 | 四个 PartialConv 残差块 |
| 骨干归一化 / 激活 | InstanceNorm / ReLU、LeakyReLU | GroupNorm / SiLU |
| 解码器 | 六层反卷积 | 残差块、最近邻放大、3×3 普通卷积 |
| RGB 输出 | Tanh | Tanh |
| MEDFE 融合 | 结构／纹理、共享 3/5/7 PConv、BPA 和辅助头 | 保留同一 `MutualFeatureBlock` 实现 |

输入仍为遮挡 RGB 加三通道已知区域指示。输入 256×256、基础宽度 64 时，
六个编码尺度是 `128/64/32/16/8/4`，通道是 `64/128/256/512/512/512`。
前三层融合纹理，后三层融合结构，在 32×32 完成均衡，再注入六级 skip。
输出接口和结构／纹理辅助监督不变，已知区域仍精确保留输入像素。

残差分支末层在全局初始化后清零，最初从 identity 或投影 skip 开始学习。
因此第一次反向中部分残差内部参数梯度为零是预期行为；末层仍能更新。
每尺度残差数量由 `model.blocks_per_scale` 控制，最深层数量沿用 `model.residual_blocks`。

## Mask 与归一化

公共输入 `mask=True` 表示洞，进入编码器时转换为 `known=True` 表示有效。
每次 PartialConv 先遮掉无效输入，以有效像素数修正卷积结果并传播有效 mask。
残差 skip 也先遮掉无效值；两个分支取有效区域并集，全无效输出保持严格为零。

融合模块收到的是新编码器第三层传播出的 mask。不能沿用旧编码器
`4×4/stride-2` 的三次下采样近似：新卷积和残差块使有效区域扩张更快。
默认两个残差块时，第三层几何感受野宽度为 127，而旧 mask 路径为 22。
例如 256 图中的居中 128 方洞，32×32 mask 的无效点从旧路径的 196 个降至新路径的 1 个。
这意味着许多随机洞进入融合模块前已被标为有效。保留的是 PCConv 模块实现，
其输入特征和 mask 分布会改变；有效 mask 表示几何支持，不表示内容已经修复正确。
零初始化时 mask 仍按卷积感受野扩张，即使新位置的初始特征还是零。

骨干 GroupNorm 使用不超过 32、且整除通道数的最大分组数。FP16/BF16 统计升为 FP32，
FP64 数值测试保持 FP64。它是普通 GroupNorm，统计包含无效零位置，并非 masked normalization。
归一化后进入下一次 PartialConv 时会再次应用 mask。
MEDFE 融合模块内原有的 InstanceNorm 保持原实现。

## 独立训练配置

`configs/sd-partial-l1-lpips-large-batch.yaml` 延用已有训练目标：

- 全图原始预测 L1=1，结构 L1=1，纹理 L1=1，VGG-LPIPS=1。
- 独立 VGG perceptual/style 系数均为 0；全图与局部原始 Patch D，RaLSGAN 上限 0.002。
- 前 5000 步关闭 GAN，随后 10000 步线性增加；随机 mask 面积 10%–40%。
- batch 128、256²、BF16；新骨干和融合模块均支持 activation checkpointing。

先将数据路径指向现有 manifest/cache，在空的新目录中启动：

```bash
medfe train --config configs/sd-partial-l1-lpips-large-batch.yaml \
  --set train.device=cuda:0 \
  --set data.train_manifest=/path/to/existing/train.jsonl \
  --set data.val_manifest=/path/to/existing/validation.jsonl \
  --set data.rtv_cache=/path/to/existing/rtv-cache
```

该命令是通用操作说明。实际 batch 容量以完整 D/G 和 LPIPS 短测为准。
新模型从新初始化开始，不能把旧骨干 checkpoint 作为 `--resume`。

### 与 CSA 相同的数据混合

`configs/sd-partial-csa-mix-large-batch.yaml` 为 CSA 数据混合配置，显式开启
`data.allow_repeated_records=true`。训练清单逐字节复制自 CSA 最终人脸混合实验：
296,145 条 Unsplash、901,730 条 Places365 Standard 和 299,469 条 HDTF 人脸采样记录，
总计 1,497,344 条。其中人脸是 30,128 张不同帧，每帧采样 9 或 10 次，占比约 20%。
去重会改变训练分布，因此读取器保留这些行以及各自的随机 mask seed。

HDTF 记录采用 `{"format":"image","path":"/absolute/frame.png"}`，普通图片和
TAR/TFRecord 图片使用相同的规范化 RGB→RTV 流程。相同图片复用同一份 RTV 缓存。
未显式开启时仍拒绝重复记录；开启后也拒绝同一源位置对应冲突的长度或格式。
验证读取器始终拒绝重复记录。
初始化使用最多 32 个线程并行读取唯一源文件的元数据，结果保存在进程内；
随后按原顺序构建记录，两遍清单的字节哈希必须一致，避免大量人脸文件的冷存储查询串行阻塞启动。

主验证集保持 CSA 的原有 2,000 张 Unsplash，不混入独立的人脸 holdout。
同数据集指样本来源、划分和采样权重相同；MEDFE 保持自己的 mask/翻转随机数和 epoch sampler，
不保证与 CSA 逐 batch 顺序和 mask 相同。

旧配置没有 `backbone` 字段时仍按 `released` 加载，原有 state_dict key 和前向路径保留。
新 checkpoint 使用独立数学语义标识，读入时同时检查骨干配置和语义。
旧模型推理继续可用；旧训练精确续跑仍应使用运行目录的冻结源码。

## 验证边界

自动检查覆盖独立 mask 传播参考、全洞输入、无效区域隔离、非方形输入、已知像素保留、
样本独立性、前后向有限性、activation checkpoint 重算一致性、新旧 checkpoint 隔离和恢复。
修改前后的小模型旧路径数值对照，五个输出的最大绝对差均为 0。
本地 CPU 套件：326 通过，5 项 CUDA 专用检查跳过。
远程 NVIDIA H100 80GB / PyTorch 2.6.0+cu124 短测：完整基础宽度 64、batch 128、256²、BF16，
真实预训练 VGG-LPIPS 加全图／局部 PatchGAN 完成两次 D/G 更新。GPU allocator 的峰值使用量
25.925 GiB、峰值保留量 30.551 GiB；它们不包含全部驱动开销。
该短测使用合成 RGB 和结构目标，直接启用 GAN 上限以检查完整路径，不改变正式配置的预热。
窄模型的 CUDA BF16 重算检查中，前向和全部参数梯度最大差均为 0；完整宽度 FP32 推理也有限且保留已知像素。
生成器参数量为 144,854,438。短测未保存模型，旧生产源码和 checkpoint 均未修改。
短测记录存于 `artifacts/medfe-sd-partial-20261029/`。
这些检查证明实现与数值接口可运行，尚不代表新架构的训练效果。
