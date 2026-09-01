# Image Anomaly Detection Reproduction

本项目用于逐步复现步兆军硕士论文《基于自监督与弱监督学习的图像异常检测研究》。

## 当前状态

已完成 GANomaly 在 MVTec AD `grid` 类别上的端到端基线：数据读取、网络、损失、训练、检查点、重建预览和 Image-level AUROC。正在进入 MF-GANomaly 复现阶段。

我们要回答的科研问题是：

> 在 GANomaly 中加入浅层特征误差，是否能让模型更容易发现面积较小的缺陷？

## 为什么先做这一部分

- GANomaly 是论文三个方法共同的基础。
- MF-GANomaly 的改动相对集中，适合第一次科研复现。
- `grid` 是规则纹理，正常与缺陷的差异较直观，便于观察热图。

## 当前结果

| 实验 | 图片级分数聚合 | Image AUROC |
|---|---|---:|
| 整图缩放到 64x64 | 单图编码差异 | 0.5196 |
| 固定四块裁剪 | 四块最大值 | 0.6817 |
| 固定四块裁剪 | 四块平均值 | **0.7026** |
| 论文报告的 GANomaly | 论文设置 | 0.658 |

论文未明确说明四块分数的图片级聚合方式，因此最大值和平均值均保留为复现假设。当前结果来自单个随机种子 42，不能视为最终统计结论。

## 已完成

- [x] 建立独立 Python/CUDA 环境并验证 GPU。
- [x] 读取 MVTec AD 正常图、异常图和像素掩码。
- [x] 实现 Encoder、Decoder、双编码器 Generator 和 Discriminator。
- [x] 实现重建、编码、对抗特征和判别器损失。
- [x] 实现交替训练、检查点、CSV日志和损失曲线。
- [x] 完成整图缩放与固定四块裁剪对照实验。
- [x] 实现测试评分与 Image-level AUROC。

## TODO

- [ ] 实现滑动窗口 PSNR 异常热图及四块热图拼接。
- [ ] 计算 Pixel-level AUROC、PSNR 和 SSIM。
- [ ] 使用至少 3 个随机种子重复 GANomaly 基线。
- [ ] 增加 DCGAN 权重初始化对照实验。
- [ ] 实现 MF-GANomaly 的浅层多尺度重建损失。
- [ ] 实现 MF-GANomaly 多尺度异常分数与消融实验。
- [ ] 实现 Mask Encoders 两阶段自监督训练。
- [ ] 实现 TS-GAN 弱监督双流训练。

## 环境安装

建议使用 Python 3.11 和支持 CUDA 的 PyTorch。先根据显卡和驱动从 PyTorch 官方源安装 `torch` 与 `torchvision`，再安装其余依赖：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

本次开发环境使用 PyTorch 2.11.0+cu128。论文环境为 PyTorch 1.8.0、CUDA 11.4，两者存在版本差异。

## 数据准备

从 MVTec AD 官方来源下载数据。本仓库不分发数据，MVTec AD 使用 CC BY-NC-SA 4.0 许可证，仅允许相应条款下的使用。

预期目录：

```text
<DATA_ROOT>/grid/
  train/good/
  test/good/
  test/bent/
  test/broken/
  test/glue/
  test/metal_contamination/
  test/thread/
  ground_truth/<defect_type>/
```

修改 `configs/*.yaml` 中的 `dataset.root` 指向本机数据目录。

## 运行

检查环境和数据：

```powershell
python -m anomaly_reproduction.check_environment
python -m anomaly_reproduction.check_four_crops
```

训练论文描述的固定四块裁剪基线：

```powershell
python -m anomaly_reproduction.train_four_crops
```

评估最终检查点：

```powershell
python -m anomaly_reproduction.evaluate --checkpoint runs/ganomaly_grid_four_crops_seed42/checkpoint_final.pt
```

运行测试：

```powershell
python -m pytest -q
```

## 在 PyCharm 中查看数据预览

运行 `src/anomaly_reproduction/inspect_grid.py`。脚本会读取 U 盘中的
`E:/datasets/mvtec_ad/grid`，并生成 `reports/grid_data_preview.png`。

## 目录说明

```text
configs/       实验参数；每次实验用什么设置都记录在这里
data/          数据说明；大型原始数据不提交到代码仓库
src/           模型、数据读取和训练代码
tests/         小型自动检查，防止代码改坏
runs/          训练日志和模型权重
reports/       图表、结果和复现实验记录
```

## 复现原则

- 先跑通，再追求论文数值。
- 每次只改变一个因素。
- 不挑最好的一次结果；保留随机种子和全部实验记录。
- 论文没有写清楚的实现细节必须标为“复现假设”。

## 重要说明

- `runs/`、模型权重、虚拟环境和数据集均被 Git 忽略。
- 当前误差图是像素残差预览，并非论文正式的滑动窗口 PSNR 热图。
- 本项目用于学习和科研复现，不提供医疗或工业生产结论。
