# Mask Encoders 运行说明

正式训练：在 PyCharm 直接运行 `src/anomaly_reproduction/train_mask.py`。
程序会先训练第一阶段 80 轮，再自动载入权重训练第二阶段 80 轮。
不要分别手动启动两个阶段。

训练完成后，直接运行 `src/anomaly_reproduction/evaluate_mask.py`。

默认设置：四象限裁剪后缩放到 128x128；第一阶段用固定随机种子抽取
50% 裁剪样本，中心 64x64 填成黑色；第二阶段用全部完整裁剪样本。
两个阶段的 Adam 独立，第二阶段继承第一阶段生成器和判别器权重。

结果位于 `runs/mask_encoders_grid_seed42/`：

- `stage1_pretrain/`：第一阶段曲线和检查点。
- `stage2_finetune/`：第二阶段曲线、检查点和测试结果。
- `config.yaml`：本次完整设置。

代码复用了已有的四块数据集、判别器损失、随机种子、曲线和检查点逻辑；
Mask 数据、128x128 网络、L2 重建损失、两阶段训练和测试为独立新文件，
原版 GANomaly 与 MF-GANomaly 未改动。

论文未明确交代判别器 100 维特征如何汇总为单个真假分数。本实现使用
`Linear(100,1) + Sigmoid`，这是明确记录的复现假设。训练仍按论文采用 BCE，
因此也可能出现此前观察到的 GAN 饱和，训练后需检查判别器曲线。

测试同时记录四块最大值和平均值的 Image AUROC，不凭测试集选择聚合方式。
PSNR/SSIM 当前按每个 128x128 裁剪块计算后平均；论文未明确其跨块汇总细节，
数值只能在确认协议一致后与论文表格严格比较。
