# MF-GANomaly 训练前说明

## 当前状态

生成器、损失、交替更新、配置、检查点、曲线和图片级测试已接好，
并已完成一次随机种子 42 的正式 150 轮训练与评估。四块最大值 Image AUROC
为 0.7703，四块平均值 Image AUROC 为 0.7510。原版 models.py、losses.py、
training.py、train.py、evaluate.py 及配置没有改动。

## 在 PyCharm 中运行

沿用项目现有 `.venv` 解释器，无需重新安装依赖。

1. 插上存有数据集的 U 盘，确认 `E:/datasets/mvtec_ad/grid` 存在。
2. 正式训练：打开 `src/anomaly_reproduction/train_mf.py`，直接点运行，无需参数。
3. 训练完成后：打开 `src/anomaly_reproduction/evaluate_mf.py`，直接点运行。

默认配置从脚本所在目录定位，与 PyCharm 的工作目录无关。
若启动时出现 `No module named anomaly_reproduction`，在 PyCharm 将项目的 `src`
标记为 Sources Root，并确认使用本项目解释器。
请清空运行配置里此前用于检查的 `--smoke-test` 参数，否则不会开始正式训练。

等价命令（在项目根目录）：

```powershell
.\.venv\Scripts\python.exe -m anomaly_reproduction.train_mf --smoke-test
.\.venv\Scripts\python.exe -m anomaly_reproduction.train_mf
.\.venv\Scripts\python.exe -m anomaly_reproduction.evaluate_mf
```

第一条只执行一个配置大小的真实训练批次并验证权重存取，临时文件随后清理。
第二条才是正式训练。每次启动创建新模型，检查用模型不会接着用于正式训练。

## 参数与输出

配置：`configs/mf_ganomaly_grid.yaml`。

- 264 张正常训练原图展开为 1056 块，每批 16 块，每轮 66 批。
- 64x64 RGB，隐向量 128 维，150 轮，种子 42。
- Adam：学习率 0.0002，beta1=0.5，beta2=0.999。
- 重建/编码/对抗权重为 40/1/1；测试 alpha=40。
- E1 和 E2 结构一致、参数独立；浅层特征取第一层模块的激活后输出。
- 默认 `encoder_variant: paper`，第一层与最后一层均增加 BN + LeakyReLU。

正式结果目录：`runs/mf_ganomaly_grid_paper_l2_seed42/`。

- `config.yaml`：实际配置。
- `history.csv`：每轮各项损失，包含 pixel/shallow/encoding/adversarial。
- `loss_curves.png`：总损失曲线。
- `checkpoint_latest.pt`：每 10 轮更新一次。
- `checkpoint_final.pt`：150 轮完成的模型。
- `checkpoint_interrupted.pt`：捕获 Ctrl+C 时保存；PyCharm 强制杀进程不保证保存。

检查点包含模型、Adam 状态、配置与历史。目前没有自动断点续训入口；
中断权重可以手动指定用于评估，但不声称支持精确恢复训练。
重复启动同名实验会拒绝覆盖；要做新实验，先修改配置的 experiment.name。

测试输出在对应检查点旁的 `evaluation_checkpoint_final/`：

- `test_patch_scores.csv`：各块的总分及三种差异。
- `test_image_scores.csv`：同一原图的 max/mean 两种聚合，及各自归一化分数。
- `test_metrics.yaml`：两种 Image AUROC，明确命名，不选取测试集表现最好的结果。
- `reconstruction_residual_preview.png`：重建和像素差预览，**不是论文的滑窗 PSNR 热图**。

只加载自己生成的可信检查点。自定义检查点可用 `evaluate_mf --checkpoint <路径>`。

## 论文对应与明确的复现假设

来源：步兆军《基于自监督与弱监督学习的图像异常检测研究》第 3.3 节，
正文页 33-36（图 3.4/3.6，表 3.1，式 3.4-3.8）；训练参数见正文页 38。

1. **L1/L2 冲突**：图 3.6 标 L1，式 3.4 和 3.8 写 L2。
   默认按公式：每个样本先展平求欧氏范数，再对 batch 求均值。
   不是 MSE，也不是每元素的平均绝对误差。
   可将 `reconstruction_metric` 改为 `l1` 做图示对照（使用 L1 范数，即绝对差之和）。
   编码与对抗特征距离仍为 L2。范数的汇总细节未见作者实现代码，属于明确实现选择。
2. **BN/激活**：默认采用正文描述的编码器每层 BN + LeakyReLU；
   `baseline` 选项保留原版第一层无 BN、最后一层无 BN/激活的实现，供后续消融。
   浅层特征按完整模块输出解释；论文没有明确 BN/激活前后取值位置。
3. **判别器**：复用基线 DCGAN 风格判别器和 BCE 真伪头，
   对抗特征仍取其分类头之前的 512x4x4 特征。论文只笼统描述与编码器结构相同，
   真伪头及特征取点细节不够完整，因此不宣称逐层严格一致。
4. **裁剪和聚合**：沿用既有四象限裁剪、缩放和归一化。论文裁剪描述出现在
   热图实验段落，不能据此断言分类 AUC 也严格使用此协议；整图聚合方式未明确。
   测试同时报告 max/mean，不按测试成绩调 alpha、选模型或选聚合。
5. **比较限制**：现有原版测试使用编码的平均绝对差，训练使用 L1/MSE；
   当前默认 MF 使用公式 L2 范数，并改变了编码器 BN 配置。
   因此新旧结果是两套完整实现的对照，不能把全部差异归因于多尺度特征。
   若研究改进来源，应另外统一结构、距离、聚合并进行多种子消融。
6. **分数归一化**：保存原始分数，并在整图聚合后按本次测试集 min-max 归一化。
   常数分数映射为 0；AUROC 用原始分数算。该归一化不是固定部署阈值的校准。

L2 范数没有除以像素数或通道数，损失数值较大是正常的；不能直接比较原版与 MF
总损失的数值大小来判断优劣。程序检查损失和梯度的 NaN/Inf。

## 检查范围

自动测试覆盖距离公式、零差异梯度、浅层梯度连通、两编码器独立、G/D 更新、
BN 更新次数、四块聚合、常数归一化、合成数据上的训练/保存/测试闭环和防覆盖。
另用真实数据在 CUDA 做单批检查；不等于验证长时间训练收敛或最终效果。

2026-09-04 验收：18 项自动测试通过；CUDA 上配置大小的 16 样本单批
更新、保存、回读和评分通过；全部 312 个测试裁剪图/掩码可读取
（21 张正常原图、57 张异常原图）。正式结果目录尚未创建。

滑窗 PSNR 热图、Pixel AUROC、多种子统计是后续实验工作，不影响现在启动
图片级 MF-GANomaly 训练；本次未上传 GitHub，也未生成正式训练结果。
