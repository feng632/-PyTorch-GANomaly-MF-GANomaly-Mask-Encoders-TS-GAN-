# 实验记录

## 实验 000：项目初始化

- 状态：完成
- 目的：建立隔离环境并确认 PyTorch 可以调用 NVIDIA GPU。
- 尚未开始模型训练。
- Python：3.11.15（项目独立环境）
- PyTorch：2.11.0+cu128
- 与论文差异：论文使用 PyTorch 1.8.0、CUDA 11.4；RTX 5070 采用新版 PyTorch/CUDA。
- GPU 验收：RTX 5070 Laptop GPU 可用，CUDA 矩阵计算通过。

## 实验 001：MVTec AD grid 数据准备

- 状态：完成
- 数据位置：`E:/datasets/mvtec_ad/grid`
- 来源：公开镜像 `foersben/mvtec-ad`，仅下载 `grid` 类别。
- 许可证：CC BY-NC-SA 4.0；仅用于非商业科研复现。
- 训练正常图：264 张
- 测试正常图：21 张
- 测试异常图：57 张
- 像素级掩码：57 张
- 异常图与掩码缺失配对：0
- 图片解码检查：通过
