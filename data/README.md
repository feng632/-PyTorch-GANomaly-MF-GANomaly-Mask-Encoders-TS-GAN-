# 数据目录

本机数据保存在 U 盘 `E:/datasets/mvtec_ad/`，项目通过配置文件引用它，不复制大型图片。

第一轮只使用 `grid`：

```text
grid/
  train/good/          训练用正常图像
  test/good/           测试用正常图像
  test/*/              各种缺陷图像
  ground_truth/*/      缺陷区域的像素级掩码
```

当前 `grid` 数据校验结果：训练正常图 264 张、测试图 78 张，其中异常图与掩码各 57 张，配对完整且均可解码。
