## 1. 项目初始化

- [ ] 1.1 创建 `src/faceverify/` 包结构（含 `__init__.py`）与 `tests/`、`models/` 目录
- [ ] 1.2 编写 `requirements.txt`（onnxruntime、numpy；标注 opencv-python/pillow 已装）
- [ ] 1.3 安装依赖并验证 `import onnxruntime` 可用；若不可用则改用 `cv2.dnn` 方案
- [ ] 1.4 下载 YuNet 检测模型与 ArcFace 识别模型到 `models/`，并在 README 记录下载源

## 2. 核心逻辑（TDD）

- [ ] 2.1 用 TDD 实现 `matcher.py`：余弦相似度、均值参考向量、阈值判定（match/no-match + score）
- [ ] 2.2 用 TDD 实现 `store.py`：保存/覆盖/删除/加载注册信息（JSON + .npy），无注册时返回空
- [ ] 2.3 为 matcher/store 补齐边界测试（空向量、损坏文件、无注册状态）

## 3. 识别流水线

- [ ] 3.1 实现 `detector.py`：加载 YuNet，输入图像 → 输出人脸框与 5 个关键点；无脸返回空
- [ ] 3.2 实现 `align.py`：用 5 关键点做相似变换，输出 112×112 对齐人脸
- [ ] 3.3 实现 `embedder.py`：加载 ArcFace ONNX，对齐人脸 → 512 维 L2 归一化向量

## 4. 界面与组装

- [ ] 4.1 实现 `gui.py`：注册（选多张照片 + 输入名字）、测试（选单张照片）、删除注册，展示结果与相似度、可调阈值
- [ ] 4.2 实现 `main.py` 组装各模块并启动 GUI
- [ ] 4.3 实现覆盖/删除的确认对话框，以及无脸/多脸/未注册的界面提示

## 5. 验证与收尾

- [ ] 5.1 用"本人两张 + 他人一张"照片做全链路手动 smoke test，核对匹配/不匹配/无脸/未注册四种结果
- [ ] 5.2 编写 README（安装、模型下载、使用说明、阈值调参提示）
- [ ] 5.3 运行全部单元测试确认通过
