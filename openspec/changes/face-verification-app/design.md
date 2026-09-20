## Context

目标是一台 Windows 本机上的轻量级 1:1 人脸验证桌面软件。已知约束：

- Python 3.13.14、OpenCV 5.0.0、numpy、pillow 已安装。
- Python 3.13 很新：`dlib`（face_recognition 的依赖）、`tensorflow`（DeepFace 的依赖）、`torch`（YOLO/ultralytics 的依赖）在该版本上要么无预编译轮子、要么体积极大，安装风险高。
- 需求要求"每个模块复用 GitHub 成熟开源方案，不自造轮子"，且"轻量、最基本功能"。

据此，识别引擎选用 **ONNX 模型 + onnxruntime**，避开 dlib/TF/torch。动机见 proposal.md - Why。

## Goals / Non-Goals

**Goals:**

- 用最少的重依赖跑通"注册 → 验证"闭环。
- 每个可拆分的步骤（检测/对齐/特征提取/比对/存储/界面）各自复用成熟开源组件。
- 可单元测试的核心逻辑（比对、存储）与依赖模型的部分分离。

**Non-Goals:**

- 不做活体检测（防照片攻击）。
- 不做 1:N 多人识别、不做数据库、不做联网服务。
- 不追求模型自训练或精度调优，直接用预训练权重。

## Decisions

### 1. 人脸检测：OpenCV 内置 YuNet（而非 YOLO / SCRFD / MTCNN）

- **选择**：`cv2.FaceDetectorYN`（YuNet，约 230KB，OpenCV 4.5.4+ 内置，模型文件单独下载）。
- **理由**：YuNet 是 OpenCV 官方维护的人脸检测器，最轻、零额外依赖，且直接输出 5 个关键点（双眼、鼻尖、两嘴角），正好给对齐步骤用，无需单独的 landmark 模型。
- **替代**：YOLO（需 torch，重且 3.13 风险）；SCRFD（InsightFace 检测器，精度略高但需 onnxruntime 跑，可作为 YuNet 的备选）；MTCNN（需 torch）。

### 2. 特征提取：InsightFace ArcFace ONNX（而非 dlib / DeepFace / insightface 包）

- **选择**：ArcFace 识别模型（如 `buffalo_l` 的 `w600k_r50.onnx`，输出 512 维 L2 归一化向量），通过 `onnxruntime` 推理。
- **理由**：ArcFace 是 GitHub 上最知名、开源、精度 SOTA 的人脸识别模型之一，完全满足"复用成熟开源方案"；ONNX + onnxruntime 对 Python 3.13 兼容、体积小（onnxruntime 约 15MB）。
- **替代**：dlib 的 face_recognition（128 维，最易用，但 dlib 在 3.13 上难装）；DeepFace（封装好但依赖 TensorFlow，重）；`insightface` 包（封装好但模型从 Google Drive 下载易失败、黑盒、依赖更多）。

### 3. 对齐：标准 5 点相似变换到 112×112 规范脸

- **选择**：用 YuNet 输出的 5 个关键点，做标准相似变换（估计仿射矩阵 → 仿射变换 → 缩放到 112×112），对齐到 ArcFace 训练用的规范坐标。
- **理由**：这是 ArcFace 的标准预处理，成熟通用；5 个关键点已由 YuNet 提供，无需额外模型。
- **替代**：不做对齐直接送原脸（精度下降，但对 v1 可作降级）。

### 4. 比对：余弦相似度 + 可调阈值

- **选择**：向量已 L2 归一化，余弦相似度 = 点积（范围 [-1,1]）。相似度 ≥ 阈值判为匹配。默认阈值 0.35，界面可调。
- **理由**：ArcFace 社区对该模型的"同人"阈值经验值约 0.3–0.4，取 0.35 作起点，允许用户在界面微调。
- **替代**：欧氏距离（等价于余弦，无额外收益）；二分类器（过重）。

### 5. 注册参考：多张特征向量的均值

- **选择**：对注册人的每张照片各提取一个 512 维向量，取算术平均（再 L2 归一化）作为参考向量。
- **理由**：均值对多角度/多光照最稳健，实现简单。
- **替代**：存全部向量逐个比对取 max（稍准但更复杂，v1 不需要）。

### 6. 界面：tkinter（而非 PySide6 / PyQt）

- **选择**：tkinter（Python 标准库自带，零额外依赖）。
- **理由**：满足"轻量化"最彻底，单窗口、几个按钮即可，无需打包体积大的 Qt 依赖。

### 7. 存储：JSON（元数据）+ .npy（参考向量）

- **选择**：`reference.npy` 存 512 维向量，`registration.json` 存名字与照片数，放在应用数据目录下。
- **理由**：比数据库轻得多，够用；numpy 已在依赖里。
- **替代**：SQLite（对单条注册过重）。

### 8. 工程结构：`src/faceverify` 七模块，职责单一

- `detector.py`（找脸+关键点）、`align.py`（对齐）、`embedder.py`（提向量）、`matcher.py`（纯比对逻辑）、`store.py`（纯持久化）、`gui.py`（界面）、`main.py`（组装）。
- 其中 `matcher` 与 `store` 不依赖模型，可纯单元测试；`detector/align/embedder` 依赖模型，靠手动 smoke test 覆盖。

## Risks / Trade-offs

- [模型文件需一次性下载（YuNet ~230KB + ArcFace ~166MB）] → 启动时检测缺失并给出下载指引；README 写明下载源（OpenCV 官方仓库 / InsightFace 模型仓库镜像）。
- [Python 3.13 下 onnxruntime 的可用性] → 实现第一步先 `pip install onnxruntime` 并跑最小推理验证，失败则回退用 OpenCV 的 `cv2.dnn` 直接读 ONNX（OpenCV 已支持 ONNX 推理）。
- [阈值 0.35 可能不适用于用户的实际照片] → 阈值做成界面可调，并在 README 说明"识别率受照片角度/光照影响"。
- [多脸检测"取最大脸"可能取错] → 在界面提示当前取的是最大脸；后续可加"手动框选"。

## Migration Plan

1. 新建项目结构，`pip install onnxruntime`。
2. 下载 YuNet 与 ArcFace 模型到 `models/`。
3. 逐模块实现（TDD 覆盖 matcher/store），最后组装 GUI。
4. 用"本人两张 + 他人一张"照片做一次手动全链路 smoke test。

无回滚需求（全新应用，非线上服务）。
