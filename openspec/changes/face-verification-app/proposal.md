## Why

用户需要一个轻量级的桌面软件，用来回答一个简单问题："这张测试照片里的人，是不是我之前注册过的那个人？"（1:1 人脸验证）。现有需求是：投喂一个人若干张照片作为"注册"，之后对任意一张测试照片给出「是/不是 + 相似度」的判断，并能随时更换或删除注册的人。目标是尽可能轻量、只做最基本功能（不做活体检测），每个模块优先复用 GitHub 上的成熟开源方案，不重复造轮子。

## What Changes

- 新增一个 Python 桌面软件 `face-verify`（tkinter 界面），运行在用户本机（Python 3.13 + OpenCV 已装）。
- 新增「人脸注册」能力：从多张照片提取一个人的人脸特征，附上名字标签，持久化到本地。
- 新增「人脸验证」能力：对单张测试照片判断是否匹配已注册的人，返回匹配结果与相似度分数。
- 新增「注册管理」能力：更换注册（覆盖，带确认）与删除注册（清空，带确认）。
- 复用成熟开源方案：OpenCV 内置 YuNet 做人脸检测、InsightFace 的 ArcFace 模型做人脸特征提取（ONNX + onnxruntime）。
- 新增依赖：`onnxruntime`、`numpy`（`opencv-python`、`pillow` 已装）。

## Capabilities

### New Capabilities

- `face-registration`: 注册一个带名字的人脸（从一张或多张照片提取特征并持久化），支持更换（覆盖）与删除。
- `face-verification`: 对单张测试照片判断是否匹配已注册的人，返回匹配结果与相似度分数，并处理无脸/多脸/未注册等异常。

### Modified Capabilities

（无 —— 全新项目，无既有规范。）

## Impact

- 新增源码目录 `src/faceverify/`（detector / align / embedder / matcher / store / gui / main 七个模块）。
- 新增单元测试 `tests/`（matcher、store 的纯逻辑）。
- 新增运行时资源 `models/`（YuNet + ArcFace 两个 ONNX 模型，需一次性下载，不入 git）。
- 新增 `requirements.txt`（onnxruntime 等）、`README.md`。
- 不涉及数据库、不涉及网络服务、不涉及活体检测。
