# face-verification-light

轻量级 1:1 人脸验证桌面软件。注册一个人若干张照片，然后对一张测试照片判断"是不是这个人"，并给出相似度分数。

## 技术栈

- 检测：OpenCV YuNet（`cv2.FaceDetectorYN`）
- 识别：InsightFace ArcFace（`w600k_r50` ONNX，onnxruntime 推理）
- 对齐：双眼 + 鼻尖三点仿射变换
- 界面：tkinter

## 安装

```bash
pip install -r requirements.txt
```

## 下载模型

模型文件较大，需单独下载到 `models/`（已 gitignore）：

```bash
# YuNet 检测模型（约 230 KB）
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# ArcFace 识别模型（约 166 MB，从 buffalo_l 压缩包中提取）
curl -L -o models/buffalo_l.zip \
  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
python -c "import zipfile; zipfile.ZipFile('models/buffalo_l.zip').extract('w600k_r50.onnx', 'models/')"
rm models/buffalo_l.zip
```

## 运行

```bash
python run.py
```

## 使用

1. 点「注册照片」，选一张或多张含人脸的图片，输入姓名。
2. 点「测试照片」，选一张图片，软件给出「是/不是 + 相似度」。
3. 阈值默认 0.35，可在界面拖动调整。
4. 重新注册会覆盖旧注册（有确认）；「删除注册」可清空。

## 测试

```bash
python -m pytest
```

## 打包成 exe（可选，方便发给朋友）

让朋友不用装 Python 也能用：

```bash
pip install pyinstaller
python -m PyInstaller --onefile --windowed --name FaceVerify \
  --paths src --add-data "models;models" \
  --collect-all onnxruntime --collect-all cv2 --noconfirm run.py
```

产物在 `dist/FaceVerify.exe`（约 240MB，已含模型），朋友双击即可运行。

## 说明

- 识别率受照片角度、光照、遮挡、年龄变化影响；注册时多选不同角度/光照的照片可提升准确率。
- 若阈值 0.35 在你照片上不准，用界面滑块微调。
- 检测到多张脸时默认取最大的一张。
- 不包含活体检测（防照片攻击）。
