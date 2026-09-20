"""Create a GitHub release and upload the packaged exe as an asset.

Usage:
  GH_TOKEN=<token> python scripts/make_release.py
"""
import json
import os
import urllib.request

REPO = "NagaZhang/face-verification-light"
TAG = "v0.1.0"
NAME = "FaceVerify v0.1.0"
BODY = """轻量级 1:1 人脸验证桌面软件：注册一个人若干张照片，然后对一张测试照片判断「是不是这个人」，并给出相似度分数。

**功能**
- 注册带名字的人脸（支持多张照片）
- 测试照片：判断是否匹配 + 相似度分数
- 更换 / 删除注册
- 可调相似度阈值

**使用**
下载 `FaceVerify.exe` 双击运行即可，无需安装 Python。

**技术栈**
检测 OpenCV YuNet · 识别 InsightFace ArcFace · 界面 tkinter
"""


def main() -> None:
    token = os.environ["GH_TOKEN"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    # 1. Create the release (this also creates the tag).
    payload = json.dumps(
        {
            "tag_name": TAG,
            "target_commitish": "main",
            "name": NAME,
            "body": BODY,
            "draft": False,
            "prerelease": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases",
        data=payload,
        method="POST",
        headers={**headers, "Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    release_id = resp["id"]
    upload_url = resp["upload_url"].split("{")[0]
    print(f"created release id={release_id}, tag={TAG}")

    # 2. Upload the exe as an asset.
    with open("dist/FaceVerify.exe", "rb") as f:
        data = f.read()
    print(f"uploading {len(data)} bytes ...")
    req = urllib.request.Request(
        f"{upload_url}?name=FaceVerify.exe",
        data=data,
        method="POST",
        headers={**headers, "Content-Type": "application/octet-stream"},
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    print(f"asset: {resp['name']}  size={resp['size']}  state={resp['state']}")
    print(f"download: {resp['browser_download_url']}")


if __name__ == "__main__":
    main()
