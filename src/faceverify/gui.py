import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import cv2
from PIL import Image, ImageTk

from faceverify.align import align_face
from faceverify.imageio import imread_unicode
from faceverify.matcher import mean_embedding, verify


class FaceVerifyApp:
    def __init__(self, detector, embedder, store):
        self.detector = detector
        self.embedder = embedder
        self.store = store
        self.registration = store.load()
        self.preview_photo = None
        self.root = tk.Tk()
        self.root.title("人脸验证 (1:1)")
        self.threshold = tk.DoubleVar(value=0.35)
        self._build_ui()
        self._refresh_registration_label()

    def _build_ui(self):
        self.status_label = tk.Label(self.root, text="", font=("", 12))
        self.status_label.pack(pady=10)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="📁 注册照片", command=self.register).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="📁 测试照片", command=self.test).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="🗑 删除注册", command=self.delete).pack(side=tk.LEFT, padx=5)

        self.preview_label = tk.Label(self.root, text="（预览）", width=56, height=20, bg="#eee")
        self.preview_label.pack(pady=10)

        self.result_label = tk.Label(self.root, text="", font=("", 14), fg="#333")
        self.result_label.pack(pady=5)

        tk.Label(self.root, text="阈值").pack()
        self.scale = tk.Scale(
            self.root, from_=0.0, to=1.0, resolution=0.01,
            orient=tk.HORIZONTAL, variable=self.threshold, length=300,
        )
        self.scale.pack(pady=5)

    def _refresh_registration_label(self):
        if self.registration:
            self.status_label.config(
                text=f"当前注册: {self.registration['name']}（{self.registration['n_photos']} 张）"
            )
        else:
            self.status_label.config(text="当前注册: 无")

    def register(self):
        paths = filedialog.askopenfilenames(
            title="选择注册照片（可多选）",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not paths:
            return
        name = simpledialog.askstring("注册", "输入姓名（标签）：", parent=self.root)
        if not name:
            return
        if self.registration and not messagebox.askyesno(
            "覆盖确认", f"已注册 {self.registration['name']}，是否覆盖？"
        ):
            return
        embeddings = []
        used = 0
        for p in paths:
            img = imread_unicode(p)
            if img is None:
                continue
            face = self.detector.detect_largest(img)
            if face is None:
                messagebox.showwarning("提示", f"未检测到人脸，已跳过: {p}")
                continue
            embeddings.append(self.embedder.embed(align_face(img, face.landmarks)))
            used += 1
        if not embeddings:
            messagebox.showerror("注册失败", "所选照片中都没有检测到人脸")
            return
        self.store.save(name, mean_embedding(embeddings), used)
        self.registration = self.store.load()
        self._refresh_registration_label()
        messagebox.showinfo("完成", f"已注册 {name}（{used} 张）")

    def test(self):
        if not self.registration:
            messagebox.showwarning("提示", "请先注册照片")
            return
        path = filedialog.askopenfilename(
            title="选择测试照片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not path:
            return
        img = imread_unicode(path)
        if img is None:
            messagebox.showerror("错误", "无法读取图片")
            return
        face = self.detector.detect_largest(img)
        if face is None:
            messagebox.showwarning("提示", "未检测到人脸")
            return
        self._show_preview(img, face)
        emb = self.embedder.embed(align_face(img, face.landmarks))
        is_match, sim = verify(emb, self.registration["embedding"], self.threshold.get())
        verdict = "✅ 是这个人" if is_match else "❌ 不是"
        self.result_label.config(text=f"{verdict}   相似度 {sim * 100:.1f}%")

    def _show_preview(self, img_bgr, face):
        h, w = img_bgr.shape[:2]
        scale = 280 / max(h, w)
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        x, y, fw, fh = face.bbox
        cv2.rectangle(
            small,
            (int(x * scale), int(y * scale)),
            (int((x + fw) * scale), int((y + fh) * scale)),
            (0, 255, 0),
            2,
        )
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        self.preview_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview_label.config(image=self.preview_photo, width=rgb.shape[1], height=rgb.shape[0])

    def delete(self):
        if not self.registration:
            messagebox.showinfo("提示", "当前没有注册")
            return
        if messagebox.askyesno("删除确认", f"确定删除注册 {self.registration['name']} 吗？"):
            self.store.clear()
            self.registration = None
            self._refresh_registration_label()
            self.result_label.config(text="")
            self.preview_label.config(image="", text="（预览）")
            self.preview_photo = None

    def run(self):
        self.root.mainloop()
