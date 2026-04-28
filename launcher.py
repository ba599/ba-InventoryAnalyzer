"""Lightweight launcher that shows a splash screen while heavy modules load."""

import sys
import threading
import tkinter as tk
from tkinter import ttk


def _show_splash_and_launch():
    root = tk.Tk()
    root.title("BA Inventory Analyzer")
    root.overrideredirect(True)  # no title bar
    root.attributes("-topmost", True)

    # Center on screen
    w, h = 360, 120
    sx = root.winfo_screenwidth() // 2 - w // 2
    sy = root.winfo_screenheight() // 2 - h // 2
    root.geometry(f"{w}x{h}+{sx}+{sy}")
    root.configure(bg="#2b2b2b")

    title = tk.Label(
        root,
        text="BA Inventory Analyzer",
        font=("Segoe UI", 14, "bold"),
        fg="#ffffff",
        bg="#2b2b2b",
    )
    title.pack(pady=(18, 4))

    status = tk.Label(
        root,
        text="로딩 중...",
        font=("Segoe UI", 9),
        fg="#aaaaaa",
        bg="#2b2b2b",
    )
    status.pack()

    style = ttk.Style()
    style.theme_use("default")
    style.configure(
        "Splash.Horizontal.TProgressbar",
        troughcolor="#3c3c3c",
        background="#4fc3f7",
        thickness=8,
    )

    bar = ttk.Progressbar(
        root,
        style="Splash.Horizontal.TProgressbar",
        orient="horizontal",
        length=300,
        mode="determinate",
        maximum=100,
    )
    bar.pack(pady=(10, 0))

    load_error = None

    def _load_modules():
        """Import heavy modules in background thread and update progress."""
        nonlocal load_error
        steps = [
            ("numpy 로딩...", lambda: __import__("numpy")),
            ("OpenCV 로딩...", lambda: __import__("cv2")),
            ("PySide6 로딩...", lambda: __import__("PySide6.QtWidgets")),
            ("ONNX Runtime 로딩...", lambda: __import__("onnxruntime")),
            ("앱 초기화...", lambda: __import__("src.desktop.app")),
        ]
        try:
            for i, (msg, importer) in enumerate(steps):
                root.after(0, lambda m=msg: status.configure(text=m))
                root.after(0, lambda v=(i * 100) // len(steps): bar.configure(value=v))
                importer()
            root.after(0, lambda: bar.configure(value=100))
            root.after(0, lambda: status.configure(text="시작!"))
            root.after(200, root.destroy)
        except Exception as e:
            load_error = e
            root.after(0, root.destroy)

    loader = threading.Thread(target=_load_modules, daemon=True)
    loader.start()

    root.mainloop()
    loader.join()

    if load_error:
        print(f"Failed to load: {load_error}", file=sys.stderr)
        sys.exit(1)


def main():
    _show_splash_and_launch()

    from src.desktop.app import run
    run()


if __name__ == "__main__":
    main()
