"""
JACKFRUIT PROBLEM
TEAM 15

PES2UG25CS051 - Akul Mittal
PES2UG25CS053 - Albin Varghese Benny
PES2UG25CS059 - Amogh Garg
PES2UG25CS059 - Anand

PROMPT: IMAGE PROCESSING
"""

"""
TASKS PER MEMBER:
Akul: Filters, Sharpening
Albin: Cartoonify, Sketch
Amogh: UI and Integration, Undo Redo Function
Anand: Flip Horizontal and Vertical, Grayscale, Invert Color
"""


import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2

img = None
original_img = None 

undo_stack = []
redo_stack = []

"""
---------------------------------------
IMAGE CONVERSIONS
---------------------------------------
"""
def cv2_to_pil(cv_img):
    if cv_img is None:
        return None
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_cv2(pil_img):
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def dodge(front, back):
    front = front.astype("float32")
    back = back.astype("float32")
    result = front / (255 - back)
    return np.clip(result * 255, 0, 255).astype('uint8')


def sketch_from_bgr(cv_bgr_image, ksize=21):
    gray = cv2.cvtColor(cv_bgr_image, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (ksize, ksize), 0)
    sketch = dodge(gray, blur)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def cartoonify_bgr(cv_bgr_image, quantization_k=8, edge_block_size=9):
    if cv_bgr_image is None:
        raise ValueError("Input image to cartoonify_bgr is None")

    img = cv_bgr_image.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    h, w = img.shape[:2]

    img_filtered = cv2.pyrMeanShiftFiltering(img, 21, 51)

    Z = img_filtered.reshape((-1, 3)).astype(np.float32)

    K = max(2, min(quantization_k, Z.shape[0]))
    try:
        _, labels, centers = cv2.kmeans(Z, K, None,
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001),
                                        10, cv2.KMEANS_RANDOM_CENTERS)
    except Exception as e:
        raise RuntimeError(f"kmeans failed: {e}")

    try:
        quant = centers[labels.flatten()].reshape(img_filtered.shape).astype(np.uint8)
    except Exception as e:
        quant = centers[labels.flatten()][:h*w].reshape((h, w, 3)).astype(np.uint8)

    gray = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)

    edges = cv2.adaptiveThreshold(gray_blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, edge_block_size, 2)

    if edges.shape[:2] != (h, w):
        try:
            edges = cv2.resize(edges, (w, h), interpolation=cv2.INTER_NEAREST)
        except Exception:
            edges = cv2.resize(edges, (w, h))

    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR).astype(np.uint8)
    if edges_bgr.shape != quant.shape:

        edges_bgr = cv2.resize(edges_bgr, (quant.shape[1], quant.shape[0]), interpolation=cv2.INTER_NEAREST)


    try:
        out = cv2.bitwise_and(quant, edges_bgr)
    except Exception as e:
        q = quant.astype(np.uint8)
        e_b = edges_bgr.astype(np.uint8)
        if q.shape == e_b.shape:
            out = cv2.bitwise_and(q, e_b)
        else:
            raise RuntimeError(f"Final bitwise_and failed due to shape mismatch: quant {q.shape} vs edges {e_b.shape}: {e}")

    return out

def filter_warm(cv_bgr):
    img = cv_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    r = np.clip(r * 1.12 + 6, 0, 255)
    b = np.clip(b * 0.95, 0, 255)

    merged = cv2.merge([b, g, r]).astype(np.uint8)
    lab = cv2.cvtColor(merged, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)
    from cv2 import createCLAHE
    clahe = createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def filter_cool(cv_bgr):
    img = cv_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    b = np.clip(b * 1.15 + 4, 0, 255)
    r = np.clip(r * 0.92, 0, 255)

    merged = cv2.merge([b, g, r]).astype(np.uint8)
    hsv = cv2.cvtColor(merged, cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv[..., 1] = np.clip(hsv[..., 1] * 0.92, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def filter_vintage(cv_bgr):
    img = cv_bgr.astype(np.float32)
    kernel = np.array([[0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]])

    sepia = cv2.transform(img, kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(sepia, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 0.85, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * 0.95, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def sharpen_cv(cv_bgr, amount=1.0, radius=1.0):
    img = cv_bgr.astype(np.float32)
    ksize = max(3, int(max(1, radius) * 2) | 1)

    blurred = cv2.GaussianBlur(img, (ksize, ksize), radius)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)

    return np.clip(sharpened, 0, 255).astype(np.uint8)

"""
---------------------------------------
IMPLEMENTATION OF UNDO/REDO USING STACKS
---------------------------------------
"""
def push_undo_state():
    if img is not None:
        undo_stack.append(img.copy())
        redo_stack.clear()


def undo():
    global img
    if undo_stack:
        redo_stack.append(img.copy())
        img = undo_stack.pop()
        update_display(img)


def redo():
    global img
    if redo_stack:
        undo_stack.append(img.copy())
        img = redo_stack.pop()
        update_display(img)


def gui():
    global tk_root, img_label, top_bar, processing_frame, image_frame, theme_buttons
    tk_root = tk.Tk()
    tk_root.title("Jackfruit Problem")
    tk_root.geometry("1200x700")

    top_bar = tk.Frame(tk_root)
    top_bar.pack(side="top", fill="x", pady=10)

    tk.Button(top_bar, text="Open Image", command=imgopen).pack(side="left", padx=5)
    tk.Button(top_bar, text="Save Image", command=imgsave).pack(side="left", padx=5)
    tk.Button(top_bar, text="Crop Image", command=open_cropper).pack(side="left", padx=5)
    tk.Button(top_bar, text="Reset", command=imgreset).pack(side="left", padx=5)
    tk.Button(top_bar, text="Undo", command=undo).pack(side="left", padx=5)
    tk.Button(top_bar, text="Redo", command=redo).pack(side="left", padx=5)
    tk.Button(top_bar, text="Exit", command=tk_root.quit).pack(side="left", padx=5)

    dark_mode_var = tk.BooleanVar(value=False)
    def toggle_dark():
        apply_theme(dark_mode_var.get())

    dark_check = tk.Checkbutton(top_bar, text="Dark Mode", variable=dark_mode_var, command=toggle_dark)
    dark_check.pack(side="right", padx=10)

    processing_frame = tk.Frame(
        tk_root,
        bd=3,
        relief="ridge",
        padx=10,
        pady=10
    )
    processing_frame.pack(side="right", fill="y", padx=20, pady=(0, 20))

    tk.Label(processing_frame, text="Image Processing").pack(pady=5)
    tk.Button(processing_frame, text="Grayscale", command=imgprocess_gray).pack(pady=5)
    tk.Button(processing_frame, text="Flip Horizontal", command=imgprocess_flip_horizontal).pack(pady=5)
    tk.Button(processing_frame, text="Flip Vertical", command=imgprocess_flip_vertical).pack(pady=5)
    tk.Button(processing_frame, text="Invert Colors", command=imgprocess_invert).pack(pady=5)
    tk.Button(processing_frame, text="Sketch Effect", command=imgprocess_sketch).pack(pady=5)
    tk.Button(processing_frame, text="Cartoonify", command=imgprocess_cartoon).pack(pady=5)

    filter_var = tk.StringVar(processing_frame)
    filter_var.set("Filters")
    filters_menu = tk.OptionMenu(
        processing_frame,
        filter_var,
        "Warm", "Cool", "Vintage", "Sharpen",
        command=lambda v: (
            imgprocess_filter_warm() if v == "Warm" else
            imgprocess_filter_cool() if v == "Cool" else
            imgprocess_filter_vintage() if v == "Vintage" else
            imgprocess_filter_sharpen()
        )
    )
    filters_menu.pack(pady=10)

    image_frame = tk.Frame(tk_root, bd=3, relief="groove", padx=10, pady=10)
    image_frame.pack(side="right", expand=True, fill="both", padx=20, pady=(0, 20))

    img_label = tk.Label(image_frame, bg="black")
    img_label.pack(expand=True, fill="both")

    theme_buttons = []

    for child in processing_frame.winfo_children():
        if isinstance(child, tk.Button) or isinstance(child, tk.Checkbutton):
            theme_buttons.append(child)

    def apply_theme(dark=False):
        # colors
        if dark:
            root_bg = "#2b2b2b"
            frame_bg = "#333333"
            text_fg = "#eeeeee"
            btn_bg = "#444444"
            btn_fg = "#ffffff"
            entry_bg = "#3b3b3b"
        else:
            root_bg = "#f0f0f0"
            frame_bg = "#ffffff"
            text_fg = "#000000"
            btn_bg = "#e0e0e0"
            btn_fg = "#000000"
            entry_bg = "#ffffff"

        # root and frames
        tk_root.configure(bg=root_bg)
        top_bar.configure(bg=root_bg)
        processing_frame.configure(bg=frame_bg)
        image_frame.configure(bg=frame_bg)

        # img_label background should contrast
        img_label.configure(bg=entry_bg)

        # labels and option menu
        for w in [processing_frame, image_frame]:
            for child in w.winfo_children():
                if isinstance(child, tk.Label):
                    child.configure(bg=frame_bg, fg=text_fg)
                if isinstance(child, tk.OptionMenu):
                    # OptionMenu is a Menu + Button; style the button part
                    try:
                        child.configure(bg=btn_bg, fg=btn_fg)
                    except Exception:
                        pass

        # style top_bar children (buttons + checkbutton)
        for child in top_bar.winfo_children():
            if isinstance(child, tk.Button) or isinstance(child, tk.Checkbutton):
                child.configure(bg=btn_bg, fg=btn_fg, activebackground=btn_bg, activeforeground=btn_fg)
            elif isinstance(child, tk.Label):
                child.configure(bg=root_bg, fg=text_fg)

        # style processing buttons
        for b in theme_buttons:
            try:
                b.configure(bg=btn_bg, fg=btn_fg, activebackground=btn_bg, activeforeground=btn_fg)
            except Exception:
                pass
    # apply initial theme (light)
    apply_theme(dark=False)

    tk_root.mainloop()


"""
---------------------------------------
IMAGE FUNCTIONS
---------------------------------------
"""
def imgopen():
    global img, original_img
    path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff")])
    if not path:
        return
    img = Image.open(path).convert("RGB")
    original_img = img.copy()

    undo_stack.clear()
    redo_stack.clear()   # COMMENT: opening new image resets history

    update_display(img)


def imgsave():
    if img is None:
        messagebox.showerror("Error", "No image to save.")
        return
    path = filedialog.asksaveasfilename(defaultextension=".jpg")
    if path:
        img.save(path)


def imgreset():
    global img
    img = original_img.copy()

    # COMMENT: Reset means new clean state
    undo_stack.clear()
    redo_stack.clear()

    update_display(img)


def update_display(pil_img):
    global imageTk

    display_width = img_label.winfo_width()
    display_height = img_label.winfo_height()

    if display_width < 50 or display_height < 50:
        display_width, display_height = 800, 600

    img_copy = pil_img.copy()
    img_copy.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)

    imageTk = ImageTk.PhotoImage(img_copy)
    img_label.config(image=imageTk)
    img_label.image = imageTk


def open_cropper():
    global img, original_img, undo_stack, redo_stack
    if img is None:
        messagebox.showinfo("No image", "Open an image first before cropping.")
        return

    pil_img = img.copy()

    win = tk.Toplevel()
    win.title("Crop Image")
    win.geometry("1000x700")

    # Toolbar
    toolbar = tk.Frame(win)
    toolbar.pack(side="top", fill="x", padx=6, pady=6)
    tk.Button(toolbar, text="Apply Crop", command=lambda: _apply()).pack(side="left", padx=4)
    tk.Button(toolbar, text="Save Crop", command=lambda: _save()).pack(side="left", padx=4)
    tk.Button(toolbar, text="Cancel", command=win.destroy).pack(side="right", padx=4)

    # Canvas area
    canvas_frame = tk.Frame(win, bd=2, relief="sunken")
    canvas_frame.pack(side="left", fill="both", expand=True, padx=8, pady=8)
    canvas = tk.Canvas(canvas_frame, cursor="cross")
    canvas.pack(fill="both", expand=True)

    # Preview pane
    side = tk.Frame(win, width=300)
    side.pack(side="right", fill="y", padx=8, pady=8)
    tk.Label(side, text="Crop Preview").pack(pady=(4,0))
    preview_label = tk.Label(side, bd=2, relief="solid")
    preview_label.pack(pady=8)
    info_label = tk.Label(side, text="Drag to select area", justify="left")
    info_label.pack(pady=8)

    # state
    state = {
        "img": pil_img,
        "photo": None,
        "canvas_image_id": None,
        "img_canvas_bbox": None,
        "sel_id": None,
        "start": None,
        "sel_box": None,
        "_preview_img": None,
    }

    def _update_canvas_image():
        img_local = state["img"]
        if img_local is None:
            canvas.delete("all")
            return
        cw = max(200, canvas.winfo_width())
        ch = max(200, canvas.winfo_height())
        iw, ih = img_local.size
        scale = min(cw/iw, ch/ih)
        display_w = int(iw * scale)
        display_h = int(ih * scale)
        resized = img_local.resize((display_w, display_h), Image.Resampling.LANCZOS)
        state["photo"] = ImageTk.PhotoImage(resized)
        canvas.delete("all")
        cid = canvas.create_image(cw//2, ch//2, image=state["photo"], anchor="center")
        state["canvas_image_id"] = cid
        bbox = canvas.bbox(cid)
        state["img_canvas_bbox"] = bbox
        if state["sel_box"]:
            _draw_selection()

    def _canvas_to_image_box(canvas_box):
        if not state["img_canvas_bbox"]:
            return None
        cx1, cy1, cx2, cy2 = state["img_canvas_bbox"]
        img_w, img_h = state["img"].size
        draw_w = cx2 - cx1
        draw_h = cy2 - cy1
        sx1 = max(canvas_box[0], cx1)
        sy1 = max(canvas_box[1], cy1)
        sx2 = min(canvas_box[2], cx2)
        sy2 = min(canvas_box[3], cy2)
        if sx2 <= sx1 or sy2 <= sy1:
            return None
        rel_x1 = (sx1 - cx1) / draw_w
        rel_y1 = (sy1 - cy1) / draw_h
        rel_x2 = (sx2 - cx1) / draw_w
        rel_y2 = (sy2 - cy1) / draw_h
        ix1 = int(rel_x1 * img_w); iy1 = int(rel_y1 * img_h)
        ix2 = int(rel_x2 * img_w); iy2 = int(rel_y2 * img_h)
        ix1 = max(0, min(ix1, img_w-1))
        iy1 = max(0, min(iy1, img_h-1))
        ix2 = max(1, min(ix2, img_w))
        iy2 = max(1, min(iy2, img_h))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        return (ix1, iy1, ix2, iy2)

    def _draw_selection():
        if not state["sel_box"]:
            return
        ix1, iy1, ix2, iy2 = state["sel_box"]
        cx1, cy1, cx2, cy2 = state["img_canvas_bbox"]
        img_w, img_h = state["img"].size
        draw_w = cx2 - cx1; draw_h = cy2 - cy1
        rx1 = cx1 + (ix1 / img_w) * draw_w
        ry1 = cy1 + (iy1 / img_h) * draw_h
        rx2 = cx1 + (ix2 / img_w) * draw_w
        ry2 = cy1 + (iy2 / img_h) * draw_h
        if state["sel_id"]:
            try: canvas.delete(state["sel_id"])
            except: pass
        state["sel_id"] = canvas.create_rectangle(rx1, ry1, rx2, ry2, outline="red", width=2)

    def _update_preview():
        if not state["sel_box"]:
            preview_label.config(image="", text="No selection")
            return
        ix1, iy1, ix2, iy2 = state["sel_box"]
        crop = state["img"].crop((ix1, iy1, ix2, iy2))
        crop_resized = crop.copy()
        crop_resized.thumbnail((260, 180), Image.Resampling.LANCZOS)
        state["_preview_img"] = ImageTk.PhotoImage(crop_resized)
        preview_label.config(image=state["_preview_img"])
        info_label.config(text=f"Selected: {ix2-ix1} x {iy2-iy1}")

    def on_press(evt):
        if state["img"] is None: return
        state["start"] = (evt.x, evt.y)
        if state["sel_id"]:
            try: canvas.delete(state["sel_id"])
            except: pass
            state["sel_id"] = None

    def on_drag(evt):
        if state["start"] is None: return
        x0, y0 = state["start"]
        x1, y1 = evt.x, evt.y
        if state["sel_id"]:
            canvas.coords(state["sel_id"], x0, y0, x1, y1)
        else:
            state["sel_id"] = canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=2)

    def on_release(evt):
        if state["start"] is None: return
        x0, y0 = state["start"]; x1, y1 = evt.x, evt.y
        box = (min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))
        img_box = _canvas_to_image_box(box)
        if img_box is None:
            messagebox.showinfo("Selection", "Selection outside image area; ignored.")
            if state["sel_id"]:
                try: canvas.delete(state["sel_id"]) 
                except: pass
                state["sel_id"] = None
            state["start"] = None
            return
        state["sel_box"] = img_box
        _update_preview()
        state["start"] = None

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    win.bind("<Configure>", lambda e: _update_canvas_image())

    def _apply():
        global img, original_img, undo_stack, redo_stack
        if not state["sel_box"]:
            messagebox.showinfo("No selection", "Draw a selection first.")
            return
        try:
            undo_stack.append(img.copy())
        except Exception:
            pass
        ix1, iy1, ix2, iy2 = state["sel_box"]
        try:
            new_img = state["img"].crop((ix1, iy1, ix2, iy2))
            img = new_img.copy()
            original_img = img.copy()
            redo_stack.clear()
            update_display(img)
            win.destroy()
        except Exception as e:
            messagebox.showerror("Crop Failed", f"Could not apply crop:\n{e}")

    def _save():
        if state["sel_box"]:
            ix1, iy1, ix2, iy2 = state["sel_box"]
            to_save = state["img"].crop((ix1, iy1, ix2, iy2))
        else:
            to_save = state["img"]
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png"),("JPEG","*.jpg")])
        if not path:
            return
        try:
            to_save.save(path)
            messagebox.showinfo("Saved", f"Saved to {path}")
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save:\n{e}")

    _update_canvas_image()


"""
---------------------------------------
IMAGE PROCESSING
---------------------------------------
"""
def imgprocess_gray():
    global img
    push_undo_state()
    img = img.convert("L").convert("RGB")
    update_display(img)

def imgprocess_flip_horizontal():
    global img
    push_undo_state()
    img = ImageOps.mirror(img)
    update_display(img)

def imgprocess_flip_vertical():
    global img
    push_undo_state()
    img = ImageOps.flip(img)
    update_display(img)

def imgprocess_invert():
    global img
    push_undo_state()
    img = ImageOps.invert(img)
    update_display(img)

def imgprocess_sketch():
    global img
    push_undo_state()
    cv = pil_to_cv2(img)
    img = cv2_to_pil(sketch_from_bgr(cv))
    update_display(img)

def imgprocess_cartoon():
    global img
    push_undo_state()
    cv = pil_to_cv2(img)
    img = cv2_to_pil(cartoonify_bgr(cv))
    update_display(img)

def imgprocess_filter_warm():
    global img
    push_undo_state()
    cv = pil_to_cv2(img)
    img = cv2_to_pil(filter_warm(cv))
    update_display(img)

def imgprocess_filter_cool():
    global img
    push_undo_state()
    cv = pil_to_cv2(img)
    img = cv2_to_pil(filter_cool(cv))
    update_display(img)

def imgprocess_filter_vintage():
    global img
    push_undo_state()
    cv = pil_to_cv2(img)
    img = cv2_to_pil(filter_vintage(cv))
    update_display(img)

def imgprocess_filter_sharpen():
    global img
    push_undo_state()
    cv = pil_to_cv2(img)
    img = cv2_to_pil(sharpen_cv(cv, amount=1.1, radius=1.2))
    update_display(img)


def main():
    gui()

if __name__ == "__main__":
    main()
