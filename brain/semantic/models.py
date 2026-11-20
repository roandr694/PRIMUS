"""Vision models of the semantic memory, sized for the Jetson Orin Nano (docs/home_autonomy_architecture.md §4.4).

    det = Detector()            # OWL-ViT B/32 (the model NanoOWL runs in TensorRT on Jetson), open vocabulary
    boxes = det(rgb, queries)   # [(x0, y0, x1, y1, score, query_index)]
    emb = Embedder()            # CLIP ViT-B/32: image crops and text -> unit vectors in one space
    emb.images([crop, ...]), emb.texts(["sofa", ...])

Here they run with PyTorch (transformers) on the desktop GPU or CPU; on the robot the same checkpoints are
exported to TensorRT (fp16). Models load lazily on first use and can be unloaded (`unload()`) to free memory for
the LLM ("service mode" vs "mapping mode", §8).
"""
import numpy as np

HOME_CLASSES = [
    "sofa", "armchair", "chair", "stool", "bed", "dining table", "coffee table", "side table", "desk", "television",
    "bookshelf", "wardrobe", "refrigerator", "oven", "stove", "kitchen sink", "bathroom sink", "toilet", "bathtub",
    "shower", "washing machine", "dishwasher", "potted plant", "floor lamp", "kitchen island", "bench",
    "nightstand", "sideboard", "sun lounger", "office chair", "kitchen cabinet", "kitchen counter", "person",
]


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class Detector:
    def __init__(self, name="google/owlvit-base-patch32", threshold=0.15, device=None):
        self.name, self.threshold, self.device = name, threshold, device
        self.model = None
        self._text_cache = {}

    def load(self):
        if self.model is None:
            import torch
            from transformers import OwlViTForObjectDetection, OwlViTProcessor
            self.device = self.device or _device()
            self.proc = OwlViTProcessor.from_pretrained(self.name)
            self.model = OwlViTForObjectDetection.from_pretrained(self.name).to(self.device).eval()
            if self.device == "cuda":
                self.model = self.model.half()
            self.torch = torch
        return self

    def unload(self):
        self.model = None
        self._text_cache = {}
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def __call__(self, rgb, queries=HOME_CLASSES, nms_iou=0.5):
        self.load()
        torch = self.torch
        H, W = rgb.shape[:2]
        prompts = [f"a photo of a {q}" for q in queries]
        inputs = self.proc(text=[prompts], images=rgb, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            out = self.model(**inputs)
        logits = out.logits[0].float().sigmoid().cpu().numpy()            # [n_boxes, n_queries]
        boxes = out.pred_boxes[0].float().cpu().numpy()                     # cx, cy, w, h normalised (padded square)
        S = max(H, W)                                                       # the processor pads to a square
        score = logits.max(1)
        cls = logits.argmax(1)
        keep = score > self.threshold
        res = []
        for (cx, cy, w, h), s, c in zip(boxes[keep], score[keep], cls[keep]):
            x0, y0, x1, y1 = (cx - w / 2) * S, (cy - h / 2) * S, (cx + w / 2) * S, (cy + h / 2) * S
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 - x0 > 4 and y1 - y0 > 4:
                res.append((x0, y0, x1, y1, float(s), int(c)))
        return nms(res, nms_iou)


def nms(dets, iou=0.5):
    dets = sorted(dets, key=lambda d: -d[4])
    out = []
    for d in dets:
        if all(_iou(d, o) < iou or d[5] != o[5] for o in out):
            out.append(d)
    return out


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class Embedder:
    def __init__(self, name="openai/clip-vit-base-patch32", device=None):
        self.name, self.device, self.model = name, device, None
        self._texts = {}

    def load(self):
        if self.model is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            self.device = self.device or _device()
            self.proc = CLIPProcessor.from_pretrained(self.name)
            self.model = CLIPModel.from_pretrained(self.name).to(self.device).eval()
            self.torch = torch
        return self

    def unload(self):
        self.model = None

    def images(self, crops):
        if not crops:
            return np.zeros((0, 512), np.float32)
        self.load()
        torch = self.torch
        inputs = self.proc(images=list(crops), return_tensors="pt").to(self.device)
        with torch.no_grad():
            f = self.model.get_image_features(**inputs)
        f = getattr(f, "pooler_output", f)
        f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()

    def texts(self, texts):
        self.load()
        miss = [t for t in texts if t not in self._texts]
        if miss:
            torch = self.torch
            inputs = self.proc(text=[f"a photo of a {t}" for t in miss], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                f = self.model.get_text_features(**inputs)
            f = getattr(f, "pooler_output", f)
            f = (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()
            for t, v in zip(miss, f):
                self._texts[t] = v
        return np.stack([self._texts[t] for t in texts])
