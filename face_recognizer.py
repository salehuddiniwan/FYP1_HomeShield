"""
Face recognition using InsightFace.

Provides:
  * Face detection + 512-d ArcFace embedding
  * Age and gender estimation (for category hints)
  * Registry matching via cosine similarity

If InsightFace isn't installed, the module still imports cleanly and
`FaceRecognizer.is_enabled()` returns False — all callers must check.
"""
import threading
import numpy as np

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


class FaceRecognizer:
    """Thread-safe face recognizer with an in-memory registry."""

    # Cosine similarity threshold for "this is person X".
    # InsightFace ArcFace: 0.40 is permissive, 0.50 is strict.
    MATCH_THRESHOLD = 0.42

    def __init__(self, use_gpu=True):
        self._app      = None
        self._registry = []          # list of {id, name, category, embedding (np.ndarray)}
        self._lock     = threading.Lock()

        if not INSIGHTFACE_AVAILABLE:
            print("[WARN] insightface not installed — face recognition disabled")
            print("       install with: pip install insightface onnxruntime-gpu")
            return

        try:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if use_gpu else ["CPUExecutionProvider"]
            )
            self._app = FaceAnalysis(name="buffalo_l", providers=providers)
            self._app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
            print("[INFO] InsightFace ready (detection + ArcFace + age + gender)")
        except Exception as e:
            print(f"[WARN] InsightFace init failed: {e}")
            self._app = None

    # ── status ───────────────────────────────────────────────
    def is_enabled(self):
        return self._app is not None

    def has_registered_persons(self):
        with self._lock:
            return len(self._registry) > 0

    def registry_size(self):
        with self._lock:
            return len(self._registry)

    # ── registry management ──────────────────────────────────
    def set_registry(self, persons):
        """Replace registry. Each person = {person_id, name, category, embedding}."""
        new_reg = []
        for p in persons:
            emb = p.get("embedding")
            if emb is None:
                continue
            if isinstance(emb, (bytes, bytearray)):
                emb = np.frombuffer(emb, dtype=np.float32).copy()
            elif isinstance(emb, list):
                emb = np.asarray(emb, dtype=np.float32)
            if not isinstance(emb, np.ndarray) or emb.size == 0:
                continue
            # Ensure unit-normalized for cosine-as-dot-product
            n = np.linalg.norm(emb)
            if n > 1e-6:
                emb = emb / n
            new_reg.append({
                "id":        p.get("person_id", p.get("id")),
                "name":      p.get("name", "?"),
                "category":  p.get("category", "adult"),
                "embedding": emb.astype(np.float32),
            })
        with self._lock:
            self._registry = new_reg

    # ── inference ────────────────────────────────────────────
    def analyze_crop(self, crop_bgr):
        """
        Run face detection + embedding on a cropped image.
        Returns the best face dict or None.
        """
        if self._app is None or crop_bgr is None or crop_bgr.size == 0:
            return None
        try:
            faces = self._app.get(crop_bgr)
        except Exception:
            return None
        if not faces:
            return None

        # Pick the largest face (biggest bbox area)
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            return None

        age    = int(face.age)    if getattr(face, "age", None)    is not None else None
        gender = int(face.gender) if getattr(face, "gender", None) is not None else None

        return {
            "embedding": np.asarray(emb, dtype=np.float32),
            "age":       age,
            "gender":    gender,
            "det_score": float(getattr(face, "det_score", 0.0)),
            "face_bbox": tuple(int(v) for v in face.bbox),  # (x1,y1,x2,y2) in crop
        }

    def analyze_person_bbox(self, frame, bbox, pad=12):
        """
        Analyze the face inside a person bounding box.
        Returns None if no face detected or face rec disabled.
        """
        if self._app is None:
            return None
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return self.analyze_crop(frame[y1:y2, x1:x2])

    # ── matching ─────────────────────────────────────────────
    def match(self, embedding):
        """
        Match a normalized embedding against the registry.
        Returns (person_dict, similarity) — person_dict is None if no match.
        """
        if embedding is None:
            return None, 0.0

        with self._lock:
            if not self._registry:
                return None, 0.0
            mat = np.stack([p["embedding"] for p in self._registry])   # (N, 512)
            sims = mat @ embedding                                     # both unit-norm → cosine
            idx  = int(np.argmax(sims))
            best = float(sims[idx])
            if best >= self.MATCH_THRESHOLD:
                return self._registry[idx], best
            return None, best

    @staticmethod
    def age_to_category(age):
        """Map an InsightFace age estimate to our three-way category."""
        if age is None:
            return None
        if age < 14:
            return "child"
        if age >= 58:
            return "elderly"
        return "adult"
