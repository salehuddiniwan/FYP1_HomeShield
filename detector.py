"""
HomeShield Detector — GPU-accelerated YOLOv8-Pose (detection + keypoints in one pass).

Replaces MediaPipe entirely.  YOLOv8-pose outputs 17 COCO keypoints on the GPU,
so there is no CPU pose step and no thread pool needed for pose estimation.

COCO-17 keypoint indices used throughout:
  0  nose          5  l_shoulder    6  r_shoulder
  7  l_elbow       8  r_elbow       9  l_wrist    10  r_wrist
  11 l_hip        12  r_hip        13  l_knee     14  r_knee
  15 l_ankle      16  r_ankle

Optimisations over previous version:
  * Single GPU→CPU transfer per result (boxes + keypoints) instead of
    one `.cpu().numpy()` per keypoint set.  On batched multi-camera
    inference this alone halves post-processing time.
  * Per-camera `PersonTracker` — fixes the cross-camera pid collision
    the shared tracker caused (a person on cam-2 could swap identities
    with one on cam-1 if their centroids happened to align).  The
    `FallDetector` / `FaceAgeEstimator` now key state on
    `(camera_id, pid)` so per-camera state is fully isolated.
  * Vectorised distance matrix in `PersonTracker.update` using numpy
    broadcasting (was an O(N·M) Python loop with `math.dist`).
  * O(1) pid→keypoint lookup via centroid hash (was an O(N·M) nested
    `min` over every tracked object each frame).
  * `draw_skeleton` precomputes the 17 pixel positions once; the
    previous version recomputed each joint up to three times (two bone
    endpoints + one circle call).
"""

import cv2
import numpy as np
import time
import math
import threading
from collections import deque
from config import Config

# ── Optional torch (for device selection only) ────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed")

OPENCV_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, "cuda") else False


# ── Device resolution ─────────────────────────────────────────────────────────

def _resolve_device() -> str:
    cfg = Config.GPU_DEVICE.strip().lower()
    if cfg != "auto":
        return cfg
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"[GPU] CUDA: {props.name}  VRAM: {props.total_memory/1024**3:.1f} GB")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[GPU] Using Apple MPS")
            return "mps"
    print("[GPU] No GPU — falling back to CPU")
    return "cpu"


DEVICE   = _resolve_device()
USE_FP16 = Config.USE_FP16 and DEVICE not in ("cpu",)


# ── Lightweight landmark wrapper ──────────────────────────────────────────────

class _KP:
    """Wraps a single COCO keypoint (x, y, conf) — mimics the .x/.y/.visibility API."""
    __slots__ = ("x", "y", "visibility")

    def __init__(self, x: float, y: float, conf: float):
        self.x          = x           # normalised [0, 1]
        self.y          = y           # normalised [0, 1]
        self.visibility = conf        # confidence [0, 1]


def _build_landmarks(kp_array, frame_w: int, frame_h: int):
    """
    Convert a (17, 3) numpy array (pixel x, pixel y, conf) to a list of 17
    _KP objects with normalised coordinates.

    YOLOv8-pose returns (0.0, 0.0) pixel coordinates for keypoints that it
    could not detect (occluded, out-of-frame, etc.).  The confidence score
    on these "ghost" keypoints is sometimes still above our visibility
    threshold, which would cause the skeleton drawer to draw a line from
    the top-left corner all the way to the person.  Zero out confidence
    whenever the keypoint is at the pixel origin — it cannot legitimately
    be there and it must be ignored by both the drawer and fall detector.
    """
    inv_w = 1.0 / frame_w
    inv_h = 1.0 / frame_h
    out = []
    for row in kp_array:
        x_px = float(row[0])
        y_px = float(row[1])
        conf = float(row[2])
        # Keypoint at pixel origin ⇒ undetected by YOLO
        if x_px < 1.0 and y_px < 1.0:
            conf = 0.0
        out.append(_KP(x_px * inv_w, y_px * inv_h, conf))
    return out


# ── COCO-17 keypoint indices ──────────────────────────────────────────────────

class KP:
    NOSE        = 0
    L_EYE, R_EYE           = 1, 2
    L_EAR, R_EAR           = 3, 4
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW,    R_ELBOW    = 7, 8
    L_WRIST,    R_WRIST    = 9, 10
    L_HIP,      R_HIP      = 11, 12
    L_KNEE,     R_KNEE     = 13, 14
    L_ANKLE,    R_ANKLE    = 15, 16


# ── Person tracker ────────────────────────────────────────────────────────────

class PersonTracker:
    """Centroid-based tracker. One instance per camera."""

    _MAX_MATCH_DIST = 150.0

    def __init__(self, max_disappeared: int = 30):
        self.next_id = 0
        self.objects: dict = {}
        self.disappeared: dict = {}
        self.histories: dict = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid, bbox):
        pid = self.next_id
        self.objects[pid]      = {"centroid": centroid, "bbox": bbox}
        self.disappeared[pid]  = 0
        self.histories[pid]    = deque(maxlen=60)
        self.histories[pid].append({"centroid": centroid, "bbox": bbox, "time": time.time()})
        self.next_id += 1

    def deregister(self, pid):
        self.objects.pop(pid, None)
        self.disappeared.pop(pid, None)
        self.histories.pop(pid, None)

    def update(self, detections):
        # No detections — bump all disappeared counters
        if not detections:
            for pid in list(self.disappeared):
                self.disappeared[pid] += 1
                if self.disappeared[pid] > self.max_disappeared:
                    self.deregister(pid)
            return self.objects

        # Build centroids and bboxes from detections
        input_centroids = []
        input_bboxes    = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            input_centroids.append(((x1 + x2) // 2, (y1 + y2) // 2))
            input_bboxes.append(det["bbox"])

        # First frame — register everything
        if not self.objects:
            for c, b in zip(input_centroids, input_bboxes):
                self.register(c, b)
            return self.objects

        obj_ids       = list(self.objects)
        obj_centroids = np.array(
            [self.objects[p]["centroid"] for p in obj_ids], dtype=np.float32
        )
        inp_centroids = np.array(input_centroids, dtype=np.float32)

        # Vectorised pairwise distance matrix (O,I)
        diff  = obj_centroids[:, None, :] - inp_centroids[None, :, :]
        dists = np.sqrt((diff * diff).sum(axis=2))

        # Greedy assignment sorted by each row's min distance
        rows = dists.min(axis=1).argsort()
        cols = dists.argmin(axis=1)[rows]
        used_r, used_c = set(), set()
        now = time.time()

        for r, c in zip(rows, cols):
            if r in used_r or c in used_c or dists[r, c] > self._MAX_MATCH_DIST:
                continue
            pid = obj_ids[r]
            self.objects[pid] = {
                "centroid": input_centroids[c],
                "bbox":     input_bboxes[c],
            }
            self.disappeared[pid] = 0
            self.histories[pid].append({
                "centroid": input_centroids[c],
                "bbox":     input_bboxes[c],
                "time":     now,
            })
            used_r.add(r)
            used_c.add(c)

        # Unmatched existing → disappear; unmatched new → register
        for r in range(len(obj_ids)):
            if r in used_r:
                continue
            pid = obj_ids[r]
            self.disappeared[pid] += 1
            if self.disappeared[pid] > self.max_disappeared:
                self.deregister(pid)

        for c in range(len(input_centroids)):
            if c not in used_c:
                self.register(input_centroids[c], input_bboxes[c])

        return self.objects


# ── Fall detector ─────────────────────────────────────────────────────────────

class FallDetector:
    """Rule-based fall detection using COCO-17 keypoints. Key: (camera_id, pid)."""

    VIS_THRESH = 0.4

    def __init__(self):
        self.pose_history:      dict = {}
        self.fall_states:       dict = {}
        self.inactivity_timers: dict = {}

    def _mid(self, lm, a, b):
        return ((lm[a].x + lm[b].x) * 0.5, (lm[a].y + lm[b].y) * 0.5)

    def _vis(self, lm, *idx):
        t = self.VIS_THRESH
        return all(lm[i].visibility >= t for i in idx)

    def _body_angle(self, lm):
        """Angle of torso from vertical (0=upright, 90=horizontal)."""
        if not self._vis(lm, KP.L_SHOULDER, KP.R_SHOULDER, KP.L_HIP, KP.R_HIP):
            return 0.0
        ms = self._mid(lm, KP.L_SHOULDER, KP.R_SHOULDER)
        mh = self._mid(lm, KP.L_HIP, KP.R_HIP)
        return abs(math.degrees(math.atan2(ms[0] - mh[0], -(ms[1] - mh[1]))))

    def _body_height_ratio(self, lm):
        """Vertical-to-horizontal span ratio (<1.2 = lying flat)."""
        idx = (KP.NOSE, KP.L_SHOULDER, KP.R_SHOULDER,
               KP.L_HIP,  KP.R_HIP,
               KP.L_ANKLE, KP.R_ANKLE)
        t = self.VIS_THRESH
        vis = [i for i in idx if lm[i].visibility >= t]
        if len(vis) < 3:
            return 2.0
        xs = [lm[i].x for i in vis]
        ys = [lm[i].y for i in vis]
        return (max(ys) - min(ys)) / (max(xs) - min(xs) + 1e-6)

    def _hip_velocity(self, pkey):
        hist = self.pose_history.get(pkey)
        if not hist or len(hist) < 3:
            return 0.0
        # deque supports O(1) indexing from either end
        latest = hist[-1]
        prev2  = hist[-3]
        dt = latest["time"] - prev2["time"]
        return (latest["hip_y"] - prev2["hip_y"]) / dt if dt > 0.01 else 0.0

    def evict(self, pkey):
        self.pose_history.pop(pkey, None)
        self.fall_states.pop(pkey, None)
        self.inactivity_timers.pop(pkey, None)

    def analyze(self, pkey, lm, frame_height):
        now = time.time()
        if pkey not in self.pose_history:
            self.pose_history[pkey]      = deque(maxlen=30)
            self.fall_states[pkey]       = {"fallen": False, "fall_time": 0}
            self.inactivity_timers[pkey] = now

        angle = self._body_angle(lm)
        ratio = self._body_height_ratio(lm)
        hip_y = (self._mid(lm, KP.L_HIP, KP.R_HIP)[1]
                 if self._vis(lm, KP.L_HIP, KP.R_HIP) else 0.5)

        hist = self.pose_history[pkey]
        hist.append({"angle": angle, "ratio": ratio, "hip_y": hip_y, "time": now})
        hip_vel = self._hip_velocity(pkey)

        if len(hist) >= 2:
            prev = hist[-2]
            if abs(hip_y - prev["hip_y"]) + abs(angle - prev["angle"]) > 0.015:
                self.inactivity_timers[pkey] = now

        is_h  = angle > 50
        is_w  = ratio < 1.3
        rapid = hip_vel > 0.5
        state = self.fall_states[pkey]

        if not state["fallen"]:
            if is_h and is_w and rapid:
                state.update({"fallen": True, "fall_time": now})
                return "fall_detected", min(0.99, 0.6 + (angle - 50) / 80 + hip_vel * 0.15)
            if is_h and is_w and len(hist) > 5:
                # Recent upright history → sudden drop = fall
                recent = list(hist)[-6:-1]
                if any(h["angle"] < 45 for h in recent):
                    state.update({"fallen": True, "fall_time": now})
                    return "fall_detected", min(0.92, 0.5 + (angle - 50) / 100)
            if angle > 70 and ratio < 0.9:
                state.update({"fallen": True, "fall_time": now})
                return "fall_detected", min(0.88, 0.55 + (angle - 70) / 100)

        if state["fallen"] and is_h:
            elapsed = now - state["fall_time"]
            return (("lying_motionless", min(0.95, 0.7 + elapsed / 100))
                    if elapsed > 5 else ("lying_after_fall", 0.85))

        if state["fallen"] and not is_h:
            state["fallen"] = False

        inactive = now - self.inactivity_timers.get(pkey, now)
        if inactive > Config.INACTIVITY_SECONDS:
            return "inactivity", min(0.95, 0.6 + inactive / 1000)

        if angle < 20 and ratio > 2.0: return "standing", 0.9
        if angle < 35 and ratio > 1.0: return "sitting",  0.85
        if angle < 30:                  return "walking",  0.8
        return "unknown", 0.5


# ── Face / age estimator ──────────────────────────────────────────────────────

class FaceAgeEstimator:
    """
    Estimates child / adult / elderly using multiple signals with temporal
    voting, and additionally runs InsightFace recognition to identify known
    persons or flag intruders.

    Priority:
      1. InsightFace match against registry → identifies the person, uses
         their registered name + category, hardens immediately.
      2. InsightFace detects a face but no registry match → accumulates
         evidence; after enough failed matches while the registry is
         non-empty, commits as 'intruder'.
      3. No face visible → falls back to pose-based voting (as before).

    Per-pid intruder state keeps the recognition stable so we don't
    oscillate between "known" and "unknown" across frames.
    """

    _CASCADE_PATH   = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    VIS             = 0.4
    VOTE_WINDOW     = 40
    MIN_VOTES       = 8
    HARDEN_AFTER    = 25
    HARDEN_MARGIN   = 0.60
    MOTION_SAMPLES  = 20

    # Face recognition state machine
    FACE_REC_INTERVAL      = 0.5    # seconds between face rec attempts per person
    FACE_REC_MAX_ATTEMPTS  = 6      # successful face detections before deciding intruder
    FACE_REC_MATCH_VOTES   = 3      # matches-to-same-person required to commit as known

    def __init__(self, face_recognizer=None):
        self._votes:     dict = {}
        self._hardened:  dict = {}
        self._motion:    dict = {}
        # Face-rec per-pid state: pkey -> dict
        self._face_state: dict = {}
        self._lock  = threading.Lock()
        self._cascade = cv2.CascadeClassifier(self._CASCADE_PATH)
        if self._cascade.empty():
            self._cascade = None
        self._face_rec = face_recognizer
        enabled = "ON " if (face_recognizer and face_recognizer.is_enabled()) else "off"
        print(f"[INFO] FaceAgeEstimator ready (voting + face-rec: {enabled})")

    # ── public api ───────────────────────────────────────────────────────
    def classify(self, pkey, frame, bbox, landmarks=None):
        """
        Returns dict:
          {
            'category':   'child' | 'adult' | 'elderly',
            'confidence': float,
            'name':       str or None,
            'is_intruder': bool,
            'new_intruder_event': bool,   # True only on transition
          }
        """
        # 1. Face recognition path (may short-circuit everything)
        fr_result = self._run_face_rec(pkey, frame, bbox)

        # 2. If face rec has committed a KNOWN identity, use it
        if fr_result and fr_result["status"] == "known":
            return {
                "category":           fr_result["category"],
                "confidence":         0.96,
                "name":               fr_result["name"],
                "is_intruder":        False,
                "new_intruder_event": False,
            }

        # 3. If face rec has committed INTRUDER, use that with age vote for category
        is_intruder = fr_result and fr_result["status"] == "intruder"
        new_intruder_event = fr_result.get("just_committed", False) if fr_result else False

        # Sticky classification from voting
        h = self._hardened.get(pkey)
        if h is not None:
            category, confidence = h
        else:
            self._record_motion(pkey, bbox)
            signals = self._gather_signals(frame, bbox, landmarks, pkey)

            # Add InsightFace age hint as an extra signal (high weight when available)
            if fr_result and fr_result.get("age_category"):
                signals.append((fr_result["age_category"], 0.90))

            votes = self._votes.get(pkey)
            if votes is None:
                votes = deque(maxlen=self.VOTE_WINDOW)
                self._votes[pkey] = votes
            for cat, weight in signals:
                votes.append((cat, weight))

            tally = {"child": 0.0, "adult": 0.0, "elderly": 0.0}
            for cat, w in votes:
                tally[cat] += w
            total = sum(tally.values()) or 1.0
            winner = max(tally, key=tally.get)
            conf   = tally[winner] / total

            if len(votes) >= self.HARDEN_AFTER and conf >= self.HARDEN_MARGIN:
                with self._lock:
                    self._hardened[pkey] = (winner, min(0.95, conf + 0.1))
                category, confidence = self._hardened[pkey]
            elif len(votes) < self.MIN_VOTES:
                category, confidence = "adult", 0.50
            else:
                category, confidence = winner, conf

        return {
            "category":           category,
            "confidence":         confidence,
            "name":               None,
            "is_intruder":        bool(is_intruder),
            "new_intruder_event": bool(new_intruder_event),
        }

    def evict(self, pkey):
        with self._lock:
            self._votes.pop(pkey, None)
            self._hardened.pop(pkey, None)
            self._motion.pop(pkey, None)
            self._face_state.pop(pkey, None)

    # ── face recognition state machine ───────────────────────────────────
    def _run_face_rec(self, pkey, frame, bbox):
        """
        Returns dict with keys:
          status:         'pending' | 'known' | 'intruder'
          name:           matched person name (if known)
          category:       matched person category (if known)
          age_category:   hint from InsightFace age estimate (child/adult/elderly)
          just_committed: True only on the frame the decision flips
        Returns None if face rec is disabled.
        """
        fr = self._face_rec
        if fr is None or not fr.is_enabled():
            return None

        st = self._face_state.get(pkey)
        if st is None:
            st = {
                "status":          "pending",
                "name":            None,
                "category":        None,
                "attempts":        0,        # successful face detections
                "match_counts":    {},       # person_id -> count
                "no_match_count":  0,
                "last_attempt":    0.0,
                "age_category":    None,
            }
            self._face_state[pkey] = st

        # If we've already committed, just return current state
        if st["status"] in ("known", "intruder"):
            return {
                "status":         st["status"],
                "name":           st["name"],
                "category":       st["category"],
                "age_category":   st["age_category"],
                "just_committed": False,
            }

        now = time.time()
        if (now - st["last_attempt"]) < self.FACE_REC_INTERVAL:
            return {
                "status":         "pending",
                "name":           None,
                "category":       None,
                "age_category":   st["age_category"],
                "just_committed": False,
            }

        st["last_attempt"] = now
        info = fr.analyze_person_bbox(frame, bbox)
        if info is None:
            # No face found this attempt — no state change
            return {
                "status":         "pending",
                "name":           None,
                "category":       None,
                "age_category":   st["age_category"],
                "just_committed": False,
            }

        # We got a face
        st["attempts"] += 1
        # Record InsightFace age hint (tends to wobble, so take running latest)
        st["age_category"] = fr.age_to_category(info.get("age"))

        matched, sim = fr.match(info["embedding"])
        just_committed = False

        if matched is not None:
            # Same person matched multiple times → commit
            mc = st["match_counts"]
            mc[matched["id"]] = mc.get(matched["id"], 0) + 1
            if mc[matched["id"]] >= self.FACE_REC_MATCH_VOTES:
                st["status"]   = "known"
                st["name"]     = matched["name"]
                st["category"] = matched["category"]
                just_committed = True
        else:
            st["no_match_count"] += 1
            # Commit as intruder only if:
            #   * registry is non-empty (otherwise EVERYONE is a stranger)
            #   * many attempts all failed
            if (fr.has_registered_persons() and
                    st["no_match_count"] >= self.FACE_REC_MAX_ATTEMPTS):
                st["status"]   = "intruder"
                st["name"]     = None
                st["category"] = st["age_category"] or "adult"
                just_committed = True

        return {
            "status":         st["status"],
            "name":           st["name"],
            "category":       st["category"],
            "age_category":   st["age_category"],
            "just_committed": just_committed,
        }

    # ── signal collection (pose / face / motion) ─────────────────────────
    def _record_motion(self, pkey, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        m = self._motion.get(pkey)
        if m is None:
            m = deque(maxlen=self.MOTION_SAMPLES)
            self._motion[pkey] = m
        m.append((cx, cy, time.time()))

    def _gait_speed(self, pkey):
        m = self._motion.get(pkey)
        if m is None or len(m) < 8:
            return None
        samples = list(m)
        dt = samples[-1][2] - samples[0][2]
        if dt < 0.1:
            return None
        dist = 0.0
        for i in range(1, len(samples)):
            dx = samples[i][0] - samples[i - 1][0]
            dy = samples[i][1] - samples[i - 1][1]
            dist += math.sqrt(dx * dx + dy * dy)
        return dist / dt

    def _gather_signals(self, frame, bbox, lm, pkey):
        signals = []
        fh = frame.shape[0]

        pose_sig = self._pose_signal(lm) if lm is not None else None
        if pose_sig:
            signals.append(pose_sig)

        if self._cascade is not None:
            face_sig = self._haar_signal(frame, bbox)
            if face_sig:
                signals.append(face_sig)

        signals.append(self._bbox_signal(bbox, fh))

        speed = self._gait_speed(pkey)
        if speed is not None:
            if speed < 15.0:
                signals.append(("elderly", 0.15))
            elif speed > 60.0:
                signals.append(("adult", 0.40))

        return signals

    def _pose_signal(self, lm):
        try:
            v = self.VIS
            if not (lm[KP.NOSE].visibility       > v and
                    lm[KP.L_SHOULDER].visibility > v and
                    lm[KP.R_SHOULDER].visibility > v):
                return None

            nose_y   = lm[KP.NOSE].y
            mid_sh_y = (lm[KP.L_SHOULDER].y + lm[KP.R_SHOULDER].y) * 0.5
            head_h   = abs(mid_sh_y - nose_y)

            if lm[KP.L_ANKLE].visibility > v and lm[KP.R_ANKLE].visibility > v:
                body_h = abs((lm[KP.L_ANKLE].y + lm[KP.R_ANKLE].y) * 0.5 - nose_y)
            elif lm[KP.L_HIP].visibility > v and lm[KP.R_HIP].visibility > v:
                body_h = abs((lm[KP.L_HIP].y + lm[KP.R_HIP].y) * 0.5 - nose_y) * 2.0
            else:
                return None

            if body_h < 0.05:
                return None

            ratio = head_h / body_h

            if ratio > 0.22:
                return ("child", min(1.0, 0.8 + (ratio - 0.22) * 3))
            if ratio > 0.18:
                return ("child", 0.55)

            return ("adult", 0.70)
        except Exception:
            return None

    def _haar_signal(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        fh, fw = frame.shape[:2]
        crop = frame[max(0, y1 - 15):min(fh, y2 + 15),
                     max(0, x1 - 15):min(fw, x2 + 15)]
        if crop.size == 0:
            return None
        person_h = y2 - y1
        if person_h <= 0:
            return None
        gray = cv2.equalizeHist(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        min_sz = max(20, person_h // 8)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(min_sz, min_sz),
        )
        if len(faces) == 0:
            return None
        _, _, _, fh_face = max(faces, key=lambda f: f[2] * f[3])
        r = fh_face / (person_h + 1e-6)
        if r > 0.30: return ("child", min(0.9, 0.75 + (r - 0.30) * 2))
        if r > 0.24: return ("child", 0.55)
        if r < 0.18: return ("adult", 0.55)
        return None

    @staticmethod
    def _bbox_signal(bbox, frame_height):
        _, y1, _, y2 = bbox
        r = (y2 - y1) / max(1, frame_height)
        if r < 0.30: return ("child",  0.35)
        if r > 0.60: return ("adult",  0.30)
        return ("adult", 0.15)


# ── Skeleton drawing ──────────────────────────────────────────────────────────

# COCO-17 bone connections for skeleton overlay
_COCO_CONNECTIONS = (
    (KP.NOSE, KP.L_EAR), (KP.NOSE, KP.R_EAR),
    (KP.L_SHOULDER, KP.R_SHOULDER),
    (KP.L_SHOULDER, KP.L_ELBOW), (KP.L_ELBOW, KP.L_WRIST),
    (KP.R_SHOULDER, KP.R_ELBOW), (KP.R_ELBOW, KP.R_WRIST),
    (KP.L_SHOULDER, KP.L_HIP),   (KP.R_SHOULDER, KP.R_HIP),
    (KP.L_HIP,      KP.R_HIP),
    (KP.L_HIP,  KP.L_KNEE),  (KP.L_KNEE,  KP.L_ANKLE),
    (KP.R_HIP,  KP.R_KNEE),  (KP.R_KNEE,  KP.R_ANKLE),
)
_VIS_DRAW = 0.50   # raised from 0.35 — low-conf keypoints have unreliable positions
_WHITE    = (255, 255, 255)
_N_JOINTS = 17
# Max legitimate bone length as a fraction of the frame's longer side.
# A real limb (shoulder-to-hip, upper-arm, thigh) never exceeds this; anything
# longer is almost certainly a ghost keypoint pairing with a real one.
_MAX_BONE_FRAC = 0.45


def draw_skeleton(frame, lm, color, fw, fh):
    """Draw COCO-17 skeleton on frame. lm = list of 17 _KP objects."""
    # Precompute pixel positions ONCE — previous version recomputed each
    # joint up to 3 times (bone start, bone end, joint circle).
    fw_minus_1 = fw - 1
    fh_minus_1 = fh - 1
    pts = [None] * _N_JOINTS
    for i in range(_N_JOINTS):
        p = lm[i]
        if p.visibility < _VIS_DRAW:
            continue
        x = int(p.x * fw)
        y = int(p.y * fh)
        # Defensive: reject pixel-origin keypoints here too, in case they
        # slipped through (different upstream sources, model updates, etc.)
        if x <= 1 and y <= 1:
            continue
        if x < 0: x = 0
        elif x > fw_minus_1: x = fw_minus_1
        if y < 0: y = 0
        elif y > fh_minus_1: y = fh_minus_1
        pts[i] = (x, y)

    # Max plausible bone length in pixels (vs. the frame, not the person —
    # a person standing close to the camera can span most of the frame).
    max_bone_px_sq = (_MAX_BONE_FRAC * max(fw, fh)) ** 2

    for i, j in _COCO_CONNECTIONS:
        p1 = pts[i]; p2 = pts[j]
        if p1 is None or p2 is None:
            continue
        # Reject bones that are implausibly long — these are almost always
        # a real keypoint connected to a ghost one that slipped the filter.
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        if dx * dx + dy * dy > max_bone_px_sq:
            continue
        cv2.line(frame, p1, p2, _WHITE, 3, cv2.LINE_AA)

    for pt in pts:
        if pt is not None:
            cv2.circle(frame, pt, 6, _WHITE, -1)
            cv2.circle(frame, pt, 4, color, -1)


# ── Main detector ─────────────────────────────────────────────────────────────

class Detector:
    """
    Single-model GPU pipeline: YOLOv8-pose outputs person bboxes + 17 COCO
    keypoints in ONE forward pass.  No MediaPipe, no thread pool.

    State that is per-person (pose history, age cache) is keyed on
    (camera_id, pid) so trackers on different cameras never collide.
    """

    _COLORS = {
        "elderly": (0, 165, 255),
        "child":   (0, 220, 80),
        "adult":   (255, 180, 0),
    }

    _FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, face_recognizer=None):
        print(f"[GPU] Inference device : {DEVICE}")
        print(f"[GPU] FP16 half-prec   : {USE_FP16}")
        print(f"[GPU] OpenCV CUDA      : {OPENCV_CUDA}")
        print(f"[GPU] Batch size       : {Config.INFERENCE_BATCH_SIZE}")

        self.yolo = None
        if YOLO_AVAILABLE:
            # Use pose model — downloads yolov8n-pose.pt automatically on first run
            pose_model = (Config.YOLO_MODEL.replace(".pt", "-pose.pt")
                          if "-pose" not in Config.YOLO_MODEL
                          else Config.YOLO_MODEL)
            try:
                self.yolo = YOLO(pose_model)
                self.yolo.to(DEVICE)
                print(f"[INFO] YOLOv8-pose on {DEVICE} (fp16={USE_FP16}): {pose_model}")
                self._warmup()
            except Exception as e:
                print(f"[WARN] Pose model failed ({e}), trying detection model")
                try:
                    self.yolo = YOLO(Config.YOLO_MODEL)
                    self.yolo.to(DEVICE)
                    print(f"[INFO] YOLOv8 (no pose) on {DEVICE}: {Config.YOLO_MODEL}")
                    self._warmup()
                except Exception as e2:
                    print(f"[WARN] YOLO load failed: {e2}")

        # HOG fallback
        if self.yolo is None:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("[INFO] Using HOG fallback")

        self.fall_detector  = FallDetector()
        self.face_recognizer = face_recognizer
        self.age_estimator   = FaceAgeEstimator(face_recognizer=face_recognizer)
        # Per-camera trackers — camera_id -> PersonTracker
        self.trackers: dict = {}

    # Backward compatibility: some callers used to read `detector.tracker`.
    # Expose an aggregate view of all per-camera trackers.
    @property
    def tracker(self):
        class _AggregateView:
            def __init__(self, trackers):
                self._t = trackers
            @property
            def objects(self):
                out = {}
                for cid, t in self._t.items():
                    for pid, info in t.objects.items():
                        out[(cid, pid)] = info
                return out
        return _AggregateView(self.trackers)

    def _get_tracker(self, camera_id):
        t = self.trackers.get(camera_id)
        if t is None:
            t = PersonTracker()
            self.trackers[camera_id] = t
        return t

    def _warmup(self):
        dummy = np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=np.uint8)
        for _ in range(Config.WARMUP_FRAMES):
            self.yolo(dummy, imgsz=Config.YOLO_IMGSZ, device=DEVICE,
                      half=USE_FP16, verbose=False)
        print(f"[GPU] YOLO warmed up ({Config.WARMUP_FRAMES} frames)")

    # ── Detection + pose ──────────────────────────────────────────────────────

    def _infer_batch(self, frames):
        """
        Returns list of (detections, landmarks_list) per frame.
        detections     = [{"bbox": (x1,y1,x2,y2), "confidence": f}, ...]
        landmarks_list = [list-of-17-_KP or None, ...]  parallel to detections
        """
        if not frames:
            return []

        if self.yolo is not None:
            results = self.yolo(
                frames,
                imgsz=Config.YOLO_IMGSZ,
                conf=Config.YOLO_CONFIDENCE,
                classes=[0],
                device=DEVICE,
                half=USE_FP16,
                verbose=False,
                stream=False,
            )
            out = []
            fh, fw = frames[0].shape[:2]
            for r in results:
                dets, kp_list = [], []
                n = len(r.boxes) if r.boxes is not None else 0
                if n == 0:
                    out.append((dets, kp_list))
                    continue

                # Single GPU→CPU transfer for boxes (was one per box).
                xyxy = r.boxes.xyxy.cpu().numpy().astype(np.int32)   # (N, 4)
                conf = r.boxes.conf.cpu().numpy()                    # (N,)

                # Single GPU→CPU transfer for ALL keypoints (was one per person).
                has_pose = (hasattr(r, "keypoints") and
                            r.keypoints is not None and
                            r.keypoints.data is not None and
                            len(r.keypoints.data) > 0)
                kp_all = r.keypoints.data.cpu().numpy() if has_pose else None  # (N,17,3)

                for i in range(n):
                    x1, y1, x2, y2 = xyxy[i]
                    dets.append({
                        "bbox":       (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": float(conf[i]),
                    })
                    if kp_all is not None and i < len(kp_all):
                        kp_list.append(_build_landmarks(kp_all[i], fw, fh))
                    else:
                        kp_list.append(None)

                out.append((dets, kp_list))
            return out

        # HOG fallback — no keypoints
        out = []
        for frame in frames:
            boxes, weights = self.hog.detectMultiScale(
                frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
            dets = [
                {"bbox": (x, y, x + w, y + h), "confidence": float(wt)}
                for (x, y, w, h), wt in zip(boxes, weights)
            ]
            out.append((dets, [None] * len(dets)))
        return out

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(self, frame, camera_id=None, zones=None):
        return self.process_frames_batch([(frame, camera_id, zones or [])])[0]

    def process_frames_batch(self, camera_inputs):
        if not camera_inputs:
            return []

        frames     = [ci[0] for ci in camera_inputs]
        cam_ids    = [ci[1] for ci in camera_inputs]
        zones_list = [ci[2] for ci in camera_inputs]

        # Single batched GPU inference pass: detection + pose for all cameras
        infer_results = self._infer_batch(frames)

        outputs = []
        for frame, camera_id, zones, (detections, kp_list) in \
                zip(frames, cam_ids, zones_list, infer_results):

            fh, fw = frame.shape[:2]
            annotated = frame.copy()
            events    = []

            # O(1) centroid→keypoint map (was O(N²) nested `min` search)
            cen_to_kp = {}
            for d, kp in zip(detections, kp_list):
                x1, y1, x2, y2 = d["bbox"]
                cen_to_kp[((x1 + x2) // 2, (y1 + y2) // 2)] = kp

            # Per-camera tracker — no cross-camera pid collisions
            tracker    = self._get_tracker(camera_id)
            before_ids = set(tracker.objects.keys())
            tracker.update(detections)
            after_ids  = set(tracker.objects.keys())

            # Evict per-person state for pids that just disappeared
            for gone in before_ids - after_ids:
                pkey = (camera_id, gone)
                self.age_estimator.evict(pkey)
                self.fall_detector.evict(pkey)

            # Partition zones into danger vs safe (precomputed np int32)
            danger_zones = [z for z in zones if z.get("zone_type", "danger") == "danger"]
            safe_zones   = [z for z in zones if z.get("zone_type")          == "safe"]

            # Annotate each tracked person
            _colors = self._COLORS
            _font   = self._FONT
            _intruder_color = (0, 0, 255)   # bright red in BGR

            for pid, info in tracker.objects.items():
                bbox       = info["bbox"]
                x1, y1, x2, y2 = bbox
                cx, cy     = info["centroid"]
                lm         = cen_to_kp.get(info["centroid"])
                pkey       = (camera_id, pid)

                # Face-aware classification: returns dict with category,
                # name (if known), is_intruder, new_intruder_event.
                clsr = self.age_estimator.classify(pkey, frame, bbox, lm)
                category    = clsr["category"]
                confidence  = clsr["confidence"]
                person_name = clsr["name"]
                is_intruder = clsr["is_intruder"]

                # Colour: intruders override category color
                color = _intruder_color if is_intruder else _colors.get(category, (200, 200, 200))

                # Skeleton
                if lm is not None:
                    draw_skeleton(annotated, lm, color, fw, fh)

                # Fall analysis
                action, action_conf = "detected", 0.5
                if lm is not None:
                    action, action_conf = self.fall_detector.analyze(pkey, lm, fh)

                # Bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Label — identity-first
                if is_intruder:
                    label = f"INTRUDER ({category}) | {action}"
                elif person_name:
                    label = f"{person_name} ({category}) | {action}"
                else:
                    label = f"{category} | {action} ({action_conf:.0%})"
                (lw, lh), _ = cv2.getTextSize(label, _font, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - lh - 10),
                              (x1 + lw + 6, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                            _font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # ── Safe-zone filter: check BEFORE firing lying-related events ──
                in_safe_zone = False
                if safe_zones and action in ("fall_detected", "lying_motionless",
                                             "lying_after_fall", "inactivity"):
                    for sz in safe_zones:
                        spoly = sz.get("polygon_np")
                        if spoly is None:
                            spoly = sz["polygon"]
                        if self._point_in_polygon(cx, cy, spoly):
                            in_safe_zone = True
                            break

                if action in ("fall_detected", "lying_motionless", "inactivity") \
                        and not in_safe_zone:
                    events.append({
                        "event_type":      action,
                        "person_id":       pid,
                        "person_category": category,
                        "person_name":     person_name,
                        "confidence":      action_conf,
                        "bbox":            bbox,
                        "camera_id":       camera_id,
                    })

                # ── Intruder: fire once on commit ──
                if clsr.get("new_intruder_event"):
                    # Crop the person's bbox from the ORIGINAL (unannotated)
                    # frame so the saved face image doesn't have the label
                    # bar or skeleton overlay baked into it.
                    fx1, fy1 = max(0, x1), max(0, y1)
                    fx2, fy2 = min(fw, x2), min(fh, y2)
                    face_crop = None
                    if fx2 > fx1 and fy2 > fy1:
                        face_crop = frame[fy1:fy2, fx1:fx2].copy()
                    events.append({
                        "event_type":      "intruder_detected",
                        "person_id":       pid,
                        "person_category": category,
                        "person_name":     None,
                        "confidence":      0.92,
                        "bbox":            bbox,
                        "camera_id":       camera_id,
                        "face_crop":       face_crop,
                    })

                # ── Child entering a DANGER zone ──
                if category == "child" and danger_zones:
                    for zone in danger_zones:
                        poly = zone.get("polygon_np")
                        if poly is None:
                            poly = zone["polygon"]
                        if self._point_in_polygon(cx, cy, poly):
                            events.append({
                                "event_type":      "zone_entry",
                                "person_id":       pid,
                                "person_category": "child",
                                "person_name":     person_name,
                                "confidence":      0.95,
                                "bbox":            bbox,
                                "camera_id":       camera_id,
                                "zone_name":       zone["zone_name"],
                            })

            # Draw zones with different colours per type
            if zones:
                for zone in zones:
                    poly = zone.get("polygon_np")
                    if poly is None:
                        poly = np.asarray(zone["polygon"], dtype=np.int32)
                    is_safe = zone.get("zone_type") == "safe"
                    zcolor  = (80, 200, 80) if is_safe else (0, 0, 255)
                    prefix  = "SAFE" if is_safe else "ZONE"
                    cv2.polylines(annotated, [poly], True, zcolor, 2)
                    if len(poly):
                        cv2.putText(annotated, f"{prefix}: {zone['zone_name']}",
                                    (int(poly[0][0]), int(poly[0][1]) - 10),
                                    _font, 0.5, zcolor, 1)

            # HUD
            hud = (f"Persons: {len(tracker.objects)}  "
                   f"[{DEVICE.upper()}{'  FP16' if USE_FP16 else ''}  POSE]")
            cv2.putText(annotated, hud, (10, 28),
                        _font, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated, hud, (10, 28),
                        _font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            outputs.append((annotated, events))

        return outputs

    @staticmethod
    def _point_in_polygon(x, y, polygon):
        """Pure-Python ray casting — faster than cv2.pointPolygonTest for
        the small polygons (4-8 points) typical of home danger zones."""
        if isinstance(polygon, np.ndarray):
            pts = polygon.tolist()
        else:
            pts = polygon
        n = len(pts)
        if n < 3:
            return False
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
            j = i
        return inside
