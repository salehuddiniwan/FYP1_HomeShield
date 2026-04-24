"""
HomeShield — Main Flask Application
Intelligent ML-Powered CCTV Surveillance for Elderly & Child Safety

Optimisations over previous version:
  * Zones are cached in-memory per camera (invalidated on add/delete).
    The previous implementation hit SQLite on every camera, every
    processing-loop iteration — with 4 cameras at 15 FPS that was
    ~60 queries/sec on the hot path.
  * Polygons are pre-converted to `np.int32` once at cache time so
    the detector doesn't re-convert on every frame.
  * `people_count` is now computed outside the camera loop (it was
    being multiplied by the number of cameras — a latent bug when the
    tracker was shared).
  * Lazy `Detector()` initialisation — the previous code loaded the
    YOLO model twice at startup (once at module import, once in
    `start_system()`).
  * Removed unused imports (`json`, `redirect`, `url_for`).
  * Snapshots are encoded+written on a background thread so the
    processing loop doesn't stall on disk I/O during an alert.
"""
import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime
from flask import (Flask, render_template, Response, jsonify, request,
                   send_from_directory)
from config import Config
from database import Database
from detector import Detector
from camera_manager import CameraManager
from alerter import Alerter
from face_recognizer import FaceRecognizer

# ── App init ─────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)

db              = Database()
cam_manager     = CameraManager()
alerter         = Alerter()
face_recognizer = FaceRecognizer(
    use_gpu=(Config.GPU_DEVICE.lower() in ("auto", "cuda")),
)
detector              = None    # (re)created in start_system()
_last_detector_config = None    # tracks model/imgsz/fp16/device


def _detector_config_tuple():
    return (Config.YOLO_MODEL, Config.YOLO_IMGSZ,
            Config.USE_FP16, Config.GPU_DEVICE)


def _sync_face_registry():
    """Push the current person list from DB to the face recognizer."""
    if face_recognizer.is_enabled():
        face_recognizer.set_registry(db.get_persons(include_embeddings=True))


_sync_face_registry()

os.makedirs(Config.SNAPSHOT_DIR, exist_ok=True)


# Apply any saved settings from DB into Config BEFORE detector loads
# so the correct model is used on startup
def _apply_saved_settings():
    _map = {
        "yolo_model":         ("YOLO_MODEL",                str),
        "yolo_confidence":    ("YOLO_CONFIDENCE",           float),
        "yolo_imgsz":         ("YOLO_IMGSZ",                int),
        "process_fps":        ("PROCESS_FPS",               int),
        "use_fp16":           ("USE_FP16",                  lambda v: str(v).lower() == "true"),
        "fall_threshold":     ("FALL_CONFIDENCE_THRESHOLD", float),
        "inactivity_seconds": ("INACTIVITY_SECONDS",        int),
        "alert_cooldown":     ("ALERT_COOLDOWN_SECONDS",    int),
    }
    for db_key, (cfg_attr, cast) in _map.items():
        val = db.get_setting(db_key, None)
        if val is not None:
            try:
                setattr(Config, cfg_attr, cast(val))
            except Exception:
                pass


_apply_saved_settings()
print(f"[INFO] Active model: {Config.YOLO_MODEL}")

# Global state
system_state = {
    "running":      False,
    "people_count": 0,
    "active_alerts": [],
}


# ── Zone cache ───────────────────────────────────────────────
# Avoids a SQLite round-trip per camera per frame.  Each cached
# zone has its polygon pre-converted to np.int32 so the detector
# can draw and hit-test it without reconversion.
_zones_cache: dict = {}          # camera_id -> list[zone_dict]
_zones_lock       = threading.Lock()


def _get_zones_cached(camera_id):
    zones = _zones_cache.get(camera_id)
    if zones is not None:
        return zones
    with _zones_lock:
        zones = _zones_cache.get(camera_id)
        if zones is None:
            zones = db.get_zones(camera_id=camera_id)
            for z in zones:
                z["polygon_np"] = np.asarray(z["polygon"], dtype=np.int32)
            _zones_cache[camera_id] = zones
    return zones


def _invalidate_zones(camera_id=None):
    with _zones_lock:
        if camera_id is None:
            _zones_cache.clear()
        else:
            _zones_cache.pop(camera_id, None)


# ── Snapshot writer (background) ─────────────────────────────
def _write_snapshot_async(path, frame):
    """Encode + write a JPEG snapshot off the processing loop."""
    try:
        cv2.imwrite(path, frame)
    except Exception as e:
        print(f"[WARN] snapshot write failed ({path}): {e}")


# ── Processing pipeline ──────────────────────────────────────
def processing_loop():
    """
    GPU-optimised main loop.

    All active camera frames are collected first, then sent to the detector
    as a single batch so YOLOv8 executes one GPU forward pass per tick
    instead of one per camera.
    """
    while system_state["running"]:
        frame_interval = 1.0 / max(1, Config.PROCESS_FPS)
        loop_start = time.time()

        # ── Collect frames from all active cameras ────────────────────────
        active = list(cam_manager.get_all_active().items())
        batch_inputs = []   # (frame, cid, zones)
        batch_cams   = []   # matching CameraStream objects

        for cid, cam in active:
            grabbed, frame = cam.read()
            if not grabbed or frame is None:
                continue
            zones = _get_zones_cached(cid)     # cached — no DB hit
            batch_inputs.append((frame, cid, zones))
            batch_cams.append((cid, cam))

        # ── Single batched GPU inference pass ─────────────────────────────
        batch_results = (detector.process_frames_batch(batch_inputs)
                         if batch_inputs else [])

        for (cid, cam), (annotated, events) in zip(batch_cams, batch_results):
            cam.annotated_frame = annotated

            # Handle events
            for event in events:
                etype = event["event_type"]
                if not alerter.should_alert(etype, cid):
                    continue

                ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_name = f"{etype}_{cid}_{ts}.jpg"
                snap_path = os.path.join(Config.SNAPSHOT_DIR, snap_name)

                # Fire-and-forget disk write — the processing loop does not
                # need to wait on JPEG encoding/flushing.
                threading.Thread(
                    target=_write_snapshot_async,
                    args=(snap_path, annotated.copy()),
                    daemon=True,
                ).start()

                # Forward slashes so the URL works on Windows
                snap_url_path = snap_path.replace("\\", "/")
                details       = event.get("zone_name", "")

                db.log_event(
                    event_type=etype,
                    camera_id=cid,
                    camera_name=cam.name,
                    person_category=event.get("person_category", "unknown"),
                    confidence=event.get("confidence", 0),
                    snapshot_path=snap_url_path,
                    alert_sent=True,
                    details=details,
                )

                # ── Special handling for intruder events: save a face crop
                # into the intruders directory + DB record so the Persons
                # page can show a gallery of seen intruders.
                if etype == "intruder_detected":
                    face_crop = event.get("face_crop")
                    if face_crop is not None and face_crop.size > 0:
                        i_name = f"intruder_{cid}_{ts}.jpg"
                        i_path = os.path.join(_INTRUDER_PHOTOS_DIR, i_name)
                        threading.Thread(
                            target=_write_snapshot_async,
                            args=(i_path, face_crop),
                            daemon=True,
                        ).start()
                        db.add_intruder(
                            camera_id=cid,
                            camera_name=cam.name,
                            category=event.get("person_category", "adult"),
                            photo_path=i_path,
                        )

                alerter.send_alert(
                    event_type=etype,
                    camera_name=f"{cam.name} ({cam.location})",
                    person_category=event.get("person_category", "unknown"),
                    confidence=event.get("confidence", 0),
                    snapshot_path=snap_path,
                    camera_id=cid,
                    details=details,
                )

        # Count people ONCE after the batch — sum per-camera tracker sizes.
        # (The previous version accumulated inside the camera loop, which
        # double-counted because the tracker was shared across cameras.)
        if detector is not None:
            system_state["people_count"] = sum(
                len(t.objects) for t in detector.trackers.values()
            )

        elapsed = time.time() - loop_start
        time.sleep(max(0, frame_interval - elapsed))


def start_system():
    global detector, _last_detector_config
    if system_state["running"]:
        return

    # Reload the detector when a model-critical setting has changed.
    # Keeps startup fast on an unchanged Stop/Start cycle, but picks up
    # a new YOLO model / image size / FP16 toggle from Settings.
    current = _detector_config_tuple()
    if detector is None or _last_detector_config != current:
        if detector is not None:
            print(f"[INFO] Reloading detector — model: {Config.YOLO_MODEL}")
        detector = Detector(face_recognizer=face_recognizer)
        _last_detector_config = current

    # Load cameras from DB
    cameras = db.get_cameras(active_only=True)
    if not cameras:
        # Add default camera if none exist
        for cam in Config.DEFAULT_CAMERAS:
            cid = db.add_camera(cam["name"], cam["url"], cam["location"])
            cam_manager.add_camera(cid, cam["name"], cam["url"], cam["location"])
    else:
        for cam in cameras:
            cam_manager.add_camera(
                cam["camera_id"], cam["name"], cam["url"], cam["location"]
            )

    system_state["running"] = True
    threading.Thread(target=processing_loop, daemon=True).start()
    print("[INFO] HomeShield system started")


def stop_system():
    system_state["running"] = False
    cam_manager.stop_all()
    print("[INFO] HomeShield system stopped")


# ── Video streaming ──────────────────────────────────────────
_JPEG_FEED_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
_JPEG_SNAP_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 80]


def generate_feed(camera_id):
    """MJPEG generator for a single camera."""
    while True:
        cam = cam_manager.cameras.get(camera_id)
        if cam is None:
            time.sleep(0.5)
            continue

        frame = getattr(cam, "annotated_frame", None)
        if frame is None:
            grabbed, frame = cam.read()
            if not grabbed or frame is None:
                time.sleep(0.1)
                continue

        ok, jpeg = cv2.imencode(".jpg", frame, _JPEG_FEED_PARAMS)
        if not ok:
            time.sleep(0.05)
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(1.0 / 15)   # 15 FPS stream


# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed/<int:camera_id>")
def video_feed(camera_id):
    return Response(
        generate_feed(camera_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/frame_snap/<int:camera_id>")
def frame_snap(camera_id):
    """Return a single JPEG frame for the zone-editor canvas."""
    grabbed, frame = cam_manager.get_frame(camera_id)
    if not grabbed or frame is None:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", placeholder)
    else:
        _, buf = cv2.imencode(".jpg", frame, _JPEG_SNAP_PARAMS)
    return Response(
        buf.tobytes(),
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.route("/api/status")
def api_status():
    cameras = cam_manager.get_status()
    online  = sum(1 for c in cameras.values() if c["active"])

    # Resolve the display name for the active model (…-pose variant).
    raw_model    = db.get_setting("yolo_model", Config.YOLO_MODEL)
    if raw_model.endswith("-pose.pt"):
        model_name = raw_model
    elif raw_model.endswith(".pt"):
        model_name = raw_model[:-3] + "-pose.pt"
    else:
        model_name = raw_model

    return jsonify({
        "running":        system_state["running"],
        "cameras_online": online,
        "cameras_total":  len(cameras),
        "cameras":        cameras,
        "people_count":   system_state["people_count"],
        "alerts_today":   db.get_today_alert_count(),
        "model":          model_name,
    })


@app.route("/api/events")
def api_events():
    limit = request.args.get("limit", 50, type=int)
    etype = request.args.get("type", None)
    cid   = request.args.get("camera_id", None, type=int)
    return jsonify(db.get_events(limit=limit, event_type=etype, camera_id=cid))


@app.route("/api/cameras", methods=["GET"])
def api_get_cameras():
    return jsonify(db.get_cameras())


@app.route("/api/cameras", methods=["POST"])
def api_add_camera():
    data     = request.json or {}
    name     = data.get("name", "Camera")
    url      = data.get("url", "0")
    location = data.get("location", "")
    cid      = db.add_camera(name, url, location)
    if system_state["running"]:
        cam_manager.add_camera(cid, name, url, location)
    return jsonify({"camera_id": cid, "status": "added"})


@app.route("/api/cameras/<int:camera_id>", methods=["DELETE"])
def api_delete_camera(camera_id):
    db.delete_camera(camera_id)
    cam_manager.remove_camera(camera_id)
    _invalidate_zones(camera_id)
    return jsonify({"status": "deleted"})


@app.route("/api/zones", methods=["GET"])
def api_get_zones():
    cid = request.args.get("camera_id", None, type=int)
    return jsonify(db.get_zones(camera_id=cid))


@app.route("/api/zones", methods=["POST"])
def api_add_zone():
    data      = request.json or {}
    zone_name = data.get("zone_name", "Danger Zone")
    camera_id = data.get("camera_id")
    polygon   = data.get("polygon", [])
    zone_type = data.get("zone_type", "danger")
    if zone_type not in ("danger", "safe"):
        zone_type = "danger"
    if not camera_id or not polygon:
        return jsonify({"error": "camera_id and polygon required"}), 400
    zid = db.add_zone(zone_name, camera_id, polygon, zone_type=zone_type)
    _invalidate_zones(camera_id)
    return jsonify({"zone_id": zid, "status": "added"})


@app.route("/api/zones/<int:zone_id>", methods=["DELETE"])
def api_delete_zone(zone_id):
    db.delete_zone(zone_id)
    # Zone id alone doesn't tell us which camera — just flush everything.
    _invalidate_zones()
    return jsonify({"status": "deleted"})


# ── Persons (face registry) ──────────────────────────────────
_PERSON_PHOTOS_DIR    = os.path.join(Config.SNAPSHOT_DIR, "persons")
_INTRUDER_PHOTOS_DIR  = os.path.join(Config.SNAPSHOT_DIR, "intruders")
os.makedirs(_PERSON_PHOTOS_DIR,   exist_ok=True)
os.makedirs(_INTRUDER_PHOTOS_DIR, exist_ok=True)


def _person_photo_url(person_id, photo_path):
    """Return a URL for the UI or empty string if no photo."""
    if not photo_path or not os.path.exists(photo_path):
        return ""
    return f"/api/persons/{person_id}/photo"


@app.route("/api/persons", methods=["GET"])
def api_get_persons():
    """Return registry without embeddings (which are binary blobs)."""
    raw = db.get_persons(include_embeddings=False)
    persons = []
    for p in raw:
        persons.append({
            "person_id":  p["person_id"],
            "name":       p["name"],
            "category":   p["category"],
            "created_at": p["created_at"],
            "photo_url":  _person_photo_url(p["person_id"], p.get("photo_path", "")),
        })
    return jsonify({
        "face_rec_enabled": face_recognizer.is_enabled(),
        "persons":          persons,
    })


@app.route("/api/persons/<int:person_id>/photo")
def api_person_photo(person_id):
    """Serve the face thumbnail for a registered person."""
    path = os.path.join(_PERSON_PHOTOS_DIR, f"{person_id}.jpg")
    if not os.path.exists(path):
        # Return a 1x1 transparent placeholder to avoid broken-image icons
        return Response(b"", mimetype="image/jpeg", status=404)
    return send_from_directory(_PERSON_PHOTOS_DIR, f"{person_id}.jpg",
                               max_age=0)


@app.route("/api/persons", methods=["POST"])
def api_add_person():
    """
    Register a person from the current frame of a given camera.
    Body: { name, category, camera_id }
    Saves a 160x160 face thumbnail for UI display.
    """
    if not face_recognizer.is_enabled():
        return jsonify({
            "error": "InsightFace is not installed. Run: "
                     "pip install insightface onnxruntime-gpu",
        }), 400

    data      = request.json or {}
    name      = (data.get("name") or "").strip()
    category  = (data.get("category") or "adult").strip().lower()
    camera_id = data.get("camera_id")

    if not name:
        return jsonify({"error": "name required"}), 400
    if category not in ("child", "adult", "elderly"):
        category = "adult"
    if not camera_id:
        return jsonify({"error": "camera_id required"}), 400

    grabbed, frame = cam_manager.get_frame(int(camera_id))
    if not grabbed or frame is None:
        return jsonify({"error": "No frame from that camera"}), 400

    info = face_recognizer.analyze_crop(frame)
    if info is None:
        return jsonify({
            "error": "No face detected. Face the camera directly and try again.",
        }), 400

    emb_bytes = info["embedding"].astype(np.float32).tobytes()
    pid = db.add_person(name, category, emb_bytes)

    # Save a small face thumbnail for the UI
    photo_url = ""
    face_bbox = info.get("face_bbox")
    if face_bbox is not None:
        try:
            fx1, fy1, fx2, fy2 = face_bbox
            fh, fw = frame.shape[:2]
            # Pad around the face by ~25% of its size
            pad_x = max(10, int((fx2 - fx1) * 0.25))
            pad_y = max(10, int((fy2 - fy1) * 0.25))
            fx1 = max(0, fx1 - pad_x); fy1 = max(0, fy1 - pad_y)
            fx2 = min(fw, fx2 + pad_x); fy2 = min(fh, fy2 + pad_y)
            face_crop = frame[fy1:fy2, fx1:fx2]
            if face_crop.size > 0:
                # Square crop then resize — avoids stretch for rectangular bboxes
                side = min(face_crop.shape[0], face_crop.shape[1])
                if side >= 32:
                    cy, cx = face_crop.shape[0] // 2, face_crop.shape[1] // 2
                    half = side // 2
                    square = face_crop[cy - half:cy + half, cx - half:cx + half]
                    thumb = cv2.resize(square, (200, 200))
                    photo_path = os.path.join(_PERSON_PHOTOS_DIR, f"{pid}.jpg")
                    cv2.imwrite(photo_path, thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                    db.update_person(pid, photo_path=photo_path)
                    photo_url = f"/api/persons/{pid}/photo"
        except Exception as e:
            print(f"[WARN] failed to save face thumbnail for {name}: {e}")

    _sync_face_registry()
    return jsonify({
        "person_id":     pid,
        "name":          name,
        "category":      category,
        "detected_age":  info.get("age"),
        "photo_url":     photo_url,
        "status":        "added",
    })


@app.route("/api/persons/<int:person_id>", methods=["PUT"])
def api_update_person(person_id):
    data = request.json or {}
    db.update_person(person_id, **{
        k: v for k, v in data.items() if k in ("name", "category")
    })
    _sync_face_registry()
    return jsonify({"status": "updated"})


@app.route("/api/persons/<int:person_id>", methods=["DELETE"])
def api_delete_person(person_id):
    db.delete_person(person_id)
    # Clean up the thumbnail on disk too
    photo_path = os.path.join(_PERSON_PHOTOS_DIR, f"{person_id}.jpg")
    try:
        if os.path.exists(photo_path):
            os.remove(photo_path)
    except OSError:
        pass
    _sync_face_registry()
    return jsonify({"status": "deleted"})


# ── Live face-detection preview (for the register panel) ─────
@app.route("/api/detect_face/<int:camera_id>")
def api_detect_face(camera_id):
    """
    Returns JSON describing the biggest detected face in the current frame.
    Used by the Persons-page register panel to light up a guide box so the
    user knows when they're framed correctly.

    { width, height, face: { x, y, w, h, age } | null }
    """
    grabbed, frame = cam_manager.get_frame(camera_id)
    if not grabbed or frame is None:
        return jsonify({"width": 0, "height": 0, "face": None, "error": "no_frame"})

    fh, fw = frame.shape[:2]
    resp = {"width": fw, "height": fh, "face": None}

    if not face_recognizer.is_enabled():
        resp["error"] = "face_rec_disabled"
        return jsonify(resp)

    info = face_recognizer.analyze_crop(frame)
    if info is None:
        return jsonify(resp)

    fx1, fy1, fx2, fy2 = info["face_bbox"]
    resp["face"] = {
        "x":   int(fx1),
        "y":   int(fy1),
        "w":   int(fx2 - fx1),
        "h":   int(fy2 - fy1),
        "age": info.get("age"),
    }
    return jsonify(resp)


# ── Intruders log ────────────────────────────────────────────
@app.route("/api/intruders", methods=["GET"])
def api_get_intruders():
    include_dismissed = request.args.get("include_dismissed", "").lower() in ("1", "true", "yes")
    rows = db.get_intruders(include_dismissed=include_dismissed, limit=100)
    out = []
    for r in rows:
        out.append({
            "intruder_id": r["intruder_id"],
            "camera_id":   r["camera_id"],
            "camera_name": r["camera_name"],
            "category":    r["category"],
            "detected_at": r["detected_at"],
            "dismissed":   bool(r["dismissed"]),
            "photo_url":   f"/api/intruders/{r['intruder_id']}/photo",
        })
    return jsonify(out)


@app.route("/api/intruders/<int:intruder_id>/photo")
def api_intruder_photo(intruder_id):
    path = os.path.join(_INTRUDER_PHOTOS_DIR, f"intruder_{intruder_id}.jpg")
    # Photos are written with a timestamped name; look it up from the DB.
    row = next((i for i in db.get_intruders(include_dismissed=True, limit=500)
                if i["intruder_id"] == intruder_id), None)
    if row and row["photo_path"] and os.path.exists(row["photo_path"]):
        dirpath = os.path.dirname(row["photo_path"])
        filename = os.path.basename(row["photo_path"])
        return send_from_directory(dirpath, filename, max_age=0)
    return Response(b"", mimetype="image/jpeg", status=404)


@app.route("/api/intruders/<int:intruder_id>/dismiss", methods=["POST"])
def api_dismiss_intruder(intruder_id):
    db.dismiss_intruder(intruder_id)
    return jsonify({"status": "dismissed"})


@app.route("/api/intruders/<int:intruder_id>", methods=["DELETE"])
def api_delete_intruder(intruder_id):
    row = db.delete_intruder(intruder_id)
    if row and row.get("photo_path"):
        try:
            if os.path.exists(row["photo_path"]):
                os.remove(row["photo_path"])
        except OSError:
            pass
    return jsonify({"status": "deleted"})


@app.route("/api/intruders/<int:intruder_id>/register", methods=["POST"])
def api_register_intruder(intruder_id):
    """
    Promote a seen intruder into the registered-persons list.
    Body: { name, category }
    Reads the stored face photo, runs InsightFace on it to get an
    embedding, saves a person thumbnail, removes the intruder record.
    """
    if not face_recognizer.is_enabled():
        return jsonify({"error": "Face recognition is not installed"}), 400

    data     = request.json or {}
    name     = (data.get("name") or "").strip()
    category = (data.get("category") or "adult").strip().lower()
    if not name:
        return jsonify({"error": "name required"}), 400
    if category not in ("child", "adult", "elderly"):
        category = "adult"

    # Find the intruder row
    row = next((i for i in db.get_intruders(include_dismissed=True, limit=500)
                if i["intruder_id"] == intruder_id), None)
    if row is None:
        return jsonify({"error": "Intruder not found"}), 404
    photo_path = row["photo_path"]
    if not photo_path or not os.path.exists(photo_path):
        return jsonify({"error": "Intruder photo missing"}), 400

    # Load the image and extract a face embedding
    face_img = cv2.imread(photo_path)
    if face_img is None:
        return jsonify({"error": "Could not read intruder photo"}), 400
    info = face_recognizer.analyze_crop(face_img)
    if info is None:
        return jsonify({"error": "No face found in intruder photo"}), 400

    emb_bytes = info["embedding"].astype(np.float32).tobytes()
    pid = db.add_person(name, category, emb_bytes)

    # Reuse the intruder photo as the person thumbnail (resize to 200x200 square)
    try:
        side = min(face_img.shape[0], face_img.shape[1])
        if side >= 32:
            cy, cx = face_img.shape[0] // 2, face_img.shape[1] // 2
            half = side // 2
            square = face_img[cy - half:cy + half, cx - half:cx + half]
            thumb = cv2.resize(square, (200, 200))
            person_photo = os.path.join(_PERSON_PHOTOS_DIR, f"{pid}.jpg")
            cv2.imwrite(person_photo, thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            db.update_person(pid, photo_path=person_photo)
    except Exception as e:
        print(f"[WARN] thumbnail for promoted intruder failed: {e}")

    # Remove the intruder record + file now that it's a real person
    db.delete_intruder(intruder_id)
    try:
        if os.path.exists(photo_path):
            os.remove(photo_path)
    except OSError:
        pass

    _sync_face_registry()
    return jsonify({"person_id": pid, "name": name, "category": category, "status": "registered"})


@app.route("/api/system/start", methods=["POST"])
def api_start():
    start_system()
    return jsonify({"status": "started"})


@app.route("/api/system/stop", methods=["POST"])
def api_stop():
    stop_system()
    return jsonify({"status": "stopped"})


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    return jsonify({
        "fall_threshold":     float(db.get_setting("fall_threshold",     Config.FALL_CONFIDENCE_THRESHOLD)),
        "inactivity_seconds": int  (db.get_setting("inactivity_seconds", Config.INACTIVITY_SECONDS)),
        "alert_cooldown":     int  (db.get_setting("alert_cooldown",     Config.ALERT_COOLDOWN_SECONDS)),
        "alert_phones":            db.get_setting("alert_phones",        ",".join(Config.ALERT_PHONE_NUMBERS)),
        # Model & GPU
        "yolo_model":              db.get_setting("yolo_model",          Config.YOLO_MODEL),
        "yolo_confidence":    float(db.get_setting("yolo_confidence",    Config.YOLO_CONFIDENCE)),
        "yolo_imgsz":         int  (db.get_setting("yolo_imgsz",         Config.YOLO_IMGSZ)),
        "process_fps":        int  (db.get_setting("process_fps",        Config.PROCESS_FPS)),
        "use_fp16":                db.get_setting("use_fp16", str(Config.USE_FP16)).lower() == "true",
    })


@app.route("/api/settings", methods=["POST"])
def api_update_settings():
    data = request.json or {}
    all_keys = ("fall_threshold", "inactivity_seconds", "alert_cooldown",
                "alert_phones", "yolo_model", "yolo_confidence",
                "yolo_imgsz", "process_fps", "use_fp16")
    for key in all_keys:
        if key in data:
            db.set_setting(key, str(data[key]))

    # Apply instantly to runtime config where possible
    if "fall_threshold"     in data: Config.FALL_CONFIDENCE_THRESHOLD = float(data["fall_threshold"])
    if "inactivity_seconds" in data: Config.INACTIVITY_SECONDS        = int  (data["inactivity_seconds"])
    if "alert_cooldown"     in data: Config.ALERT_COOLDOWN_SECONDS    = int  (data["alert_cooldown"])
    if "alert_phones"       in data:
        Config.ALERT_PHONE_NUMBERS = [
            p.strip() for p in data["alert_phones"].split(",") if p.strip()
        ]
    if "yolo_confidence"    in data: Config.YOLO_CONFIDENCE = float(data["yolo_confidence"])
    if "process_fps"        in data: Config.PROCESS_FPS    = int  (data["process_fps"])
    # yolo_model / yolo_imgsz / use_fp16 require a system restart
    if "yolo_model"         in data: Config.YOLO_MODEL = data["yolo_model"]
    if "yolo_imgsz"         in data: Config.YOLO_IMGSZ = int(data["yolo_imgsz"])
    if "use_fp16"           in data: Config.USE_FP16   = str(data["use_fp16"]).lower() == "true"

    return jsonify({"status": "updated"})


@app.route("/api/events/clear", methods=["POST"])
def api_clear_events():
    db.clear_events()
    system_state["people_count"] = 0
    return jsonify({"status": "cleared"})


@app.route("/snapshots/<path:filename>")
def serve_snapshot(filename):
    return send_from_directory(Config.SNAPSHOT_DIR, filename)


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("""
        ▄▄▄  ▄▄▄ ▄▄▄▄▄▄▄ ▄▄   ▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄   ▄▄ ▄▄▄ ▄▄▄▄▄▄▄ ▄▄▄     ▄▄▄▄▄▄  
        █   ██   █       █  █ █  █       █       █  █ █  █   █       █   █   █      █ 
        █   ██   █   ▄   █  █▄█  █    ▄▄▄█  ▄▄▄▄▄█  █▄█  █   █    ▄▄▄█   █   █  ▄    █
        █▄▄▄▄▄▄▄▄█  █ █  █       █   █▄▄▄█ █▄▄▄▄▄█       █   █   █▄▄▄█   █   █ █ █   █
        █        █  █▄█  █       █    ▄▄▄█▄▄▄▄▄  █       █   █    ▄▄▄█   █▄▄▄█ █▄█   █
        █   ▄▄   █       █   ▄   █   █▄▄▄ ▄▄▄▄▄█ █   ▄   █   █   █▄▄▄█       █       █
        █▄▄▄█ █▄▄█▄▄▄▄▄▄▄█▄▄█ █▄▄█▄▄▄▄▄▄▄█▄▄▄▄▄▄▄█▄▄█ █▄▄█▄▄▄█▄▄▄▄▄▄▄█▄▄▄▄▄▄▄█▄▄▄▄▄▄█ 

        ► ML-Powered CCTV Surveillance System
        ► Elderly & Child Safety Monitoring Active
        ==============================================================================
        """)

    start_system()
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=False,       # debug=False with threaded camera capture
        threaded=True,
    )
