"""
HomeShield Camera Manager — multi-camera RTSP streaming with threaded capture.

Optimisations over previous version:
  * `cv2.resize` is done OUTSIDE the lock so `read()` is never blocked
    by the expensive resize operation.
  * Removed the hard 30 FPS sleep that was throttling RTSP streams
    (cap.read() already blocks until a frame is ready; for webcams the
    driver throttles). A short sleep only happens on read failure.
  * Cached Config constants to local variables in the hot loop.
"""
import cv2
import threading
import time
from config import Config


class CameraStream:
    """Threaded camera capture to avoid blocking the main pipeline."""

    def __init__(self, camera_id, name, url, location=""):
        self.camera_id = camera_id
        self.name = name
        self.url = url
        self.location = location
        self.frame = None
        self.grabbed = False
        self.running = False
        self.lock = threading.Lock()
        self.cap = None
        self.fps = 0.0
        self._frame_count = 0
        self._fps_time = time.time()

    def start(self):
        src = int(self.url) if self.url.isdigit() else self.url
        self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open camera: {self.name} ({self.url})")
            return False

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()
        print(f"[INFO] Camera started: {self.name} ({self.url})")
        return True

    def _update(self):
        # Cache once to avoid repeated attribute lookups in the hot loop
        target_w = Config.FRAME_WIDTH
        target_h = Config.FRAME_HEIGHT

        while self.running:
            grabbed, frame = self.cap.read()

            if not grabbed or frame is None:
                with self.lock:
                    self.grabbed = False
                time.sleep(0.01)      # brief back-off on read failure
                continue

            # Resize OUTSIDE the lock — this is the expensive part and
            # holding the lock here would block every reader.
            resized = cv2.resize(frame, (target_w, target_h))

            with self.lock:
                self.grabbed = True
                self.frame = resized
                self._frame_count += 1
                elapsed = time.time() - self._fps_time
                if elapsed >= 1.0:
                    self.fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._fps_time = time.time()

    def read(self):
        with self.lock:
            if self.frame is not None:
                # .copy() prevents callers from seeing a frame that the
                # capture thread may overwrite mid-read.
                return True, self.frame.copy()
            return False, None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        print(f"[INFO] Camera stopped: {self.name}")

    @property
    def is_active(self):
        return self.running and self.grabbed


class CameraManager:
    """Manages multiple camera streams."""

    def __init__(self):
        self.cameras = {}    # camera_id -> CameraStream

    def add_camera(self, camera_id, name, url, location=""):
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
        stream = CameraStream(camera_id, name, url, location)
        if stream.start():
            self.cameras[camera_id] = stream
            return True
        return False

    def remove_camera(self, camera_id):
        cam = self.cameras.pop(camera_id, None)
        if cam is not None:
            cam.stop()

    def get_frame(self, camera_id):
        cam = self.cameras.get(camera_id)
        if cam is not None:
            return cam.read()
        return False, None

    def get_all_active(self):
        return {cid: cam for cid, cam in self.cameras.items() if cam.is_active}

    def get_status(self):
        return {
            cid: {
                "name":     cam.name,
                "location": cam.location,
                "active":   cam.is_active,
                "fps":      round(cam.fps, 1),
            }
            for cid, cam in self.cameras.items()
        }

    def stop_all(self):
        for cam in self.cameras.values():
            cam.stop()
        self.cameras.clear()
