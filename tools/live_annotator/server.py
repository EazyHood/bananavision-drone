"""Live detection + correction tool for BananaVision.

Upload an aerial image; the trained keypoint model places a box + crown dot on
each plant. You drag/resize/delete/add them, then save. Saved corrections become
YOLO-pose labels (+ the image) under ./corrections, ready to retrain on.

Pure standard library (http.server) + ultralytics. Run:
    python server.py [port]   (default 8000)
"""
import base64
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from PIL import Image
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
CORR = os.path.join(HERE, "corrections")
os.makedirs(CORR, exist_ok=True)

MODEL_PATH = os.environ.get(
    "CROWN_MODEL", os.path.join(HERE, "..", "..", "models", "banana_crown_pose.pt")
)
DEVICE = os.environ.get("ANNOTATOR_DEVICE", "cpu")  # cpu = do not fight GPU training
print(f"Loading model {MODEL_PATH} on {DEVICE} ...", flush=True)
model = YOLO(MODEL_PATH)
DEFAULT_CONF = 0.40


def _decode_image(data_url: str) -> Image.Image:
    raw = base64.b64decode(data_url.split(",")[-1])
    return Image.open(io.BytesIO(raw)).convert("RGB")


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(json.dumps(obj).encode(), "application/json", code)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(f.read(), "text/html; charset=utf-8")
        elif path == "/sample":
            import glob
            import random
            pics = sorted(glob.glob(os.path.join(HERE, "samples", "*.jpg")))
            if not pics:
                return self._json({"error": "no samples"}, 404)
            with open(random.choice(pics), "rb") as f:
                self._send(f.read(), "image/jpeg")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            return self._json({"error": f"bad json: {e}"}, 400)

        if path == "/detect":
            try:
                img = _decode_image(data["image"])
            except Exception as e:
                return self._json({"error": f"bad image: {e}"}, 400)
            W, H = img.size
            conf = float(data.get("conf", DEFAULT_CONF))
            r = model.predict(img, conf=conf, imgsz=640, device=DEVICE, verbose=False)[0]
            boxes = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
            kpts = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else []
            dets = []
            for i, b in enumerate(boxes):
                if i < len(kpts) and len(kpts[i]) > 0:
                    crown = [float(kpts[i][0][0]), float(kpts[i][0][1])]
                else:
                    crown = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
                dets.append({"box": [float(v) for v in b], "crown": crown})
            return self._json({"w": W, "h": H, "detections": dets})

        if path == "/save":
            W, H = float(data["w"]), float(data["h"])
            dets = data.get("detections", [])
            name = os.path.splitext(os.path.basename(data.get("name", "corrected")))[0]
            name = "".join(c for c in name if c.isalnum() or c in "-_") or "corrected"
            lines = []
            for d in dets:
                x1, y1, x2, y2 = d["box"]
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                bw, bh = abs(x2 - x1) / W, abs(y2 - y1) / H
                kx, ky = d["crown"][0] / W, d["crown"][1] / H
                clamp = lambda v: max(0.0, min(1.0, v))
                lines.append(
                    f"0 {clamp(cx):.6f} {clamp(cy):.6f} {clamp(bw):.6f} {clamp(bh):.6f} "
                    f"{clamp(kx):.6f} {clamp(ky):.6f} 2"
                )
            with open(os.path.join(CORR, name + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            if data.get("image"):
                try:
                    _decode_image(data["image"]).save(os.path.join(CORR, name + ".jpg"), quality=92)
                except Exception:
                    pass
            return self._json({"saved": name, "count": len(dets), "dir": CORR})

        self._json({"error": "not found"}, 404)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Live annotator ready -> http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
