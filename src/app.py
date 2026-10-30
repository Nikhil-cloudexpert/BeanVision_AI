import io, logging, threading
import cv2, numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from pipeline import ROOT, Pipeline, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("beanvision")
cfg = load_config()
app = Flask(__name__, template_folder=str(ROOT / "templates"))
app.config["MAX_CONTENT_LENGTH"] = cfg["max_upload_mb"] * 1024 * 1024
pipe, state = None, {"status": "loading", "error": None}
stats = {"images": 0, "failed": 0, "beans": 0, "class_counts": {c: 0 for c in cfg["classes"]}, "times_ms": []}
slock = threading.Lock()

def load():
    global pipe
    try:
        pipe = Pipeline(cfg); state["status"] = "ready"
    except Exception as e:
        log.exception("model load failed"); state.update(status="error", error=str(e))
threading.Thread(target=load, daemon=True).start()

def err(msg, code): return jsonify({"success": False, "error": msg}), code

@app.get("/")
def index(): return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify({"app": "up", "models": state["status"], "error": state["error"],
                    "device": cfg["device"], "offline": True})

@app.get("/api/models")
def models():
    return jsonify({"segmentation": cfg["segmentation"]["model"].split("/")[-1], "imgsz": cfg["segmentation"]["imgsz"],
                    "conf": cfg["segmentation"]["conf"], "feature_extractor": "DINOv2-base (CLS token, 768-D)",
                    "classifier": "MLP 768-512-256-7", "classes": cfg["classes"], "device": cfg["device"],
                    "reported_metrics": cfg["reported_metrics"], "status": state["status"]})

@app.errorhandler(413)
def too_big(e): return err(f"File too large (max {cfg['max_upload_mb']} MB)", 413)

@app.post("/api/predict")
def predict():
    if state["status"] != "ready": return err("Models not ready: " + (state["error"] or "still loading"), 503)
    f = request.files.get("image")
    if not f: return err("No image uploaded", 400)
    data = f.read()
    try:
        im = Image.open(io.BytesIO(data)); fmt = im.format; im.verify()
        if fmt not in ("JPEG", "PNG", "WEBP"): return err("Unsupported format (use JPG, PNG or WEBP)", 415)
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None: raise ValueError
    except Exception:
        return err("File is not a valid image", 400)
    m = max(bgr.shape[:2])
    if m > cfg["max_side_px"]:
        s = cfg["max_side_px"] / m; bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    try:
        res = pipe.predict(bgr)
    except Exception:
        log.exception("inference failed")
        with slock: stats["failed"] += 1
        return err("Inference failed; see server log", 500)
    with slock:
        stats["images"] += 1; stats["beans"] += res["summary"]["beans_detected"]
        stats["times_ms"].append(res["inference"]["duration_ms"])
        for c, n in res["class_counts"].items(): stats["class_counts"][c] += n
    return jsonify(res)  # zero detections is a valid success with beans_detected = 0

@app.get("/api/session/summary")
def summary():
    with slock: return jsonify(stats)

@app.post("/api/session/reset")
def reset():
    with slock:
        stats.update(images=0, failed=0, beans=0, times_ms=[]); stats["class_counts"] = {c: 0 for c in cfg["classes"]}
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
