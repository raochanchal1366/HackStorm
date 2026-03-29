from flask import Flask, request, jsonify
from flask_cors import CORS
import hashlib
import datetime
import os
import requests
from PIL import Image
from PIL.ExifTags import TAGS
import math

# -----------------------------------------------
# TruthLens Backend — Production Ready
# Upload this file to GitHub then deploy on Railway
# -----------------------------------------------

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Token is read from Railway environment variables
# You set this in Railway → Variables → HF_TOKEN
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# 3 backup AI models — tries each until one works
HF_MODELS = [
    "https://api-inference.huggingface.co/models/Ateeqq/ai-image-detector",
    "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector",
    # "https://api-inference.huggingface.co/models/microsoft/resnet-50",
]


# -----------------------------------------------
# HOME ROUTE — serves the HTML interface
# -----------------------------------------------
@app.route("/")
def home():
    # Serve the index.html file from the same directory
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    return jsonify({
        "status":  "TruthLens is running!",
        "version": "Production — EXIF + ELA + AI detection",
        "note": "index.html not found in same directory"
    })


# -----------------------------------------------
# ANALYZE ROUTE — receives uploaded file
# -----------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():

    if "file" not in request.files:
        return jsonify({"error": "No file received"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename  = file.filename
    filepath  = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    uploader_name = request.form.get("name",   "Anonymous")
    reason        = request.form.get("reason", "Not specified")

    file_hash       = hash_file(filepath)
    identity_record = create_identity_record(
                          uploader_name, filename, file_hash, reason)
    checks          = run_all_checks(filepath, filename)
   

    result          = calculate_verdict(checks, filename)

    # print("FILENAME=",filename)
    # print("FINAL RESULT=",result)
    # print("CHECKS=",checks)

    # try:
    #     result = calculate_verdict(checks, filename)
    # except Exception as e:
    #     print("ERROR IN VERDICT:", e)
    #     result={
    #     "label":"ERROR",
    #     "confidence": 0
    # }

    try:
        os.remove(filepath)
    except:
        pass

    return jsonify({
        "verdict":         result["label"],
        "confidence":      result["confidence"],
        "checks":          checks,
        "identity_record": identity_record,
        "file_hash":       file_hash,
        "timestamp":       identity_record["timestamp"]
    })


# -----------------------------------------------
# SHA-256 FILE FINGERPRINT
# Tamper-proof — 1 pixel change = different hash
# -----------------------------------------------
def hash_file(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# -----------------------------------------------
# IDENTITY RECORD
# Protects innocent people from false accusations
# -----------------------------------------------
def create_identity_record(uploader_name, filename, file_hash, reason):
    now       = datetime.datetime.utcnow()
    timestamp = now.isoformat() + "Z"
    name_hash = hashlib.sha256(uploader_name.encode()).hexdigest()[:16]
    return {
        "timestamp":  timestamp,
        "uploader":   uploader_name,
        "name_hash":  name_hash,
        "filename":   filename,
        "file_hash":  file_hash[:16] + "...",
        "reason":     reason,
        "record_id":  hashlib.sha256(
                          (timestamp + file_hash).encode()
                      ).hexdigest()[:12]
    }


# -----------------------------------------------
# RUN ALL CHECKS
# -----------------------------------------------
def run_all_checks(filepath, filename):
    ext    = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    checks = []

    checks.append(check_file_type(ext))
    checks.append(check_file_size(filepath))

    if ext in ["jpg", "jpeg", "png", "webp"]:
        checks.append(check_exif_metadata(filepath))
        checks.append(check_exif_camera(filepath))
        checks.append(check_exif_gps(filepath))
        checks.append(check_ela(filepath))
        checks.append(check_ai_model(filepath))

    return checks


# -----------------------------------------------
# CHECK 1: File type
# -----------------------------------------------
def check_file_type(ext):
    allowed = ["jpg", "jpeg", "png", "webp",
               "gif", "mp4", "mov", "avi", "webm"]
    return {
        "name":   "File type",
        "status": "ok" if ext in allowed else "warn",
        "detail": f".{ext.upper()} — {'supported format' if ext in allowed else 'unusual format'}"
    }


# -----------------------------------------------
# CHECK 2: File size
# -----------------------------------------------
def check_file_size(filepath):
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    return {
        "name":   "File size",
        "status": "ok" if size_mb < 50 else "warn",
        "detail": f"{size_mb:.2f} MB"
    }


# -----------------------------------------------
# CHECK 3: EXIF metadata exists?
# Real cameras always write EXIF.
# AI images have no EXIF at all.
# Note: WhatsApp/Telegram strip EXIF too.
# -----------------------------------------------
def check_exif_metadata(filepath):
    try:
        img  = Image.open(filepath)
        exif = img._getexif()
        if exif is None:
            return {
                "name":   "EXIF metadata",
                "status": "fail",
                "detail": "No EXIF found — AI images have none (note: WhatsApp also strips EXIF)"
            }
        return {
            "name":   "EXIF metadata",
            "status": "ok",
            "detail": f"EXIF present — {len(exif)} fields found"
        }
    except Exception:
        return {
            "name":   "EXIF metadata",
            "status": "warn",
            "detail": "Could not read EXIF (PNG or unsupported format)"
        }


# -----------------------------------------------
# CHECK 4: Camera make and model
# Real photos always have camera brand + model.
# AI images never have this.
# -----------------------------------------------
def check_exif_camera(filepath):
    try:
        img      = Image.open(filepath)
        exif     = img._getexif()
        if exif is None:
            return {
                "name":   "Camera model",
                "status": "fail",
                "detail": "No camera info found (messaging apps remove this)"
            }
        exif_data = {TAGS.get(t, t): v for t, v in exif.items()}
        make  = exif_data.get("Make")
        model = exif_data.get("Model")
        if make or model:
            return {
                "name":   "Camera model",
                "status": "ok",
                "detail": f"{make or ''} {model or ''}".strip()
            }
        return {
            "name":   "Camera model",
            "status": "warn",
            "detail": "EXIF exists but no camera make/model found"
        }
    except Exception:
        return {
            "name":   "Camera model",
            "status": "warn",
            "detail": "Could not check camera info"
        }


# -----------------------------------------------
# CHECK 5: GPS data
# GPS presence = real device took this photo.
# Absence is weak signal — many people turn GPS off.
# -----------------------------------------------
def check_exif_gps(filepath):
    try:
        img      = Image.open(filepath)
        exif     = img._getexif()
        if exif is None:
            return {
                "name":   "GPS data",
                "status": "warn",
                "detail": "No GPS — no EXIF at all"
            }
        exif_data = {TAGS.get(t, t): v for t, v in exif.items()}
        if exif_data.get("GPSInfo"):
            return {
                "name":   "GPS data",
                "status": "ok",
                "detail": "GPS coordinates found — real device confirmed"
            }
        return {
            "name":   "GPS data",
            "status": "warn",
            "detail": "No GPS (location may have been turned off)"
        }
    except Exception:
        return {
            "name":   "GPS data",
            "status": "warn",
            "detail": "Could not check GPS"
        }


# -----------------------------------------------
# CHECK 6: Error Level Analysis (ELA)
# Re-saves image at lower quality and compares.
# Edited/AI regions show different compression.
# -----------------------------------------------
def check_ela(filepath):
    try:
        ext = filepath.rsplit(".", 1)[-1].lower()
        if ext not in ["jpg", "jpeg"]:
            return {
                "name":   "Error Level Analysis",
                "status": "warn",
                "detail": "ELA works best on JPEG — skipped for this format"
            }

        original     = Image.open(filepath).convert("RGB")
        temp_path    = filepath + "_ela_temp.jpg"
        original.save(temp_path, "JPEG", quality=75)
        recompressed = Image.open(temp_path).convert("RGB")

        orig_px = list(original.getdata())
        comp_px = list(recompressed.getdata())

        diffs    = [sum(abs(o - c) for o, c in zip(a, b))
                    for a, b in zip(orig_px, comp_px)]
        avg      = sum(diffs) / len(diffs)
        variance = sum((d - avg) ** 2 for d in diffs) / len(diffs)
        std      = math.sqrt(variance)

        try:
            os.remove(temp_path)
        except:
            pass

        if avg > 15 and std < 5:
            return {
                "name":   "Error Level Analysis",
                "status": "fail",
                "detail": f"Uniform high error (avg={avg:.1f}) — possible AI generation"
            }
        elif std > 20:
            return {
                "name":   "Error Level Analysis",
                "status": "fail",
                "detail": f"Uneven error levels (std={std:.1f}) — possible editing"
            }
        elif avg < 2:
            return {
                "name":   "Error Level Analysis",
                "status": "warn",
                "detail": f"Very low error (avg={avg:.1f}) — heavily processed"
            }
        return {
            "name":   "Error Level Analysis",
            "status": "ok",
            "detail": f"Normal error levels (avg={avg:.1f}, std={std:.1f})"
        }

    except Exception as e:
        return {
            "name":   "Error Level Analysis",
            "status": "warn",
            "detail": f"ELA could not run: {str(e)}"
        }


# -----------------------------------------------
# CHECK 7: Hugging Face AI Model
# Tries 3 models — if one is down, tries next.
# -----------------------------------------------
def check_ai_model(filepath):

    if not HF_TOKEN:
        return {
            "name":   "AI model detection",
            "status": "warn",
            "detail": "HF_TOKEN not set in environment variables"
        }

    with open(filepath, "rb") as f:
        image_bytes = f.read()

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    for model_url in HF_MODELS:
        try:
            response = requests.post(
                model_url,
                headers=headers,
                data=image_bytes,
                timeout=30
            )

            if response.status_code in [404, 410]:
                continue

            if response.status_code == 503:
                return {
                    "name":   "AI model detection",
                    "status": "warn",
                    "detail": "AI model warming up — wait 20 seconds and retry"
                }

            if response.status_code != 200:
                continue

            result = response.json()

            if isinstance(result, dict) and "error" in result:
                continue

            if not result or not isinstance(result, list):
                continue

            scores = {item["label"].lower(): item["score"]
                      for item in result
                      if "label" in item and "score" in item}

            ai_score   = 0
            real_score = 0

            for label, score in scores.items():
                if any(w in label for w in
                       ["artificial", "fake", "generated", "ai", "synthetic"]):
                    ai_score = max(ai_score, score)
                if any(w in label for w in
                       ["real", "authentic", "natural", "human"]):
                    real_score = max(real_score, score)

            if ai_score > 0 or real_score > 0:
                ai_pct   = round(ai_score * 100, 1)
                real_pct = round(real_score * 100, 1)

                if ai_score > 0.85:
                    return {"name": "AI model detection", "status": "fail",
                            "detail": f"AI GENERATED — {ai_pct}% confidence"}
                elif ai_score > 0.60:
                    return {"name": "AI model detection", "status": "fail",
                            "detail": f"Likely AI generated — {ai_pct}% AI score"}
                elif ai_score > 0.40:
                    return {"name": "AI model detection", "status": "warn",
                            "detail": f"Uncertain — {ai_pct}% AI, {real_pct}% real"}
                else:
                    return {"name": "AI model detection", "status": "ok",
                            "detail": f"Looks real — {real_pct}% authentic score"}

            top = result[0] if result else {}
            return {
                "name":   "AI model detection",
                "status": "warn",
                "detail": f"Model result: {top.get('label','?')} ({round(top.get('score',0)*100,1)}%)"
            }

        except requests.exceptions.Timeout:
            continue
        except Exception:
            continue

    return {
        "name":   "AI model detection",
        "status": "warn",
        "detail": "All AI models unavailable — other checks still valid"
    }


# -----------------------------------------------
# VERDICT CALCULATOR
# Smart scoring — ignores weak signals like GPS
# -----------------------------------------------
# def calculate_verdict(checks):

#     exif_check = next((c for c in checks if c["name"] == "EXIF metadata"),       None)
#     cam_check  = next((c for c in checks if c["name"] == "Camera model"),         None)
#     ela_check  = next((c for c in checks if c["name"] == "Error Level Analysis"), None)
#     gps_check  = next((c for c in checks if c["name"] == "GPS data"),             None)
#     ai_check   = next((c for c in checks if c["name"] == "AI model detection"),   None)

#     exif_fail  = exif_check and exif_check["status"] == "fail"
#     cam_fail   = cam_check  and cam_check["status"]  == "fail"
#     ela_fail   = ela_check  and ela_check["status"]  == "fail"
#     gps_warn   = gps_check  and gps_check["status"]  == "warn"

#     ai_flagged = (
#         ai_check and
#         ai_check["status"] == "fail" and
#         "unavailable" not in ai_check.get("detail", "").lower() and
#         "warming"     not in ai_check.get("detail", "").lower()
#     )

#     if ai_flagged and exif_fail and cam_fail:
#         return {"label": "LIKELY FAKE", "confidence": 97}
#     elif ai_flagged and (exif_fail or cam_fail):
#         return {"label": "LIKELY FAKE", "confidence": 93}
#     elif ai_flagged:
#         return {"label": "LIKELY FAKE", "confidence": 86}
#     elif exif_fail and cam_fail and ela_fail:
#         return {"label": "LIKELY FAKE", "confidence": 91}
#     elif exif_fail and cam_fail:
#         return {"label": "LIKELY FAKE", "confidence": 82}
#     elif exif_fail and ela_fail:
#         return {"label": "LIKELY FAKE", "confidence": 79}
#     elif exif_fail:
#         return {"label": "SUSPICIOUS",  "confidence": 63}
#     elif ela_fail and gps_warn:
#         return {"label": "SUSPICIOUS",  "confidence": 58}
#     elif ela_fail:
#         return {"label": "SUSPICIOUS",  "confidence": 54}
#     elif cam_fail and gps_warn:
#         return {"label": "SUSPICIOUS",  "confidence": 48}
#     elif gps_warn and not exif_fail and not cam_fail:
#         return {"label": "LOOKS REAL",  "confidence": 74}
#     else:
#         return {"label": "LOOKS REAL",  "confidence": 88}


def calculate_verdict(checks, filename=""):
    if not checks:
        return{
            "label":"ERROR",
            "confidence": 0
        }
    
    filename = (filename or "").lower()

    exif_check = next((c for c in checks if c["name"] == "EXIF metadata"), None)
    cam_check = next((c for c in checks if c["name"] == "Camera model"), None)
    ela_check = next((c for c in checks if c["name"] == "Error Level Analysis"), None)
    ai_check = next((c for c in checks if c["name"] == "AI model detection"), None)

    exif_fail = exif_check and exif_check["status"] == "fail"
    cam_fail = cam_check and cam_check["status"] == "fail"
    ela_fail = ela_check and ela_check["status"] == "fail"

    ai_flagged = (
        ai_check and
        ai_check["status"] == "fail" and
        "unavailable" not in ai_check.get("detail", "").lower()
    )

   

    # ✅ screenshot detection
    if any (x in filename for x in["screenshot","screen","ss","whatsapp","img"]):
        return {
            "label": "SCREENSHOT / DIGITAL IMAGE",
            "confidence": 95
        }

    # ✅ whatsapp detection
    if exif_fail and cam_fail and not ela_fail and not ai_flagged:
        return {
            "label": "POSSIBLY REAL (METADATA STRIPPED)",
            "confidence": 70
        }

    # ✅ strong fake only when AI + ELA suspicious
    if ai_flagged and ela_fail:
        return {
            "label": "LIKELY FAKE",
            "confidence": 94
        }

    if ai_flagged:
        return {
            "label": "LIKELY FAKE",
            "confidence": 88
        }

    if ela_fail:
        return {
            "label": "SUSPICIOUS EDITED",
            "confidence": 75
        }
    if exif_fail and cam_fail and ela_fail:
        return{
            "label": "LIKELY AI GENERATED",
            "confidence": 85
        }

    # return {
    #     "label": "LIKELY REAL",
    #     "confidence": 90
    # }
    return{
        "label": "SUSPICIOUS (NO METADATA)",
        "confidence": 75
    }
# -----------------------------------------------
# START SERVER
# host="0.0.0.0" and PORT from env = required
# for Railway deployment
# -----------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 50)
    print(f"  TruthLens running on port {port}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=True)

    