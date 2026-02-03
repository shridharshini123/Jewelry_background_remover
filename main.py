from flask import Flask, render_template, request, send_from_directory
import os
import warnings
from PIL import Image
import rawpy
import uuid

# Import your background removal function
from app import remove_background

# ======================
# Flask App Setup
# ======================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/output"
BACKGROUND_FOLDER = "static/backgrounds"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ======================
# Pillow Configuration
# ======================
# Disable Pillow’s pixel limit and warning to keep full resolution
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

def get_background_choices():
    return sorted([
        f for f in os.listdir(BACKGROUND_FOLDER)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
def composite_with_background(foreground_path, background_path, output_path):
    fg = Image.open(foreground_path).convert("RGBA")
    bg = Image.open(background_path).convert("RGBA")
    bg = bg.resize(fg.size, Image.LANCZOS)
    composed = Image.alpha_composite(bg, fg)
    composed.save(output_path, "PNG")
    return output_path
def composite_with_color(foreground_path, hex_color, output_path):
    fg = Image.open(foreground_path).convert("RGBA")
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) + (255,)
    bg = Image.new("RGBA", fg.size, rgb)
    composed = Image.alpha_composite(bg, fg)
    composed.save(output_path, "PNG")
    return output_path
# ======================
# Routes
# ======================
@app.route("/", methods=["GET", "POST"])
def index():
    backgrounds = get_background_choices()
    original_output = None
    if request.method == "POST":
        # accept both old & new hidden field names
        original_output = request.form.get("original_output")
        if not original_output:
            original_output = request.form.get("orig_filename")
        file = request.files.get("image")

        # ----------- Upload & Background removal -----------
        if file and file.filename != "":
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

            ext = os.path.splitext(filename)[1].lower()
            if ext in [".raw", ".raf", ".nef", ".cr2", ".arw", ".dng"]:
                with rawpy.imread(upload_path) as raw:
                    rgb = raw.postprocess()
                img = Image.fromarray(rgb)
                converted = filename.rsplit(".", 1)[0] + "_conv.png"
                converted_path = os.path.join(UPLOAD_FOLDER, converted)
                img.save(converted_path)
                input_path = converted_path
            else:
                input_path = upload_path

            original_output = filename.rsplit(".", 1)[0] + "_removed.png"
            removed_path = os.path.join(OUTPUT_FOLDER, original_output)
            remove_background(input_path, removed_path)
        else:
            removed_path = os.path.join(OUTPUT_FOLDER, original_output)

       # ----------- Background Apply -----------
        selected_bg = request.form.get("background", "none")
        hex_color = request.form.get("bg_color", "")

        if selected_bg == "color" and hex_color:
            final_name = f"composited_color_{hex_color}_{original_output}"
            final_path = os.path.join(OUTPUT_FOLDER, final_name)
            composite_with_color(removed_path, hex_color, final_path)
            return render_template("index.html", output_image=final_name, backgrounds=backgrounds,
                                   orig_filename=original_output)

        if selected_bg in backgrounds:
            bg_path = os.path.join(BACKGROUND_FOLDER, selected_bg)
            final_name = f"composited_{selected_bg}_{original_output}"
            final_path = os.path.join(OUTPUT_FOLDER, final_name)
            composite_with_background(removed_path, bg_path, final_path)
            return render_template("index.html", output_image=final_name, backgrounds=backgrounds,
                                   orig_filename=original_output)

        return render_template("index.html", output_image=original_output, backgrounds=backgrounds,
                               orig_filename=original_output)

    return render_template("index.html", backgrounds=backgrounds,
                           orig_filename=None)


@app.route("/output/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


# ======================
# Run Flask (Single Thread Mode)
# ======================
if __name__ == "__main__":
    # Disable reloader to avoid “Thread-2” warnings
    app.run(debug=False, use_reloader=False)
