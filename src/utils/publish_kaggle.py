import os
import subprocess
import json
import sty


def generate_slug(name):
    slug = "".join([c if c.isalnum() or c == "-" else "-" for c in name])
    return slug


data_type = "full"

if data_type == "lite":
    folder_name = "datasets_lite"
else:
    folder_name = "datasets"

title = "MindSet" + ("-lite" if data_type == "lite" else "")

print(sty.fg.red + f"Generating {title}" + sty.rs.fg)

metadata = {
    "title": title,
    "id": f"Valerio1988/{title}",
    "licenses": [{"name": "CC0-1.0"}],
}

metadata_path = os.path.join(f"{folder_name}", "dataset-metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f)

subprocess.call(
    [
        "kaggle",
        "datasets",
        "create",
        "-p",
        f"{folder_name}",
        "-r",
        "zip",  # Upload folder as a zip
    ]
)
