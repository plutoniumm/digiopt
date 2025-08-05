from super_image import EdsrModel, ImageLoader as IL
import torch.nn.functional as F
import math, os, string
from PIL import Image
import sys

model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=2)
targ = (335, 335)
steps = math.ceil(math.log2(335 / 28))


def output_images(path, out_dir, name):
    out = IL.load_image(Image.open(path).convert("L"))
    IL.save_image(out, f"{out_dir}/{name}.png")


def input_images(path, out_dir, name):
    out = IL.load_image(Image.open(path))
    for _ in range(steps):
        out = model(out)

    out = F.interpolate(out, size=targ, mode="bicubic", align_corners=False)

    IL.save_image(out, f"{out_dir}/{name}.png")


chars = {"A": string.ascii_uppercase + string.digits, "B": [str(i) for i in range(36)]}


def toChar(char):
    if char not in string.digits:
        char_int = int(char)
        if char_int >= 10:
            char_int = string.ascii_uppercase[char_int - 10]
        else:
            char_int = str(char_int)
    else:
        char_int = str(char)

    return char_int


grp = sys.argv[1].upper()
if grp not in chars:
    print(f"Invalid group '{grp}'. Use 'A' or 'B'.")
    sys.exit(1)

for char in chars[grp]:

    if grp == "B":
        char_int = toChar(char)
        out_dir = f"./data/output{grp}/{char_int}/"
    else:
        out_dir = f"./data/output{grp}/{char}/"
    in_dir = f"./data/input{grp}/{char}/"

    for fname in os.listdir(in_dir):
        if not fname.lower().endswith(".png") and not fname.lower().endswith(".jpg"):
            continue

        base, ext = fname.rsplit(".", 1)
        key = f"{grp}{char}{base}".strip()
        out_path = "out_dir" + fname

        input_images(in_dir + fname, "./inputs/", key)

        if not os.path.exists(out_path):
            ext = "png" if out_path.endswith(".jpg") else "jpg"
            out_path = out_path[:-3] + ext
        # endif

        base = toChar(base)
        fname = f"{base}.{ext}"
        try:
            output_images(out_dir + fname, "./outputs/", key)
        except Exception as e:
            print(f"ENOENT {out_dir + fname}")
            continue
