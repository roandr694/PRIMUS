"""CC0 textures for the CMA house: download (ambientCG, Poly Haven), bake (tint, tile grout) and write PNGs.

Raw downloads are cached outside the repository (<workspace>/assets/cc0_textures, see docs/workspace.md); the
processed PNGs used by MuJoCo and the USD export go to sim/models/house/textures/ (generated, not in Git).

    python3 house/textures.py            # download what is missing and bake every texture

Each material in materials.py names one source texture here (ambientCG asset id + resolution) and how to bake it.
"""
import io
import json
import os
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
WORKSPACE = Path(os.environ.get("PRIMUS_WORKSPACE", SIM.parents[1]))
CACHE = WORKSPACE / "assets" / "cc0_textures"
OUT = SIM / "models" / "house" / "textures"
UA = {"User-Agent": "PRIMUS-sim-texture-fetch/1.0"}

# ambientCG asset id -> resolution. All CC0 (https://ambientcg.com/license).
AMBIENTCG = {
    "WoodFloor051": "2K",     # narrow teak strips: slatted wood ceilings
    "WoodFloor064": "2K",     # warm planks: vertical wall panelling, joinery
    "WoodFloor043": "1K",     # reddish planks: darker deck strip, stair treads
    "Planks023A": "2K",       # weathered grey planks: outdoor decks
    "Granite003B": "2K",      # light granite: varanda, upper floor, kitchen
    "Concrete031": "2K",      # dark grey porcelain: living room floor
    "Concrete033": "1K",      # dark stone: kitchen counters
    "Concrete045": "1K",      # white vertical boards: TV wall
    "Concrete046": "1K",      # white plaster
    "Concrete034": "2K",      # light concrete: garden pavers, pool coping
    "Rocks025": "2K",         # rubble stone masonry walls
    "Granite007A": "1K",      # rough granite blocks: garden steps
    "Rock020": "1K",          # shore boulders
    "Ground037": "1K",        # grass and soil: slopes
    "Ground108": "1K",        # rocky soil
    "ScatteredLeaves009": "1K",  # planting beds
    "Grass004": "2K",         # lawn
    "Gravel023": "1K",        # white pebbles
    "Tiles132A": "1K",        # pool mosaic
    "Marble014": "1K",        # bathroom stone
    "Fabric062": "1K",        # white woven upholstery
    "Fabric036": "1K",        # linen, tinted for cushions
    "Carpet016": "1K",        # woven rug
    "Wicker008A": "1K",       # rattan: pendant lamps, baskets
    "Wicker005": "1K",        # rope weave: armchairs
    "Wicker013": "1K",        # rush seats
    "Metal046A": "1K",        # dark steel
    "Metal055A": "1K",        # brushed stainless
    "CorrugatedSteel009": "1K",  # roof sheet
    "Bark002": "1K",          # palm trunk
    "Bark001": "1K",          # tree trunk
    "Leather037": "1K",       # leather
}
POLYHAVEN_HDRI = {"simons_town_rocks": "4k"}     # sky for the MuJoCo skybox and the Isaac dome light
MAPS = ("Color", "NormalGL", "Roughness")


def _get(url, timeout=120):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def fetch_ambientcg(asset, res):
    dst = CACHE / asset
    if all((dst / f"{m}.jpg").exists() for m in MAPS[:1]):
        return asset, "cached"
    dst.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(_get(f"https://ambientcg.com/get?file={asset}_{res}-JPG.zip")))
    for name in z.namelist():
        for m in MAPS:
            if name.endswith(f"_{m}.jpg"):
                (dst / f"{m}.jpg").write_bytes(z.read(name))
    (dst / "source.json").write_text(json.dumps({"asset": asset, "res": res, "license": "CC0 ambientCG"}))
    return asset, "downloaded"


def fetch_hdri(name, res):
    dst = CACHE / "hdri" / f"{name}_{res}.hdr"
    if dst.exists():
        return name, "cached"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(_get(f"https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/{res}/{name}_{res}.hdr", 600))
    return name, "downloaded"


def fetch_all():
    jobs = [(fetch_ambientcg, a, r) for a, r in AMBIENTCG.items()]
    jobs += [(fetch_hdri, n, r) for n, r in POLYHAVEN_HDRI.items()]
    with ThreadPoolExecutor(6) as ex:
        for name, status in ex.map(lambda j: j[0](j[1], j[2]), jobs):
            print(f"  {name}: {status}")


# --------------------------------------------------------------------------------------------------------------
# Baking
# --------------------------------------------------------------------------------------------------------------
def _load(asset, m="Color"):
    p = CACHE / asset / f"{m}.jpg"
    return Image.open(p).convert("RGB") if p.exists() else None


def _tint(im, rgb, amount):
    """Multiply-blend towards a colour, keeping the texture's luminance detail."""
    a = np.asarray(im, np.float32) / 255.0
    lum = a.mean(axis=2, keepdims=True)
    target = lum / max(lum.mean(), 1e-3) * np.array(rgb, np.float32)
    out = (1 - amount) * a + amount * target
    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))


def _gain(im, gain=1.0, gamma=1.0):
    a = (np.asarray(im, np.float32) / 255.0) ** gamma * gain
    return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))


def _grout(im, n, width_frac, colour, offset_rows=False, random_shade=0.0, seed=0, ny=None):
    """Draws an n x ny tile grid (grout lines) and optionally shades tiles slightly differently."""
    a = np.asarray(im, np.float32).copy()
    h, w, _ = a.shape
    ny = ny or n
    rng = np.random.default_rng(seed)
    tw, th = w / n, h / ny
    if random_shade:
        for j in range(ny):
            for i in range(n):
                s = 1 + rng.uniform(-random_shade, random_shade)
                x0 = int((i + (0.5 if offset_rows and j % 2 else 0)) * tw) % w
                xs = (np.arange(int(tw)) + x0) % w
                a[int(j * th):int((j + 1) * th)][:, xs] *= s
    g = max(1, int(round(width_frac * w)))
    col = np.array(colour, np.float32)
    for j in range(ny):
        y = int(j * th)
        a[max(0, y - g // 2):y + (g + 1) // 2] = col
        shift = 0.5 if (offset_rows and j % 2) else 0.0
        for i in range(n):
            x = int(((i + shift) * tw) % w)
            a[int(j * th):int((j + 1) * th), max(0, x - g // 2):x + (g + 1) // 2] = col
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _noise_tex(size, base, var, scale, seed):
    rng = np.random.default_rng(seed)
    small = rng.normal(0, 1, (max(2, size // scale), max(2, size // scale), 1))
    im = Image.fromarray(((small - small.min()) / (np.ptp(small) + 1e-9) * 255).astype(np.uint8)[:, :, 0])
    im = np.asarray(im.resize((size, size), Image.BICUBIC), np.float32)[:, :, None] / 255.0
    fine = rng.normal(0, 0.5, (size, size, 1))
    a = np.array(base, np.float32) * (1 + var * (im - 0.5) * 2 + 0.25 * var * fine)
    return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))


def _slats(size, n, base, gap_colour, var, seed, vertical=True, grain=0.0, src=None):
    """Planks/slats: n boards per tile with per-board shade variation and dark gaps; optional wood source."""
    rng = np.random.default_rng(seed)
    if src is not None:
        a = np.asarray(src.resize((size, size), Image.BICUBIC), np.float32) / 255.0
    else:
        a = np.ones((size, size, 3), np.float32) * np.array(base, np.float32)
    bw = size / n
    for i in range(n):
        s = 1 + rng.uniform(-var, var)
        x0, x1 = int(i * bw), int((i + 1) * bw)
        if vertical:
            a[:, x0:x1] *= s
        else:
            a[x0:x1] *= s
    g = max(1, int(size / n * 0.08))
    for i in range(n):
        x = int(i * bw)
        if vertical:
            a[:, x:x + g] = gap_colour
        else:
            a[x:x + g] = gap_colour
    if not vertical:
        pass
    return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))


def bake_specs():
    """Texture name -> (function returning a PIL image, size in metres covered by one texture tile)."""
    T = {}

    def tex(name, metres):
        def deco(fn):
            T[name] = (fn, metres)
            return fn
        return deco

    @tex("wood_ceiling", 1.2)
    def _():
        im = _load("WoodFloor051")
        im = _gain(_tint(im, (0.80, 0.52, 0.28), 0.45), 1.28)
        return _slats(2048, 24, None, (0.22, 0.13, 0.07), 0.10, 1, vertical=False, src=im)

    @tex("wood_panel", 1.6)        # vertical boards ~14 cm (rotate UVs in geometry: boards along V)
    def _():
        im = _load("WoodFloor064").rotate(90, expand=True)
        im = _gain(_tint(im, (0.80, 0.53, 0.30), 0.40), 1.22)
        return _slats(2048, 12, None, (0.25, 0.15, 0.08), 0.08, 2, vertical=True, src=im)

    @tex("wood_dark", 1.0)
    def _():
        return _gain(_tint(_load("WoodFloor043"), (0.52, 0.26, 0.14), 0.45), 1.08)

    @tex("wood_light", 1.0)
    def _():
        return _gain(_tint(_load("WoodFloor064"), (0.78, 0.57, 0.35), 0.35), 1.18)

    @tex("deck_grey", 1.6)
    def _():
        return _gain(_tint(_load("Planks023A"), (0.55, 0.52, 0.48), 0.25), 1.05)

    @tex("deck_brown", 1.6)
    def _():
        return _tint(_load("WoodFloor043"), (0.46, 0.27, 0.17), 0.4)

    @tex("granite_tiles", 1.8)      # 90 cm light granite tiles
    def _():
        im = _gain(_tint(_load("Granite003B"), (0.86, 0.84, 0.80), 0.55), 1.12)
        return _grout(im, 2, 0.0025, (150, 148, 142), random_shade=0.025, seed=3)

    @tex("dark_tiles", 2.4)         # 120 cm dark grey porcelain
    def _():
        im = _gain(_tint(_load("Concrete031"), (0.36, 0.36, 0.35), 0.5), 1.0)
        return _grout(im, 2, 0.0018, (52, 52, 50), random_shade=0.03, seed=4)

    @tex("stone_counter", 1.0)
    def _():
        return _gain(_tint(_load("Concrete033"), (0.20, 0.20, 0.20), 0.6), 0.85)

    @tex("white_boards", 1.2)
    def _():
        return _gain(_tint(_load("Concrete045"), (0.96, 0.95, 0.92), 0.65), 1.28)

    @tex("plaster", 2.0)
    def _():
        return _gain(_tint(_load("Concrete046"), (0.95, 0.94, 0.91), 0.75), 1.30)

    @tex("concrete_light", 2.0)
    def _():
        return _gain(_tint(_load("Concrete034"), (0.74, 0.72, 0.68), 0.3), 1.0)

    @tex("pavers", 2.4)             # 60 x 240 cm concrete slabs with gravel joints (upper garden)
    def _():
        im = _gain(_tint(_load("Concrete034"), (0.76, 0.74, 0.70), 0.3), 1.02)
        return _grout(im, 1, 0.012, (196, 192, 184), random_shade=0.04, seed=5, ny=4)

    @tex("stone_wall", 2.5)
    def _():
        return _gain(_tint(_load("Rocks025"), (0.62, 0.58, 0.52), 0.2), 0.95)

    @tex("stone_blocks", 1.6)       # rough granite blocks (steps, plinths)
    def _():
        im = _gain(_tint(_load("Granite007A"), (0.58, 0.56, 0.48), 0.35), 0.95)
        return _grout(im, 2, 0.01, (70, 66, 58), offset_rows=True, random_shade=0.08, seed=6, ny=4)

    @tex("rock", 3.0)
    def _():
        return _tint(_load("Rock020"), (0.48, 0.47, 0.45), 0.2)

    @tex("soil_grass", 4.0)
    def _():
        return _gain(_tint(_load("Grass004"), (0.24, 0.38, 0.14), 0.45), 0.95)

    @tex("ground_forest", 3.0)
    def _():
        return _gain(_tint(_load("ScatteredLeaves009"), (0.30, 0.34, 0.16), 0.55), 0.85)

    @tex("soil_rocky", 3.0)
    def _():
        return _gain(_tint(_load("Ground037"), (0.36, 0.38, 0.22), 0.35), 0.9)

    @tex("planting_bed", 2.0)
    def _():
        return _gain(_load("ScatteredLeaves009"), 0.9)

    @tex("lawn", 3.0)
    def _():
        return _gain(_tint(_load("Grass004"), (0.30, 0.52, 0.18), 0.25), 1.1)

    @tex("gravel_white", 1.0)
    def _():
        return _gain(_tint(_load("Gravel023"), (0.86, 0.85, 0.82), 0.3), 1.05)

    @tex("pool_mosaic", 0.6)
    def _():
        return _tint(_load("Tiles132A"), (0.30, 0.58, 0.56), 0.75)

    @tex("bath_stone", 1.2)
    def _():
        return _gain(_tint(_load("Marble014"), (0.86, 0.82, 0.74), 0.3), 1.0)

    @tex("fabric_white", 0.25)
    def _():
        return _gain(_tint(_load("Fabric062"), (0.92, 0.90, 0.86), 0.4), 1.08)

    for nm, rgb in {"fabric_orange": (0.92, 0.36, 0.18), "fabric_yellow": (0.95, 0.85, 0.25),
                    "fabric_teal": (0.30, 0.55, 0.52), "fabric_turquoise": (0.10, 0.62, 0.68),
                    "fabric_grey": (0.55, 0.56, 0.54), "fabric_sage": (0.58, 0.64, 0.57),
                    "fabric_beige": (0.80, 0.74, 0.64), "fabric_navy": (0.16, 0.22, 0.34)}.items():
        T[nm] = ((lambda rgb=rgb: _tint(_load("Fabric036"), rgb, 0.85)), 0.3)

    @tex("rug", 1.0)
    def _():
        return _gain(_tint(_load("Carpet016"), (0.80, 0.75, 0.66), 0.4), 1.05)

    @tex("rattan", 0.4)
    def _():
        return _gain(_load("Wicker008A"), 1.1)

    @tex("rope_weave", 0.4)
    def _():
        return _gain(_tint(_load("Wicker005"), (0.80, 0.76, 0.68), 0.4), 1.05)

    @tex("rush", 0.3)
    def _():
        return _load("Wicker013")

    @tex("steel_dark", 1.0)
    def _():
        return _gain(_tint(_load("Metal046A"), (0.17, 0.17, 0.18), 0.6), 0.9)

    @tex("stainless", 1.0)
    def _():
        return _gain(_load("Metal055A"), 1.0)

    @tex("roof_metal", 1.2)
    def _():
        return _tint(_load("CorrugatedSteel009"), (0.36, 0.25, 0.20), 0.75)

    @tex("palm_bark", 1.0)
    def _():
        im = _gain(_tint(_load("Bark002"), (0.55, 0.52, 0.47), 0.3), 1.0)
        return _grout(im, 1, 0.006, (70, 64, 56), ny=6)        # ring scars

    @tex("tree_bark", 1.0)
    def _():
        return _load("Bark001")

    @tex("leather_tan", 0.6)
    def _():
        return _tint(_load("Leather037"), (0.55, 0.36, 0.22), 0.6)

    # procedural (no download)
    T["water_sea"] = (lambda: _noise_tex(512, (0.08, 0.30, 0.32), 0.10, 16, 7), 8.0)
    T["water_pool"] = (lambda: _noise_tex(256, (0.45, 0.78, 0.80), 0.06, 8, 8), 2.0)
    T["leaf_green"] = (lambda: _noise_tex(256, (0.20, 0.42, 0.12), 0.25, 8, 9), 0.5)
    T["mashrabiya"] = (lambda: _slats(1024, 16, (0.62, 0.40, 0.22), (0.06, 0.04, 0.03), 0.10, 10), 1.0)
    return T


def bake(force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name, (fn, metres) in bake_specs().items():
        dst = OUT / f"{name}.png"
        meta[name] = {"metres": metres}
        if dst.exists() and not force:
            continue
        im = fn()
        if im is None:
            print(f"  {name}: source missing, skipped")
            continue
        im.save(dst, optimize=False, compress_level=3)
        print(f"  baked {name} {im.size}")
    # normal / roughness maps (Isaac): copied from the sources that have them
    for name, asset in SOURCE_OF.items():
        for m in ("NormalGL", "Roughness"):
            src = CACHE / asset / f"{m}.jpg"
            dst = OUT / f"{name}_{m.lower()}.jpg"
            if src.exists() and (force or not dst.exists()):
                dst.write_bytes(src.read_bytes())
    (OUT / "textures.json").write_text(json.dumps(meta, indent=1))
    return meta


SOURCE_OF = {"wood_ceiling": "WoodFloor051", "wood_panel": "WoodFloor064", "wood_dark": "WoodFloor043",
             "wood_light": "WoodFloor064", "deck_grey": "Planks023A", "deck_brown": "WoodFloor043",
             "granite_tiles": "Granite003B", "dark_tiles": "Concrete031", "stone_counter": "Concrete033",
             "white_boards": "Concrete045", "plaster": "Concrete046", "concrete_light": "Concrete034",
             "pavers": "Concrete034", "stone_wall": "Rocks025", "stone_blocks": "Granite007A", "rock": "Rock020",
             "soil_grass": "Grass004", "soil_rocky": "Ground037", "ground_forest": "ScatteredLeaves009", "planting_bed": "ScatteredLeaves009",
             "lawn": "Grass004", "gravel_white": "Gravel023", "pool_mosaic": "Tiles132A",
             "bath_stone": "Marble014", "fabric_white": "Fabric062", "rug": "Carpet016", "rattan": "Wicker008A",
             "rope_weave": "Wicker005", "rush": "Wicker013", "steel_dark": "Metal046A", "stainless": "Metal055A",
             "roof_metal": "CorrugatedSteel009", "palm_bark": "Bark002", "tree_bark": "Bark001",
             "leather_tan": "Leather037"}


if __name__ == "__main__":
    print(f"cache: {CACHE}")
    fetch_all()
    print(f"baking into {OUT}")
    bake(force="--force" in sys.argv)
