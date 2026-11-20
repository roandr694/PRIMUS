"""Material catalogue: texture, tint, how many metres one texture tile covers, rendering and contact parameters.

`tile` sets the texture scale (texture coordinates are generated in metres). `friction` is MuJoCo's
(slide, torsion, roll) used by the colliders built with that material. The USD export reads `roughness`,
`metallic` and `opacity`, plus the normal/roughness maps that textures.py copies next to the colour map.
"""
from dataclasses import dataclass, field


@dataclass
class Material:
    name: str
    texture: str = None                 # baked texture name (textures.py) or None for a flat colour
    rgba: tuple = (1, 1, 1, 1)
    tile: float = 1.0                   # metres per texture tile (taken from textures.py when None)
    specular: float = 0.2
    shininess: float = 0.2
    reflectance: float = 0.0
    emission: float = 0.0
    friction: tuple = (1.0, 0.02, 0.002)
    roughness: float = 0.7
    metallic: float = 0.0
    opacity: float = 1.0
    double_sided: bool = False
    extra: dict = field(default_factory=dict)


def M(name, texture=None, rgba=(1, 1, 1, 1), tile=None, **kw):
    return Material(name, texture, rgba, tile if tile is not None else -1.0, **kw)


POLISHED_STONE = (0.75, 0.02, 0.002)     # dry polished granite / porcelain (rubber wheel)
WOOD = (0.85, 0.02, 0.002)
DECK = (0.95, 0.02, 0.002)
GRASS = (0.80, 0.03, 0.004)
GRAVEL = (0.70, 0.05, 0.010)
CONCRETE = (1.00, 0.02, 0.002)
GLASS = (0.45, 0.01, 0.001)
FABRIC = (0.90, 0.02, 0.002)

CATALOGUE = [
    # structure and finishes
    M("wood_ceiling", "wood_ceiling", specular=0.15, shininess=0.25, friction=WOOD, roughness=0.6),
    M("wood_panel", "wood_panel", specular=0.15, shininess=0.3, friction=WOOD, roughness=0.55),
    M("wood_dark", "wood_dark", specular=0.25, shininess=0.4, friction=WOOD, roughness=0.45),
    M("wood_light", "wood_light", specular=0.2, shininess=0.35, friction=WOOD, roughness=0.5),
    M("wood_floor", "wood_light", tile=1.4, specular=0.25, shininess=0.4, friction=WOOD, roughness=0.45),
    M("deck_grey", "deck_grey", specular=0.05, shininess=0.1, friction=DECK, roughness=0.85),
    M("deck_brown", "deck_brown", specular=0.08, shininess=0.15, friction=DECK, roughness=0.8),
    M("granite_tiles", "granite_tiles", specular=0.35, shininess=0.5, reflectance=0.03, friction=POLISHED_STONE,
      roughness=0.35),
    M("dark_tiles", "dark_tiles", specular=0.3, shininess=0.45, reflectance=0.03, friction=POLISHED_STONE,
      roughness=0.4),
    M("stone_counter", "stone_counter", specular=0.4, shininess=0.6, friction=POLISHED_STONE, roughness=0.3),
    M("white_boards", "white_boards", specular=0.1, shininess=0.2, emission=0.08, roughness=0.8),
    M("plaster", "plaster", specular=0.05, shininess=0.1, emission=0.08, roughness=0.9),
    M("concrete_light", "concrete_light", specular=0.08, shininess=0.15, friction=CONCRETE, roughness=0.8),
    M("pavers", "pavers", specular=0.08, shininess=0.15, friction=CONCRETE, roughness=0.8),
    M("stone_wall", "stone_wall", specular=0.05, shininess=0.1, friction=CONCRETE, roughness=0.9),
    M("stone_blocks", "stone_blocks", specular=0.05, shininess=0.1, friction=CONCRETE, roughness=0.9),
    M("rock", "rock", specular=0.1, shininess=0.2, friction=CONCRETE, roughness=0.8),
    M("soil_grass", "soil_grass", specular=0.02, shininess=0.05, friction=GRASS, roughness=0.95),
    M("ground_forest", "ground_forest", specular=0.02, shininess=0.05, friction=GRASS, roughness=0.95),
    M("soil_rocky", "soil_rocky", specular=0.02, shininess=0.05, friction=GRAVEL, roughness=0.95),
    M("planting_bed", "planting_bed", specular=0.02, shininess=0.05, friction=GRAVEL, roughness=0.95),
    M("lawn", "lawn", specular=0.02, shininess=0.05, friction=GRASS, roughness=0.95),
    M("gravel_white", "gravel_white", specular=0.05, shininess=0.1, friction=GRAVEL, roughness=0.9),
    M("pool_mosaic", "pool_mosaic", specular=0.5, shininess=0.7, friction=(0.6, 0.02, 0.002), roughness=0.2),
    M("bath_stone", "bath_stone", specular=0.4, shininess=0.6, friction=POLISHED_STONE, roughness=0.3),
    M("steel_dark", "steel_dark", specular=0.4, shininess=0.5, friction=(0.6, 0.02, 0.002), roughness=0.4,
      metallic=0.8),
    M("stainless", "stainless", specular=0.8, shininess=0.8, reflectance=0.1, friction=(0.5, 0.02, 0.002),
      roughness=0.25, metallic=1.0),
    M("roof_metal", "roof_metal", specular=0.3, shininess=0.4, roughness=0.5, metallic=0.6),
    M("mashrabiya", "mashrabiya", specular=0.1, shininess=0.2, friction=WOOD, roughness=0.6),
    M("asphalt", None, (0.20, 0.20, 0.21, 1), 1.0, specular=0.05, shininess=0.1, friction=(1.0, 0.02, 0.002)),
    M("curb", "concrete_light", tile=1.0, specular=0.05),
    M("terracotta_roof", None, (0.62, 0.30, 0.20, 1), 1.0, specular=0.1, shininess=0.2),
    M("neighbour_wall", "plaster", tile=3.0, rgba=(0.96, 0.92, 0.84, 1), specular=0.05),
    # water and glass (visual only)
    M("water_sea", "water_sea", (0.75, 0.95, 0.95, 1), specular=0.9, shininess=0.9, reflectance=0.25,
      roughness=0.05, extra={"water": True}),
    M("water_pool", "water_pool", (0.85, 1.0, 1.0, 0.75), specular=0.9, shininess=0.9, reflectance=0.2,
      roughness=0.05, opacity=0.6, extra={"water": True}),
    M("water_dark", None, (0.10, 0.16, 0.17, 0.9), 1.0, specular=0.9, shininess=0.9, reflectance=0.35,
      roughness=0.05, opacity=0.9),
    M("glass", None, (0.72, 0.84, 0.86, 0.16), 1.0, specular=0.9, shininess=0.95, reflectance=0.08,
      friction=GLASS, roughness=0.02, opacity=0.15, double_sided=True),
    M("glass_frosted", None, (0.92, 0.95, 0.95, 0.55), 1.0, specular=0.5, shininess=0.6, friction=GLASS,
      roughness=0.4, opacity=0.55),
    M("mirror", None, (0.75, 0.80, 0.82, 1), 1.0, specular=1.0, shininess=1.0, reflectance=0.5, roughness=0.0,
      metallic=1.0),
    # furniture and decor
    M("fabric_white", "fabric_white", specular=0.05, shininess=0.1, friction=FABRIC, roughness=0.95),
    M("fabric_orange", "fabric_orange", specular=0.05, shininess=0.1, friction=FABRIC, roughness=0.95),
    M("fabric_yellow", "fabric_yellow", specular=0.05, shininess=0.1, friction=FABRIC, roughness=0.95),
    M("fabric_teal", "fabric_teal", specular=0.05, shininess=0.1, friction=FABRIC, roughness=0.95),
    M("fabric_turquoise", "fabric_turquoise", specular=0.05, shininess=0.1, friction=FABRIC, roughness=0.95),
    M("fabric_grey", "fabric_grey", specular=0.05, shininess=0.1, friction=FABRIC, roughness=0.95),
    M("fabric_sage", "fabric_sage", specular=0.05, shininess=0.1, friction=FABRIC, roughness=0.95),
    M("fabric_beige", "fabric_beige", specular=0.05, shininess=0.1, friction=FABRIC, roughness=0.95),
    M("fabric_navy", "fabric_navy", specular=0.05, shininess=0.1, friction=FABRIC, roughness=0.95),
    M("rug", "rug", specular=0.02, shininess=0.05, friction=(1.0, 0.03, 0.004), roughness=0.95),
    M("rattan", "rattan", specular=0.1, shininess=0.2, roughness=0.8),
    M("rope_weave", "rope_weave", specular=0.05, shininess=0.1, roughness=0.9),
    M("rush", "rush", specular=0.05, shininess=0.1, roughness=0.9),
    M("leather_tan", "leather_tan", specular=0.3, shininess=0.4, roughness=0.5),
    M("white_gloss", None, (0.93, 0.93, 0.92, 1), 1.0, specular=0.6, shininess=0.7, roughness=0.25),
    M("ceramic_white", None, (0.96, 0.96, 0.95, 1), 1.0, specular=0.8, shininess=0.9, reflectance=0.05,
      roughness=0.1),
    M("ceramic_green", None, (0.27, 0.40, 0.36, 1), 1.0, specular=0.7, shininess=0.8, reflectance=0.05,
      roughness=0.15),
    M("black_matte", None, (0.04, 0.04, 0.045, 1), 1.0, specular=0.1, shininess=0.2, roughness=0.7),
    M("black_glass", None, (0.02, 0.02, 0.025, 1), 1.0, specular=0.9, shininess=0.95, reflectance=0.15,
      roughness=0.05),
    M("screen_on", None, (0.10, 0.16, 0.24, 1), 1.0, specular=0.9, shininess=0.9, emission=0.25, roughness=0.05),
    M("brass", None, (0.78, 0.62, 0.33, 1), 1.0, specular=0.8, shininess=0.8, roughness=0.3, metallic=1.0),
    M("lamp_glow", None, (1.0, 0.86, 0.62, 1), 1.0, specular=0.0, shininess=0.0, emission=0.9, roughness=0.9),
    M("lamp_paper", None, (0.97, 0.93, 0.85, 1), 1.0, specular=0.05, shininess=0.1, emission=0.45,
      roughness=0.9),
    M("book_red", None, (0.55, 0.16, 0.12, 1), 1.0, specular=0.1),
    M("book_blue", None, (0.14, 0.25, 0.42, 1), 1.0, specular=0.1),
    M("book_cream", None, (0.88, 0.82, 0.70, 1), 1.0, specular=0.1),
    M("book_green", None, (0.20, 0.36, 0.26, 1), 1.0, specular=0.1),
    M("art_1", None, (0.62, 0.28, 0.36, 1), 1.0, specular=0.05),
    M("art_2", None, (0.22, 0.30, 0.38, 1), 1.0, specular=0.05),
    M("wine_bottle", None, (0.12, 0.20, 0.10, 1), 1.0, specular=0.8, shininess=0.9, roughness=0.1),
    M("car_paint", None, (0.16, 0.18, 0.20, 1), 1.0, specular=0.9, shininess=0.9, reflectance=0.1,
      roughness=0.15, metallic=0.6),
    M("tyre", None, (0.05, 0.05, 0.05, 1), 1.0, specular=0.05),
    M("generator_yellow", None, (0.92, 0.70, 0.12, 1), 1.0, specular=0.4, shininess=0.4),
    M("boat_white", None, (0.95, 0.95, 0.94, 1), 1.0, specular=0.7, shininess=0.8, roughness=0.2),
    M("rope", None, (0.80, 0.72, 0.58, 1), 1.0, specular=0.05),
    M("net_black", None, (0.05, 0.05, 0.05, 1), 1.0, specular=0.1),
    M("felt_green", None, (0.10, 0.36, 0.20, 1), 1.0, specular=0.02),
    # vegetation and terrain
    M("palm_bark", "palm_bark", specular=0.05, shininess=0.1, roughness=0.9),
    M("tree_bark", "tree_bark", specular=0.05, shininess=0.1, roughness=0.9),
    M("leaf_palm", None, (0.30, 0.52, 0.16, 1), 1.0, specular=0.25, shininess=0.4, roughness=0.6,
      double_sided=True),
    M("leaf_dark", None, (0.13, 0.30, 0.10, 1), 1.0, specular=0.25, shininess=0.4, roughness=0.6,
      double_sided=True),
    M("leaf_light", None, (0.42, 0.62, 0.20, 1), 1.0, specular=0.25, shininess=0.4, roughness=0.6,
      double_sided=True),
    M("leaf_variegated", None, (0.56, 0.66, 0.30, 1), 1.0, specular=0.3, shininess=0.4, roughness=0.6,
      double_sided=True),
    M("flower_pink", None, (0.92, 0.45, 0.58, 1), 1.0, specular=0.1, roughness=0.8),
    M("flower_white", None, (0.98, 0.95, 0.86, 1), 1.0, specular=0.1, roughness=0.8),
    M("flower_yellow", None, (0.98, 0.80, 0.18, 1), 1.0, specular=0.1, roughness=0.8),
    M("forest", "leaf_green", (0.36, 0.50, 0.33, 1), tile=40.0, specular=0.02, shininess=0.05, roughness=0.95),
    M("forest_far", None, (0.20, 0.34, 0.22, 1), 1.0, specular=0.0, shininess=0.0, roughness=1.0),
    M("bamboo", None, (0.55, 0.62, 0.30, 1), 1.0, specular=0.2, shininess=0.3),
]


def catalogue(tiles):
    """Material dict with texture tile sizes filled in from textures.py (metres per tile)."""
    out = {}
    for m in CATALOGUE:
        if m.tile < 0:
            m.tile = tiles.get(m.texture, {}).get("metres", 1.0) if m.texture else 1.0
        out[m.name] = m
    return out
