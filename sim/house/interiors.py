"""Furnishing of every room, following the photographs where they show the room (varanda: photo 8; living room:
photos 9, 12, 14; gourmet kitchen: photo 10; pool deck: photo 5; sea deck: photos 1, 4, 13) and plausible,
in-style choices elsewhere (internal kitchen, suites, bathrooms, service rooms, wine cellar, pavilion, study).
"""
import math

import numpy as np

import furniture as FU
import structure as ST
import vegetation as V
from geom import Frame, rot_x, rot_z

PI = math.pi
E, N, W, S = -PI / 2, 0.0, PI / 2, PI          # yaw so that a piece's front (local +Y) faces east/north/west/south


def F(x, y, z=0.0, yaw=0.0):
    return Frame((x, y, z), yaw=yaw)


def spear(s, x, y, z0, h=2.6):
    """Decorative paddle/spear on the gourmet kitchen wall (photo 10)."""
    s.tube("wood_dark", (x, y, z0), (x, y, z0 + h), 0.012)
    for k in range(4):
        s.box(["book_red", "black_matte", "book_cream", "black_matte"][k], (x, y, z0 + 0.9 + 0.12 * k),
              (0.016, 0.016, 0.05), collide=False)
    s.sphere("black_matte", (x, y, z0 + h + 0.18), 0.1, scale=(0.25, 1.0, 2.0))


def varanda(s):
    z = 0.0
    # gourmet kitchen (photo 10): back counter with sink, island with cooktop, rush stools, linear pendant
    FU.kitchen_run(s, F(0.6, 0.2, z, N), [("drawers3", 0.6), ("door2", 0.9), ("sink", 0.9), ("dishwasher", 0.6),
                                         ("drawers3", 0.6), ("door", 0.6), ("drawers2", 0.9)],
                   splash="black_glass", prefix="gourmet")
    s.box("stainless", (3.15, 0.27, 1.45), (2.5, 0.06, 0.008), collide=False)                 # glass shelf
    FU.island(s, F(3.0, 2.5, z, N), L=3.6, W=1.0, prefix="gourmet_isl")
    for x in (1.8, 2.6, 3.4, 4.2):
        FU.bar_stool(s, F(x, 3.35, z, S))
    FU.pendant_linear(s, F(3.0, 2.5, z), ST.Z_CEIL_SOC, 1.75, L=2.4)
    spear(s, 6.15, 1.0, 0.25)
    spear(s, 6.15, 1.5, 0.25)
    s.note("varanda", "gourmet kitchen: counter, sink, dishwasher, island with cooktop, 4 stools")
    # lounge 1 (photo 8): built-in sofa, two timber block tables, woven armchairs, ceramic stools, globe lamp
    FU.sofa_builtin(s, F(5.45, 8.0, z, W), L=5.0, note="varanda")
    FU.coffee_table_log(s, F(3.9, 7.62, z, N))
    FU.coffee_table_log(s, F(3.9, 8.32, z, N))
    FU.armchair_woven(s, F(2.25, 7.3, z, E), note="varanda")
    FU.armchair_woven(s, F(2.25, 8.8, z, E))
    FU.stool_ceramic(s, F(0.4, 7.75, z))
    FU.stool_ceramic(s, F(0.42, 8.6, z))
    FU.floor_lamp(s, F(5.3, 11.0, z), 1.3)
    FU.side_table_round(s, F(5.3, 5.2, z))
    FU.plant_small(s, F(5.3, 5.2, 0.5), 0.3)
    # lounge 2: sofa, lounge armchairs, timber table, lamp
    FU.sofa_sectional(s, F(5.35, 13.6, z, W), L=2.8, D=1.0, note="varanda")
    FU.armchair_lounge(s, F(2.6, 12.9, z, E), "fabric_sage")
    FU.armchair_lounge(s, F(2.6, 14.3, z, E), "fabric_sage")
    FU.coffee_table_log(s, F(3.9, 13.6, z, W), L=1.4, W=0.7, H=0.36)
    FU.side_table_round(s, F(5.4, 15.4, z))
    FU.table_lamp(s, F(5.4, 15.4, 0.5))
    FU.plant_pot(s, F(0.6, 15.6, z), 1.4)
    FU.plant_pot(s, F(0.6, 4.6, z), 1.2)
    for y in (6.5, 10.0, 13.6):
        FU.track_light(s, (1.0, y), (5.5, y), ST.Z_CEIL_SOC - 0.02, 3)


def living(s):
    z = 0.0
    zc = ST.Z_CEIL_UP
    # TV wall (photos 9, 12): white boards, long low bench, floating shelves with objects, TV
    FU.bench_shelf(s, F(10.0, 16.15, z, S), L=6.0)
    FU.tv(s, F(10.0, 16.39, 1.55, S))
    FU.shelf_wall(s, F(12.6, 16.25, 1.15, S), L=2.4)
    FU.shelf_wall(s, F(7.9, 16.25, 1.85, S), L=1.8)
    FU.shelf_wall(s, F(12.3, 16.25, 2.05, S), L=3.0)
    FU.shelf_wall(s, F(10.0, 16.25, 2.7, S), L=6.4, decor=True)
    # main sofa group facing the TV
    FU.rug(s, F(10.2, 12.9, z), L=4.4, W=3.4)
    FU.sofa_sectional(s, F(9.8, 11.3, z, N), L=3.4, D=1.05, ret=1.7, note="living room")
    FU.coffee_table_round(s, F(9.9, 13.3, z), r=0.62)
    FU.side_table_glass(s, F(9.8, 10.45, z, N))
    FU.chair_windsor(s, F(12.7, 14.2, z, W + 0.4))
    FU.basket(s, F(12.6, 15.0, z))
    FU.armchair_lounge(s, F(7.5, 13.2, z, E), "fabric_grey")
    FU.floor_lamp(s, F(7.4, 14.4, z), 1.5)
    FU.plant_pot(s, F(13.4, 15.7, z), 1.6)
    # second group near the pivot doors (photo 14)
    FU.armchair_lounge(s, F(7.3, 8.1, z, E), "fabric_sage")
    FU.armchair_lounge(s, F(7.3, 9.5, z, E), "fabric_sage")
    FU.sofa_sectional(s, F(9.7, 8.8, z, W), L=2.2, D=0.95)
    FU.coffee_table_log(s, F(8.5, 8.8, z, W), L=1.0, W=0.6, H=0.36)
    FU.side_table_round(s, F(7.3, 10.4, z))
    FU.table_lamp(s, F(7.3, 10.4, 0.5))
    # dining with wicker pendants (photo 14)
    FU.table_with_chairs(s, F(11.0, 6.0, z), 3.2, 1.05, 3, chair=FU.chair_wood)
    for x in (10.0, 11.0, 12.0):
        FU.pendant_wicker(s, F(x, 6.0), zc, 1.6)
    FU.sideboard(s, F(8.1, 4.55, z, N), L=2.0)
    FU.plant_pot(s, F(6.8, 4.7, z), 1.3)
    for x in (9.0, 13.0):
        FU.track_light(s, (x, 5.0), (x, 15.8), zc - 0.03, 4)
    s.note("living room", "TV wall with shelves, sectional sofa, lounge chairs, dining for 8, wicker pendants")


def kitchen(s):
    z = 0.0
    FU.kitchen_run(s, F(14.9, 4.1, z, S), [("fridge", 0.9), ("oven_tall", 0.6), ("tall", 0.6), ("drawers3", 0.6),
                                          ("cooktop", 0.9), ("drawers3", 0.6), ("door2", 0.9), ("sink", 0.9),
                                          ("dishwasher", 0.6), ("drawers3", 0.6), ("door2", 0.9)],
                   splash="stone_counter", prefix="kit_n")
    FU.range_hood(s, F(14.9 - 2.85, 3.75, 0.0), 1.65, ST.Z_CEIL_SOC)
    s.box("wood_panel", (8.5, 3.92, 2.25), (1.7, 0.18, 0.36), collide=True)                     # wall cabinets
    FU.kitchen_run(s, F(6.6, 0.2, z, N), [("drawers3", 0.6), ("door2", 0.9), ("sink", 1.0), ("dishwasher", 0.6),
                                         ("drawers2", 0.9), ("door2", 0.9), ("drawers3", 0.6)],
                   prefix="kit_s")
    s.cylinder("wood_light", (13.4, 2.1, 0.73), 0.5, 0.025, collide=False)
    s.cylinder("wood_dark", (13.4, 2.1, 0.36), 0.06, 0.36, collide=False)
    s.collider("cylinder", (13.4, 2.1, 0.375), (0.5, 0.375))
    for a in (0.3, 2.4, 4.5):
        FU.chair_wood(s, F(13.4 + 0.75 * math.cos(a), 2.1 + 0.75 * math.sin(a), z, a + PI / 2))
    FU.vase(s, F(13.4, 2.1, 0.755), 0.2, 0.06, "ceramic_green")
    s.note("kitchen", "galley kitchen: fridge, oven, cooktop + hood, two sinks, dishwashers, breakfast table")


def service(s):
    z = 0.0
    FU.washer(s, F(15.6, 0.55, z, N))
    FU.washer(s, F(16.25, 0.55, z, N))
    FU.kitchen_run(s, F(16.65, 0.2, z, N), [("door2", 0.9), ("sink", 0.8)], prefix="laundry")
    FU.bookcase(s, F(18.1, 3.25, z, S), W=1.4, H=2.2)
    FU.vanity(s, F(16.6, 7.38, z, S), L=0.9, sinks=1)
    FU.toilet(s, F(18.14, 6.4, z, W))
    FU.vanity(s, F(20.95, 6.6, z, W), L=0.7, sinks=1)
    FU.toilet(s, F(19.6, 7.38, z, S))
    FU.shower(s, F(20.95, 5.4, z, W), W=1.0, D=0.9)
    FU.bookcase(s, F(18.6, 9.85, z, S), W=4.6, H=2.4, rows=6)
    FU.bookcase(s, F(19.6, 7.62, z, N), W=2.6, H=2.4, rows=6)
    s.note("service", "laundry, guest powder room, staff bathroom, pantry shelves")
    # wine cellar
    FU.wine_rack(s, F(20.95, 13.3, z, W), W=5.6, H=2.6)
    FU.wine_rack(s, F(18.6, 16.35, z, S), W=4.4, H=2.6)
    FU.table_with_chairs(s, F(18.2, 12.9, z, W), 2.2, 0.9, 2, chair=FU.chair_wood, ends=False)
    FU.pendant_wicker(s, F(18.2, 12.9), ST.Z_CEIL_SOC, 1.9, r=0.22)
    s.note("wine cellar", "wine walls, tasting table for 4")


def pavilion(s):
    z = ST.Z_DECK
    # spa
    FU.massage_table(s, F(-14.7, 3.9, z, N))
    FU.armchair_lounge(s, F(-16.7, 4.8, z, E), "fabric_white")
    FU.shower(s, F(-13.62, 1.3, z, W), W=1.6, D=0.9)
    FU.bookcase(s, F(-16.1, 5.7, z, S), W=1.4, H=1.8, rows=4)
    FU.plant_pot(s, F(-17.0, 3.0, z), 1.0)
    # games room
    FU.rug(s, F(-15.2, 10.2, z), L=3.0, W=3.2)
    FU.sofa_sectional(s, F(-16.6, 10.2, z, E), L=3.0, D=0.95, fabric="fabric_navy")
    FU.tv(s, F(-13.52, 10.2, z + 1.5, W), w=1.9, h=1.08)
    FU.bench_shelf(s, F(-13.75, 10.2, z, W), L=2.8, d=0.4)
    FU.coffee_table_log(s, F(-15.3, 10.2, z, W), L=1.2, W=0.6, H=0.34)
    FU.foosball(s, F(-15.2, 7.3, z, W))
    FU.kitchen_run(s, F(-13.7, 13.85, z, S), [("door2", 0.9), ("sink", 0.9), ("drawers3", 0.6)], prefix="bar")
    for x in (-14.3, -15.0, -15.7):
        FU.bar_stool(s, F(x, 12.95, z, N))
    FU.armchair_woven(s, F(-16.8, 12.9, z, S))
    s.note("pavilion", "games room: sofa, TV, foosball, bar with sink and stools")


def suites(s):
    z = ST.Z_UP
    # master (y 10.25 .. 16.4)
    FU.rug(s, F(0.6, 13.3, z), L=3.2, W=3.6)
    FU.bed(s, F(1.33, 13.3, z, W), note="suite 1")
    FU.nightstand(s, F(2.2, 11.95, z, W))
    FU.nightstand(s, F(2.2, 14.65, z, W))
    FU.bench_end(s, F(-0.15, 13.3, z, W), L=1.7)
    FU.armchair_lounge(s, F(-1.5, 15.5, z, W + 0.6), "fabric_beige")
    FU.side_table_round(s, F(-0.7, 15.9, z))
    FU.desk(s, F(-0.9, 10.75, z, N), L=1.4, W=0.6)
    FU.chair_wood(s, F(-0.9, 11.3, z, S))
    FU.wardrobe(s, F(4.3, 10.6, z, N), W=3.2, H=2.6, D=0.6)
    FU.bench_end(s, F(4.3, 12.2, z, N), L=1.4)
    FU.vanity(s, F(4.3, 16.4, z, S), L=1.8, sinks=2)
    FU.bathtub(s, F(6.4, 15.35, z, W))
    FU.toilet(s, F(7.64, 14.0, z, W))
    FU.shower(s, F(2.65, 14.0, z, E), W=1.1, D=1.1)
    FU.towel_rail(s, F(3.2, 13.44, z, N))
    FU.plant_pot(s, F(-1.9, 10.7, z), 1.2)
    # suite 2 (y 5.05 .. 10.15)
    FU.bed(s, F(1.33, 8.45, z, W), throw="fabric_teal", note="suite 2")
    FU.nightstand(s, F(2.2, 7.15, z, W))
    FU.nightstand(s, F(2.2, 9.75, z, W))
    FU.armchair_lounge(s, F(-1.5, 5.8, z, W - 0.5), "fabric_grey")
    FU.wardrobe(s, F(4.3, 5.4, z, N), W=3.2, H=2.6, D=0.55)
    FU.vanity(s, F(6.0, 8.6, z, W), L=1.2, sinks=1)
    FU.toilet(s, F(3.3, 10.12, z, S))
    FU.shower(s, F(4.9, 10.12, z, S), W=1.1, D=1.0)
    FU.rug(s, F(0.4, 8.45, z), L=2.6, W=3.0)
    # suite 3 (y 0.15 .. 4.97)
    FU.bed(s, F(1.33, 1.95, z, W), throw="fabric_orange", note="suite 3")
    FU.nightstand(s, F(2.2, 0.6, z, W))
    FU.armchair_lounge(s, F(-1.5, 4.2, z, W + 0.5), "fabric_sage")
    FU.wardrobe(s, F(4.3, 4.58, z, S), W=3.2, H=2.6, D=0.55)
    FU.vanity(s, F(6.0, 1.5, z, W), L=1.2, sinks=1)
    FU.toilet(s, F(3.2, 0.18, z, N))
    FU.shower(s, F(5.0, 0.18, z, N), W=1.1, D=1.0)
    FU.rug(s, F(0.4, 1.95, z), L=2.6, W=3.0)
    # curtains along the sea glazing
    for (y0, y1) in ((0.3, 1.3), (3.8, 4.8), (5.2, 6.2), (9.0, 10.0), (10.4, 11.4), (15.3, 16.3)):
        FU.curtain(s, (-2.2, y0), (-2.2, y1), z + 0.02, ST.Z_CEIL_UP - 0.05)
    s.note("suites", "3 suites: king beds, nightstands, wardrobes, armchairs; bathrooms with vanities, "
           "toilets, walk-in showers, freestanding tub (master)")


def upper_common(s):
    z = ST.Z_UP
    # lounge over the kitchen (photo 14): long table, sofa, armchairs, sideboard, plants
    FU.table_with_chairs(s, F(10.6, 1.4, z), 3.0, 0.95, 3, chair=FU.chair_wood)
    FU.sofa_sectional(s, F(14.9, 0.7, z, N), L=2.4, D=0.95, fabric="fabric_beige")
    FU.armchair_lounge(s, F(13.9, 2.5, z, S), "fabric_grey")
    FU.armchair_lounge(s, F(15.9, 2.5, z, S), "fabric_grey")
    FU.coffee_table_log(s, F(14.9, 1.6, z, N), L=1.2, W=0.55, H=0.34)
    FU.plant_pot(s, F(8.3, 3.6, z), 1.5)
    FU.plant_pot(s, F(17.3, 3.6, z), 1.3)
    FU.floor_lamp(s, F(17.3, 1.8, z), 1.5)
    # corridor console
    FU.sideboard(s, F(6.35, 8.0, z, E), L=1.6, D=0.35, H=0.8)
    # upper powder room
    FU.vanity(s, F(18.3, 4.0, z, S), L=0.7, sinks=1)
    FU.toilet(s, F(17.8, 3.3, z, E))
    # study
    FU.rug(s, F(23.3, 2.0, z), L=2.6, W=2.0)
    FU.desk(s, F(23.5, 1.45, z, S), L=1.8, W=0.75)
    FU.office_chair(s, F(23.5, 0.8, z, N))
    FU.bookcase(s, F(21.6, 0.32, z, N), W=1.4, H=2.2)
    FU.armchair_lounge(s, F(25.0, 3.3, z, W + 0.5), "leather_tan")
    FU.floor_lamp(s, F(25.3, 2.4, z), 1.4)
    s.note("upper lounge", "long table for 8, sofa, armchairs; study with desk, bookcase, armchair")


def outdoors(s):
    # pool deck (photo 5): four turquoise loungers, two woven turquoise parasols, side tables
    for k, y in enumerate((3.0, 4.3, 5.6, 6.9)):
        FU.lounger(s, F(-8.6, y, 0.0, W))
    for y in (3.65, 6.25):
        FU.parasol(s, F(-7.75, y, 0.0), r=1.25)
    for y in (3.65, 6.25):
        FU.side_table_round(s, F(-9.75, y, 0.0), r=0.18, H=0.4, mat="steel_dark")
    s.note("pool deck", "4 loungers, 2 parasols")
    # lawn: young staked trees (photo 8), frangipani, palms
    V.broadleaf_tree(s, (-3.0, 5.5, 0.0), 5.0, 2.0, seed=1, staked=True)
    V.broadleaf_tree(s, (-2.3, 10.4, 0.0), 5.5, 2.2, seed=2, staked=True)
    V.broadleaf_tree(s, (-4.2, 13.8, 0.0), 4.8, 2.0, seed=3, staked=True)
    V.frangipani(s, (-4.5, 0.6, 0.0), 3.8, seed=4)
    V.coconut_palm(s, (-6.8, -0.9, 0.0), 10.5, lean=(-0.8, 0.2), seed=5)
    V.coconut_palm(s, (-6.6, 14.6, 0.0), 9.0, lean=(-0.6, -0.3), seed=6)
    V.hedge(s, (-13.0, 16.95, 0.0), (-6.0, 17.6, 0.0), height=1.4, depth=0.8, seed=7)
    V.hedge(s, (-0.3, 17.0, 0.0), (4.7, 17.9, 0.0), height=1.6, depth=0.9, seed=8)
    for k, x in enumerate(np.arange(-7.0, -0.8, 1.2)):
        V.ginger_clump(s, (x, -1.35, 0.0), 0.5, n=6, height=1.3, seed=80 + k)
    # sea deck (photos 1, 4, 13)
    zd = ST.Z_DECK
    for (x, y, yaw) in ((-23.6, 2.6, E), (-21.4, 2.6, W), (-23.6, 4.0, E), (-21.4, 4.0, W)):
        FU.deck_lounge_chair(s, F(x, y, zd, yaw))
    FU.coffee_table_log(s, F(-22.5, 3.3, zd, N), L=0.9, W=0.9, H=0.32)
    for y in (13.6, 15.2):
        FU.lounger(s, F(-25.6, y, zd, W), mattress="fabric_grey", pillow="fabric_white")
    for (x, y) in ST.DECK_PALMS:
        V.coconut_palm(s, (x, y, zd - 0.03), 10.0 + 2 * ((x * 7 + y) % 1), lean=(-0.9, 0.15 * math.sin(y)),
                       seed=int(abs(x * 10 + y)))
    FU.hammock(s, (-19.4, 5.25, zd + 1.8), (-19.6, 8.75, zd + 1.8))
    FU.hammock(s, (-19.6, 9.25, zd + 1.8), (-19.8, 12.75, zd + 1.8), mat="fabric_beige")
    s.note("sea deck", "4 timber lounge chairs, 2 loungers, 2 hammocks between the palms")
    FU.boat(s, Frame((-34.6, 6.0, ST.Z_SEA - 0.25), yaw=0.08))
    # upper garden bench and east terrace planters
    s.box("stone_blocks", (20.4, 6.0, ST.Z_UP + 0.22), (0.9, 0.25, 0.22))
    for y in (2.0, 8.0, 12.5):
        ST.planter(s, 27.6, 28.4, y, y + 1.6, ST.Z_UP, 0.6, wall_mat="concrete_light", seed=int(y * 3))


def furnish(s):
    varanda(s)
    living(s)
    kitchen(s)
    service(s)
    pavilion(s)
    suites(s)
    upper_common(s)
    outdoors(s)
