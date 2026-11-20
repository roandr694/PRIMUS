"""Fast unit tests of the home-autonomy stack (no simulator, no GPU):

    python3 -m unittest discover -s brain/tests -v        (from PRIMUS/)
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "robot", ROOT / "control"):
    sys.path.insert(0, str(p))

import link  # noqa: E402
from brain.agent.intents import parse  # noqa: E402
from brain.mapping.grid import Grid  # noqa: E402
from brain.mapping.matcher import global_search, match  # noqa: E402
from brain.nav.astar import astar, simplify  # noqa: E402
from brain.nav.follower import PathFollower  # noqa: E402
from brain.semantic import rooms as R  # noqa: E402


def box_world(res=0.05):
    """10 x 6 m: two rooms joined by a 1 m doorway, walls 10 cm thick."""
    g = Grid((0, 10, 0, 6), res)
    occ = np.zeros((g.ny, g.nx), bool)
    occ[:2, :] = occ[-2:, :] = occ[:, :2] = occ[:, -2:] = True
    occ[:, 98:100] = True                                     # dividing wall at x = 4.9-5.0
    occ[50:70, 98:100] = False                                # doorway y 2.5-3.5
    occ[80:96, 150:180] = True                                # a sideboard in the right room (breaks the symmetry)
    occ[20:30, 20:26] = True                                  # a column in the left room
    g.L[:] = np.where(occ, 3.0, -2.5)
    return g


class TestLink(unittest.TestCase):
    def test_roundtrip(self):
        s = link.State(stamp=12.5, x=1.0, y=-2.0, z=0.4, yaw=0.3, mode=2, battery=0.8, pad_buttons=3)
        s2, seq = link.unpack(link.pack(s, 7))
        self.assertEqual(seq, 7)
        self.assertAlmostEqual(s2.y, -2.0, places=5)
        self.assertEqual(s2.mode, 2)
        c = link.Command(stamp=1.0, speed=0.5, skill=True, source=2, heartbeat=99)
        c2, _ = link.unpack(link.pack(c))
        self.assertTrue(c2.skill)
        self.assertEqual(c2.heartbeat, 99)
        g = np.full((60, 60), np.nan, np.float32)
        g[10, 10] = 0.123
        p = link.ElevationPatch(3.0, 1.0, 2.0, 0.04, g, np.full((60, 60), np.inf, np.float32))
        p2, _ = link.unpack(link.pack(p))
        self.assertAlmostEqual(float(p2.ground[10, 10]), 0.123, places=3)
        self.assertTrue(np.isnan(p2.ground[0, 0]))
        self.assertTrue(np.isinf(p2.ceiling_at(np.array([[1.5, 2.5]])))[0])


class TestNav(unittest.TestCase):
    def test_astar_through_door(self):
        g = box_world()
        cost = np.where(g.occupied(), np.inf, 1.0)
        path = astar(cost, g.ij(np.array([2.0, 1.0])), g.ij(np.array([8.0, 1.0])))
        self.assertIsNotNone(path)
        xy = g.xy(np.array([p[0] for p in path]), np.array([p[1] for p in path]))
        crossing = xy[np.argmin(np.abs(xy[:, 0] - 4.95))]
        self.assertTrue(2.5 <= crossing[1] <= 3.5)                 # goes through the doorway
        self.assertLess(len(simplify(path, cost)), len(path))

    def test_follower_reaches_goal(self):
        f = PathFollower()
        f.set_path([(0, 0), (2, 0), (2, 2)])
        x, y, th = 0.0, 0.0, 0.0
        for _ in range(2000):
            v, w = f.step((x, y, th))
            th += w * 0.02
            x += v * math.cos(th) * 0.02
            y += v * math.sin(th) * 0.02
            if f.done:
                break
        self.assertTrue(f.done)
        self.assertLess(math.hypot(x - 2, y - 2), 0.2)


class TestMapping(unittest.TestCase):
    def scan(self, g, x, y, th, n=360):
        occ = g.occupied()
        pts = []
        for a in np.arange(n) * 2 * math.pi / n:
            for r in np.arange(0.15, 8, 0.02):
                i, j = g.ij(np.array([x + r * math.cos(th + a), y + r * math.sin(th + a)]))
                if not g.inside(i, j) or occ[i, j]:
                    pts.append((r * math.cos(a), r * math.sin(a)))
                    break
        return np.array(pts)

    def test_match_recovers_pose(self):
        g = box_world()
        pts = self.scan(g, 2.3, 4.1, 0.4)
        p, s = match(g, pts, (2.45, 3.95, 0.48))
        self.assertLess(math.hypot(p[0] - 2.3, p[1] - 4.1), 0.05)
        self.assertLess(abs(p[2] - 0.4), 0.03)
        self.assertGreater(s, 0.5)

    def test_global_relocalisation(self):
        g = box_world()
        pts = self.scan(g, 7.2, 1.8, -1.0)
        free = g.free()
        ii, jj = np.nonzero(free[::6, ::6])
        best = global_search(g, pts, g.xy(ii * 6, jj * 6))
        s, x, y, th = best[0]
        self.assertLess(math.hypot(x - 7.2, y - 1.8), 0.15)


class TestRooms(unittest.TestCase):
    def test_two_rooms(self):
        g = box_world()
        lab = R.segment_rooms(g)
        self.assertEqual(len(np.unique(lab[lab > 0])), 2)
        rooms = R.describe(lab, g, None, "f0")
        self.assertEqual(len(rooms), 2)
        self.assertEqual(len(rooms[0].neighbours), 1)               # connected through the doorway


class TestIntents(unittest.TestCase):
    def test_commands(self):
        self.assertEqual(parse("pare!")[0]["tool"], "stop")
        self.assertEqual(parse("vai para a cozinha")[0]["args"], {"target": "cozinha"})
        c = parse("Primus, vai pra sala e fica do lado do sofá")[0]
        self.assertEqual(c["args"], {"object": "sofa", "room": "sala"})
        self.assertEqual(parse("fica mais baixo")[0]["tool"], "set_body")
        self.assertEqual(parse("me segue")[0]["tool"], "follow")
        self.assertEqual(parse("sobe aquela escada")[0]["args"]["kind"], "stairs_up")
        self.assertEqual(parse("o que você está vendo?")[0]["tool"], "describe")
        self.assertIsNone(parse("conta uma piada"))


if __name__ == "__main__":
    unittest.main()
