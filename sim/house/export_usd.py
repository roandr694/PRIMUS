"""USD export of the CMA house for Isaac Sim (camera / perception work), from the same `Scene` as the MuJoCo model.

    cd sim && python3 house/build.py --usd        # MuJoCo scene + sim/isaac/usd/house/cma_house.usd

Layout of the stage (Z up, metres, same frame as MuJoCo):
    /World/Looks/<material>        UsdPreviewSurface: base colour, normal and roughness maps (baked textures)
    /World/Visual/<material>       merged static visual meshes (vertex normals, st texture coordinates)
    /World/Water/*                 sea, pool and reflecting pool surfaces (translucent, no collision)
    /World/Collision/*             primitive colliders (invisible, UsdPhysics.CollisionAPI + friction materials)
    /World/Terrain/collision       hillside triangle mesh collider (from the MuJoCo heightfield)
    /World/Articulated/<body>      doors, drawers, appliance doors, lift cabin + gates: rigid bodies with
                                   revolute / prismatic joints (limits, damping drives; lift with a position drive)
    /World/Lights/*                HDRI dome (same sky as MuJoCo), sun, warm interior lights
    /World/Cameras/*               the preview viewpoints (photo-matched)
    /World/Spawns/*                robot spawn points (same names as the MuJoCo keyframes)

The robot is not part of this stage: reference sim/isaac/usd/robo_bipedal.usd and place it at a spawn point.
The PhysX friction model differs from MuJoCo's; the friction values are the MuJoCo sliding coefficients.
"""
import math
import os
import shutil
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

from geom import mat2quat

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "isaac" / "usd" / "house"
TEX_DIR = HERE.parent / "models" / "house" / "textures"


def _safe(name):
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def _rel(path, start):
    return os.path.relpath(str(path), str(start)).replace(os.sep, "/")


def _quatf(R):
    w, x, y, z = mat2quat(R)
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


def _xform(prim, pos, R=None, scale=None):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    if R is not None and not np.allclose(R, np.eye(3)):
        x.AddOrientOp().Set(_quatf(R))
    if scale is not None:
        x.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))


class Exporter:
    def __init__(self, scene, out_dir=OUT_DIR):
        self.s = scene
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / "cma_house.usd"
        self.stage = Usd.Stage.CreateNew(str(self.path)) if not self.path.exists() else Usd.Stage.Open(str(self.path))
        self.stage.GetRootLayer().Clear()
        st = self.stage
        UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(st, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(st, 1.0)
        world = UsdGeom.Xform.Define(st, "/World")
        st.SetDefaultPrim(world.GetPrim())
        scn = UsdPhysics.Scene.Define(st, "/World/PhysicsScene")
        scn.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scn.CreateGravityMagnitudeAttr(9.81)
        for p in ("Looks", "Visual", "Water", "Collision", "Terrain", "Articulated", "Lights", "Cameras", "Spawns",
                  "PhysicsMaterials"):
            UsdGeom.Scope.Define(st, f"/World/{p}")
        self.looks = {}
        self.phys_mats = {}

    # ------------------------------------------------------------------ materials
    def look(self, name):
        if name in self.looks:
            return self.looks[name]
        m = self.s.materials[name]
        path = f"/World/Looks/{_safe(name)}"
        mat = UsdShade.Material.Define(self.stage, path)
        sh = UsdShade.Shader.Define(self.stage, path + "/Surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        rgba = m.rgba
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(m.roughness))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(m.metallic))
        if m.opacity < 1.0 or rgba[3] < 1.0:
            sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(min(m.opacity, rgba[3])))
        if m.emission > 0:
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*[float(c * m.emission) for c in rgba[:3]]))
        tex = TEX_DIR / f"{m.texture}.png" if m.texture else None
        if tex is not None and tex.exists():
            reader = UsdShade.Shader.Define(self.stage, path + "/st")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            stout = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

            def texture(fname, key, raw=False, out="rgb", ttype=Sdf.ValueTypeNames.Float3):
                t = UsdShade.Shader.Define(self.stage, f"{path}/{key}")
                t.CreateIdAttr("UsdUVTexture")
                t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(_rel(fname, self.out))
                t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(stout)
                t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw" if raw else "sRGB")
                return t, t.CreateOutput(out, ttype)

            col, col_out = texture(tex, "BaseColor")
            col.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*map(float, rgba[:3]), 1.0))
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(col_out)
            nrm = TEX_DIR / f"{m.texture}_normalgl.jpg"
            if nrm.exists():
                n, n_out = texture(nrm, "Normal", raw=True)
                n.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                n.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
                sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(n_out)
            rough = TEX_DIR / f"{m.texture}_roughness.jpg"
            if rough.exists():
                r, r_out = texture(rough, "Roughness", raw=True, out="r", ttype=Sdf.ValueTypeNames.Float)
                sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(r_out)
        else:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgba[:3])))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        self.looks[name] = mat
        return mat

    def phys_mat(self, friction):
        key = round(float(friction[0]), 3)
        if key in self.phys_mats:
            return self.phys_mats[key]
        path = f"/World/PhysicsMaterials/mu_{str(key).replace('.', '_')}"
        mat = UsdShade.Material.Define(self.stage, path)
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(key)
        api.CreateDynamicFrictionAttr(key)
        api.CreateRestitutionAttr(0.0)
        self.phys_mats[key] = mat
        return mat

    # ------------------------------------------------------------------ geometry
    def mesh(self, path, mat, v, n, t, f, double=False):
        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(v, np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(f, np.int32).ravel()))
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(n, np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        st = np.asarray(t, np.float32).copy()
        st[:, 1] = -st[:, 1]                       # MuJoCo v runs down the image rows, USD v runs up
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(st))
        m.CreateSubdivisionSchemeAttr("none")
        m.CreateDoubleSidedAttr(bool(double or self.s.materials[mat].double_sided))
        lo, hi = np.asarray(v).min(0), np.asarray(v).max(0)
        m.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(self.look(mat))
        return m

    def collider(self, path, c, visible=False):
        kind, size, R = c["kind"], c["size"], c["R"]
        if kind == "box":
            g = UsdGeom.Cube.Define(self.stage, path)
            g.CreateSizeAttr(2.0)
            _xform(g.GetPrim(), c["pos"], R, size[:3])
        elif kind in ("cylinder", "capsule"):
            g = (UsdGeom.Cylinder if kind == "cylinder" else UsdGeom.Capsule).Define(self.stage, path)
            g.CreateRadiusAttr(float(size[0]))
            g.CreateHeightAttr(float(2 * size[1]))
            g.CreateAxisAttr("Z")
            _xform(g.GetPrim(), c["pos"], R)
        else:
            g = UsdGeom.Sphere.Define(self.stage, path)
            g.CreateRadiusAttr(float(size[0]))
            _xform(g.GetPrim(), c["pos"], R)
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(self.phys_mat(c["friction"]),
                                                            UsdShade.Tokens.weakerThanDescendants, "physics")
        g.CreatePurposeAttr(UsdGeom.Tokens.guide)
        if not visible:
            g.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        return g

    # ------------------------------------------------------------------ parts
    def statics(self):
        for mat, buf in sorted(self.s.meshes.items()):
            arr = buf.arrays()
            if arr is not None:
                self.mesh(f"/World/Visual/{_safe(mat)}", mat, *arr)
        for i, c in enumerate(self.s.colliders):
            self.collider(f"/World/Collision/c{i}", c)
        for p in self.s.planes:
            cx, cy, cz = p["pos"]
            hx, hy = p["half"]
            v = np.array([[cx - hx, cy - hy, cz], [cx + hx, cy - hy, cz], [cx + hx, cy + hy, cz], [cx - hx, cy + hy, cz]])
            tile = self.s.materials[p["mat"]].tile
            t = v[:, :2] / tile
            self.mesh(f"/World/Water/{_safe(p['name'])}", p["mat"], v, np.tile([0, 0, 1.0], (4, 1)), t,
                      np.array([[0, 1, 2], [0, 2, 3]]))

    def terrain(self):
        for h in self.s.hfields:
            H = np.asarray(h["heights"], float)
            ny, nx = H.shape
            cx, cy = h["centre"]
            rx, ry = h["half"]
            xs = np.linspace(cx - rx, cx + rx, nx)
            ys = np.linspace(cy - ry, cy + ry, ny)
            X, Y = np.meshgrid(xs, ys)
            v = np.stack([X, Y, H], -1).reshape(-1, 3)
            idx = np.arange(nx * ny).reshape(ny, nx)
            a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
            f = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
            m = UsdGeom.Mesh.Define(self.stage, f"/World/Terrain/{_safe(h['name'])}_collision")
            m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
            m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
            m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.astype(np.int32).ravel()))
            m.CreatePurposeAttr(UsdGeom.Tokens.guide)
            m.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(m.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(self.phys_mat(h["friction"]),
                                                                UsdShade.Tokens.weakerThanDescendants, "physics")

    def bodies(self):
        def world_frame(b, parent_T):
            T = np.eye(4)
            T[:3, :3] = b.frame.R
            T[:3, 3] = b.frame.t
            return parent_T @ T

        def rec(b, parent_path, parent_T):
            T = world_frame(b, parent_T)
            path = f"/World/Articulated/{_safe(b.name)}"
            x = UsdGeom.Xform.Define(self.stage, path)
            _xform(x.GetPrim(), T[:3, 3], T[:3, :3])
            prim = x.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            mass = UsdPhysics.MassAPI.Apply(prim)
            mass.CreateMassAttr(float(b.mass))
            mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, b.com)))
            if b.inertia is not None:
                I = list(b.inertia)
                for k in range(3):
                    I[k] = min(I[k], 0.995 * (I[(k + 1) % 3] + I[(k + 2) % 3]))
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, I)))
            for mat, buf in sorted(b.meshes.items()):
                arr = buf.arrays()
                if arr is not None:
                    self.mesh(f"{path}/vis_{_safe(mat)}", mat, *arr)
            for i, c in enumerate(b.colliders):
                self.collider(f"{path}/col{i}", c)
            if b.joint:
                j = b.joint
                jt = j.get("type", "hinge")
                jp = f"{path}/joint"
                axis = np.asarray(j.get("axis", (0, 0, 1)), float)
                axis /= np.linalg.norm(axis)
                # joint frame: X along the joint axis (prismatic) / Z along the axis (revolute)
                if jt == "hinge":
                    joint = UsdPhysics.RevoluteJoint.Define(self.stage, jp)
                    ref = np.array([0, 0, 1.0])
                    joint.CreateAxisAttr("Z")
                    lo, hi = j.get("range", (-math.pi, math.pi))
                    joint.CreateLowerLimitAttr(math.degrees(lo))
                    joint.CreateUpperLimitAttr(math.degrees(hi))
                    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
                    drive.CreateDampingAttr(float(j.get("damping", 1.0)))
                    drive.CreateStiffnessAttr(0.0)
                else:
                    joint = UsdPhysics.PrismaticJoint.Define(self.stage, jp)
                    ref = np.array([1.0, 0, 0])
                    joint.CreateAxisAttr("X")
                    lo, hi = j.get("range", (0, 1))
                    joint.CreateLowerLimitAttr(float(lo))
                    joint.CreateUpperLimitAttr(float(hi))
                    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
                    if b.name == "lift_cabin":
                        drive.CreateStiffnessAttr(60000.0)
                        drive.CreateDampingAttr(8000.0)
                        drive.CreateTargetPositionAttr(0.0)
                    elif b.name.startswith("lift_gate"):
                        drive.CreateStiffnessAttr(2000.0)
                        drive.CreateDampingAttr(200.0)
                        drive.CreateTargetPositionAttr(float(j.get("qpos0", 0.0)))
                    else:
                        drive.CreateStiffnessAttr(0.0)
                        drive.CreateDampingAttr(float(j.get("damping", 1.0)))
                v = np.cross(ref, axis)
                if np.linalg.norm(v) < 1e-9:
                    Ra = np.eye(3) if ref @ axis > 0 else np.diag([1.0, -1.0, -1.0]) if jt != "hinge" else np.diag([-1.0, 1.0, -1.0])
                else:
                    ang = math.acos(max(-1.0, min(1.0, float(ref @ axis))))
                    k = v / np.linalg.norm(v)
                    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                    Ra = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
                # body0 = parent (or the world), body1 = this body; joint at the body origin
                if parent_path:
                    joint.CreateBody0Rel().SetTargets([parent_path])
                    rel = np.linalg.inv(parent_T) @ T
                    joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, rel[:3, 3])))
                    joint.CreateLocalRot0Attr(_quatf(rel[:3, :3] @ Ra))
                else:
                    joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, T[:3, 3])))
                    joint.CreateLocalRot0Attr(_quatf(T[:3, :3] @ Ra))
                joint.CreateBody1Rel().SetTargets([path])
                joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
                joint.CreateLocalRot1Attr(_quatf(Ra))
                prim.CreateAttribute("primus:initialJointPosition", Sdf.ValueTypeNames.Float).Set(
                    float(j.get("qpos0", 0.0)))
            for ch in b.children:
                rec(ch, path, T)

        for b in self.s.bodies:
            rec(b, None, np.eye(4))

    def lights(self, hdr_path, sun_dir, sky_rotation_deg):
        dome = UsdLux.DomeLight.Define(self.stage, "/World/Lights/Sky")
        hdr_dst = self.out / Path(hdr_path).name
        if not hdr_dst.exists():
            shutil.copy(hdr_path, hdr_dst)
        dome.CreateTextureFileAttr(hdr_dst.name)
        dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
        dome.CreateIntensityAttr(1000.0)
        _xform(dome.GetPrim(), (0, 0, 0), np.array([[math.cos(math.radians(sky_rotation_deg)),
                                                    -math.sin(math.radians(sky_rotation_deg)), 0],
                                                   [math.sin(math.radians(sky_rotation_deg)),
                                                    math.cos(math.radians(sky_rotation_deg)), 0], [0, 0, 1]]))
        sun = UsdLux.DistantLight.Define(self.stage, "/World/Lights/Sun")
        sun.CreateIntensityAttr(3000.0)
        sun.CreateAngleAttr(0.53)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.88))
        d = -np.asarray(sun_dir, float)                    # light travels along -sun_dir; DistantLight shines -Z
        z = -d / np.linalg.norm(d)
        x = np.cross([0, 0, 1.0], z)
        x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0])
        _xform(sun.GetPrim(), (0, 0, 50), np.c_[x, np.cross(z, x), z])
        for L in self.s.lights:
            if L.get("directional"):
                continue
            sp = UsdLux.SphereLight.Define(self.stage, f"/World/Lights/{_safe(L['name'])}")
            sp.CreateRadiusAttr(0.15)
            sp.CreateIntensityAttr(30000.0)
            sp.CreateColorAttr(Gf.Vec3f(1.0, 0.85, 0.65))
            _xform(sp.GetPrim(), L["pos"])

    def cameras(self, views):
        for name, (lookat, dist, az, el, fovy) in views.items():
            a, e = math.radians(az), math.radians(el)
            fwd = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
            pos = np.asarray(lookat, float) - fwd * dist
            z = -fwd                                      # USD cameras look down -Z
            x = np.cross([0, 0, 1.0], z)
            x /= np.linalg.norm(x) + 1e-12
            y = np.cross(z, x)
            cam = UsdGeom.Camera.Define(self.stage, f"/World/Cameras/{_safe(name)}")
            vfov = math.radians(fovy)
            cam.CreateVerticalApertureAttr(20.25)
            cam.CreateHorizontalApertureAttr(36.0)
            cam.CreateFocalLengthAttr(float(20.25 / (2 * math.tan(vfov / 2))))
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20000.0))
            _xform(cam.GetPrim(), pos, np.c_[x, y, z])

    def spawns(self):
        for name, (x, y, zf, heading) in self.s.spawns.items():
            p = UsdGeom.Xform.Define(self.stage, f"/World/Spawns/{_safe(name)}")
            _xform(p.GetPrim(), (x, y, zf), np.array([[math.cos(heading), -math.sin(heading), 0],
                                                     [math.sin(heading), math.cos(heading), 0], [0, 0, 1]]))
            p.GetPrim().CreateAttribute("primus:heading", Sdf.ValueTypeNames.Float).Set(float(heading))

    def save(self):
        self.stage.GetRootLayer().Save()
        return self.path


def export(scene, hdr_path, sun_dir, sky_rotation_deg, views):
    ex = Exporter(scene)
    ex.statics()
    ex.terrain()
    ex.bodies()
    ex.lights(hdr_path, sun_dir, sky_rotation_deg)
    ex.cameras(views)
    ex.spawns()
    return ex.save()
