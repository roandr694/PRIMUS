import adsk.core, adsk.fusion, json, sys, os, struct, time
OUT = os.path.expanduser('~/Desktop/robo_sim_export')
def color_of(app_, cache):
    if app_ is None: return (0.6, 0.6, 0.6)
    k = app_.name
    if k in cache: return cache[k]
    c = None
    try:
        for p in app_.appearanceProperties:
            cp = adsk.core.ColorProperty.cast(p)
            if cp and cp.value:
                v = cp.value; c = (v.red / 255.0, v.green / 255.0, v.blue / 255.0); break
    except: pass
    if c is None: c = (0.6, 0.6, 0.6)
    cache[k] = c
    return c
def run(context):
    t0 = time.time()
    import simlib
    app = adsk.core.Application.get()
    des = adsk.fusion.Design.cast(app.activeProduct)
    root = des.rootComponent
    os.makedirs(os.path.join(OUT, 'raw'), exist_ok=True)
    st = getattr(sys, '_claude_exp', None)
    if st is None:
        bodies = [(p, b) for p, b in simlib.leaf_bodies(root) if b.isVisible and b.isSolid]
        st = {'bodies': bodies, 'i': 0, 'meta': [], 'buf': {}, 'cc': {}, 'ntri': 0}
        sys._claude_exp = st
    bodies = st['bodies']
    LQ = adsk.fusion.TriangleMeshQualityOptions.LowQualityTriangleMesh
    NQ = adsk.fusion.TriangleMeshQualityOptions.NormalQualityTriangleMesh
    while st['i'] < len(bodies) and time.time() - t0 < 40:
        p, b = bodies[st['i']]; st['i'] += 1
        try:
            L = simlib.link_of(p, b.name)
            top = (p[0].rsplit(':', 1)[0] if p else 'root')
            sub = (p[1].rsplit(':', 1)[0] if len(p) > 1 else '')
            pp = b.physicalProperties
            ok, xx, yy, zz, xy, yz, xz = pp.getXYZMomentsOfInertia()
            c = pp.centerOfMass
            col = color_of(b.appearance, st['cc'])
            bb = b.boundingBox
            diag = bb.minPoint.distanceTo(bb.maxPoint)
            calc = b.meshManager.createMeshCalculator()
            calc.setQuality(NQ if diag > 3.0 else LQ)
            mesh = calc.calculate()
            xyz = mesh.nodeCoordinatesAsDouble; idx = mesh.nodeIndices
            key = L + '|%.2f,%.2f,%.2f' % (round(col[0] * 8) / 8, round(col[1] * 8) / 8, round(col[2] * 8) / 8)
            buf = st['buf'].setdefault(key, [])
            for t in range(0, len(idx), 3):
                a, b2, c2 = idx[t], idx[t + 1], idx[t + 2]
                buf.append((xyz[3 * a] * 10, xyz[3 * a + 1] * 10, xyz[3 * a + 2] * 10, xyz[3 * b2] * 10, xyz[3 * b2 + 1] * 10, xyz[3 * b2 + 2] * 10, xyz[3 * c2] * 10, xyz[3 * c2 + 1] * 10, xyz[3 * c2 + 2] * 10))
            st['ntri'] += len(idx) // 3
            st['meta'].append({'link': L, 'top': top, 'sub': sub, 'body': b.name, 'mat': b.material.name if b.material else None, 'app': b.appearance.name if b.appearance else None,
                               'mass': pp.mass, 'vol': pp.volume, 'com': [c.x, c.y, c.z], 'I': [xx, yy, zz, xy, yz, xz], 'diag': diag})
        except Exception as e:
            st['meta'].append({'err': str(e)[:100], 'i': st['i']})
    done = st['i'] >= len(bodies)
    if done and not st.get('written'):
        for key, tris in st['buf'].items():
            L, col = key.split('|')
            fn = '%s__%s.stl' % (L, col.replace(',', '_').replace('.', 'p'))
            with open(os.path.join(OUT, 'raw', fn), 'wb') as f:
                f.write(b'robo bipedal export'.ljust(80, b' '))
                f.write(struct.pack('<I', len(tris)))
                for tr in tris:
                    f.write(struct.pack('<12fH', 0, 0, 0, *tr, 0))
        with open(os.path.join(OUT, 'raw', 'bodies.json'), 'w') as f:
            json.dump(st['meta'], f)
        st['written'] = True
    print(json.dumps({'i': st['i'], 'n': len(bodies), 'ntri': st['ntri'], 'files': len(st['buf']), 'written': st.get('written', False), 'errs': [m for m in st['meta'] if 'err' in m][:3], 't': round(time.time() - t0)}))
