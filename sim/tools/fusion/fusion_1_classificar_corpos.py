import adsk.core, adsk.fusion, json, sys, types
SRC = r'''
LEG = {'1': 'xp', '2': 'xm'}
ARMS = {'upper arm:1': 'armYp_xp', 'upper arm espelhado:2': 'armYm_xp', 'upper arm espelhado:1': 'armYp_xm', 'upper arm:2': 'armYm_xm'}
AXLES = {'1': 'armYp_xp', '3': 'armYm_xp', '2': 'armYp_xm', '4': 'armYm_xm'}
WHEEL_BODIES = ('Body5', 'Body6', 'Body7', 'Body10', 'Body11')
def link_of(path, bname):
    parts = list(path) if path else []
    top = parts[0] if parts else ''
    if top.startswith('lower new:'):
        leg = LEG[top.split(':')[1]]
        rest = parts[1:]
        if not rest:
            return ('wheel_' + leg) if bname in WHEEL_BODIES else ('shin_' + leg)
        r0 = rest[0]
        if r0.startswith('lower link :'): return 'linkYm_' + leg
        if r0.startswith('Fixadores - roda + link Y+'): return 'wheel_' + leg
        return 'shin_' + leg
    if top in ARMS: return ARMS[top]
    if top.startswith('Conjunto eixo joelho'): return AXLES[top.split(':')[1]]
    if top.startswith('Barriga + trem retratil'):
        r0 = parts[1] if len(parts) > 1 else ''
        if r0.startswith('Trem dianteiro'): return 'gear_front'
        if r0.startswith('Trem traseiro'): return 'gear_rear'
        return 'base'
    return 'base'
def leaf_bodies(root):
    for b in root.bRepBodies:
        yield '', b
    for o in root.allOccurrences:
        if not o.isVisible: continue
        if o.bRepBodies.count == 0: continue
        ch = []; x = o
        while x is not None:
            ch.append(x.name); x = x.assemblyContext
        ch.reverse()
        for b in o.bRepBodies:
            yield ch, b
'''
def run(context):
    m = types.ModuleType('simlib'); exec(SRC, m.__dict__); sys.modules['simlib'] = m
    app = adsk.core.Application.get()
    des = adsk.fusion.Design.cast(app.activeProduct)
    root = des.rootComponent
    st = {}
    n = 0
    for p, b in m.leaf_bodies(root):
        if not b.isVisible or not b.isSolid: continue
        L = m.link_of(p, b.name)
        mass = b.physicalProperties.mass
        s = st.setdefault(L, [0, 0.0])
        s[0] += 1; s[1] += mass
        n += 1
    print(json.dumps({'n': n, 'links': {k: [v[0], round(v[1], 3)] for k, v in sorted(st.items())}, 'total': round(sum(v[1] for v in st.values()), 3)}))
