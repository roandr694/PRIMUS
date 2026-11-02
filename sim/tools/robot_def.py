"""Robot kinematic definition (from the joints of the Fusion assembly).
Units: mm in the CAD frame (X right, Y forward, Z up). All joints rotate about +X."""
LEGS = {
    'xp': dict(hipYp=(128.35, 90.0, -0.35), hipYm=(214.35, -50.0, -0.35),
               kneeYp=(115.35, 266.08, -132.24), kneeYm=(201.35, -213.92, -147.08),
               foot=(203.35, 33.95, -394.18)),
    'xm': dict(hipYp=(-151.65, 90.0, -0.35), hipYm=(-237.65, -50.0, -0.35),
               kneeYp=(-138.65, 266.08, -132.24), kneeYm=(-224.65, -213.92, -147.08),
               foot=(-226.65, 33.95, -394.18)),
}
GEAR = {
    'gear_front': dict(hinge=(-9.65, 80.0, -154.0), range_deg=(-105.0, 0.0)),
    'gear_rear': dict(hinge=(-10.9, -64.0, -154.0), range_deg=(0.0, 105.0)),
}
# origin of each body (= joint point with its parent); the base is at the CAD origin
def link_origin(name):
    if name == 'base': return (0.0, 0.0, 0.0)
    if name in GEAR: return GEAR[name]['hinge']
    kind, leg = name.rsplit('_', 1)
    L = LEGS[leg]
    return {'armYp': L['hipYp'], 'armYm': L['hipYm'], 'shin': L['kneeYp'], 'linkYm': L['kneeYm'], 'wheel': L['foot']}[kind]

# mass corrections: imported models / no real material in the CAD
# (top-level, sub, body) -> real total mass of the group [kg] or density factor
MASS_OVERRIDES = [
    # (filtro, modo, valor, motivo)
    (dict(top='M18 Battery 5Ah'), 'total', 0.68, 'Milwaukee M18 5.0 Ah battery ~0.68 kg (imported model was steel)'),
    (dict(top='Estrutura perfis 20x20'), 'scale', 2.70 / 7.85, '20x20 profiles are aluminium (CAD in steel)'),
    (dict(top='Orin Nano dev kit + case'), 'total', 0.28, 'Jetson Orin Nano dev kit + case ~0.28 kg (model in steel)'),
    (dict(top='rplidara1'), 'total', 0.17, 'RPLIDAR A1 ~0.17 kg (model in steel)'),
    (dict(top='Raspberry Pi 5'), 'total', 0.06, 'Raspberry Pi 5 + heatsink ~0.06 kg (model in steel)'),
    (dict(top='lower new', sub='', body='Body10'), 'scale', 1.27 / 7.85, 'wheel rim/hub assumed printed in PETG (CAD had no material, was steel) - CONFIRM'),
]
