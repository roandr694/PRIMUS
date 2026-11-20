"""The owner's page (phone / desktop / Quest browser), served by the robot: chat with the agent, the house map with
rooms, objects and the robot, the camera view and the status (docs/home_autonomy_architecture.md §4.6, D3: phone /
web first). Speech will plug into the same /say endpoint (push-to-talk audio -> ASR on the Jetson).

    app = make_app(hub)        # hub: object with handle(text) -> reply, state() -> dict, map_png() / camera_jpg() -> bytes
    aiohttp.web.run_app(app, port=8090)
"""
import json

from aiohttp import web

PAGE = """<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>PRIMUS</title>
<style>
:root{--bg:#f6f5f2;--fg:#1d1d1f;--card:#fff;--line:#ddd;--acc:#2f6fdf;--me:#e8f0fe;--bot:#f1f1f1}
@media (prefers-color-scheme: dark){:root{--bg:#141416;--fg:#ececec;--card:#1e1e22;--line:#333;--acc:#6ea0ff;--me:#22314d;--bot:#2a2a2e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
header b{font-size:18px}#status{opacity:.8;font-size:13px}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.2fr);gap:12px;padding:12px 16px}
@media (max-width:800px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;min-width:0}
#log{height:52vh;overflow:auto;display:flex;flex-direction:column;gap:6px}
.msg{padding:7px 10px;border-radius:10px;max-width:85%}.me{background:var(--me);align-self:flex-end}.bot{background:var(--bot)}
form{display:flex;gap:8px;margin-top:8px}input{flex:1;min-width:0;padding:10px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--fg);font-size:16px}
button{padding:10px 14px;border:0;border-radius:8px;background:var(--acc);color:#fff;font-size:15px}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}.chips button{background:var(--bot);color:var(--fg);padding:6px 10px;font-size:13px}
img{width:100%;border-radius:8px;display:block;image-rendering:pixelated}
</style></head><body>
<header><b>PRIMUS</b><span id="status">conectando…</span></header>
<main>
<section class="card"><div id="log"></div>
<form id="f"><input id="t" placeholder="fale com o robô… (ex.: vai para a cozinha)" autocomplete="off"><button>Enviar</button></form>
<div class="chips"><button>vai para a cozinha</button><button>vai para a sala e fica do lado do sofá</button>
<button>fica mais baixo</button><button>fica mais alto</button><button>me segue</button><button>onde você está?</button>
<button>o que você está vendo?</button><button>para</button></div></section>
<section class="card"><img id="cam" alt="câmera do robô"><div style="height:8px"></div><img id="map" alt="mapa da casa"></section>
</main>
<script>
const log=document.getElementById('log'),t=document.getElementById('t');
function add(cls,txt){const d=document.createElement('div');d.className='msg '+cls;d.textContent=txt;log.appendChild(d);log.scrollTop=1e9}
async function say(x){if(!x)return;add('me',x);t.value='';
 const r=await fetch('/say',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({text:x})});
 const j=await r.json();add('bot',j.reply)}
document.getElementById('f').onsubmit=e=>{e.preventDefault();say(t.value.trim())};
document.querySelectorAll('.chips button').forEach(b=>b.onclick=()=>say(b.textContent));
async function poll(){try{const s=await (await fetch('/state')).json();
 document.getElementById('status').textContent=s.line;(s.events||[]).forEach(e=>add('bot',e))}catch(e){}
 document.getElementById('cam').src='/camera.jpg?'+Date.now();document.getElementById('map').src='/map.png?'+Date.now()}
setInterval(poll,700);poll();
</script></body></html>"""


def make_app(hub):
    async def index(_):
        return web.Response(text=PAGE, content_type="text/html")

    async def say(req):
        body = await req.json()
        reply = hub.handle(str(body.get("text", ""))[:300])
        return web.json_response({"reply": reply})

    async def state(_):
        return web.json_response(hub.state())

    async def map_png(_):
        return web.Response(body=hub.map_png(), content_type="image/png")

    async def camera(_):
        return web.Response(body=hub.camera_jpg(), content_type="image/jpeg")

    app = web.Application()
    app.add_routes([web.get("/", index), web.post("/say", say), web.get("/state", state),
                    web.get("/map.png", map_png), web.get("/camera.jpg", camera)])
    return app


def render_map(stack, size=520):
    """PNG of the current floor: free / occupied, rooms tinted and named, confirmed objects, the robot and its path."""
    import io

    import numpy as np
    from PIL import Image, ImageDraw
    fid = stack.slam.floor
    if fid is None:
        return b""
    g = stack.slam.floors[fid].grid
    bb = g.known_bbox(pad=0.5)
    if bb is None:
        return b""
    (x0, y0), (x1, y1) = bb
    i0, j0 = g.ij(np.array([x0, y0]))
    i1, j1 = g.ij(np.array([x1, y1]))
    i0, j0, i1, j1 = max(i0, 0), max(j0, 0), min(i1, g.ny), min(j1, g.nx)
    L = g.L[i0:i1, j0:j1]
    img = np.full(L.shape + (3,), 150, np.uint8)
    img[L < -0.5] = (242, 242, 238)
    img[L > 0.8] = (40, 40, 40)
    sg = stack.scene
    palette = [(255, 214, 165), (189, 224, 254), (202, 255, 191), (255, 198, 255), (253, 255, 182), (155, 246, 255),
               (255, 173, 173), (214, 204, 255)]
    if sg is not None and fid in sg.regions:
        lab = sg.regions[fid][i0:i1, j0:j1]
        for k, r in enumerate(sg.rooms[fid]):
            m = (lab == r.id) & (L < -0.5)
            img[m] = (0.55 * img[m] + 0.45 * np.array(palette[k % len(palette)])).astype(np.uint8)
    im = Image.fromarray(img[::-1])
    s = size / max(im.size)
    im = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    H = im.size[1]

    def px(x, y):
        return ((x - g.origin[0]) / g.res - j0) * s, H - ((y - g.origin[1]) / g.res - i0) * s

    if stack.memory is not None:
        for o in stack.memory.confirmed(fid):
            u, v = px(*o.centroid[:2])
            dr.ellipse([u - 3, v - 3, u + 3, v + 3], fill=(220, 90, 30))
    if sg is not None and fid in sg.rooms:
        for r in sg.rooms[fid]:
            u, v = px(*r.centroid)
            dr.text((u - 20, v - 6), (r.label or r.pt)[:18], fill=(20, 20, 20))
    if stack.nav.path is not None and stack.nav.status == "driving":
        pts = [px(*p) for p in stack.nav.path]
        dr.line(pts, fill=(47, 111, 223), width=2)
    x, y, th = stack.pose()
    u, v = px(x, y)
    import math
    dr.ellipse([u - 6, v - 6, u + 6, v + 6], fill=(47, 111, 223))
    dr.line([u, v, u + 12 * math.cos(th), v - 12 * math.sin(th)], fill=(255, 255, 255), width=2)
    if stack.people is not None:
        for t in stack.people.tracks.values():
            u, v = px(*t.xy)
            dr.rectangle([u - 4, v - 4, u + 4, v + 4], outline=(200, 0, 120), width=2)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def to_jpeg(rgb, q=70):
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=q)
    return buf.getvalue()


_ = json
