"""The agent's tools: validated against the map and the robot's state, each one starts a behaviour or answers
directly. The same schema is given to the LLM (JSON) and used by the intent layer.

    tb = ToolBox(stack)
    behaviour, text = tb.execute({"tool": "go_to", "args": {"room": "cozinha"}})
"""
import math

import numpy as np

from .. import behaviors as B

TOOLS = {
    "go_to": ("ir até um cômodo, um lugar nomeado ou perto de um objeto",
              {"room": "nome do cômodo (opcional)", "object": "objeto para ficar ao lado (opcional)",
               "place": "lugar nomeado (opcional)", "target": "texto livre se não souber separar"}),
    "find": ("procurar um objeto pela casa e ir até ele", {"object": "o objeto"}),
    "follow": ("seguir a pessoa que está na frente", {}),
    "come_here": ("vir até a pessoa que chamou", {}),
    "stop": ("parar tudo agora", {}),
    "set_body": ("mudar a altura do corpo", {"height": "baixo | normal | alto | +0.05 | -0.05 (metros)"}),
    "park": ("estacionar (descer o trem de pouso)", {}),
    "stand": ("levantar e ficar em pé", {}),
    "carry": ("ligar/desligar o modo carga (alguém pôs um objeto em cima)", {"on": "true | false"}),
    "cross": ("atravessar o obstáculo à frente", {"kind": "stairs_up | stairs_down | curb | gap | overhang | any"}),
    "turn": ("virar no lugar", {"degrees": "graus, positivo = esquerda"}),
    "remember_place": ("guardar a posição atual com um nome", {"name": "nome do lugar"}),
    "label_room": ("dar um nome ao cômodo atual", {"name": "nome do cômodo"}),
    "where_am_i": ("dizer onde estou", {}),
    "describe": ("dizer o que estou vendo", {}),
    "status": ("bateria e estado", {}),
    "say": ("só responder, sem agir", {"text": "a resposta"}),
}


def tools_text():
    lines = []
    for name, (desc, args) in TOOLS.items():
        a = ", ".join(f'"{k}": {v}' for k, v in args.items())
        lines.append(f"- {name}: {desc}" + (f"; args {{{a}}}" if a else ""))
    return "\n".join(lines)


class ToolBox:
    def __init__(self, stack):
        self.stack = stack

    def execute(self, call):
        name = call.get("tool")
        args = call.get("args") or {}
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return None, f"(não conheço a ação '{name}')"
        try:
            return fn(**args)
        except TypeError as e:
            return None, f"(argumentos inválidos para {name}: {e})"

    # ------------------------------------------------------------------------------------------------ motion
    def t_go_to(self, room=None, object=None, place=None, target=None, **_):
        st, sg = self.stack, self.stack.scene
        if sg is None:
            return None, "ainda não tenho o mapa da casa"
        if target and not (room or object or place):
            # free text: a known place, then a room, then an object
            if sg.places.get(target.lower()) or any(p.name.lower() == target.lower() for p in sg.places.values()):
                place = target
            elif sg.find_room(target, st.floor) is not None:
                room = target
            else:
                object = target
        goal, why = sg.resolve(st, room=room, obj=object, place=place)
        if goal is None:
            if object and st.memory is not None:              # not seen yet: search for it
                cls = sg.object_class(object)
                rooms = [sg.find_room(room, st.floor)] if room else None
                return B.Find(cls, object, [r for r in (rooms or []) if r]), f"ainda não vi {object}, vou procurar"
            return None, why
        if goal.floor != st.floor:
            return None, f"{goal.label} fica em outro andar; ainda não sei mudar de andar sozinho, me leve até a escada"
        return B.GoTo(goal), f"indo para {goal.label}"

    def t_find(self, object, **_):
        st, sg = self.stack, self.stack.scene
        cls = sg.object_class(object)
        rooms = sorted(sg.all_rooms(), key=lambda r: -r.area) if sg else []
        return B.Find(cls, object, rooms), f"vou procurar {object}"

    def t_follow(self, **_):
        st = self.stack
        if st.people is None:
            return None, "não consigo ver pessoas agora"
        now = st.brain.state.stamp
        t = st.people.lock(now, st.pose())
        if t is None:
            return B.Follow(), "não estou te vendo; fique na minha frente que eu te sigo"
        return B.Follow(), "ok, estou te seguindo"

    def t_come_here(self, **_):
        return self.t_follow()

    def t_stop(self, **_):
        return B.Stop(), "parando"

    def t_turn(self, degrees=90, **_):
        return B.Turn(math.radians(float(degrees))), "virando"

    def t_set_body(self, height="normal", **_):
        h = str(height).strip().lower()
        if h.startswith(("+", "-")):
            return B.SetBody(delta=float(h)), "ajustando a altura"
        if h in B.HEIGHTS:
            return B.SetBody(leg=B.HEIGHTS[h]), "ajustando a altura"
        try:
            v = float(h)
            return B.SetBody(leg=v if v < 1 else v / 100), "ajustando a altura"
        except ValueError:
            return None, f"não entendi a altura '{height}'"

    def t_park(self, **_):
        return B.Park(True), "estacionando"

    def t_stand(self, **_):
        return B.Park(False), "levantando"

    def t_carry(self, on=True, **_):
        on = on if isinstance(on, bool) else str(on).lower() in ("true", "1", "sim", "yes")
        st = self.stack
        st.carrying = on
        if on:
            st.leg = min(st.leg, 0.36)
            kg = st.brain.state.payload_kg if st.brain.state is not None else 0.0
            extra = f" (sinto uns {kg * 1000:.0f} g a mais)" if kg > 0.05 else ""
            return None, f"modo carga ligado{extra}: vou devagar e sem pular degraus"
        return None, "modo carga desligado"

    def t_cross(self, kind="any", **_):
        return B.Cross(kind), "vou tentar atravessar"

    # ------------------------------------------------------------------------------------------------ memory
    def t_remember_place(self, name, **_):
        st = self.stack
        x, y, th = st.pose()
        st.scene.add_place(name, (x, y), th, st.floor)
        return None, f"guardei este lugar como '{name}'"

    def t_label_room(self, name, **_):
        st = self.stack
        r = st.scene.label_room(st.pose()[:2], st.floor, name)
        if r is None:
            return None, "não sei em que cômodo estou para dar esse nome"
        return None, f"ok, este cômodo agora se chama '{name}'"

    # ------------------------------------------------------------------------------------------------ questions
    def t_where_am_i(self, **_):
        st = self.stack
        r = st.scene.room_of(st.pose()[:2], st.floor) if st.scene else None
        if r is None:
            return None, "não tenho certeza de onde estou"
        return None, f"estou em {r.label or r.pt}"

    def t_describe(self, **_):
        st = self.stack
        if st.memory is None:
            return None, "não estou com a visão ligada"
        x, y, th = st.pose()
        near = []
        for o in st.memory.confirmed(st.floor):
            d = o.centroid[:2] - np.array([x, y])
            dist = float(np.linalg.norm(d))
            ang = abs((math.atan2(d[1], d[0]) - th + math.pi) % (2 * math.pi) - math.pi)
            if dist < 4.0 and ang < math.radians(45):
                near.append((dist, o))
        if not near:
            return None, "não vejo nada que eu conheça na minha frente"
        from ..semantic.scene_graph import OBJECT_WORDS
        pt = {v: k for k, v in OBJECT_WORDS.items() if " " not in k or True}
        near.sort(key=lambda p: p[0])
        items = [f"{pt.get(o.cls, o.cls)} a {d:.1f} m" for d, o in near[:4]]
        return None, "vejo " + ", ".join(items)

    def t_status(self, **_):
        s = self.stack.brain.state
        if s is None:
            return None, "ainda estou ligando"
        modes = ("estacionado", "levantando", "em pé", "pousando", "caído", "me recuperando")
        txt = f"bateria {s.battery * 100:.0f} %, {modes[s.mode] if s.mode < len(modes) else 'ok'}"
        if self.stack.carrying:
            txt += ", modo carga"
        return None, txt

    def t_say(self, text="", **_):
        return None, text
