"""The conversational agent: text in (chat page now, speech later), actions and a short reply out.

    agent = Agent(stack, llm=TransformersLLM())       # llm=None: rules only
    reply = agent.handle("vai para a sala e fica do lado do sofá")
    agent.poll()  -> text when the running behaviour finishes ("cheguei: sofá")

Order: the deterministic intent layer first (never waits for the LLM, works without it); the LLM only for what the
rules do not understand. Tool calls are validated by the ToolBox (map, robot state) before anything moves; several
calls become one Sequence behaviour. Physical safety never depends on the LLM (stop is handled below it).
"""
import json

from .. import behaviors as B
from .intents import parse
from .robot_card import robot_card
from .llm import extract_json
from .tools import TOOLS, ToolBox, tools_text

SYSTEM = """{card}

Mapa da casa que você conhece:
{scene}

Seu estado agora: {state}

Ferramentas (ações) disponíveis:
{tools}

Responda SOMENTE com um objeto JSON, sem texto fora dele:
{{"say": "<frase curta em português>", "calls": [{{"tool": "<nome>", "args": {{...}}}}]}}
- "calls" pode ser vazio (só conversa) ou ter várias ações em sequência.
- Use nomes de cômodos e objetos como o dono falou (em português); para objetos você pode usar o nome em inglês.
- Se o pedido for impossível para o seu corpo, explique em "say" e não chame ferramentas.
- "Leva X para Y" significa: alguém coloca X em cima de você (carry on), você vai até Y; você não pega nada.

Exemplos:
pedido: "me espera perto da geladeira" -> {{"say": "Vou te esperar na geladeira.", "calls": [{{"tool": "go_to", "args": {{"object": "geladeira"}}}}]}}
pedido: "leva esse copo para a cozinha e volta" -> {{"say": "Ok, coloque o copo em cima de mim.", "calls": [{{"tool": "carry", "args": {{"on": true}}}}, {{"tool": "go_to", "args": {{"room": "cozinha"}}}}, {{"tool": "go_to", "args": {{"place": "aqui"}}}}]}}
pedido: "dá uma olhada no quarto" -> {{"say": "Vou dar uma olhada.", "calls": [{{"tool": "go_to", "args": {{"room": "quarto"}}}}, {{"tool": "describe", "args": {{}}}}]}}
pedido: "agacha" -> {{"say": "Agachando.", "calls": [{{"tool": "set_body", "args": {{"height": "baixo"}}}}]}}"""


class Agent:
    def __init__(self, stack, llm=None, history=6):
        self.stack, self.llm, self.tools = stack, llm, ToolBox(stack)
        self.history, self.turns = history, []
        self.log = []                        # (user text, source "rules"/"llm", calls, reply)
        self._announced = None

    def state_text(self):
        st = self.stack
        r = st.scene.room_of(st.pose()[:2], st.floor) if st.scene is not None else None
        b = st.behaviour
        parts = [f"em {r.label or r.pt}" if r else "local desconhecido", f"perna {st.leg * 100:.0f} cm"]
        if st.carrying:
            parts.append("carregando um objeto")
        if b is not None and not b.done:
            parts.append(f"fazendo: {b.progress() or b.name}")
        s = st.brain.state
        if s is not None:
            parts.append(f"bateria {s.battery * 100:.0f}%")
        return ", ".join(parts)

    def _llm_calls(self, text):
        st = self.stack
        sys_msg = SYSTEM.format(card=robot_card(), scene=st.scene.to_text(st) if st.scene else "(sem mapa)",
                                state=self.state_text(), tools=tools_text())
        msgs = [{"role": "system", "content": sys_msg}] + self.turns[-self.history:] + [{"role": "user", "content": text}]
        err = None
        for _ in range(2):
            raw = self.llm.chat(msgs if err is None else msgs + [{"role": "assistant", "content": raw},
                                                                  {"role": "user", "content": f"Erro: {err}. Responda só o JSON."}])
            try:
                obj = extract_json(raw)
                calls = obj.get("calls") or []
                if not isinstance(calls, list):
                    raise ValueError("'calls' deve ser uma lista")
                norm = []
                for c in calls:                       # tolerate {"go_to": {...}} for {"tool": "go_to", "args": {...}}
                    if isinstance(c, dict) and "tool" not in c and len(c) == 1:
                        k, v = next(iter(c.items()))
                        c = {"tool": k, "args": v if isinstance(v, dict) else {}}
                    norm.append(c)
                calls = norm
                for c in calls:
                    if c.get("tool") not in TOOLS:
                        raise ValueError(f"ferramenta desconhecida {c.get('tool')}")
                return str(obj.get("say", "")), calls, raw
            except (ValueError, AttributeError) as e:
                err = str(e)
        return "desculpe, não entendi", [], raw

    def handle(self, text):
        st = self.stack
        if st.scene is not None and st.brain.state is not None and st.floor is not None and \
                (st.behaviour is None or st.behaviour.done):
            x, y, th = st.pose()                      # "volta" / "aqui" = where the robot was when asked
            st.scene.add_place("aqui", (x, y), th, st.floor)
        calls = parse(text)
        source, say = "rules", ""
        if calls is None and self.llm is not None:
            say, calls, raw = self._llm_calls(text)
            source = "llm"
        if calls is None:
            return "não entendi; pode repetir de outro jeito?"
        behaviours, texts = [], []
        for c in calls:
            if c.get("tool") == "say":
                say = say or c.get("args", {}).get("text", "")
                continue
            b, t = self.tools.execute(c)
            if b is not None:
                behaviours.append(b)
            if t and (b is not None or not say):
                texts.append(t)
            elif t and b is None:
                texts.append(t)
        if behaviours:
            stop_first = isinstance(behaviours[0], B.Stop)
            self.stack.start(behaviours[0] if len(behaviours) == 1 else B.Sequence(behaviours))
            if stop_first:
                self.stack.nav.cancel()
            self._announced = None
        reply = say or ("; ".join(texts) if texts else "ok")
        if source == "llm" and texts and say and any("não" in t for t in texts):
            reply = f"{say} ({'; '.join(t for t in texts if 'não' in t)})"
        self.turns += [{"role": "user", "content": text},
                       {"role": "assistant", "content": json.dumps({"say": reply, "calls": calls}, ensure_ascii=False)}]
        self.log.append((text, source, calls, reply))
        return reply

    def poll(self):
        """Result of the behaviour that just finished (once), else None."""
        b = self.stack.behaviour
        msg = self.stack.message
        if msg and msg != self._announced and (b is None or b.done):
            self._announced = msg
            return msg
        return None
