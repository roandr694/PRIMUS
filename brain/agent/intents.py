"""Deterministic intent layer: common Portuguese commands parsed by rules, so they never wait for (or depend on)
the LLM (docs/home_autonomy_architecture.md §4.6). Returns tool calls in the same format the LLM produces, or None
when the sentence is not a simple command (then the LLM handles it).

    parse("vai para a cozinha")  -> [{"tool": "go_to", "args": {"room": "cozinha"}}]
    parse("fica do lado do sofá na sala") -> [{"tool": "go_to", "args": {"object": "sofá", "room": "sala"}}]
"""
import re
import unicodedata


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower().strip())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[!?.,;]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


WAKE = r"^((ei|oi|ola|o)[ ,]+)?(primus|robo|robozinho)\b[ ,]*"
GO = r"(vai|va|ir|anda|leva-me|me leva|navega|volta|vem|venha|segue|siga)"
ART = r"(?:a |o |as |os |pra |para |pro |ate |até |na |no |em |ao |à )*"


def _strip_wake(t):
    t = re.sub(WAKE, "", t).strip()
    return re.sub(r"^(ei|oi|ola)[ ,]+", "", t).strip()


def parse(text):
    t = _strip_wake(norm(text))
    if not t:
        return None
    # --- stop / cancel (also handled below the LLM by the keyword spotter)
    if re.fullmatch(r"(para|pare|parar|stop|chega|cancela|cancelar|espera|fica parado|fica ai)( agora)?( ja)?", t):
        return [{"tool": "stop", "args": {}}]
    # --- body height
    if re.search(r"\b(fica|fique|ficar|abaixa|abaixe|sobe|suba|levanta|levante)\b.*\b(mais )?(baixo|baixinho|agachado)\b", t) \
            or re.fullmatch(r"(abaixa|abaixe|agacha|agache)( um pouco)?", t):
        return [{"tool": "set_body", "args": {"height": "baixo" if "pouco" not in t else "-0.06"}}]
    if re.search(r"\b(fica|fique|ficar)\b.*\b(mais )?(alto|altinho|em pe alto)\b", t) or re.fullmatch(r"(cresce|estica)( um pouco)?", t):
        return [{"tool": "set_body", "args": {"height": "alto" if "pouco" not in t else "+0.06"}}]
    if re.search(r"\b(altura normal|altura media|fica normal)\b", t):
        return [{"tool": "set_body", "args": {"height": "normal"}}]
    # --- park / stand
    if re.search(r"\b(estaciona|estacione|deita|pousa|desliga as pernas|abaixa o trem)\b", t):
        return [{"tool": "park", "args": {}}]
    if re.fullmatch(r"(levanta|levante|fica de pe|acorda|em pe)( ai)?", t):
        return [{"tool": "stand", "args": {}}]
    # --- follow / come
    if re.search(r"\b(me segue|me siga|segue-me|siga-me|vem comigo|venha comigo|me acompanha|vem atras de mim)\b", t):
        return [{"tool": "follow", "args": {}}]
    if re.fullmatch(r"(vem aqui|vem ca|venha aqui|venha ca|aqui)", t):
        return [{"tool": "come_here", "args": {}}]
    # --- carry
    if re.search(r"\b(vou (botar|colocar|por)|coloquei|botei|pus)\b.*\b(em cima de voce|em voce|nas suas costas|em cima)\b", t):
        return [{"tool": "carry", "args": {"on": True}}]
    if re.search(r"\b(tirei|ja tirei|pode (parar|sair) de carregar|terminou de carregar)\b", t):
        return [{"tool": "carry", "args": {"on": False}}]
    # --- stairs / skills
    m = re.search(r"\b(sobe|suba|subir)\b.*\b(escada|degrau|degraus)\b", t)
    if m:
        return [{"tool": "cross", "args": {"kind": "stairs_up"}}]
    if re.search(r"\b(desce|desca|descer)\b.*\b(escada|degrau|degraus)\b", t):
        return [{"tool": "cross", "args": {"kind": "stairs_down"}}]
    if re.search(r"\b(pula|pule|pular)\b", t):
        return [{"tool": "cross", "args": {"kind": "gap"}}]
    # --- remember / label
    m = re.search(r"\b(lembra|lembre|memoriza|grava|salva)\b.*\b(lugar|ponto|aqui)\b.*\b(como|de) (.+)$", t)
    if m:
        return [{"tool": "remember_place", "args": {"name": m.group(4).strip()}}]
    m = re.search(r"\b(esta|essa|este|aqui) (e|eh) (a |o )?(.+)$", t)
    if m and len(m.group(4).split()) <= 4:
        return [{"tool": "label_room", "args": {"name": m.group(4).strip()}}]
    # --- questions
    if re.search(r"\b(onde (voce )?(esta|ta)|em que comodo)\b", t):
        return [{"tool": "where_am_i", "args": {}}]
    if re.search(r"\b(o que (voce )?(ve|esta vendo|ta vendo))\b", t):
        return [{"tool": "describe", "args": {}}]
    if re.search(r"\b(bateria|status|como (voce )?esta)\b", t):
        return [{"tool": "status", "args": {}}]
    # --- turn
    if re.fullmatch(r"(vira|gira|da uma volta)( para| pra)? (a )?(esquerda|direita)", t) or re.fullmatch(r"(vira|gira)( para| pra)? tras", t):
        ang = 90 if "esquerda" in t else (-90 if "direita" in t else 180)
        return [{"tool": "turn", "args": {"degrees": ang}}]
    # --- go to: "vai para a cozinha", "vai para a sala e fica do lado do sofá", "vai até a geladeira"
    m = re.match(GO + r"\b(.*)$", t)
    if m:
        rest = m.group(2).strip()
        rest = re.sub(r"^(para|pra|pro|ate|até|a|o|na|no|em|ao)\s+", "", rest)
        rest = re.sub(r"^(a|o|as|os)\s+", "", rest)
        rel = re.search(r"(?:e )?(?:fica|fique|para|pare|chega)?\s*(?:do lado|ao lado|perto|junto|na frente|em frente)\s+(?:d[aeo]s?\s+)?(.+)$", rest)
        if rel:
            obj = rel.group(1).strip()
            room_part = rest[:rel.start()].strip()
            room_part = re.sub(r"\s+e$", "", room_part).strip()
            m2 = re.search(r"^(.+?)\s+(?:na|no|da|do)\s+(.+)$", obj)
            room = room_part or None
            if m2 and room is None:
                obj, room = m2.group(1), m2.group(2)
            args = {"object": obj}
            if room:
                args["room"] = room
            return [{"tool": "go_to", "args": args}]
        if rest:
            m2 = re.search(r"^(.+?)\s+(?:na|no|da|do)\s+(sala|cozinha|quarto|suite|banheiro|varanda|escritorio|lavanderia)(.*)$", rest)
            if m2:
                return [{"tool": "go_to", "args": {"object": m2.group(1), "room": (m2.group(2) + m2.group(3)).strip()}}]
            return [{"tool": "go_to", "args": {"target": rest}}]
    m = re.match(r"(procura|ache|acha|encontra|encontre|onde (esta|fica|ta))\s+(a |o |minha |meu |as |os )?(.+)$", t)
    if m:
        return [{"tool": "find", "args": {"object": m.group(4).strip()}}]
    return None
