"""Multi-agent orchestration with LangGraph:
Router -> Retriever -> Analyst -> Guardrail -> (Finalize | Human escalation) ; Router -> Blocked."""
import os, re, time, socket
from typing import TypedDict
from langgraph.graph import StateGraph, END

class S(TypedDict, total=False):
    question: str; intent: str; hits: list; answer: str
    flags: list; confidence: float; status: str; trace: list; general: bool

INJECTION = re.compile(r"ignore (all |previous |prior )?instructions|system prompt|reveal.*(key|secret|password)", re.I)
PII = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
T = lambda s, agent, msg: s.get("trace", []) + [{"agent": agent, "msg": msg}]

_OL = {"t": 0.0, "up": False}
def ollama_up():
    """Cheap cached check (0.3s max) so a missing Ollama never slows requests."""
    if time.time() - _OL["t"] > 10:
        try: socket.create_connection(("127.0.0.1", 11434), timeout=0.3).close(); up = True
        except OSError: up = False
        _OL.update(t=time.time(), up=up)
    return _OL["up"]

def llm(prompt):
    """Local Ollama (free) if running, else Azure OpenAI / OpenAI if configured; None -> offline extractive mode."""
    try:
        if not ollama_up(): raise OSError
        import json, urllib.request
        req = urllib.request.Request("http://127.0.0.1:11434/api/generate", headers={"Content-Type": "application/json"},
              data=json.dumps({"model": os.getenv("OLLAMA_MODEL", "llama3.2"), "prompt": prompt, "stream": False,
                               "options": {"temperature": 0}}).encode())
        return json.load(urllib.request.urlopen(req, timeout=180))["response"]
    except Exception:
        pass
    try:
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            from openai import AzureOpenAI
            c, m = AzureOpenAI(api_version="2024-06-01"), os.environ["AZURE_OPENAI_DEPLOYMENT"]
        elif os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            c, m = OpenAI(), os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        else:
            return None
        r = c.chat.completions.create(model=m, temperature=0, messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content
    except Exception:
        return None

def build(index):
    def router(s):
        if INJECTION.search(s["question"]):
            return {"intent": "blocked", "flags": ["prompt_injection"], "trace": T(s, "Router", "Prompt-injection pattern detected -> request blocked")}
        intent = "compare" if re.search(r"\b(compare|difference|versus|vs)\b", s["question"], re.I) else "lookup"
        return {"intent": intent, "flags": [], "trace": T(s, "Router", f"Intent = {intent}")}

    def retriever(s):
        hits = [h for h in index.search(s["question"], k=6 if s["intent"] == "compare" else 4) if h["score"] > 0.05]
        top = hits[0]["score"] if hits else 0
        return {"hits": hits, "trace": T(s, "Retriever", f"BM25 + dense -> RRF -> rerank: {len(hits)} relevant passages, top score {top}")}

    def analyst(s):
        hits = s["hits"]
        if not hits:
            gen = llm(f"Answer the question accurately in 3-5 sentences.\n\nQuestion: {s['question']}")
            if gen:
                return {"answer": gen.strip(), "confidence": 0.5, "general": True,
                        "trace": T(s, "Analyst", "No policy evidence -> general-knowledge answer from LLM (not from documents)")}
            return {"answer": "No supporting passage was found in the policy documents, and no LLM is running for general questions. "
                              "Start Ollama to enable general answers. Escalating to a human reviewer.",
                    "confidence": 0.0, "trace": T(s, "Analyst", "No evidence and no LLM available -> refusing to answer")}
        conf = round(min(1.0, hits[0]["score"] * 1.3), 2)
        ctx = "\n".join(f"[{i+1}] ({h['doc']}) {h['text']}" for i, h in enumerate(hits))
        ans = llm(f"Answer ONLY from the sources and cite them like [1]. If unsupported, say so.\n\nSources:\n{ctx}\n\nQuestion: {s['question']}")
        mode = "LLM grounded generation" if ans else "offline extractive mode"
        ans = ans or " ".join(f"{h['text']} [{i+1}]" for i, h in enumerate(hits[:2]))
        return {"answer": ans, "confidence": conf, "trace": T(s, "Analyst", f"{mode}, confidence {conf}")}

    def guardrail(s):
        flags, a = list(s["flags"]), s["answer"]
        if PII.search(a): a = PII.sub("[REDACTED]", a); flags.append("pii_redacted")
        if s.get("general"): flags.append("general_knowledge")
        else:
            if not re.search(r"\[\d+\]", a): flags.append("no_citation")
            if s["confidence"] < 0.45: flags.append("low_confidence")
        return {"answer": a, "flags": flags, "trace": T(s, "Guardrail", f"Output validation flags: {flags or 'none'}")}

    route = lambda s: "escalate" if {"no_citation", "low_confidence"} & set(s["flags"]) else "finalize"
    finalize = lambda s: {"status": "answered", "trace": T(s, "Finalize", "General answer released (not source-grounded)" if s.get("general") else "Grounded, cited answer released")}
    escalate = lambda s: {"status": "needs_human_review", "trace": T(s, "Escalation", "Held for human-in-the-loop review")}
    blocked = lambda s: {"status": "blocked", "answer": "Request blocked by safety policy.", "trace": T(s, "Blocked", "Refused")}

    g = StateGraph(S)
    for name, fn in [("router", router), ("retriever", retriever), ("analyst", analyst), ("guardrail", guardrail),
                     ("finalize", finalize), ("escalate", escalate), ("blocked", blocked)]:
        g.add_node(name, fn)
    g.set_entry_point("router")
    g.add_conditional_edges("router", lambda s: "blocked" if s["intent"] == "blocked" else "retriever")
    g.add_edge("retriever", "analyst"); g.add_edge("analyst", "guardrail")
    g.add_conditional_edges("guardrail", route)
    for n in ("finalize", "escalate", "blocked"): g.add_edge(n, END)
    return g.compile()
