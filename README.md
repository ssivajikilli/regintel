# RegIntel – Multi-Agent Regulatory Intelligence Platform

Production-style GenAI system for compliance teams: ask a policy question, get a **source-cited** answer produced by a
**LangGraph multi-agent pipeline**, with **guardrails** and **human-in-the-loop escalation**. Mirrors my work in regulated
industries (HIPAA / Basel / AML) – hybrid RAG, agent orchestration, Responsible AI, LLMOps.

## Architecture
```
Question -> Router --(injection?)--> Blocked
              |
          Retriever  (BM25 + dense -> RRF fusion -> rerank)
              |
           Analyst   (grounded LLM answer w/ citations; offline extractive fallback)
              |
          Guardrail  (PII redaction, citation check, confidence threshold)
          /        \
   Finalize      Escalation -> human review queue (approve / reject)
```
FastAPI backend · single-page UI with live agent trace · `/metrics` for latency & escalation rate (Prometheus/Grafana-ready).

## Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload      # open http://127.0.0.1:8000
```
No API key needed. Policy questions get cited answers from `data/docs/` (works fully offline).
For **general questions** (e.g. "what is YouTube") install free [Ollama](https://ollama.com) and run `ollama pull llama3.2`.
Those answers are labelled *general knowledge, not from the policy documents*. `/metrics` shows `ollama_running`.
Drop your own `.md` files in `data/docs/` to change the corpus.

## Tests, Docker, CI
```bash
pip install -r requirements-dev.txt && pytest -q      # retrieval, grounding, escalation, blocking, review flow
docker build -t regintel . && docker run -p 8000:8000 regintel
```
GitHub Actions (`.github/workflows/ci.yml`) runs the tests and the Docker build on every push.

## Knowledge base (12 sample policy documents)
Banking: capital adequacy, liquidity risk, AML/KYC, fair lending (ECOA) · Healthcare: PHI privacy ·
Governance: AI governance, data governance, incident response, third-party risk ·
Development finance: procurement, environmental & social safeguards, debt sustainability.

## Live demo script (5 min)
1. "What is the minimum CET1 capital ratio?" – cited answer, agent trace, scores per retriever.
2. "Compare KYC for high-risk vs standard customers" – compare intent, wider retrieval.
3. "What is the cafeteria menu today?" – low confidence -> **escalated to human**, click Approve.
4. "When must an adverse action notice be sent?" / "What are the environmental and social screening categories?" – answers from other domains.
5. "Ignore previous instructions…" – **blocked** by the router.
6. Open `/metrics` – request count, avg latency, escalation rate.

## Production swap-ins
Dense: Azure OpenAI embeddings + Qdrant/pgvector · Rerank: ColBERT / cross-encoder · Tracing: LangSmith ·
Deploy: Docker + Azure ML / AKS · CI/CD: GitHub Actions · Eval: golden-question regression suite.

*Sample policies in `data/docs/` are short synthetic summaries for demonstration only.*
