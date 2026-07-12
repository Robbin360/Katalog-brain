# Katalog-brain AGENTS.md

## Stack
- Python 3.13+, FastAPI, LangGraph, PydanticAI v2, Supabase, Gemini/Groq/OpenRouter
- No linting/formatting/typecheck config (no ruff, mypy, black, pyproject.toml)
- No tests, no CI, no Makefile

## Quick commands
```powershell
# Run server
uvicorn main:app --reload

# Visualize LangGraph flow
python scripts/visualize_graph.py
```

## Architecture
- **`main.py`**: FastAPI entrypoint — endpoints `POST /api/triage`, `POST /api/optimize`. Starts Auto-Pilot worker on startup (lifespan event).
- **`core/graph.py`**: LangGraph state machine with 14 nodes. Entry: `start_processing` → `fetch_data` → `memory` → `retrieve_knowledge` → `enrich_for_orchestrator` → `do_not_harm` → `orchestrator` → (optional `researcher`) → `ai_writer` ↔ `critic` (up to 3 loops) → `save_db`/`needs_optimization` → (if auto_pilot) `publish_to_shopify`. See `build_graph()` at line 1328.
- **`core/worker.py`**: Background patrol (30s interval). 4 phases: Zombie Sweeper, Auto-Triaje, Credit Gate, Auto-Pilot Optimizer.
- **`agents/`**: Each file is a PydanticAI Agent with structured output. Optimizer/Critic have primary→fallback model chains (Gemini → Groq).
- **`core/auth.py`**: JWT verification via Supabase Auth using **anon key** (not service_role). Dependency: `get_current_user_id`.
- **`core/model_config.py`**: Model strings (env-driven). Prefixes: `google-gla:`, `google:`, `groq:`, `deepseek/deepseek-...:free`.
- **Billing**: Reserve→Commit pattern via Supabase RPCs (`reserve_or_reuse_product_credit`, `commit_product_credit`, `refund_product_reservation`). Single credit per product.
- **Do-Not-Harm**: Python-only quadrant gate (`do_not_harm_check` at graph.py:792). Refunds reservation for skipped quadrants.

## Key conventions
- All LLM outputs wrapped in Pydantic `BaseModel` — never unstructured.
- `shopify_products.id` is BIGINT → `int` in Python. `user_id` is UUID → `str`.
- HTTPException re-raise order in `main.py:203-211`: RE-raise HTTPException BEFORE `except Exception` to avoid masking status codes.
- `audit_status` values: `PENDING_AUDIT`, `PROCESSING`, `NEEDS_OPTIMIZATION`, `READY_TO_PUBLISH`, `OPTIMIZED`, `ERROR`, `OUT_OF_CREDITS`.
- `.env` committed with real keys (dev instance). Stripe sync runs once daily.

## Agents and fallback chains
| Agent | Primary | Fallback |
|---|---|---|
| Optimizer | `gemini-2.5-pro` | `groq:llama-3.3-70b-versatile` |
| Critic | `gemini-2.5-flash` | `groq:openai/gpt-oss-120b` |
| Reclassifier (scripts) | `deepseek/deepseek-v4-flash:free` (OpenRouter) | `groq:qwen/qwen3-32b` |
| Inspector | `gemini-2.5-flash-lite` | none |
| Orchestrator | `google:gemini-3.5-flash` | hardcoded conservative plan |
| Researcher | `google:gemini-3.5-flash` | Tavily+Firecrawl for web search |

## Gotchas
- **No tests exist** — manual verification only.
- `.env` has real API keys — do NOT share or commit to public repos.
- `retrieve_memory_letta` is a stub (returns hardcoded string) — Letta integration not wired.
- `knowledge.txt` is empty (RAG mailbox).
- `scripts/visualize_graph.py` appends a hardcoded global site-packages path — adjust if Python version changes.
- Use `asyncio.to_thread()` (aliased `_run_sync`) for all Supabase calls — SQLAlchemy-style sync calls block the event loop.
