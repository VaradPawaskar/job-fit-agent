# Job Fit Agent

[![tests](https://github.com/VaradPawaskar/job-fit-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/VaradPawaskar/job-fit-agent/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A **LangChain tool-calling agent** that compares your CV with a job posting. It fetches the posting, extracts the requirements, searches your CV for evidence of each one, and returns a structured report: a fit score, a status for every requirement, your top gaps and honest suggestions for improving the CV.

Built with LangChain, FastAPI, Streamlit and Docker. Works with Gemini, Claude or OpenAI models by changing one environment variable.

<!-- Add a screenshot or GIF here:  ![demo](docs/demo.gif) -->

## What it does

Input: your CV (PDF, `.txt` or `.md`) and a job posting (URL or pasted text).

Output:
- A **fit score** from 0 to 100
- A table of 6-12 requirements, each marked `match`, `partial` or `missing`, with the evidence found in your CV
- **Top gaps**, most important first
- **CV suggestions** that only reorder, reword or emphasise what the CV already shows (it never invents experience)
- The list of **tool calls** the agent made, so every claim can be traced

## How it works

```
 Streamlit UI ──► FastAPI /analyze ──► LangChain agent (create_agent)
                      │                      │
        CV upload ────┤                      ├─ tool: fetch_job_posting(url)    SSRF-guarded web fetch
        (PDF / txt)   │                      ├─ tool: get_cv_overview()
                      │                      ├─ tool: search_cv_many(queries)   batch evidence lookup
                      │                      └─ tool: search_cv(query)          single follow-up lookup
                      ▼
             structured output (Pydantic: AgentReport)
                      ▼
         score computed in code from the matches  ──►  JSON + tool-call trace
```

1. The agent gets the posting text, either pasted or fetched by a tool.
2. It lists 6-12 concrete requirements and marks each `must_have` or `nice_to_have`.
3. It checks all of them against the CV in one batched call, then does at most one follow-up with synonyms.
4. It returns a validated `AgentReport`. The **score is computed in plain Python** (must-haves weigh 2x, partial matches count 0.5), so it is reproducible and not a number the LLM made up.

## Quick start

You need Docker and an API key for a supported model provider.

```bash
git clone https://github.com/VaradPawaskar/job-fit-agent.git
cd job-fit-agent
cp .env.example .env          # Windows cmd: copy .env.example .env
# open .env and set MODEL and the matching API key
docker compose up --build
```

Then open:
- **App:** http://localhost:8501
- **API docs:** http://localhost:8000/docs

Stop with `Ctrl+C`, or `docker compose down`.

### Without Docker

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
export $(grep -v '^#' .env | xargs)                   # or set the variables yourself
uvicorn app.api:app --reload                          # terminal 1
API_URL=http://localhost:8000 streamlit run ui/streamlit_app.py    # terminal 2
```

### API example

```bash
curl -X POST http://localhost:8000/analyze \
  -F cv_file=@my_cv.pdf \
  -F job_url="https://example.com/jobs/123"
```

Use `job_text` instead of `job_url` to send the posting text. If both are sent, `job_text` wins. If `API_KEY` is set in `.env`, add `-H "X-API-Key: <your key>"`.

## Configuration

All settings are environment variables (see `.env.example`).

| Variable | Default | Purpose |
|---|---|---|
| `MODEL` | `google_genai:gemini-3.5-flash-lite` | `<provider>:<model name>`, e.g. `anthropic:<model>` or `openai:<model>` |
| `GOOGLE_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | none | Key for the provider you chose |
| `MODEL_KWARGS` | none | Extra model options as JSON, e.g. `{"max_output_tokens": 4096}` |
| `API_KEY` | none | If set, `/analyze` requires an `X-API-Key` header |
| `AGENT_RECURSION_LIMIT` | `25` | Maximum agent steps before it is stopped |
| `MAX_UPLOAD_MB` | `10` | Maximum CV upload size |

**Choosing a model:** this agent makes several sequential model calls, so latency adds up. Prefer a fast, low-cost model (a "flash" or "lite" class model, or a small Claude or GPT model). Large reasoning models are slower here and rarely more accurate for this task. Every run logs its duration and number of model calls, so you can compare models: `docker compose logs api`.

**Local models:** `pip install langchain-ollama`, then `MODEL=ollama:<model>`. Smaller local models may follow the tool workflow less reliably.

## Why Docker

- **Same environment everywhere.** The image pins the Python version and all dependencies, so "works on my machine" problems disappear.
- **One command to run.** `docker compose up` starts both the API and the UI, wired together, with a health check so the UI waits for the API.
- **No local setup.** You do not need to install Python packages or manage virtual environments to try the project.
- **Isolation.** The app runs as a non-root user in its own container and does not touch the rest of your system.
- **Deployment-ready.** The same image runs on a laptop, a CI server or a cloud service such as AWS.

## Tests and evaluation

```bash
pytest -q                 # 41 offline tests, no API key needed
python -m eval.run_eval   # live check against known answers; needs an API key
```

- Unit tests cover CV search, URL safety (SSRF), HTML cleaning, scoring, API validation and error mapping.
- Agent tests replay a scripted fake model, so the full tool-calling loop and the step limit are tested without paying for tokens.
- `eval/cases.jsonl` holds postings with known expected statuses against `samples/sample_cv.txt` (for example, Kubernetes must come back `missing`). Run it after changing the prompt or the model.
- GitHub Actions runs the test suite on every push.

## Design decisions

| Decision | Why |
|---|---|
| Agent with tools instead of one big prompt | The CV is searched on demand, so it scales to long CVs, and every claim can be traced to a tool call. |
| Batch CV search tool | One call checks all requirements, which cuts model round trips and latency. |
| Structured output (Pydantic) | The API returns validated JSON, not free text. |
| Score computed in code | LLM-chosen scores are inconsistent between runs. |
| Tools return `ERROR: ...` strings | The agent can recover (for example by asking the user to paste the text) instead of crashing. |
| Recursion limit | A confused agent cannot loop and burn tokens. Hitting it returns a clear error. |
| SSRF guard on the URL fetcher | Only public http(s) hosts, redirects re-validated, size and content-type limits. |
| Prompt-injection stance | The posting and tool output are declared untrusted data, and the posting is wrapped in tags. |
| Optional API key header | Every call costs money. Set it before exposing the API beyond localhost. |

## Limitations

- CV search is keyword-based with light normalisation, not semantic. It can miss unusual wording. An embeddings retriever is the natural next step.
- Pages that need JavaScript or a login cannot be fetched. Paste the text instead.
- Scanned PDFs without a text layer are rejected.
- Your CV is sent to the model provider you configure. Do not use a provider you would not trust with it.
- The agent's judgement is only as good as the model. Treat the score as a prompt for your own review, not a verdict.
- No rate limiting or user accounts.

## Project layout

```
app/        agent.py (LangChain agent) · tools.py · schemas.py · cv.py · api.py · config.py
ui/         streamlit_app.py
tests/      offline pytest suite
eval/       live evaluation cases + runner
samples/    fictional CV and posting
.github/    CI workflow
```

## Roadmap

- [ ] Replace keyword search with embeddings (FAISS or Chroma) and compare eval results
- [ ] Draft a tailored cover-letter paragraph per gap, grounded in CV facts
- [ ] Store past analyses (SQLite) and show which skills are most often missing
- [ ] LangSmith tracing

## License

MIT. See [LICENSE](LICENSE).
