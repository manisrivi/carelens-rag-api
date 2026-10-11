# CareLens RAG API

A Python FastAPI backend that lets users upload a blood test report (PDF) and ask questions about it in plain language. It uses Retrieval-Augmented Generation (RAG): answers come only from the user's own report, with sources.

This is the Python rebuild of the AI backend from [CareLens](https://github.com/manisrivi/carelens), originally built in Node.js / Next.js.

The same RAG pipeline is available in four ways: a hand-built version, a LangChain version, a LangGraph agent, and an MCP server that AI assistants can use as a tool.

## How it works

```
Upload:  PDF → extract text → clean → line-aware chunking → Gemini embeddings → MongoDB Atlas
Ask:     question → embed → Atlas Vector Search (filtered to one report) → Gemini answer + sources
```

## Features

- **PDF upload** with text extraction and whitespace cleanup for table-style lab reports
- **Line-aware chunking**: each lab result stays whole (no values split mid-number)
- **Semantic search** with MongoDB Atlas Vector Search, filtered by report ID for privacy
- **Grounded answers**: answers only from the report, says when something isn't there, and doesn't diagnose
- **Sources returned** with every answer, so results can be verified
- **LangChain version** of the same pipeline, for comparison with the hand-built one
- **LangGraph agent** that routes each message: report question → search, small talk → reply, vague question → ask for clarification
- **MCP server** that exposes the report search as a tool for AI assistants (tested with Gemini CLI)
- **API key protection** on all endpoints except `/health`
- **Auto setup**: the collection and vector index are created from code at startup
- **Health check** that reports MongoDB and vector index status
- **Docker** support

## Tech stack

| Layer | Tool |
|---|---|
| API | FastAPI, Pydantic |
| PDF parsing | pypdf |
| Embeddings | Gemini `gemini-embedding-001` (3072 dimensions) |
| LLM | Gemini (configurable with `GEMINI_MODEL`) |
| Vector DB | MongoDB Atlas Vector Search |
| Frameworks | LangChain, LangGraph |
| Tool protocol | MCP (Model Context Protocol) Python SDK |
| Deployment | Docker |

## Project structure

```
app/
├── main.py               # App setup, routers, startup
├── config.py             # Env variables, Gemini and MongoDB clients
├── schemas.py            # Pydantic request models
├── security.py           # API key check
├── routes/
│   ├── reports.py        # POST /reports/upload
│   └── ask.py            # POST /ask, /ask-langchain, /agent
└── services/
    ├── pdf.py            # Extract, clean, chunk
    ├── embeddings.py     # Gemini embeddings
    ├── vector_store.py   # Index, save, search
    ├── llm.py            # Answer generation (hand-built)
    ├── langchain_rag.py  # Same RAG with LangChain
    └── agent.py          # LangGraph routing agent
mcp_server.py             # MCP server with report tools
Dockerfile
```

## API

All endpoints except `/health` require an `X-API-Key` header.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server, MongoDB, and vector index status |
| POST | `/reports/upload` | Upload a PDF report, returns a `report_id` |
| POST | `/ask` | Hand-built RAG answer (uses the latest report if no `report_id` is given) |
| POST | `/ask-langchain` | Same RAG, built with LangChain |
| POST | `/agent` | LangGraph agent that decides whether to search, reply, or ask for clarification |

Example request:

```json
POST /ask
{ "question": "What is my hemoglobin level?" }
```

Example answer:

> Your hemoglobin (Hb) level is 12.5 g/dL. The reference range is 13.0 - 17.0 g/dL, which means your hemoglobin level is low. You should discuss this result with your doctor.

## MCP server

`mcp_server.py` exposes two tools to any MCP-compatible AI app:

| Tool | Description |
|---|---|
| `ask_report(question)` | Answers a question about the latest blood report |
| `latest_report_info()` | Returns the latest report's filename and chunk count |

Test it with the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector venv/bin/python mcp_server.py
```

## Run locally

```bash
git clone https://github.com/manisrivi/carelens-rag-api.git
cd carelens-rag-api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
GEMINI_API_KEY=your-gemini-key
MONGODB_URI=your-mongodb-atlas-uri
API_KEY=your-secret-api-key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Start the server:

```bash
fastapi dev app/main.py
```

Open http://127.0.0.1:8000/docs to try the API.

## Run with Docker

```bash
docker build -t carelens-rag-api .
docker run --rm -p 8000:8000 --env-file .env carelens-rag-api
```

Then open http://127.0.0.1:8000/docs

## Roadmap

- [x] Docker support
- [x] LangChain / LangGraph version for comparison
- [x] MCP server so AI assistants can query reports
- [x] API key protection
- [ ] Deploy with a public URL
- [ ] User authentication (filter search by user)
- [ ] Streaming answers
- [ ] Evaluation set to measure answer quality