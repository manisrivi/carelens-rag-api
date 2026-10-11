# CareLens RAG API

A Python FastAPI backend that lets users upload a blood test report (PDF) and ask questions about it in plain language. It uses Retrieval-Augmented Generation (RAG): answers come only from the user's own report, with sources.

This is the Python rebuild of the AI backend from [CareLens](https://github.com/manisrivi/carelens), originally built in Node.js / Next.js.

## How it works

```
Upload:  PDF → extract text → clean → line-aware chunking → Gemini embeddings → MongoDB Atlas
Ask:     question → embed → Atlas Vector Search (filtered to one report) → Gemini answer + sources
```

## Features

- **PDF upload** with text extraction and whitespace cleanup for table-style lab reports
- **Line-aware chunking**: each lab result stays whole (no values split mid-number)
- **Semantic search** with MongoDB Atlas Vector Search, filtered by report ID for privacy
- **Grounded answers**: Gemini answers only from the report, says when something isn't there, and doesn't diagnose
- **Sources returned** with every answer, so results can be verified
- **Auto setup**: the vector index is created from code at startup
- **Health check** that reports MongoDB and vector index status

## Tech stack

| Layer | Tool |
|---|---|
| API | FastAPI, Pydantic |
| PDF parsing | pypdf |
| Embeddings | Gemini `gemini-embedding-001` (3072 dimensions) |
| LLM | Gemini (`gemini-2.5-flash` by default, configurable) |
| Vector DB | MongoDB Atlas Vector Search |

## Project structure

```
app/
├── main.py            # App setup, routers, startup
├── config.py          # Env variables, Gemini and MongoDB clients
├── schemas.py         # Pydantic request models
├── routes/
│   ├── reports.py     # POST /reports/upload
│   └── ask.py         # POST /ask
└── services/
    ├── pdf.py         # Extract, clean, chunk
    ├── embeddings.py  # Gemini embeddings
    ├── vector_store.py # Index, save, search
    └── llm.py         # Answer generation
```

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server, MongoDB, and vector index status |
| POST | `/reports/upload` | Upload a PDF report, returns a `report_id` |
| POST | `/ask` | Ask a question (uses the latest report if no `report_id` is given) |

Example request:

```json
POST /ask
{ "question": "What is my hemoglobin level?" }
```

Example answer:

> Your hemoglobin (Hb) level is 12.5 g/dL, which is low according to the reference range of 13.0 - 17.0 g/dL. You should discuss this with your doctor.

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
GEMINI_MODEL=gemini-2.5-flash
```

Start the server:

```bash
fastapi dev app/main.py
```

Open http://127.0.0.1:8000/docs to try the API.

## Roadmap

- [ ] Docker support
- [ ] User authentication (filter search by user)
- [ ] Streaming answers
- [ ] LangChain / LangGraph version for comparison
- [ ] MCP server so AI assistants can query reports
- [ ] Evaluation set to measure answer quality