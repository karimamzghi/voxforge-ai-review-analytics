# VoxForge AI — Phase A Foundation

A simple full-stack foundation for the Ironhack AI Engineering capstone.

## Stack

- FastAPI + Pydantic backend
- React + Vite + TypeScript frontend
- TailwindCSS for styling
- Recharts for visualisations
- Single Vercel project deployment

## Structure

```text
api/index.py              Vercel Python entry point
backend/app/              FastAPI application
backend/data/             Deployment-ready analytics JSON
frontend/src/             React application
frontend/public/          VoxForge logo
```

The frontend contains no ML or business logic. It consumes typed FastAPI endpoints. The backend reads precomputed dashboard data so heavy model training and inference remain in your existing notebooks and reusable `src/` pipeline.

## Run locally

### Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

Backend: `http://127.0.0.1:8000`
Swagger: `http://127.0.0.1:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend: `http://localhost:5173`

Vite proxies `/api` calls to the local FastAPI server.

## Deploy to Vercel

1. Push this folder to GitHub.
2. Import the repository into Vercel.
3. Keep the repository root as the project root.
4. Vercel reads `vercel.json`, builds `frontend/dist`, and exposes FastAPI through `api/index.py`.

## Replace seed data

`backend/data/dashboard.json` is intentionally lightweight and immediately runnable. In Phase B, replace it with JSON generated from your final review, sentiment and topic outputs. The API contract and frontend do not need to change.
