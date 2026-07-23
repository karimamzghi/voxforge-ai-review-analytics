# VoxForge AI 


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

The frontend contains no ML or business logic. It consumes typed FastAPI endpoints. The backend reads precomputed dashboard data so heavy model training and inference remain in the notebooks directory and reusable `src/` pipeline.


### Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```
API documentation:
http://127.0.0.1:8000/docs

API response 
http://127.0.0.1:8000/redoc

Health:
http://127.0.0.1:8000/api/health

Dashboard data:
http://127.0.0.1:8000/api/dashboard

Topics:
http://127.0.0.1:8000/api/topics

Recommendations:
http://127.0.0.1:8000/api/recommendations

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend: `http://localhost:5173`

Vite proxies `/api` calls to the local FastAPI server.

