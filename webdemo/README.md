# CVQA web demo

The demo has two independent processes:

```text
webdemo/frontend (Vite) -> webdemo/backend (FastAPI) -> src/online + checkpoint
```

## Backend

From the repository root:

```powershell
pip install -r requirements.txt
$env:CVQA_CHECKPOINT="checkpoints\exp01\best.pt"
python -m uvicorn webdemo.backend.app:app --reload --port 8000
```

The backend reads `configs/online.yaml` by default. Override it with
`CVQA_CONFIG`. The REST endpoint is `POST /api/infer` and accepts multipart
fields `image`, `question`, `history` (JSON array), and `sample_id`.

## Frontend

In another terminal:

```powershell
cd webdemo\frontend
npm install
npm run dev
```

Open the Vite URL (normally `http://localhost:5173`). Set
`VITE_API_URL` if the backend is not running on `http://localhost:8000`.
