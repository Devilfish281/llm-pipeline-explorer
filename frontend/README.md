# Run the complete project

You need **two PowerShell terminals running at the same time**:

```text
Terminal 1 → Python/FastAPI backend → port 8000
Terminal 2 → TypeScript/Vite frontend → port 5173
```

FastAPI requires an ASGI server such as Uvicorn. Poetry’s `poetry run` command runs Uvicorn inside the project’s virtual environment. ([FastAPI][1])

## Step 1 — Start the backend

Open **PowerShell Terminal 1**.

Go to the backend:

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\backend"
```

Confirm the location:

```powershell
Get-Location
```

Install the backend dependencies if you have not already done so:

```powershell
poetry install
```

Verify FastAPI:

```powershell
poetry run python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
```

Verify that your package imports correctly:

```powershell
poetry run python -c "import how_llms_work; print(how_llms_work.__file__)"
```

Start FastAPI:

```powershell
poetry run uvicorn how_llms_work.main:app `
    --app-dir src `
    --reload `
    --host 127.0.0.1 `
    --port 8000
```

The command means:

```text
how_llms_work.main → src/how_llms_work/main.py
app                → app = FastAPI(...)
--app-dir src       → add the src directory to Python's import path
--reload            → restart when Python code changes
--port 8000         → listen on port 8000
```

Expected output:

```text
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

Keep Terminal 1 open.

## Step 2 — Test the backend directly

Open a browser and visit:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "healthy"
}
```

You can also open FastAPI’s API documentation:

```text
http://127.0.0.1:8000/docs
```

Or test the health endpoint from another PowerShell terminal:

```powershell
Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/health" `
    -Method Get
```

Expected PowerShell output:

```text
status
------
healthy
```

With the `main.py` you showed, FastAPI currently registers:

```text
GET  /health
POST /simple-chat
```

The other Python routers must be implemented and included in `main.py` before those API endpoints will work.

## Step 3 — Start the frontend

Open **PowerShell Terminal 2**.

Go to the frontend:

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\frontend"
```

Confirm the location:

```powershell
Get-Location
```

Install the frontend packages if needed:

```powershell
pnpm install
```

Run the TypeScript check:

```powershell
pnpm typecheck
```

If type checking succeeds, start Vite:

```powershell
pnpm dev
```

The `vite` command starts the development server using the current directory as the project root and automatically loads `vite.config.ts`. ([vitejs][2])

Expected output:

```text
VITE v8.1.5 ready

➜  Local: http://127.0.0.1:5173/
➜  press h + enter to show help
```

Keep Terminal 2 open.

## Step 4 — Open the application

Open this address in Chrome or Edge:

```text
http://127.0.0.1:5173/
```

The request flow is:

```text
Browser
   ↓
Vite frontend on port 5173
   ↓ /api/simple-chat
Vite development proxy
   ↓ /simple-chat
FastAPI backend on port 8000
```

Vite’s `server.proxy` forwards requests whose paths match the configured proxy key, such as `/api`. ([vitejs][3])

## Step 5 — Test the Vite proxy

While both servers are running, open a third PowerShell window and run:

```powershell
Invoke-RestMethod `
    -Uri "http://127.0.0.1:5173/api/health" `
    -Method Get
```

Expected output:

```text
status
------
healthy
```

This proves that:

```text
Vite is running
       +
The proxy is configured
       +
FastAPI is running
```

The complete request is:

```text
http://127.0.0.1:5173/api/health
                    ↓
Vite removes /api
                    ↓
http://127.0.0.1:8000/health
```

## Step 6 — Test Simple Chat through Vite

Run:

```powershell
curl.exe -N `
    -X POST `
    "http://127.0.0.1:5173/api/simple-chat" `
    -H "Content-Type: application/json" `
    --data '{"message":"hello"}'
```

Expected SSE output will resemble:

```text
event: start
data: {}

event: word
data: {"word":"Hello!"}

event: word
data: {"word":"How"}

event: word
data: {"word":"can"}

event: done
data: {}
```

Then test it through the browser:

1. Open `http://127.0.0.1:5173/`.
2. Enter `hello`.
3. Select **Send**.
4. Confirm that the assistant response appears word by word.

## Everyday startup commands

You do not need to reinstall everything each time.

### Terminal 1 — Backend

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\backend"

poetry run uvicorn how_llms_work.main:app `
    --app-dir src `
    --reload `
    --host 127.0.0.1 `
    --port 8000
```

### Terminal 2 — Frontend

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\frontend"

pnpm dev
```

### Browser

```text
http://127.0.0.1:5173/
```

## Stop the project

In the backend terminal, press:

```text
Ctrl+C
```

In the frontend terminal, press:

```text
Ctrl+C
```

## Common errors

### `No module named fastapi`

Run:

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\backend"

poetry install
poetry run python -c "import fastapi; print(fastapi.__version__)"
```

Always start the backend with `poetry run`.

### `ECONNREFUSED 127.0.0.1:8000`

The frontend is running, but the backend is not.

Start Terminal 1:

```powershell
poetry run uvicorn how_llms_work.main:app `
    --app-dir src `
    --reload `
    --port 8000
```

### Port 8000 is already in use

Find the process:

```powershell
Get-NetTCPConnection -LocalPort 8000 |
    Select-Object LocalAddress, LocalPort, State, OwningProcess
```

Stop it:

```powershell
Stop-Process -Id PROCESS_ID -Force
```

Replace `PROCESS_ID` with the displayed number.

### Port 5173 is already in use

Find the process:

```powershell
Get-NetTCPConnection -LocalPort 5173 |
    Select-Object LocalAddress, LocalPort, State, OwningProcess
```

Stop it:

```powershell
Stop-Process -Id PROCESS_ID -Force
```

### `/api/health` gives `404`

Confirm `vite.config.ts` contains:

```typescript
proxy: {
  "/api": {
    target: "http://127.0.0.1:8000",
    changeOrigin: true,
    rewrite: path => path.replace(/^\/api/, ""),
  },
},
```

Then restart Vite:

```powershell
pnpm dev -- --force
```

### The other demonstrations give `404`

That is expected until you implement and register these backend routers:

```text
/bpe-tokenize
/neural-net
/train-embed
/train-transformer
```

Eventually, `main.py` will need imports and `include_router()` calls for each router.

[1]: https://fastapi.tiangolo.com/deployment/manually/?utm_source=chatgpt.com "Run a Server Manually"
[2]: https://vite.dev/guide/cli?utm_source=chatgpt.com "Command Line Interface"
[3]: https://vite.dev/config/server-options?utm_source=chatgpt.com "Server Options"
