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

# Run the complete suite

```powershell
poetry run pytest
```

# Then run quality checks:

## Run formatting, linting, and typing checks

```powershell
poetry run black --check .
```

```powershell
poetry run ruff check .
```

```powershell
poetry run mypy src
```

```powershell

poetry run black --check .
poetry run ruff check .
poetry run mypy src
```

# Use the generated OpenAPI path map instead of directly iterating over app.routes:

```powershell
poetry run python -c "import how_llms_work.main as m; methods={'get','post','put','patch','delete','options','head','trace'}; paths=m.app.openapi()['paths']; print('Loaded:', m.__file__); print(*sorted((path, sorted(method.upper() for method in operations if method in methods)) for path, operations in paths.items()), sep='\n')"
```

## Expected output should include approximately:

````text
/health: GET
/api/bpe-tokenize: POST
/api/neural-net: POST
/api/train-embed: POST
/api/train-transformer: POST
```S

# Train Embeddings

king queen dog cat | 10 4 1 1
king queen man woman cat dog | 100 16 1 2
king queen man woman cat dog | 500 16 1 2
king queen man woman cat dog | 500 32 1 2

king queen man woman cat dog | 1000 32 1 2

| Test                 | Epochs | Dimensions | Window | Negative samples |
| -------------------- | -----: | ---------: | -----: | ---------------: |
| Connection test      |     10 |          4 |      1 |                1 |
| Fast quality test    |    100 |         16 |      1 |                2 |
| Better quality test  |    500 |         16 |      1 |                2 |
| Higher-capacity test |    500 |         32 |      1 |                2 |
````

# XOR

## single-layer

```text
single-layer 5000
```

## multi-layer

```text
multi-layer 5000
```

# Basic Tokenizer

```text
cat cat car
```

```text
the cat sat on the mat
```

# Train Embeddings

## Your web input format i

```text
words | epochs dimensions window-size negative-samples
```

## the smallest and fastest valid test first:

```text
king queen dog cat | 10 4 1 1
```

# Train Transformer

```text
50 1.0 0.6 1 3
```

```text
300 0.8 0.9 2 40
```

```text
2000 1 0.6 6 3
```

## First test: 300 epochs, 0.8 learning rate, 0.9 dropout, 2 attention heads, 40 hidden units

```text
50 1.0 0.6 1 3
```

### Understand the five numbers

The format is:

```text
epochs temperature top-p layers max-tokens
```

Your test means:

| Value | Setting        | Meaning                                         |
| ----: | -------------- | ----------------------------------------------- |
|  `50` | Epochs         | Train through epochs `0–50`                     |
| `1.0` | Temperature    | Normal sampling temperature                     |
| `0.6` | Top-P          | Sample from the highest-probability 60% nucleus |
|   `1` | Layers         | Use the smallest one-layer Transformer          |
|   `3` | Maximum tokens | Generate three new sample tokens                |

Your current accepted ranges are:

| Setting        | Allowed range |
| -------------- | ------------: |
| Epochs         |     `50–2000` |
| Temperature    |     `0.1–2.0` |
| Top-P          |     `0.1–1.0` |
| Layers         |         `1–6` |
| Maximum tokens |       `3–500` |

The request fields sent to FastAPI are `epochs`, `temperature`, `topP`, `numLayers`, and `maxTokens`.

## What `50 1.0 0.6 1 3` means

This is the smallest supported Transformer-training configuration in your current application:

| Number | Setting        | Meaning                                                                   |
| -----: | -------------- | ------------------------------------------------------------------------- |
|   `50` | Epochs         | Train through epoch 50                                                    |
|  `1.0` | Temperature    | Do not sharpen or flatten token probabilities                             |
|  `0.6` | Top-P          | Sample from the most likely tokens whose probabilities total at least 60% |
|    `1` | Layers         | Build a one-layer Transformer                                             |
|    `3` | Maximum tokens | Generate three new tokens for each sample                                 |

## 2. Inspect the saved configuration

Avoid printing the entire model because it contains thousands of weight values.

```powershell
$modelPath = ".data\transformer-weights-e50-l1-d32-h2-ff128-ctx32.json"

$model = Get-Content $modelPath -Raw |
    ConvertFrom-Json

$model.type
$model.config
$model.vocab.Count
$model.weights.blocks.Count
```

Expected general result:

```text
decoder-transformer

vocabSize  : 392
contextLen : 32
embDim     : 32
numHeads   : 2
ffDim      : 128
numLayers  : 1

392
1
```

This confirms that the file contains a one-layer model rather than merely an empty completion marker.

## 3. Record this as your baseline

Write down:

```text
Input:      50 1.0 0.6 1 3
Layers:     1
Epochs:     0–50 inclusive
Final loss: 4.6349
Final sample:
once upon a to young sky
```

This gives you a reference for comparing later runs.

## 4. Recommended next training experiment

Keep everything the same and change only the epoch count:

```text
100 1.0 0.6 1 3
```

This creates a separate file:

```text
.data\transformer-weights-e100-l1-d32-h2-ff128-ctx32.json
```

## 5. Inspect the saved configuration

Avoid printing the entire model because it contains thousands of weight values.

```powershell
$modelPath = ".data\transformer-weights-e100-l1-d32-h2-ff128-ctx32.json"

$model = Get-Content $modelPath -Raw |
    ConvertFrom-Json

$model.type
$model.config
$model.vocab.Count
$model.weights.blocks.Count
```

## 6. Record this as your baseline

Write down:

```text
Input:      100 1.0 0.6 1 3
Layers:     1
Epochs:     0–100 inclusive
Final loss: 3.7488
Final sample:
once upon a tall king man
```

This gives you a reference for comparing later runs.

## 7. Recommended next training experiment

Keep everything the same and change only the epoch count:

```text
1000 0.8 0.9 1 20
```

## 8. Inspect the saved configuration

Avoid printing the entire model because it contains thousands of weight values.

```powershell
$modelPath = ".data\transformer-weights-e1000-l1-d32-h2-ff128-ctx32.json"

$model = Get-Content $modelPath -Raw |
    ConvertFrom-Json

$model.type
$model.config
$model.vocab.Count
$model.weights.blocks.Count
```

## 9. Record this as your baseline

Write down:

```text
Input:      1000 0.8 0.9 1 20
Layers:     1
Epochs:     0–1000 inclusive
Final loss: 0.3074
Final sample:
once upon a time a small cat lived in a village with an old woman. the cat was a loyal pet who
```

This gives you a reference for comparing later runs.

## Important consequence

### The maximum configuration:

```text
1000 0.8 0.9 6 20
```

#### FILE: .data\transformer-weights-e1000-l6-d32-h2-ff128-ctx32.json

transformer-weights-e1000-l6-d32-h2-ff128-ctx32.json

```text
1000 1 0.6 6 3
```

## Check whether it is really working

### While the page appears unchanged, open another PowerShell window:

```powershell
Get-Process python* -ErrorAction SilentlyContinue |
    Select-Object `
        Id,
        ProcessName,
        CPU,
        @{Name = "MemoryMB"; Expression = {
            [math]::Round($_.WorkingSet64 / 1MB, 1)
        }}
```

### Wait 30 seconds:

```text
Run the process command again.

If the worker processes’ cumulative CPU numbers increase, the model is still training.
```
