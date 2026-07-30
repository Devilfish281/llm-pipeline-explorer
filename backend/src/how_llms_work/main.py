# backend/src/how_llms_work/main.py

from fastapi import FastAPI
from how_llms_work.routes.bpe_tokenize import router as bpe_tokenize_router
from how_llms_work.routes.neural_net import router as neural_net_router
from how_llms_work.routes.simple_chat import router as simple_chat_router
from how_llms_work.routes.train_embed import router as train_embed_router
from how_llms_work.routes.train_transformer import router as train_transformer_router

app = FastAPI(title="How LLMs Work")

app.include_router(simple_chat_router)
app.include_router(bpe_tokenize_router)
app.include_router(neural_net_router)
app.include_router(train_embed_router)
app.include_router(train_transformer_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}
