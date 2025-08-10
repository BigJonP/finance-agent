import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from api.routes.generate import router as generate_router
from api.routes.user import router as users_router
from api.routes.holding import router as holding_router

app = FastAPI(title="Finance Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0"]
)

app.include_router(generate_router)
app.include_router(users_router)
app.include_router(holding_router)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the finance agent API!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
