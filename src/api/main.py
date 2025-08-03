import uvicorn
from fastapi import FastAPI

from api.routes.generate import router as generate_router

app = FastAPI()
app.include_router(generate_router)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the finance agent API!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
