from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import APP_NAME, APP_VERSION
from backend.app.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description="Customer review intelligence API.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api")
    return app


app = create_app()
