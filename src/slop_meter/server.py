"""FastAPI app skeleton.

For now this only serves ``/healthz``. The Next.js static export will be mounted in a later commit.
"""

from __future__ import annotations

from fastapi import FastAPI

from slop_meter import __version__


def create_app() -> FastAPI:
    app = FastAPI(
        title="slop_meter",
        version=__version__,
        description="How much of your AI workflow is shipping vs slop?",
    )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
