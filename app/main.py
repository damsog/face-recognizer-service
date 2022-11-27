import asyncio
from fastapi import FastAPI

from app.configurations.information import *
from aiortc.contrib.media import MediaRelay
from videoAnalytics.processor import processor

from app.controllers import detectionController
from app.controllers import encoderController

def create_app():
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        contact=contact,
    )

    app.include_router(encoderController.router)
    app.include_router(detectionController.router)

    @app.on_event("startup")
    async def startup():
        app.state.processor = processor()
        app.state.pcs = set()
        app.state.relay = MediaRelay()

    @app.on_event("shutdown")
    async def shutdown():
        coros = [pc.close() for pc in app.state.pcs]
        await asyncio.gather(*coros)
        app.state.pcs.clear()

    return app

app = create_app()