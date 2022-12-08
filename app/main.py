import asyncio
import os
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
import pyfiglet
import iridi

from app.configurations.information import *
from aiortc.contrib.media import MediaRelay
from videoAnalytics.processor import processor

from app.controllers import detectionController
from app.controllers import encoderController

def create_app():
    # Create the FastAPI app. Setting server information
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        contact=contact,
    )

    # Set the routes
    app.include_router(encoderController.router)
    app.include_router(detectionController.router)

    # Set the startup and shutdown events. and Global variables
    # Load the processor
    # Create a set of PeerConnections
    # Create a MediaRelay
    @app.on_event("startup")
    async def startup():
        # Service title definition
        serverTitle = pyfiglet.figlet_format("Gnosis", font="isometric2", width=200)
        serverSubtitle = pyfiglet.figlet_format("Recognizer Service", font="alligator2", width=300)

        iridi.print(serverTitle, ["#8A2387", "#E94057", "#F27121"], bold=True)
        iridi.print(serverSubtitle, ["#8A2387", "#E94057", "#F27121"], bold=True)


        load_dotenv(find_dotenv())

        # Load the processor, this will load the model. 0-N for GPU, -1 for CPU
        app.state.processor = processor(
            detector_load_device= int(os.getenv("DETECTOR_LOAD_DEVICE")),
            recognizer_load_device= int(os.getenv("RECOGNIZER_LOAD_DEVICE"))
        )

        app.state.pcs = set()
        app.state.relay = MediaRelay()

    @app.on_event("shutdown")
    async def shutdown():
        coros = [pc.close() for pc in app.state.pcs]
        await asyncio.gather(*coros)
        app.state.pcs.clear()

    return app

app = create_app()