import uvicorn
from dotenv import load_dotenv, find_dotenv
import os

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    if(os.getenv("ENV") == "development"):
        uvicorn.run("app.main:app", host=os.getenv("SERVER_IP"), port=int(os.getenv("SERVER_PORT")), log_level=os.getenv("LOGGER_LEVEL"), reload=True)
    else:
        uvicorn.run("app.main:app", host=os.getenv("SERVER_IP"), port=int(os.getenv("SERVER_PORT")), log_level=os.getenv("LOGGER_LEVEL"))