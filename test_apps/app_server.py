# Mini fastapi server to serve the streaming test app
# Start this using uvicorn on port 5500 as:
# uvicorn app_server:app --host 127.0.0.1 --port 5500 --reload

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    with open("./app_stream.html", "r") as file:
        content = file.read()
    return HTMLResponse(content=content)