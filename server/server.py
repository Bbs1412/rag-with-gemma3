# FastAPI server which will handle all the backend and GenAI aspects of the application
# uvicorn server:app --reload
# Avoid using --reload flag, because, LLMs will keep reloading and system will overheat.

from fastapi import FastAPI, File, UploadFile, Form, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import json
from typing import Literal
from pydantic import BaseModel
from contextlib import asynccontextmanager

# llm system imports:
from llm_system.core.llm import get_llm, get_output_parser  # Functions
from llm_system.core.llm import get_dummy_response          # Function
from llm_system.core.llm import get_dummy_response_stream   # Function
from llm_system.core.database import VectorDB               # Class
from llm_system.core.history import HistoryStore            # Class
from llm_system.chains.rag import build_rag_chain           # Function
from llm_system import config                               # Constants
from llm_system.core.ingestion import ingest_file           # Function

# Helper Modules:
import sq_db
import files

# Type hinting imports:
from langchain_core.vectorstores import VectorStore as T_VECTOR_STORE

import logger
log = logger.get_logger("rag_server", log_to_console=False, log_to_file=True)


# ------------------------------------------------------------------------------
# Constants:
# ------------------------------------------------------------------------------

# UPLOADS_DIR: str = "user_uploads"
OLD_FILE_THRESHOLD: int = 3600 * 1  # 24 hours in seconds
# OLD_FILE_THRESHOLD: int = 20         # 1 min


# ------------------------------------------------------------------------------
# FastAPI Startup:
# ------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Define the lifespan context manager for startup/shutdown"""

    # [ Startup ]
    log.info("[LifeSpan] Starting the server components.")

    app.state.llm_chat = get_llm(
        model_name=config.LLM_CHAT_MODEL_NAME,
        context_size=config.MAX_CONTENT_SIZE,
        temperature=config.LLM_CHAT_TEMPERATURE,
        verify_connection=config.VERIFY_LLM_CONNECTION
    )

    # app.state.llm_summary = get_llm(...)
    app.state.llm_summary = app.state.llm_chat

    app.state.output_parser = get_output_parser()
    app.state.vector_db = VectorDB(
        embed_model=config.EMB_MODEL_NAME,
        retriever_num_docs=config.DOCS_NUM_COUNT,
        verify_connection=config.VERIFY_EMB_CONNECTION,
    )
    app.state.history_store = HistoryStore()

    app.state.rag_chain = build_rag_chain(
        llm_chat=app.state.llm_chat,
        llm_summary=app.state.llm_summary,
        retriever=app.state.vector_db.get_retriever(),
        get_history_fn=app.state.history_store.get_session_history,
    )

    log.info("[LifeSpan] All LLM components initialized.")

    # sq_db.delete_database()
    sq_db.create_tables()

    # Files
    files.check_create_uploads_folder()

    # [ Lifespan ]
    yield

    # [ Shutdown ]
    log.info("[LifeSpan] Shutting down LLM server...")
    # Add any cleanup part here
    # Like saving vector DB, or shutting down subprocesses


# Make one FastAPI app instance with the lifespan context manager
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:5500",
        # "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


# ------------------------------------------------------------------------------
# Basic API Endpoints:
# ------------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint to check if the server is running."""
    return {
        "message": "LLM RAG Server is running!",
        "further": "Proceed to code ur application :)",
        "thought": "You really are not supposed to be reading this waste of time, but if you are, then you are a curious person. I like that! 😄",
    }


# Define data model for chat request
class BasicChatRequest(BaseModel):
    query: str
    session_id: str
    dummy: bool = False


@app.post("/simple")
async def simple(request: Request, chat_request: BasicChatRequest):
    """Endpoint to handle ont time generation queries.
    - Post request expects JSON `{"query": "", "session_id": "", "dummy":T/F}` structure.
    - Return JSON with `{"response": "", "session_id": ""}` structure.
    """

    llm = request.app.state.llm_chat | request.app.state.output_parser
    session_id = chat_request.session_id.strip() or "unknown_session"

    try:
        query = chat_request.query
        dummy = chat_request.dummy
        log.info(f"/simple Requested by '{session_id}'")

        if dummy:
            log.info(f"/simple Dummy response returned for '{session_id}'")
            return get_dummy_response()

        else:
            result = await llm.ainvoke(input=query)

            log.info(f"/simple Response generated for '{session_id}'.")
            return {"response": result, "session_id": session_id}

    except Exception as e:

        log.exception(f"/simple Error {e} for '{session_id}'")
        return JSONResponse(status_code=500, content={"error": str(e)})


# Make one streaming endpoint for the Simple LLM response:
class StreamChatRequest(BaseModel):
    query: str
    session_id: str
    dummy: bool = False


@app.post("/simple/stream")
async def chat_stream(request: Request, chat_request: StreamChatRequest):
    """Endpoint to handle streaming responses for one time generation queries.
    - Post request expects JSON `{"query": "", "session_id": "", "dummy":T/F}` structure.
    - Return NDJSON with types "metadata", "content", or "error".
    """
    llm = request.app.state.llm_chat | request.app.state.output_parser
    session_id = chat_request.session_id.strip() or "unknown_session"

    async def token_streamer():
        try:
            dummy = chat_request.dummy
            s = 'dummy' if dummy else 'real'
            log.info(f"/simple/stream {s} response requested by '{session_id}'")

            # Start be sending meta data first.
            yield json.dumps({
                "type": "metadata",
                "data": {"session_id": session_id}
            }) + "\n"
            # NDJSON (newline-delimited JSON) - Frontend will merge full response my splitting this

            #  Then send the actual response content:
            if dummy:
                # If dummy is True, stream dummy response
                resp = get_dummy_response_stream(
                    batch_tokens=config.BATCH_TOKEN_PS,
                    token_rate=config.TOKENS_PER_SEC
                )
                for chunk in resp:
                    if await request.is_disconnected():
                        log.warning(f"/simple/stream client disconnected for '{session_id}'")
                        break

                    yield json.dumps({
                        "type": "content",
                        "data": chunk
                    }) + "\n"

            else:
                async for chunk in llm.astream(chat_request.query):
                    if await request.is_disconnected():
                        log.warning(f"/simple/stream client disconnected for '{session_id}'")
                        break

                    yield json.dumps({
                        "type": "content",
                        "data": chunk
                    }) + "\n"

            # In the end, you can send some "Done" etc if u need some conditional logic
            # Server will auto send EOF to mark end of generator response.
            # yield json.dumps({
            #     "type": "end",
            #     "data": "done"
            # }) + "\n"
            log.info(f"/simple/stream Streaming completed for '{session_id}'")

        except Exception as e:
            log.exception(f"/simple/stream Error {e} for '{session_id}'")
            yield json.dumps({
                "type": "error",
                "data": str(e)
            }) + "\n"

    # Return a StreamingResponse with the token streamer generator (basically enable streaming)
    return StreamingResponse(token_streamer(), media_type="text/plain")


# ------------------------------------------------------------------------------
# Initialization End-points:
# ------------------------------------------------------------------------------

# First end-point to call on client initialization:
class LoginRequest(BaseModel):
    login_id: str
    password: str


@app.post("/login")
async def login(request: Request, login_request: LoginRequest):
    """Endpoint to handle user login.
    + Client sends login_id and password for login
    + Based on it, server sends back one user_id
    + But, for now, it is skipped and we will send one dummy user_id
    + Folder is created for user_id, older files are removed

    - Post request expects JSON `{"login_id": "", "password": ""}` structure.
    - Return JSON with `{"user_id": "dummy_user_id"}` structure.
    """

    login_id = login_request.login_id.strip()
    password = login_request.password.strip()

    # For now, we will just return a dummy user_id
    # In future, can implement actual user authentication and return a real user_id
    user_id = login_id
    log.info(f"/login requested by '{user_id}'")

    # Check if folder exists in UPLOADS_DIR with user_id
    files.create_user_uploads_folder(user_id=user_id)

    # Old any older data if exists (older than 24 hours)
    old = sq_db.get_old_files(user_id=user_id, time=OLD_FILE_THRESHOLD)
    if old['files']:
        log.info(f"/login Removing old files for user '{user_id}': {old['files']}")

        for file in old['files']:
            status = files.delete_file(user_id=user_id, file_name=file)
            if status:
                file_id = sq_db.get_file_id_by_name(user_id=user_id, file_name=file)
                sq_db.mark_file_removed(user_id=user_id, file_id=file_id)

    if old['embeddings']:
        log.info(f"/login Removing old embeddings for user '{user_id}'")
        # Add FAISS deletion code here:
        db: T_VECTOR_STORE = app.state.vector_db.get_vector_store()
        resp = db.delete(old['embeddings'])
        if resp == True:
            sq_db.mark_embeddings_removed(vector_ids=old['embeddings'])
            log.info(f"/login Old embeddings removed for user '{user_id}'")
        else:
            log.error(f"/login Failed to remove old embeddings for user '{user_id}': {resp}")
    else:
        log.info(f"/login No old files found for user '{user_id}'")

    return {"user_id": user_id}


# ------------------------------------------------------------------------------
# File handling endpoints:
# ------------------------------------------------------------------------------

# Endpoint to receive file uploads:
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    try:
        log.info(f"/upload Received file: {file.filename} from user: {user_id}")
        filename = file.filename if file.filename else "unknown_file"

        status, message = files.save_file(
            user_id=user_id,
            file_value_binary=await file.read(),
            file_name=filename
        )

        if status:
            filename = message
            sq_db.add_file(user_id=user_id, filename=filename)
            return JSONResponse(content={"message": filename}, status_code=200)
        else:
            log.error(f"/upload File upload failed for user {user_id}: {filename}")
            return JSONResponse(content={"error": message}, status_code=500)

    except Exception as e:
        log.error(f"/upload File upload failed: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Endpoint to embed the uploaded file:
# takes user_id and file_name as input
class EmbedRequest(BaseModel):
    user_id: str
    file_name: str


@app.post("/embed")
async def embed_file(embed_request: EmbedRequest, request: Request):
    """Endpoint to embed the uploaded file.
    - Post request expects JSON `{"user_id": "", "file_name": ""}` structure.
    - Return JSON with `{"status": "success"}` or `{"error": "message"}` structure.
    """
    user_id = embed_request.user_id.strip()
    file_name = embed_request.file_name.strip()

    log.info(f"/embed Requested by '{user_id}' for file '{file_name}'")

    # Call the ingest_file function to process the file
    status, doc_ids, message = ingest_file(
        user_id=user_id,
        file_path=files.get_file_path(user_id=user_id, file_name=file_name),
        vectorstore=request.app.state.vector_db,
        embeddings=request.app.state.vector_db.get_embeddings()
    )

    if status:
        file_id = sq_db.get_file_id_by_name(user_id=user_id, file_name=file_name)
        for vid in doc_ids:
            sq_db.add_embedding(file_id=file_id, vector_id=vid)

        log.info(f"/embed Embedding completed for '{user_id}' and file '{file_name}'")
        return JSONResponse(content={"status": "success"}, status_code=200)
    else:
        log.error(f"/embed Embedding failed for '{user_id}' and file '{file_name}': {message}")
        return JSONResponse(content={"error": message}, status_code=500)


# End point to get all the files uploaded by user:
# This will be called first at initialization, and then after each file upload
@app.get("/uploads")
async def get_files(user_id: str = Query(...)):
    """Endpoint to get all the files uploaded by user.
    - Get request expects `user_id` as query parameter.
    - Return JSON with `{"files": ["file1", "file2", ...]}` structure.
    """
    log.info(f"/uploads Requested by '{user_id}'")
    files_list = sq_db.get_user_files(user_id=user_id)
    return {"files": files_list}


# Send pdf iframe based on user and file name:
# params: type=pdf/ppt/txt, user_id, file_name, num_pages
class FileIframeRequest(BaseModel):
    # type: Literal["pdf", "ppt", "txt"]
    user_id: str
    file_name: str
    num_pages: int = 5


@app.post("/iframe")
async def get_file_iframe(file_request: FileIframeRequest):
    """Endpoint to get the iframe for the file.
    - Post request expects JSON `{"user_id": "", "file_name": "", "num_pages": 5}` structure.
    - Return JSON with `{"iframe": "<iframe>...</iframe>"}` structure.
    """

    user_id = file_request.user_id.strip()
    file_name = file_request.file_name.strip()
    num_pages = file_request.num_pages

    log.info(f"/iframe Requested by '{user_id}' for file '{file_name}'")

    # Get the iframe for the requested file
    status, message = files.get_pdf_iframe(
        user_id=user_id,
        file_name=file_name,
        num_pages=num_pages
    )

    if status:
        return JSONResponse(content={"iframe": message}, status_code=200)
    else:
        return JSONResponse(content={"error": message}, status_code=404)


# ------------------------------------------------------------------------------
# RAG Chain Endpoint:
# ------------------------------------------------------------------------------

# Create endpoint for rag:
# input = {
#     query: str,
#     session_id: str,
#     dummy: bool = False
# }
# Output will be streamed in same format as the simple/streaming chat endpoint.


class RagChatRequest(BaseModel):
    query: str
    session_id: str
    dummy: bool = False


@app.post("/rag")
async def rag(request: Request, chat_request: RagChatRequest):
    """Endpoint to handle RAG (Retrieval-Augmented Generation) queries.
    - Post request expects JSON `{"query": "", "session_id": "", "dummy":T/F}` structure.
    - Return NDJSON with types "metadata", "content", "context", or "error".
    """
    rag_chain = request.app.state.rag_chain
    session_id = chat_request.session_id.strip() or "unknown_session"

    async def token_streamer():
        try:
            dummy = chat_request.dummy
            log.info(f"/rag {'dummy' if dummy else 'real'} response requested by '{session_id}'")

            # Start by sending meta data first.
            yield json.dumps({
                "type": "metadata",
                "data": {"session_id": session_id}
            }) + "\n"

            if dummy:
                # If dummy is True, stream dummy response
                resp = get_dummy_response_stream(
                    batch_tokens=config.BATCH_TOKEN_PS,
                    token_rate=config.TOKENS_PER_SEC
                )
                for chunk in resp:
                    if await request.is_disconnected():
                        log.warning(f"/rag client disconnected for '{session_id}'")
                        break

                    yield json.dumps({
                        "type": "content",
                        "data": chunk
                    }) + "\n"

            else:
                async for chunk in rag_chain.astream(
                    input={"input": chat_request.query},
                    config={"configurable": {"session_id": session_id}}
                ):
                    if await request.is_disconnected():
                        log.warning(f"/rag client disconnected for '{session_id}'")
                        break

                    # there is answer/input/context
                    if "answer" in chunk:
                        yield json.dumps({
                            "type": "content",
                            "data": chunk["answer"]
                        }) + "\n"

                    elif "context" in chunk:
                        for document in chunk["context"]:
                            if await request.is_disconnected():
                                log.warning(f"/rag client disconnected for '{session_id}'")
                                break

                            if "user_id" in document.metadata:
                                # Hide user_id from metadata
                                document.metadata.pop("user_id")
                                
                            yield json.dumps({
                                "type": "context",
                                "data": {
                                    "metadata": document.metadata,
                                    "page_content": document.page_content
                                }
                            }) + "\n"

            log.info(f"/rag Streaming completed for '{session_id}'")

        except Exception as e:
            log.exception(f"/rag Error {e} for '{session_id}'")
            yield json.dumps({
                "type": "error",
                "data": str(e)
            }) + "\n"

    return StreamingResponse(token_streamer(), media_type="text/plain")
