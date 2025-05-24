# FastAPI server which will handle all the backend and GenAI aspects of the application
# Serve this with uvicorn using: uvicorn server:app --reload


from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager


# Logger:
import logging

# LLM System Imports:
from llm_system.core.llm import get_llm, get_output_parser  # Functions
from llm_system.core.llm import get_dummy_response          # Function
from llm_system.core.llm import get_dummy_response_stream   # Function
from llm_system.core.database import VectorDB               # Class
from llm_system.core.history import HistoryStore            # Class
from llm_system.chains.rag import build_rag_chain           # Function
from llm_system import config                               # Constants

# Types for Type Hinting, Safety, and IDE Support:
from llm_system.core.llm import T_LLM

# import my logger here:
logger = logging.getLogger("rag_server")
logging.basicConfig(level=logging.INFO)


# ------------------------------------------------------------------------------
# FastAPI Startup:
# ------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Define the lifespan context manager for startup/shutdown"""

    # Startup
    app.state.llm_chat = get_llm(
        model_name=config.LLM_MODEL_NAME,
        context_size=config.MAX_CONTENT_SIZE,
        temperature=config.LLM_TEMPERATURE,
    )

    # app.state.llm_summary = get_llm(...)
    app.state.llm_summary = app.state.llm_chat

    app.state.output_parser = get_output_parser()
    app.state.vector_db = VectorDB(embed_model=config.EMB_MODEL_NAME)
    app.state.history_store = HistoryStore(logger=logger)

    app.state.rag_chain = build_rag_chain(
        llm_chat=app.state.llm_chat,
        llm_summary=app.state.llm_summary,
        retriever=app.state.vector_db.get_retriever(),
        get_history_fn=app.state.history_store.get_session_history,
    )

    logger.info("All LLM systems initialized.")

    # Lifespan
    yield

    # Shutdown
    logger.info("Shutting down LLM server...")
    # Add any cleanup part here
    # Like saving vector DB, or shutting down subprocesses


# Make one FastAPI app instance with the lifespan context manager
app = FastAPI(lifespan=lifespan)


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
    """

    llm = request.app.state.llm_chat | request.app.state.output_parser
    try:
        # Access the query and session_id from the request body
        query = chat_request.query
        session_id = chat_request.session_id
        dummy = chat_request.dummy

        if dummy:
            return get_dummy_response()

        else:
            result = await llm.ainvoke(
                input=query,
            )

            return {"response": result, "session_id": session_id}

    except Exception as e:
        logger.exception("LLM response failed.")
        return JSONResponse(status_code=500, content={"error": str(e)})


# Make one streaming endpoint for the Simple LLM response:
class StreamChatRequest(BaseModel):
    query: str
    session_id: str


@app.post("/simple/stream")
async def chat_stream(request: Request, chat_request: StreamChatRequest):
    """Endpoint to handle streaming responses for one time generation queries.
    - Post request expects JSON `{"query": "", "session_id": ""}` structure.
    """
    llm = request.app.state.llm_chat | request.app.state.output_parser

    async def token_streamer():
        try:
            async for chunk in llm.astream(chat_request.query):
                yield chunk

        except Exception as e:
            logger.exception("Streaming failed")
            yield f"[ERROR] {str(e)}"

    return StreamingResponse(token_streamer(), media_type="text/plain")
