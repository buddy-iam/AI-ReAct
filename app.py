import os
import datetime
import json
import asyncio
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# --- Environment Variables ---
# Set these in your hosting dashboard or local .env file
os.environ.setdefault("WIKIPEDIA_USER_AGENT", "LangChain-ReAct-Agent/1.0")

app = FastAPI(title="LangGraph ReAct Backend")

# --- Enable CORS for GitHub Pages & Localhost ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows GitHub Pages and local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define Agent Tools ---
@tool
def get_current_datetime() -> str:
    """Returns the current date, time, day of the week, and UTC offset."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("Current UTC Time: %Y-%m-%d %H:%M:%S (Day: %A, %Z)")

@tool
def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b

wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1200))
tavily = TavilySearchResults(max_results=3)

tools = [get_current_datetime, wiki, tavily, add, multiply]

# --- Agent Initialization ---
llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)
prompt_template = (
    "You are an expert AI assistant with access to real-time tools. "
    "Whenever a question depends on the current year, month, or time, use `get_current_datetime` first. "
    "When you use Wikipedia or Tavily to gather information, always state the sources or URLs you consulted in your answer."
)
agent_executor = create_react_agent(llm, tools, prompt=prompt_template)

class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def health_check():
    return {"status": "ok", "message": "LangGraph ReAct Agent Backend is running."}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    async def event_generator():
        sources_found = []

        async for chunk in agent_executor.astream({"messages": [("user", request.prompt)]}):
            # Handle AI reasoning & tool calls
            if "agent" in chunk:
                messages = chunk["agent"]["messages"]
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for call in msg.tool_calls:
                            log = f"🛠️ **Tool Chosen:** `{call['name']}`  \n📥 **Arguments Passed:** `{json.dumps(call['args'])}`"
                            yield f"data: {json.dumps({'type': 'thought', 'content': log})}\n\n"

                    if msg.content:
                        yield f"data: {json.dumps({'type': 'message', 'content': msg.content})}\n\n"

            # Handle tool observations & resource parsing
            elif "tools" in chunk:
                messages = chunk["tools"]["messages"]
                for msg in messages:
                    raw_content = str(msg.content)

                    # 1. Tavily output extraction
                    try:
                        if raw_content.startswith("["):
                            parsed = eval(raw_content)
                            for item in parsed:
                                if isinstance(item, dict) and "url" in item:
                                    title = item.get("title", "Web Page")
                                    url = item["url"]
                                    log = f"🔗 **[RESOURCE EXTRACTED]:** [{title}]({url})"
                                    yield f"data: {json.dumps({'type': 'thought', 'content': log})}\n\n"

                                    if url not in [s["url"] for s in sources_found]:
                                        sources_found.append({"title": title, "url": url})
                                        yield f"data: {json.dumps({'type': 'source', 'title': title, 'url': url})}\n\n"
                    except Exception:
                        pass

                    # 2. Wikipedia output extraction
                    if getattr(msg, "name", None) == "wikipedia":
                        log = "📚 **[RESOURCE EXTRACTED]:** Wikipedia search query executed."
                        yield f"data: {json.dumps({'type': 'thought', 'content': log})}\n\n"

                        if not any(s["title"] == "Wikipedia Database" for s in sources_found):
                            sources_found.append({"title": "Wikipedia Database", "url": "#"})
                            yield f"data: {json.dumps({'type': 'source', 'title': 'Wikipedia Database', 'url': '#'})}\n\n"

                    # 3. Raw observation snippet
                    preview = raw_content[:300].replace("\n", " ")
                    yield f"data: {json.dumps({'type': 'thought', 'content': f'👀 **Raw Observation:** `{preview}...`'})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")