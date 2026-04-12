import os
import re
import torch
import requests
import asyncio
import uuid
import json
import time
import base64
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime

import redis.asyncio as redis
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from bs4 import BeautifulSoup
import pypdf
from duckduckgo_search import DDGS
import docker

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rich.console import Console
import uvicorn

# Ensure python-multipart is available for file uploads
try:
    import multipart  # noqa: F401
except ImportError:
    raise RuntimeError(
        "python-multipart is required for file uploads.\n"
        "Install it with:  pip install python-multipart"
    )

from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TextIteratorStreamer
)
import ollama
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser

console = Console()

# ─────────────────────────── Lifespan ───────────────────────────
# Forward-declared; hexai instance is created after class definitions.
_hexai_instance: "HexAI | None" = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _hexai_instance
    await _hexai_instance.startup()
    yield
    await _hexai_instance.close()

app = FastAPI(title="HexAi API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────── Pydantic Models ───────────────────────────

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    stream: bool = False

class ChatResponse(BaseModel):
    session_id: str
    response: str

class EngineRequest(BaseModel):
    engine: str

class ModeRequest(BaseModel):
    mode: str

class RagRequest(BaseModel):
    query: str

class IngestRequest(BaseModel):
    source: str
    chunk_size: int = 500
    chunk_overlap: int = 100

class ImageRequest(BaseModel):
    prompt: str

class StatsResponse(BaseModel):
    engine: str
    mode: str
    vram_used_gb: Optional[float]
    vram_total_gb: Optional[float]
    active_sessions: int
    history_len: int
    model_loaded: bool
    model_idle_seconds: float

# ─────────────────────────── Model Manager ───────────────────────────

class ModelManager:
    def __init__(self, idle_timeout: int = 600):
        self.idle_timeout = idle_timeout
        self.model = None
        self.tokenizer = None
        self.last_used: Optional[float] = None
        self.loading = False
        self.lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def get_model(self):
        async with self.lock:
            if self.model is None:
                await self._load_model()
            self.last_used = time.time()
            return self.model, self.tokenizer

    async def _load_model(self):
        if self.loading:
            while self.loading:
                await asyncio.sleep(0.1)
            return
        self.loading = True
        try:
            console.print("[bold green]Loading Transformers model...[/bold green]")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_id = "mistralai/Mistral-7B-v0.1"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto"
            )
            self.last_used = time.time()
            console.print("[bold green]Model loaded.[/bold green]")
        finally:
            self.loading = False

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(60)
            async with self.lock:
                if (
                    self.model is not None
                    and self.last_used is not None
                    and (time.time() - self.last_used) > self.idle_timeout
                ):
                    console.print("[bold yellow]Unloading idle model...[/bold yellow]")
                    del self.model
                    del self.tokenizer
                    self.model = None
                    self.tokenizer = None
                    torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self.model is not None

    def idle_seconds(self) -> float:
        if self.last_used is None:
            return 0.0
        return time.time() - self.last_used

# ─────────────────────────── Advanced RAG ───────────────────────────

class AdvancedRAG:
    def __init__(self, chroma_client, collection_name: str = "hexai_docs"):
        self.chroma_client = chroma_client
        self.collection = chroma_client.get_or_create_collection(collection_name)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.bm25_index: Optional[BM25Okapi] = None
        self.documents: List[str] = []
        self.metadata: List[Dict] = []
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def add_documents(self, texts: List[str], metadata: Optional[List[Dict]] = None):
        if not texts:
            return
        if metadata is None:
            metadata = [{}] * len(texts)
        ids = [str(uuid.uuid4()) for _ in texts]
        embeddings = self.embedding_fn(texts)
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadata,
        )
        self.documents.extend(texts)
        self.metadata.extend(metadata)
        tokenized_docs = [doc.split() for doc in self.documents]
        self.bm25_index = BM25Okapi(tokenized_docs)

    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.5) -> List[str]:
        query_embedding = self.embedding_fn([query])[0]
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, max(1, self.collection.count())),
            include=["documents", "metadatas", "distances"],
        )
        combined: List[Dict] = []
        for i, doc in enumerate(vector_results["documents"][0]):
            dist = vector_results["distances"][0][i]
            combined.append({"text": doc, "score": (1 - alpha) * (1 - dist), "source": "vector"})

        if self.bm25_index is not None and self.documents:
            bm25_scores = self.bm25_index.get_scores(query.split())
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
            max_score = max((bm25_scores[i] for i in top_indices), default=1e-6)
            for idx in top_indices:
                combined.append({
                    "text": self.documents[idx],
                    "score": alpha * (bm25_scores[idx] / (max_score + 1e-6)),
                    "source": "bm25",
                })

        combined.sort(key=lambda x: x["score"], reverse=True)
        top_k = combined[:k]
        if not top_k:
            return []

        pairs = [(query, doc["text"]) for doc in top_k]
        rerank_scores = self.cross_encoder.predict(pairs)
        for i, score in enumerate(rerank_scores):
            top_k[i]["rerank_score"] = float(score)
        top_k.sort(key=lambda x: x["rerank_score"], reverse=True)
        return [doc["text"] for doc in top_k]

    def search(self, query: str, k: int = 5) -> List[str]:
        if self.collection.count() == 0:
            return []
        query_embedding = self.embedding_fn([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=["documents"],
        )
        return results["documents"][0]

# ─────────────────────────── Docker Executor ───────────────────────────

class DockerExecutor:
    def __init__(self, image: str = "python:3.11-slim", timeout: int = 10):
        self.image = image
        self.timeout = timeout
        try:
            self.docker_client = docker.from_env()
        except Exception:
            self.docker_client = None

    async def run_code(self, code: str) -> str:
        if self.docker_client is None:
            return "Docker nie jest dostępny."
        try:
            safe_code = code.replace('"""', '\\"\\"\\"')
            script = f'exec("""{safe_code}""")'
            container = self.docker_client.containers.run(
                self.image,
                command=["python", "-c", script],
                detach=True,
                mem_limit="128m",
                memswap_limit="256m",
                cpu_period=100000,
                cpu_quota=50000,
                network_disabled=True,
                remove=False,
                user="nobody",
            )
            try:
                container.wait(timeout=self.timeout)
                logs = container.logs().decode("utf-8", errors="replace")
            finally:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
            return logs or "(brak wyjścia)"
        except Exception as e:
            return f"Błąd wykonania: {e}"

# ─────────────────────────── User Profiler ───────────────────────────

class UserProfiler:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self._fallback: dict[str, list] = {}

    async def add_fact(self, user_id: str, fact: str):
        if user_id not in self._fallback:
            self._fallback[user_id] = []
        self._fallback[user_id].insert(0, fact)
        try:
            key = f"user_facts:{user_id}"
            await self.redis.lpush(key, fact)
            await self.redis.expire(key, 86400 * 30)
        except Exception:
            pass

    async def get_facts(self, user_id: str, limit: int = 10) -> List[str]:
        try:
            key = f"user_facts:{user_id}"
            facts = await self.redis.lrange(key, 0, limit - 1)
            return [f.decode() if isinstance(f, bytes) else f for f in facts]
        except Exception:
            return self._fallback.get(user_id, [])[:limit]

    async def update_profile(self, user_id: str, message: str, _response: str):
        patterns = [
            (r"(?:preferuję|wolę|używam)\s+([\w\s]+)", "preference"),
            (r"(?:pracuję|używam)\s+(?:na|w)\s+([\w\s]+)", "tech_stack"),
        ]
        for pattern, fact_type in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                await self.add_fact(user_id, f"{fact_type}: {match.group(1).strip()}")

# ─────────────────────────── Persistent Memory ───────────────────────────

class PersistentMemory:
    def __init__(self, redis_client: redis.Redis, ttl: int = 86400):
        self.redis = redis_client
        self.ttl = ttl
        # In-memory fallback when Redis is unavailable
        self._fallback: dict[str, list] = {}

    async def _redis_ok(self) -> bool:
        try:
            await self.redis.ping()
            return True
        except Exception:
            return False

    async def get_history(self, session_id: str) -> List[tuple]:
        try:
            key = f"session:{session_id}"
            raw = await self.redis.lrange(key, 0, -1)
            history = []
            for item in raw:
                data = item.decode() if isinstance(item, bytes) else item
                pair = json.loads(data)
                history.append((pair[0], pair[1]))
            return history
        except Exception:
            return [(u, a) for u, a in self._fallback.get(session_id, [])]

    async def add_message(self, session_id: str, user_msg: str, assistant_msg: str):
        # Always update in-memory fallback
        if session_id not in self._fallback:
            self._fallback[session_id] = []
        self._fallback[session_id].append((user_msg, assistant_msg))
        # Keep fallback bounded
        if len(self._fallback[session_id]) > 50:
            self._fallback[session_id] = self._fallback[session_id][-50:]
        try:
            key = f"session:{session_id}"
            await self.redis.rpush(key, json.dumps([user_msg, assistant_msg]))
            await self.redis.expire(key, self.ttl)
        except Exception:
            pass  # Already stored in fallback

    async def clear_session(self, session_id: str):
        self._fallback.pop(session_id, None)
        try:
            await self.redis.delete(f"session:{session_id}")
        except Exception:
            pass

    async def list_sessions(self) -> List[str]:
        try:
            keys = await self.redis.keys("session:*")
            redis_sessions = [k.decode().split(":", 1)[1] if isinstance(k, bytes) else k.split(":", 1)[1] for k in keys]
            # Merge with fallback sessions
            all_sessions = list(set(redis_sessions) | set(self._fallback.keys()))
            return all_sessions
        except Exception:
            return list(self._fallback.keys())

# ─────────────────────────── HexAI Core ───────────────────────────

class HexAI:
    def __init__(self):
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=False)
        self.persistent_memory = PersistentMemory(self.redis)
        self.user_profiler = UserProfiler(self.redis)
        self.model_manager = ModelManager(idle_timeout=600)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.advanced_rag = AdvancedRAG(self.chroma_client)
        self.docker_executor = DockerExecutor()
        self.engine = "transformers"
        self.mode = "general"
        self.max_tool_iterations = 3
        self.system_prompts = {
            "general": (
                "Jesteś HexAi – inteligentnym, pomocnym asystentem AI. "
                "Odpowiadaj precyzyjnie, zwięźle i uprzejmie."
            ),
            "programista": (
                "Jesteś HexAi – ekspertem programistycznym. "
                "Twój kod jest czysty, wydajny, dobrze skomentowany i zgodny z najlepszymi praktykami. "
                "Zawsze podajesz kompletne przykłady z obsługą błędów. "
                "Wyjaśniaj swoje rozwiązania krok po kroku."
            ),
        }
        # Lazy-loaded models
        self.whisper_model = None
        self.diffuser_pipe = None
        self.tts_model = None
        self.vision_model = None
        self.vision_processor = None

    async def startup(self):
        await self.model_manager.start()
        # Check Redis connectivity – warn but don't crash if unavailable
        try:
            await self.redis.ping()
            console.print("[bold green]✓ Redis połączony[/bold green]")
        except Exception:
            console.print(
                "[bold yellow]⚠ Redis niedostępny (localhost:6379) – "
                "historia sesji będzie przechowywana tylko w pamięci RAM.[/bold yellow]\n"
                "[yellow]Uruchom Redis: docker run -d -p 6379:6379 redis:7-alpine[/yellow]"
            )

    async def close(self):
        await self.model_manager.stop()
        await self.redis.aclose()

    # ── Sanitization ──

    def sanitize_prompt(self, prompt: str) -> str:
        prompt = re.sub(
            r"(?i)(zignoruj|ignore|forget|przestań|nie\s+słuchaj).*?instrukcje",
            "[FILTERED]",
            prompt,
        )
        prompt = re.sub(
            r"(?i)(podaj|give|show).*(hasło|password|secret|token|klucz|key)",
            "[FILTERED]",
            prompt,
        )
        return prompt.strip()

    # ── Long-term memory ──

    async def index_conversation(self, session_id: str, user_msg: str, assistant_msg: str):
        text = f"Użytkownik: {user_msg}\nAsystent: {assistant_msg}"
        meta = {"session_id": session_id, "timestamp": datetime.now().isoformat(), "type": "conversation"}
        self.advanced_rag.add_documents([text], [meta])

    async def recall_past_conversations(self, query: str, limit: int = 3) -> List[str]:
        return self.advanced_rag.search(query, k=limit)

    # ── Generation (Transformers) ──

    async def _generate_transformers(
        self, prompt: str, history: List[tuple], stream: bool = False
    ) -> AsyncGenerator[str, None]:
        """Always an async generator – yields chunks or single full response."""
        model, tokenizer = await self.model_manager.get_model()
        system_prompt = self.system_prompts.get(self.mode, self.system_prompts["general"])

        formatted = f"[INST] {system_prompt} [/INST]\n"
        for u, a in history:
            formatted += f"[INST] {u} [/INST] {a} </s>\n"
        formatted += f"[INST] {prompt} [/INST]"

        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        if stream:
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=30)
            import threading
            gen_kwargs = dict(**inputs, max_new_tokens=512, streamer=streamer, do_sample=True, temperature=0.7)
            thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()
            for token in streamer:
                yield token
        else:
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = text.split("[/INST]")[-1].strip()
            yield response

    # ── Generation (Ollama) ──

    async def _generate_ollama(
        self, prompt: str, history: List[tuple], stream: bool = False
    ) -> AsyncGenerator[str, None]:
        system_prompt = self.system_prompts.get(self.mode, self.system_prompts["general"])
        messages = []
        for u, a in history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": prompt})

        if stream:
            response_stream = ollama.chat(
                model=self._ollama_model(),
                messages=messages,
                system=system_prompt,
                stream=True,
            )
            for chunk in response_stream:
                yield chunk["message"]["content"]
        else:
            response = ollama.chat(
                model=self._ollama_model(),
                messages=messages,
                system=system_prompt,
            )
            yield response["message"]["content"]

    def _ollama_model(self) -> str:
        return "codellama:7b-instruct" if self.mode == "programista" else "llama2"

    # ── Generator dispatcher ──

    def _get_generator(self, prompt: str, history: List[tuple], stream: bool) -> AsyncGenerator[str, None]:
        if self.engine == "transformers":
            return self._generate_transformers(prompt, history, stream)
        return self._generate_ollama(prompt, history, stream)

    # ── Tool helpers ──

    def parse_tool_call(self, text: str) -> List[tuple]:
        pattern = r"\{tool:(\w+)\s+(.+?)\}"
        matches = re.findall(pattern, text)
        result = []
        for tool_name, args_str in matches:
            args = dict(re.findall(r'(\w+)="([^"]+)"', args_str))
            result.append((tool_name, args))
        return result

    def execute_tool(self, tool_name: str, args: Dict) -> str:
        tools = {
            "get_weather": self.get_weather,
            "search_web": self.search_web,
        }
        fn = tools.get(tool_name)
        if fn is None:
            return f"Narzędzie '{tool_name}' nie istnieje."
        try:
            return fn(**args)
        except Exception as e:
            return f"Błąd narzędzia: {e}"

    def get_weather(self, city: str = "Warsaw") -> str:
        return f"Pogoda w {city}: słonecznie, 22°C."

    def search_web(self, query: str = "") -> str:
        api_key = os.environ.get("SERPER_API_KEY")
        if api_key:
            try:
                url = "https://google.serper.dev/search"
                headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
                data = json.dumps({"q": query, "num": 5})
                resp = requests.post(url, headers=headers, data=data, timeout=10)
                if resp.ok:
                    items = resp.json().get("organic", [])
                    return "\n".join(f"{i['title']}: {i['snippet']}" for i in items)
            except Exception as e:
                return f"Błąd Serper: {e}"
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
            return "\n".join(f"{r['title']}: {r['body']}" for r in results)
        except Exception as e:
            return f"Błąd wyszukiwania: {e}"

    # ── Core chat helpers ──

    async def _build_prompt_with_context(self, session_id: str, prompt: str) -> tuple[str, List[tuple]]:
        """Returns (enriched_prompt, history_list)."""
        prompt = self.sanitize_prompt(prompt)
        history = await self.persistent_memory.get_history(session_id)

        if re.search(r"(pamiętasz|co ustaliliśmy|co mówiliśmy|przypomnij)", prompt, re.IGNORECASE):
            past = await self.recall_past_conversations(prompt)
            if past:
                prompt += "\n\nPoprzednie rozmowy:\n" + "\n".join(past)

        facts = await self.user_profiler.get_facts(session_id)
        if facts:
            prompt = "Fakty o użytkowniku: " + ", ".join(facts) + "\n\n" + prompt

        return prompt, list(history)

    async def _run_tool_loop_stream(
        self, session_id: str, prompt: str, history: List[tuple]
    ) -> AsyncGenerator[str, None]:
        current_prompt = prompt
        history_copy = history + [(prompt, "")]
        final_response = ""

        for iteration in range(self.max_tool_iterations):
            gen = self._get_generator(current_prompt, history_copy[:-1], stream=True)
            chunks: List[str] = []
            async for chunk in gen:
                chunks.append(chunk)
                yield chunk
            full = "".join(chunks)

            tool_calls = self.parse_tool_call(full)
            if not tool_calls:
                history_copy[-1] = (prompt, full)
                final_response = full
                break

            tool_results = []
            for tname, targs in tool_calls:
                res = await asyncio.to_thread(self.execute_tool, tname, targs)
                tool_results.append(f"Narzędzie {tname}: {res}")

            clean = re.sub(r"\{tool:.*?\}", "", full).strip()
            history_copy[-1] = (prompt, clean)
            summary = "\n".join(tool_results)
            history_copy.append(("(tool results)", summary))
            current_prompt = f"Kontynuuj uwzględniając wyniki narzędzi: {summary}"

        if final_response:
            await self.persistent_memory.add_message(session_id, prompt, final_response)
            await self.index_conversation(session_id, prompt, final_response)
            await self.user_profiler.update_profile(session_id, prompt, final_response)

    async def _run_tool_loop_sync(
        self, session_id: str, prompt: str, history: List[tuple]
    ) -> str:
        current_prompt = prompt
        history_copy = history + [(prompt, "")]
        final_response = ""

        for iteration in range(self.max_tool_iterations):
            gen = self._get_generator(current_prompt, history_copy[:-1], stream=False)
            chunks: List[str] = []
            async for chunk in gen:
                chunks.append(chunk)
            full = "".join(chunks)

            tool_calls = self.parse_tool_call(full)
            if not tool_calls:
                history_copy[-1] = (prompt, full)
                final_response = full
                break

            tool_results = []
            for tname, targs in tool_calls:
                res = await asyncio.to_thread(self.execute_tool, tname, targs)
                tool_results.append(f"Narzędzie {tname}: {res}")

            clean = re.sub(r"\{tool:.*?\}", "", full).strip()
            history_copy[-1] = (prompt, clean)
            summary = "\n".join(tool_results)
            history_copy.append(("(tool results)", summary))
            current_prompt = f"Kontynuuj uwzględniając wyniki narzędzi: {summary}"

        if final_response:
            await self.persistent_memory.add_message(session_id, prompt, final_response)
            await self.index_conversation(session_id, prompt, final_response)
            await self.user_profiler.update_profile(session_id, prompt, final_response)

        return final_response

    async def chat(self, session_id: str, prompt: str, stream: bool = False):
        enriched, history = await self._build_prompt_with_context(session_id, prompt)
        if stream:
            return self._run_tool_loop_stream(session_id, enriched, history)
        return await self._run_tool_loop_sync(session_id, enriched, history)

    # ── RAG ──

    async def rag_query(self, query: str) -> str:
        results = self.advanced_rag.hybrid_search(query, k=3)
        if not results:
            context = "Brak dokumentów w bazie wiedzy."
        else:
            context = "\n".join(results)
        prompt = f"Odpowiedz na pytanie na podstawie kontekstu:\n\n{context}\n\nPytanie: {query}"
        chunks: List[str] = []
        async for chunk in self._get_generator(prompt, [], stream=False):
            chunks.append(chunk)
        return "".join(chunks)

    async def ingest_document(self, source: str, chunk_size: int = 500, chunk_overlap: int = 100) -> str:
        text = ""
        try:
            if source.startswith("http://") or source.startswith("https://"):
                resp = requests.get(source, timeout=30)
                ct = resp.headers.get("Content-Type", "")
                if "application/pdf" in ct:
                    reader = pypdf.PdfReader(BytesIO(resp.content))
                    text = "".join(p.extract_text() or "" for p in reader.pages)
                else:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
            else:
                if not os.path.exists(source):
                    return f"Plik {source} nie istnieje."
                if source.endswith(".pdf"):
                    reader = pypdf.PdfReader(source)
                    text = "".join(p.extract_text() or "" for p in reader.pages)
                else:
                    with open(source, encoding="utf-8") as f:
                        text = f.read()
        except Exception as e:
            return f"Błąd pobierania: {e}"

        if not text.strip():
            return "Dokument jest pusty lub nie można wyodrębnić tekstu."

        parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = parser.get_nodes_from_documents([Document(text=text)])
        texts = [n.text for n in nodes]
        metas = [{"source": source, "chunk": i} for i in range(len(texts))]
        self.advanced_rag.add_documents(texts, metas)
        return f"Zindeksowano {len(texts)} fragmentów z {source}"

    # ── Vision ──

    async def _load_vision_model(self):
        if self.vision_model is None:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            model_id = "llava-hf/llava-1.5-7b-hf"
            self.vision_model = LlavaForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map="auto"
            )
            self.vision_processor = AutoProcessor.from_pretrained(model_id)

    async def analyze_image(self, image_bytes: bytes, prompt: str = "Describe this image.") -> str:
        await self._load_vision_model()
        from PIL import Image
        image = Image.open(BytesIO(image_bytes))
        inputs = self.vision_processor(images=image, text=prompt, return_tensors="pt").to(self.vision_model.device)
        with torch.inference_mode():
            outputs = self.vision_model.generate(**inputs, max_new_tokens=200)
        return self.vision_processor.decode(outputs[0], skip_special_tokens=True)

    # ── TTS ──

    async def _load_tts(self):
        if self.tts_model is None:
            from melo.api import TTS
            self.tts_model = TTS(language="PL")

    async def text_to_speech(self, text: str) -> bytes:
        await self._load_tts()
        path = f"/tmp/hexai_tts_{uuid.uuid4().hex}.wav"
        spk_id = self.tts_model.hps.data.spk2id.get("PL", 0)
        self.tts_model.tts_to_file(text, spk_id, path, speed=1.0)
        with open(path, "rb") as f:
            data = f.read()
        os.remove(path)
        return data

    # ── Whisper ──

    async def _load_whisper(self):
        if self.whisper_model is None:
            import whisper
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = whisper.load_model("small", device=device)

    async def transcribe_audio(self, file_path: str) -> str:
        await self._load_whisper()
        result = await asyncio.to_thread(self.whisper_model.transcribe, file_path)
        return result["text"]

    # ── Image generation ──

    async def _load_diffuser(self):
        if self.diffuser_pipe is None:
            from diffusers import StableDiffusionPipeline
            self.diffuser_pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
            )
            if torch.cuda.is_available():
                self.diffuser_pipe = self.diffuser_pipe.to("cuda")

    async def generate_image(self, prompt: str) -> str:
        await self._load_diffuser()
        image_result = await asyncio.to_thread(self.diffuser_pipe, prompt)
        image = image_result.images[0]
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    # ── Stats ──

    async def get_stats(self) -> Dict:
        vram_used = vram_total = None
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024 ** 3
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        try:
            sessions = await self.persistent_memory.list_sessions()
            n_sessions = len(sessions)
        except Exception:
            n_sessions = 0
        return {
            "engine": self.engine,
            "mode": self.mode,
            "vram_used_gb": vram_used,
            "vram_total_gb": vram_total,
            "active_sessions": n_sessions,
            "history_len": 0,
            "model_loaded": self.model_manager.is_loaded(),
            "model_idle_seconds": self.model_manager.idle_seconds(),
        }


# ─────────────────────────── App Lifecycle ───────────────────────────

hexai = HexAI()
_hexai_instance = hexai  # wire into lifespan


# ─────────────────────────── Endpoints ───────────────────────────

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    if request.stream:
        generator = await hexai.chat(session_id, request.message, stream=True)
        return StreamingResponse(generator, media_type="text/plain")
    else:
        response = await hexai.chat(session_id, request.message, stream=False)
        return ChatResponse(session_id=session_id, response=response)


@app.post("/engine")
async def set_engine(request: EngineRequest):
    if request.engine not in ("transformers", "ollama"):
        raise HTTPException(400, "Nieprawidłowy silnik")
    hexai.engine = request.engine
    return {"message": f"Silnik: {request.engine}"}


@app.post("/mode")
async def set_mode(request: ModeRequest):
    if request.mode not in ("general", "programista"):
        raise HTTPException(400, "Nieprawidłowy tryb")
    hexai.mode = request.mode
    return {"message": f"Tryb: {request.mode}"}


@app.post("/rag")
async def rag_query(request: RagRequest):
    response = await hexai.rag_query(request.query)
    return {"response": response}


@app.post("/ingest")
async def ingest_document(request: IngestRequest):
    result = await hexai.ingest_document(request.source, request.chunk_size, request.chunk_overlap)
    return {"message": result}


@app.post("/generate_image")
async def generate_image(request: ImageRequest):
    img = await hexai.generate_image(request.prompt)
    return {"image_base64": img}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    path = f"/tmp/hexai_audio_{uuid.uuid4().hex}_{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    try:
        transcription = await hexai.transcribe_audio(path)
    finally:
        if os.path.exists(path):
            os.remove(path)
    return {"transcription": transcription}


@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...), prompt: str = Form("Describe this image.")):
    data = await file.read()
    description = await hexai.analyze_image(data, prompt)
    return {"description": description}


@app.get("/tts")
async def text_to_speech(text: str):
    audio = await hexai.text_to_speech(text)
    return StreamingResponse(BytesIO(audio), media_type="audio/wav")


@app.get("/stats", response_model=StatsResponse)
async def stats():
    data = await hexai.get_stats()
    return StatsResponse(**data)


@app.get("/sessions")
async def list_sessions():
    sessions = await hexai.persistent_memory.list_sessions()
    return {"sessions": sessions}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    await hexai.persistent_memory.clear_session(session_id)
    return {"message": f"Sesja {session_id} usunięta"}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
