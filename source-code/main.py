import os
import re
import torch
import requests
import asyncio
import uuid
import json
import time
import base64
from io import BytesIO
from collections import deque
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime, timedelta

import redis.asyncio as redis
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from bs4 import BeautifulSoup
import pypdf
from duckduckgo_search import DDGS
import docker
from docker.types import DeviceRequest

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rich.console import Console
import uvicorn

# Importy modeli
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    pipeline, TextIteratorStreamer
)
import ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.node_parser import SimpleNodeParser
import whisper  # openai-whisper
from diffusers import StableDiffusionPipeline
import torchaudio
import soundfile as sf
from melo.api import TTS

app = FastAPI(title="HexAi API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
console = Console()

# ------------------------- Modele danych Pydantic -------------------------
class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    stream: bool = False

class ChatResponse(BaseModel):
    session_id: str
    response: str

class EngineRequest(BaseModel):
    engine: str  # "transformers" lub "ollama"

class ModeRequest(BaseModel):
    mode: str  # "general" lub "programista"

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
    vram_used_gb: float | None
    vram_total_gb: float | None
    active_sessions: int
    history_len: int
    model_loaded: bool
    model_idle_seconds: float

# ------------------------- Zarządzanie modelami (TTL) -------------------------
class ModelManager:
    def __init__(self, idle_timeout: int = 600):
        self.idle_timeout = idle_timeout
        self.model = None
        self.tokenizer = None
        self.last_used = None
        self.loading = False
        self.lock = asyncio.Lock()
        self._cleanup_task = None

    async def start(self):
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            await self._cleanup_task

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
                if self.model is not None and self.last_used and (time.time() - self.last_used) > self.idle_timeout:
                    console.print("[bold yellow]Unloading idle model...[/bold yellow]")
                    del self.model
                    del self.tokenizer
                    self.model = None
                    self.tokenizer = None
                    torch.cuda.empty_cache()

    def is_loaded(self):
        return self.model is not None

    def idle_seconds(self):
        if self.last_used is None:
            return 0
        return time.time() - self.last_used

# ------------------------- Zaawansowany RAG z chunkingiem, hybrydą i rerankingiem -------------------------
class AdvancedRAG:
    def __init__(self, chroma_client, collection_name="hexai_docs"):
        self.chroma_client = chroma_client
        self.collection = chroma_client.get_or_create_collection(collection_name)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.bm25_index = None
        self.bm25_docs = []
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.documents = []  # store raw text for BM25
        self.metadata = []   # store metadata per doc

    def add_documents(self, texts: List[str], metadata: List[Dict] = None):
        """Add documents to both Chroma (vector) and BM25 index."""
        if metadata is None:
            metadata = [{}] * len(texts)
        # Add to Chroma
        ids = [str(uuid.uuid4()) for _ in texts]
        embeddings = self.embedding_fn(texts)
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadata
        )
        # Update BM25
        self.documents.extend(texts)
        self.metadata.extend(metadata)
        # Rebuild BM25 index if we have enough docs
        if len(self.documents) > 0:
            tokenized_docs = [doc.split() for doc in self.documents]
            self.bm25_index = BM25Okapi(tokenized_docs)

    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.5):
        """
        Perform hybrid search: vector + BM25, then rerank.
        """
        # Vector search
        query_embedding = self.embedding_fn([query])[0]
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        # BM25 search
        if self.bm25_index is not None:
            tokenized_query = query.split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            # Get top k BM25 indices
            top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
            bm25_docs = [self.documents[i] for i in top_bm25_indices]
            bm25_scores_list = [bm25_scores[i] for i in top_bm25_indices]
        else:
            bm25_docs = []
            bm25_scores_list = []

        # Combine results (simple linear combination)
        combined = []
        # Add vector results
        for i, doc in enumerate(vector_results['documents'][0]):
            combined.append({
                'text': doc,
                'score': (1 - alpha) * (1 - vector_results['distances'][0][i]),  # convert distance to similarity
                'source': 'vector'
            })
        # Add BM25 results
        for doc, score in zip(bm25_docs, bm25_scores_list):
            combined.append({
                'text': doc,
                'score': alpha * (score / (max(bm25_scores_list) + 1e-6)),
                'source': 'bm25'
            })
        # Sort by combined score
        combined.sort(key=lambda x: x['score'], reverse=True)
        top_k = combined[:k]

        # Rerank with cross-encoder
        pairs = [(query, doc['text']) for doc in top_k]
        rerank_scores = self.cross_encoder.predict(pairs)
        for i, score in enumerate(rerank_scores):
            top_k[i]['rerank_score'] = score
        # Sort by rerank score
        top_k.sort(key=lambda x: x['rerank_score'], reverse=True)
        return [doc['text'] for doc in top_k]

    def search(self, query: str, k: int = 5):
        """Simple vector search (for compatibility)."""
        query_embedding = self.embedding_fn([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents"]
        )
        return results['documents'][0]

# ------------------------- Bezpieczne wykonywanie kodu (Docker) -------------------------
class DockerExecutor:
    def __init__(self, image: str = "python:3.10-slim", timeout: int = 5):
        self.docker_client = docker.from_env()
        self.image = image
        self.timeout = timeout

    async def run_code(self, code: str) -> str:
        """Execute Python code in a temporary container with resource limits."""
        try:
            # Create a temporary script
            script = f"""
import sys
import io
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()
try:
    {code}
    print(sys.stdout.getvalue())
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    print(sys.stderr.getvalue())
            """
            # Run container
            container = self.docker_client.containers.run(
                self.image,
                command=["python", "-c", script],
                detach=True,
                mem_limit="128m",
                memswap_limit="256m",
                cpu_period=100000,
                cpu_quota=50000,
                network_disabled=True,  # no network access
                remove=True,
                user="nobody",  # run as non-root
            )
            # Wait for container to finish
            result = container.wait(timeout=self.timeout)
            if result['StatusCode'] != 0:
                logs = container.logs().decode()
                return f"Execution failed with status {result['StatusCode']}:\n{logs}"
            logs = container.logs().decode()
            return logs
        except docker.errors.ContainerError as e:
            return f"Container error: {e}"
        except docker.errors.APIError as e:
            return f"Docker API error: {e}"
        except Exception as e:
            return f"Unexpected error: {e}"

# ------------------------- Profilowanie użytkownika (fakty) -------------------------
class UserProfiler:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def add_fact(self, user_id: str, fact: str):
        """Store a fact about the user."""
        key = f"user_facts:{user_id}"
        await self.redis.lpush(key, fact)
        await self.redis.expire(key, 86400 * 30)  # keep for 30 days

    async def get_facts(self, user_id: str, limit: int = 10) -> List[str]:
        """Retrieve stored facts for a user."""
        key = f"user_facts:{user_id}"
        facts = await self.redis.lrange(key, 0, limit - 1)
        return [fact.decode() for fact in facts]

    async def update_profile(self, user_id: str, message: str, response: str):
        """Extract facts from conversation (simple keyword-based)."""
        # Simple extraction: look for patterns like "preferuję X" or "używam Y"
        patterns = [
            (r"(?:preferuję|wolę|preferuję|używam) ([\w\s]+)", "preference"),
            (r"(?:pracuję|używam) (?:na|w) ([\w\s]+)", "tech_stack"),
        ]
        for pattern, fact_type in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                fact = f"{fact_type}: {match.group(1)}"
                await self.add_fact(user_id, fact)

# ------------------------- Pamięć trwała (Redis) -------------------------
class PersistentMemory:
    def __init__(self, redis_client: redis.Redis, ttl: int = 86400):
        self.redis = redis_client
        self.ttl = ttl

    async def get_history(self, session_id: str) -> List[tuple]:
        """Get conversation history as list of (user, assistant) tuples."""
        key = f"session:{session_id}"
        raw = await self.redis.lrange(key, 0, -1)
        history = []
        for item in raw:
            user, assistant = json.loads(item)
            history.append((user, assistant))
        return history

    async def add_message(self, session_id: str, user_msg: str, assistant_msg: str):
        """Append a message pair to history."""
        key = f"session:{session_id}"
        value = json.dumps([user_msg, assistant_msg])
        await self.redis.rpush(key, value)
        await self.redis.expire(key, self.ttl)

    async def clear_session(self, session_id: str):
        """Delete all history for a session."""
        key = f"session:{session_id}"
        await self.redis.delete(key)

    async def list_sessions(self) -> List[str]:
        """Get all session keys (pattern: session:*)."""
        keys = await self.redis.keys("session:*")
        return [k.decode().split(":")[1] for k in keys]

# ------------------------- Główna klasa HexAI -------------------------
class HexAI:
    def __init__(self):
        # Połączenie z Redis
        self.redis = redis.Redis(host='localhost', port=6379, decode_responses=False)
        self.persistent_memory = PersistentMemory(self.redis)
        self.user_profiler = UserProfiler(self.redis)

        # Model manager dla Transformers
        self.model_manager = ModelManager(idle_timeout=600)  # 10 minut bezczynności
        asyncio.create_task(self.model_manager.start())

        # RAG
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.advanced_rag = AdvancedRAG(self.chroma_client)

        # Bezpieczne wykonywanie kodu
        self.docker_executor = DockerExecutor()

        # Silnik i tryb
        self.engine = 'transformers'
        self.mode = 'general'

        # Narzędzia
        self.tools = {
            'get_weather': self.get_weather,
            'search_web': self.search_web,
            'run_python': self.run_python
        }
        self.max_tool_iterations = 3

        # System prompts
        self.system_prompts = {
            'general': "Jesteś pomocnym asystentem AI.",
            'programista': (
                "Jesteś ekspertem programistycznym na poziomie Claude. "
                "Twój kod jest czysty, wydajny, dobrze skomentowany i zgodny z najlepszymi praktykami. "
                "Zawsze podajesz kompletne przykłady, uwzględniasz obsługę błędów i wyjaśniasz swoje rozwiązania. "
                "Jeśli to możliwe, podajesz kod w różnych językach programowania i dostosowujesz go do kontekstu."
            )
        }

        # Inicjalizacja innych komponentów (lazy loading)
        self.whisper_model = None
        self.diffuser_pipe = None
        self.tts_model = None
        self.vision_model = None
        self.vision_processor = None

    async def close(self):
        await self.model_manager.stop()
        await self.redis.close()

    # ------------------------- Bezpieczeństwo: sanityzacja promptów -------------------------
    def sanitize_prompt(self, prompt: str) -> str:
        """Detect and neutralize prompt injection attempts."""
        # Proste reguły: usuwanie prób zmiany roli asystenta
        prompt = re.sub(r"(?i)(zignoruj|ignore|forget|przestań|nie słuchaj).*poprzednie.*instrukcje", "[FILTERED]", prompt)
        prompt = re.sub(r"(?i)(podaj|give|show).*(hasło|password|secret|token|klucz|key)", "[FILTERED]", prompt)
        return prompt

    # ------------------------- Pamięć długoterminowa (indeksowanie rozmów) -------------------------
    async def index_conversation(self, session_id: str, user_msg: str, assistant_msg: str):
        """Automatically index conversation snippets into ChromaDB for long-term recall."""
        # Create a combined text with metadata
        text = f"Użytkownik: {user_msg}\nAsystent: {assistant_msg}"
        metadata = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "type": "conversation"
        }
        self.advanced_rag.add_documents([text], [metadata])

    async def recall_past_conversations(self, query: str, limit: int = 3) -> List[str]:
        """Retrieve relevant past conversations from vector DB."""
        results = self.advanced_rag.search(query, k=limit)
        return results

    # ------------------------- Generowanie odpowiedzi (z streamingiem) -------------------------
    def _build_system_prompt(self) -> str:
        """Zwraca systemowy prompt dla bieżącego trybu."""
        return self.system_prompts.get(self.mode, self.system_prompts['general'])

    async def generate_response_transformers(self, prompt: str, history: List[tuple], stream: bool = False):
        """Generowanie odpowiedzi przez Transformers z opcją streamingu."""
        model, tokenizer = await self.model_manager.get_model()
        system_prompt = self._build_system_prompt()
        formatted_history = f"[INST] {system_prompt} [/INST] (System prompt) </s>\n"
        for user_msg, assistant_msg in history:
            formatted_history += f"[INST] {user_msg} [/INST] {assistant_msg} </s>\n"
        formatted_history += f"[INST] {prompt} [/INST]"
        inputs = tokenizer(formatted_history, return_tensors="pt").to(model.device)

        if stream:
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=10)
            generation_kwargs = dict(inputs, max_new_tokens=150, streamer=streamer)
            # Run generation in a separate thread
            import threading
            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            # Yield tokens as they arrive
            for text in streamer:
                yield text
        else:
            outputs = model.generate(**inputs, max_new_tokens=150)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("[/INST]")[-1].strip()
            return response

    async def generate_response_ollama(self, prompt: str, history: List[tuple], stream: bool = False):
        """Generowanie odpowiedzi przez Ollama z opcją streamingu."""
        system_prompt = self._build_system_prompt()
        messages = []
        for user_msg, assistant_msg in history:
            messages.append({'role': 'user', 'content': user_msg})
            messages.append({'role': 'assistant', 'content': assistant_msg})
        messages.append({'role': 'user', 'content': prompt})

        if stream:
            # Ollama supports streaming
            response_stream = ollama.chat(
                model=self._get_ollama_model(),
                messages=messages,
                system=system_prompt,
                stream=True
            )
            for chunk in response_stream:
                yield chunk['message']['content']
        else:
            response = ollama.chat(
                model=self._get_ollama_model(),
                messages=messages,
                system=system_prompt
            )
            return response['message']['content']

    def _get_ollama_model(self):
        """Zwraca nazwę modelu Ollama w zależności od trybu."""
        if self.mode == 'programista':
            return "codellama:7b-instruct"
        return "llama2"

    # ------------------------- Główna metoda chat (rozdzielona) -------------------------
    async def _chat_nonstream(self, session_id: str, prompt: str) -> str:
        """Non‑streaming version of chat."""
        # Sanityzacja promptu
        prompt = self.sanitize_prompt(prompt)

        # Pobierz historię sesji z Redis
        history = await self.persistent_memory.get_history(session_id)
        history_list = [(h[0], h[1]) for h in history]

        # Sprawdź, czy pytanie dotyczy przeszłych rozmów (pamięć długoterminowa)
        recall_trigger = re.search(r'(pamiętasz|co ustaliliśmy|co mówiliśmy|przypomnij)', prompt, re.IGNORECASE)
        if recall_trigger:
            past_conversations = await self.recall_past_conversations(prompt)
            if past_conversations:
                context = "\n\nPoprzednie rozmowy:\n" + "\n".join(past_conversations)
                prompt = f"{prompt}\n\n{context}"

        # Pobierz fakty o użytkowniku
        facts = await self.user_profiler.get_facts(session_id)
        if facts:
            prompt = f"Fakty o użytkowniku: {', '.join(facts)}\n\n{prompt}"

        # Dodaj nową wiadomość do historii (tymczasowo)
        history_list.append((prompt, ""))

        # Pętla narzędzi
        current_prompt = prompt
        final_response = ""
        iteration = 0
        while iteration < self.max_tool_iterations:
            if self.engine == 'transformers':
                response = await self.generate_response_transformers(current_prompt, history_list, stream=False)
            else:
                response = await self.generate_response_ollama(current_prompt, history_list, stream=False)

            tool_calls = self.parse_tool_call(response)
            if not tool_calls:
                # Koniec
                history_list[-1] = (prompt, response)
                final_response = response
                break

            # Wykonaj narzędzia
            tool_results = []
            for tool_name, args in tool_calls:
                result = await asyncio.to_thread(self.execute_tool, tool_name, args)
                tool_results.append(f"Narzędzie {tool_name} zwróciło: {result}")

            clean_response = re.sub(r'\{tool:.*?\}', '', response).strip()
            history_list[-1] = (prompt, clean_response)
            tool_summary = "\n".join(tool_results)
            history_list.append(("(tool results)", tool_summary))
            current_prompt = f"Kontynuuj, uwzględniając wyniki narzędzi: {tool_summary}"
            iteration += 1

        # Zapisz w pamięci trwałej
        if final_response:
            await self.persistent_memory.add_message(session_id, prompt, final_response)
            await self.index_conversation(session_id, prompt, final_response)
            await self.user_profiler.update_profile(session_id, prompt, final_response)

        return final_response

    async def _chat_stream(self, session_id: str, prompt: str) -> AsyncGenerator[str, None]:
        """Streaming version of chat."""
        # Sanityzacja promptu
        prompt = self.sanitize_prompt(prompt)

        # Pobierz historię sesji z Redis
        history = await self.persistent_memory.get_history(session_id)
        history_list = [(h[0], h[1]) for h in history]

        # Sprawdź, czy pytanie dotyczy przeszłych rozmów (pamięć długoterminowa)
        recall_trigger = re.search(r'(pamiętasz|co ustaliliśmy|co mówiliśmy|przypomnij)', prompt, re.IGNORECASE)
        if recall_trigger:
            past_conversations = await self.recall_past_conversations(prompt)
            if past_conversations:
                context = "\n\nPoprzednie rozmowy:\n" + "\n".join(past_conversations)
                prompt = f"{prompt}\n\n{context}"

        # Pobierz fakty o użytkowniku
        facts = await self.user_profiler.get_facts(session_id)
        if facts:
            prompt = f"Fakty o użytkowniku: {', '.join(facts)}\n\n{prompt}"

        # Dodaj nową wiadomość do historii (tymczasowo)
        history_list.append((prompt, ""))

        # Pętla narzędzi
        current_prompt = prompt
        final_response = ""
        iteration = 0
        while iteration < self.max_tool_iterations:
            if self.engine == 'transformers':
                generator = self.generate_response_transformers(current_prompt, history_list, stream=True)
            else:
                generator = self.generate_response_ollama(current_prompt, history_list, stream=True)

            # Zbieramy odpowiedź w całości (do wykrycia narzędzi) i jednocześnie strumieniujemy
            collected_chunks = []
            async for chunk in generator:
                collected_chunks.append(chunk)
                yield chunk
            full_response = "".join(collected_chunks)

            tool_calls = self.parse_tool_call(full_response)
            if not tool_calls:
                # Koniec
                history_list[-1] = (prompt, full_response)
                final_response = full_response
                break

            # Wykonaj narzędzia
            tool_results = []
            for tool_name, args in tool_calls:
                result = await asyncio.to_thread(self.execute_tool, tool_name, args)
                tool_results.append(f"Narzędzie {tool_name} zwróciło: {result}")

            clean_response = re.sub(r'\{tool:.*?\}', '', full_response).strip()
            history_list[-1] = (prompt, clean_response)
            tool_summary = "\n".join(tool_results)
            history_list.append(("(tool results)", tool_summary))
            current_prompt = f"Kontynuuj, uwzględniając wyniki narzędzi: {tool_summary}"
            iteration += 1

        # Zapisz w pamięci trwałej
        if final_response:
            await self.persistent_memory.add_message(session_id, prompt, final_response)
            await self.index_conversation(session_id, prompt, final_response)
            await self.user_profiler.update_profile(session_id, prompt, final_response)

    async def chat(self, session_id: str, prompt: str, stream: bool = False) -> Any:
        """Główna metoda – wywołuje odpowiednią wersję w zależności od stream."""
        if stream:
            return self._chat_stream(session_id, prompt)
        else:
            return await self._chat_nonstream(session_id, prompt)

    # ------------------------- Narzędzia -------------------------
    def get_weather(self, city: str) -> str:
        return f"Pogoda w {city}: słonecznie, 22°C."

    def search_web(self, query: str) -> str:
        """Wyszukiwanie web z użyciem Google Search API (Serper) zamiast DuckDuckGo."""
        api_key = os.environ.get("SERPER_API_KEY")
        if api_key:
            url = "https://google.serper.dev/search"
            headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
            payload = json.dumps({"q": query, "num": 5})
            response = requests.post(url, headers=headers, data=payload)
            if response.status_code == 200:
                data = response.json()
                snippets = [f"{item['title']}: {item['snippet']}" for item in data.get("organic", [])]
                return "\n".join(snippets)
            else:
                return f"Błąd API Google: {response.status_code}"
        else:
            # Fallback do DuckDuckGo
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=3))
                    snippets = [f"{r['title']}: {r['body']}" for r in results]
                    return "\n".join(snippets)
            except Exception as e:
                return f"Błąd wyszukiwania: {e}"

    async def run_python(self, code: str) -> str:
        """Bezpieczne wykonanie kodu w Dockerze."""
        return await self.docker_executor.run_code(code)

    def parse_tool_call(self, text: str) -> List[tuple]:
        pattern = r'\{tool:(\w+)\s+(.+?)\}'
        matches = re.findall(pattern, text)
        tool_calls = []
        for tool_name, args_str in matches:
            args = {}
            arg_pattern = r'(\w+)="([^"]+)"'
            for key, value in re.findall(arg_pattern, args_str):
                args[key] = value
            tool_calls.append((tool_name, args))
        return tool_calls

    def execute_tool(self, tool_name: str, args: Dict) -> str:
        if tool_name in self.tools:
            return self.tools[tool_name](**args)
        else:
            return f"Narzędzie {tool_name} nie istnieje."

    # ------------------------- Zaawansowany RAG -------------------------
    async def rag_query(self, query: str, use_hybrid: bool = True) -> str:
        """Zapytanie do bazy wiedzy z opcją hybrydowego wyszukiwania."""
        if use_hybrid:
            results = self.advanced_rag.hybrid_search(query, k=3)
        else:
            results = self.advanced_rag.search(query, k=3)
        context = "\n".join(results)
        prompt = f"Odpowiedz na pytanie na podstawie poniższego kontekstu:\n\n{context}\n\nPytanie: {query}"
        if self.engine == 'transformers':
            response = await self.generate_response_transformers(prompt, [], stream=False)
        else:
            response = await self.generate_response_ollama(prompt, [], stream=False)
        return response

    async def ingest_document(self, source: str, chunk_size: int = 500, chunk_overlap: int = 100):
        """Dodaje dokument (URL, plik lokalny) z podziałem na fragmenty."""
        if source.startswith('http://') or source.startswith('https://'):
            try:
                resp = requests.get(source)
                if 'application/pdf' in resp.headers.get('Content-Type', ''):
                    pdf_file = BytesIO(resp.content)
                    pdf_reader = pypdf.PdfReader(pdf_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                else:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    text = soup.get_text()
            except Exception as e:
                return f"Błąd pobierania: {e}"
        else:
            if not os.path.exists(source):
                return f"Plik {source} nie istnieje."
            if source.endswith('.pdf'):
                pdf_reader = pypdf.PdfReader(source)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            else:
                with open(source, 'r', encoding='utf-8') as f:
                    text = f.read()

        # Podział na fragmenty
        parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = parser.get_nodes_from_documents([Document(text=text)])
        texts = [node.text for node in nodes]
        metadatas = [{"source": source, "chunk": i} for i in range(len(texts))]
        self.advanced_rag.add_documents(texts, metadatas)
        return f"Zindeksowano {len(texts)} fragmentów z {source}"

    # ------------------------- Multimodalność: Vision -------------------------
    async def load_vision_model(self):
        if self.vision_model is None:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            model_id = "llava-hf/llava-1.5-7b-hf"
            self.vision_model = LlavaForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map="auto"
            )
            self.vision_processor = AutoProcessor.from_pretrained(model_id)

    async def analyze_image(self, image_bytes: bytes, prompt: str = "Describe this image.") -> str:
        """Opis obrazu przy użyciu modelu LLaVA."""
        await self.load_vision_model()
        from PIL import Image
        image = Image.open(BytesIO(image_bytes))
        inputs = self.vision_processor(images=image, text=prompt, return_tensors="pt").to(self.vision_model.device)
        outputs = self.vision_model.generate(**inputs, max_new_tokens=200)
        description = self.vision_processor.decode(outputs[0], skip_special_tokens=True)
        return description

    # ------------------------- TTS (Text-to-Speech) -------------------------
    async def load_tts_model(self):
        if self.tts_model is None:
            self.tts_model = TTS(language='PL')
            self.speed = 1.0

    async def text_to_speech(self, text: str) -> bytes:
        """Konwersja tekstu na mowę (zwraca bytes audio)."""
        await self.load_tts_model()
        output_path = "output.wav"
        self.tts_model.tts_to_file(text, self.tts_model.hps.data.spk2id['PL'], output_path, speed=self.speed)
        with open(output_path, 'rb') as f:
            audio_bytes = f.read()
        os.remove(output_path)
        return audio_bytes

    # ------------------------- Audio (Whisper) -------------------------
    async def load_whisper_model(self):
        if self.whisper_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = whisper.load_model("small", device=device)

    async def transcribe_audio(self, file_path: str) -> str:
        await self.load_whisper_model()
        result = await asyncio.to_thread(self.whisper_model.transcribe, file_path)
        return result["text"]

    # ------------------------- Obrazy (Stable Diffusion) -------------------------
    async def load_diffuser_pipe(self):
        if self.diffuser_pipe is None:
            self.diffuser_pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
            )
            if torch.cuda.is_available():
                self.diffuser_pipe = self.diffuser_pipe.to("cuda")

    async def generate_image(self, prompt: str) -> str:
        await self.load_diffuser_pipe()
        image = await asyncio.to_thread(self.diffuser_pipe, prompt)
        image = image.images[0]
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return img_base64

    # ------------------------- Statystyki -------------------------
    def get_stats(self):
        vram_used = None
        vram_total = None
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024**3
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            "engine": self.engine,
            "mode": self.mode,
            "vram_used_gb": vram_used,
            "vram_total_gb": vram_total,
            "active_sessions": len(asyncio.run(self.persistent_memory.list_sessions())),
            "history_len": 0,
            "model_loaded": self.model_manager.is_loaded(),
            "model_idle_seconds": self.model_manager.idle_seconds()
        }

# ------------------------- Inicjalizacja -------------------------
hexai = HexAI()

# ------------------------- Endpointy -------------------------
@app.on_event("shutdown")
async def shutdown():
    await hexai.close()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Główny endpoint do konwersacji. Jeśli stream=True, zwraca strumień."""
    session_id = request.session_id or str(uuid.uuid4())
    if request.stream:
        return StreamingResponse(
            hexai.chat(session_id, request.message, stream=True),
            media_type="text/plain"
        )
    else:
        response = await hexai.chat(session_id, request.message, stream=False)
        return ChatResponse(session_id=session_id, response=response)

@app.post("/engine")
async def set_engine(request: EngineRequest):
    hexai.engine = request.engine
    return {"message": f"Silnik zmieniony na {request.engine}"}

@app.post("/mode")
async def set_mode(request: ModeRequest):
    hexai.mode = request.mode
    return {"message": f"Tryb zmieniony na {request.mode}"}

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
    img_base64 = await hexai.generate_image(request.prompt)
    return {"image_base64": img_base64}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    transcription = await hexai.transcribe_audio(temp_path)
    os.remove(temp_path)
    return {"transcription": transcription}

@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...), prompt: str = Form("Describe this image.")):
    image_bytes = await file.read()
    description = await hexai.analyze_image(image_bytes, prompt)
    return {"description": description}

@app.get("/tts")
async def text_to_speech(text: str):
    audio_bytes = await hexai.text_to_speech(text)
    return StreamingResponse(BytesIO(audio_bytes), media_type="audio/wav")

@app.get("/stats", response_model=StatsResponse)
async def stats():
    stats_data = hexai.get_stats()
    return StatsResponse(**stats_data)

@app.get("/sessions")
async def list_sessions():
    sessions = await hexai.persistent_memory.list_sessions()
    return {"sessions": sessions}

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    await hexai.persistent_memory.clear_session(session_id)
    return {"message": f"Sesja {session_id} usunięta"}

# ------------------------- Uruchomienie -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
