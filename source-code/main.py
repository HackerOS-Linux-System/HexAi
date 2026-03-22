import os
import re
import torch
import requests
import asyncio
import uuid
from collections import deque
from bs4 import BeautifulSoup
from io import BytesIO
import pypdf
from duckduckgo_search import DDGS
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from rich.console import Console
import uvicorn

# Importy dla podanych bibliotek (importujemy na górze, ale inicjalizacja lazy)
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb
from faster_whisper import WhisperModel
from diffusers import StableDiffusionPipeline
import base64

app = FastAPI(title="HexAi API", version="1.0.0")
console = Console()

# ------------------------- Modele danych Pydantic -------------------------
class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

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

class ImageRequest(BaseModel):
    prompt: str

class StatsResponse(BaseModel):
    engine: str
    mode: str
    vram_used_gb: float | None
    vram_total_gb: float | None
    active_sessions: int
    history_len: int  # dla przykładowej sesji (można rozszerzyć)

# ------------------------- Klasa HexAI (z sesjami) -------------------------
class HexAI:
    def __init__(self):
        self.accelerator = None

        # Atrybuty dla modeli (lazy loading)
        self.tokenizer = None
        self.model = None
        self.ollama_model = 'llama2'          # Domyślny model Ollama
        self.code_ollama_model = 'codellama'  # Możesz użyć modelu kodowego (np. codellama:7b-instruct)
        self.index = None
        self.whisper_model = None
        self.diffuser_pipe = None

        # Persystentny klient ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.chroma_collection = self.chroma_client.get_or_create_collection("hexai_docs")

        # Domyślny silnik i tryb
        self.engine = 'transformers'
        self.mode = 'general'                 # 'general' lub 'programista'

        # Narzędzia
        self.tools = {
            'get_weather': self.get_weather,
            'search_web': self.search_web,
            'run_python': self.run_python
        }
        self.max_tool_iterations = 3

        # Sesje: session_id -> { 'history': deque, 'engine': str (opcjonalnie), 'mode': str (opcjonalnie) }
        self.sessions = {}

        # Definicje systemowych promptów dla trybów
        self.system_prompts = {
            'general': "Jesteś pomocnym asystentem AI.",
            'programista': (
                "Jesteś ekspertem programistycznym na poziomie Claude. "
                "Twój kod jest czysty, wydajny, dobrze skomentowany i zgodny z najlepszymi praktykami. "
                "Zawsze podajesz kompletne przykłady, uwzględniasz obsługę błędów i wyjaśniasz swoje rozwiązania. "
                "Jeśli to możliwe, podajesz kod w różnych językach programowania i dostosowujesz go do kontekstu."
            )
        }

    def _init_accelerator(self):
        if self.accelerator is None:
            self.accelerator = Accelerator()

    # ------------------------- Ładowanie modeli (lazy) -------------------------
    def _load_transformers_model(self):
        if self.model is None:
            self._init_accelerator()
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            # Wybór modelu w zależności od trybu (można rozszerzyć)
            if self.mode == 'programista':
                # Użyj modelu zorientowanego na kod, jeśli dostępny
                model_id = "codellama/CodeLlama-7b-Instruct-hf"
                console.print("[bold blue]Ładowanie CodeLlama (tryb programista)[/bold blue]")
            else:
                model_id = "mistralai/Mistral-7B-v0.1"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto"
            )

    def _unload_transformers_model(self):
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()

    def _get_ollama_model(self):
        """Zwraca nazwę modelu Ollama w zależności od trybu."""
        if self.mode == 'programista':
            return self.code_ollama_model
        return self.ollama_model

    def _build_system_prompt(self):
        """Zwraca systemowy prompt dla bieżącego trybu."""
        return self.system_prompts.get(self.mode, self.system_prompts['general'])

    def generate_response_transformers(self, prompt, history):
        """Generowanie odpowiedzi z uwzględnieniem historii sesji i trybu."""
        self._load_transformers_model()
        # Formatowanie historii z systemowym promptem jako pierwszą wiadomością
        system_prompt = self._build_system_prompt()
        formatted_history = f"[INST] {system_prompt} [/INST] (System prompt) </s>\n"
        for user_msg, assistant_msg in history:
            formatted_history += f"[INST] {user_msg} [/INST] {assistant_msg} </s>\n"
        formatted_history += f"[INST] {prompt} [/INST]"
        inputs = self.tokenizer(formatted_history, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=150)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Wyciągamy ostatnią część po [/INST]
        response = response.split("[/INST]")[-1].strip()
        self._unload_transformers_model()
        return response

    def generate_response_ollama(self, prompt, history):
        """Generowanie odpowiedzi przez Ollama z historią i systemowym promptem."""
        system_prompt = self._build_system_prompt()
        messages = []
        # Dodaj system prompt jako pierwszą wiadomość (tylko jeśli nie ma go w historii)
        # Ollama nie przechowuje system promptu w historii, więc przekazujemy go osobno
        for user_msg, assistant_msg in history:
            messages.append({'role': 'user', 'content': user_msg})
            messages.append({'role': 'assistant', 'content': assistant_msg})
        messages.append({'role': 'user', 'content': prompt})
        # Wywołanie z system promptem
        response = ollama.chat(
            model=self._get_ollama_model(),
            messages=messages,
            system=system_prompt
        )
        return response['message']['content']

    async def chat_response(self, session_id, prompt):
        """Główna metoda generowania odpowiedzi dla danej sesji."""
        # Pobierz lub utwórz sesję
        if session_id not in self.sessions:
            self.sessions[session_id] = {'history': deque(maxlen=10)}
        session = self.sessions[session_id]
        history = session['history']

        # Dodaj pytanie użytkownika (tymczasowo)
        history.append((prompt, ""))

        current_prompt = prompt
        iteration = 0
        final_response = ""

        while iteration < self.max_tool_iterations:
            if self.engine == 'transformers':
                try:
                    response = await asyncio.to_thread(
                        self.generate_response_transformers, current_prompt, history
                    )
                except Exception as e:
                    if "OutOfMemory" in str(e) or "CUDA out of memory" in str(e):
                        console.print("[bold yellow]Transformers OOM. Switching to Ollama...[/bold yellow]")
                        self.switch_engine('ollama')
                        response = await asyncio.to_thread(
                            self.generate_response_ollama, current_prompt, history
                        )
                    else:
                        raise
            else:
                response = await asyncio.to_thread(
                    self.generate_response_ollama, current_prompt, history
                )

            # Sprawdź wywołania narzędzi
            tool_calls = self.parse_tool_call(response)
            if not tool_calls:
                # Koniec – aktualizujemy historię
                history[-1] = (prompt, response)
                final_response = response
                break

            # Wykonaj narzędzia
            tool_results = []
            for tool_name, args in tool_calls:
                result = await asyncio.to_thread(self.execute_tool, tool_name, args)
                tool_results.append(f"Narzędzie {tool_name} zwróciło: {result}")

            # Usuń wywołania narzędzi z odpowiedzi i zapisz w historii
            clean_response = re.sub(r'\{tool:.*?\}', '', response).strip()
            history[-1] = (prompt, clean_response)

            # Dodaj wyniki narzędzi jako wiadomość systemową
            tool_summary = "\n".join(tool_results)
            history.append(("(tool results)", tool_summary))

            # Przygotuj kontynuację
            current_prompt = f"Kontynuuj, uwzględniając wyniki narzędzi: {tool_summary}"
            iteration += 1

        if not final_response:
            final_response = history[-1][1]

        # Usuń wiadomość systemową (jeśli została dodana) – możemy pozostawić, ale dla czystości usuńmy
        if len(history) > 0 and history[-1][0] == "(tool results)":
            history.pop()

        return final_response

    # ------------------------- RAG -------------------------
    def _load_rag_index(self):
        if self.index is None:
            vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            if os.path.exists("docs"):
                documents = SimpleDirectoryReader("docs").load_data()
                self.index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            else:
                self.index = VectorStoreIndex([], storage_context=storage_context)
                console.print("[bold yellow]No docs directory found. RAG will be empty.[/bold yellow]")

    def _unload_rag_index(self):
        if self.index is not None:
            del self.index
            self.index = None
            torch.cuda.empty_cache()

    async def rag_query(self, query):
        self._load_rag_index()
        query_engine = self.index.as_query_engine()
        response = await asyncio.to_thread(query_engine.query, query)
        self._unload_rag_index()
        return str(response)

    async def ingest_document(self, source):
        """Dodaje dokument (plik lokalny lub URL) do bazy wiedzy."""
        self._load_rag_index()
        if source.startswith('http://') or source.startswith('https://'):
            try:
                response = requests.get(source)
                if 'application/pdf' in response.headers.get('Content-Type', ''):
                    pdf_file = BytesIO(response.content)
                    pdf_reader = pypdf.PdfReader(pdf_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                else:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = soup.get_text()
                temp_path = "temp_doc.txt"
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write(text)
                reader = SimpleDirectoryReader(input_files=[temp_path])
                documents = reader.load_data()
                for doc in documents:
                    self.index.insert(doc)
                os.remove(temp_path)
                return f"Zindeksowano dokument z {source}"
            except Exception as e:
                return f"Błąd podczas pobierania/indeksowania: {e}"
        else:
            if not os.path.exists(source):
                return f"Plik {source} nie istnieje."
            reader = SimpleDirectoryReader(input_files=[source])
            documents = reader.load_data()
            for doc in documents:
                self.index.insert(doc)
            return f"Zindeksowano plik {source}"

    # ------------------------- Narzędzia -------------------------
    def get_weather(self, city):
        return f"Pogoda w {city}: słonecznie, 22°C."

    def search_web(self, query):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                snippets = [f"{r['title']}: {r['body']}" for r in results]
                return "\n".join(snippets)
        except Exception as e:
            return f"Błąd wyszukiwania: {e}"

    def run_python(self, code):
        try:
            local_vars = {}
            exec(code, {}, local_vars)
            return "Kod wykonany pomyślnie."
        except Exception as e:
            return f"Błąd wykonania: {e}"

    def parse_tool_call(self, text):
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

    def execute_tool(self, tool_name, args):
        if tool_name in self.tools:
            return self.tools[tool_name](**args)
        else:
            return f"Narzędzie {tool_name} nie istnieje."

    # ------------------------- Zarządzanie silnikiem i trybem -------------------------
    def switch_engine(self, engine_name):
        if engine_name not in ['transformers', 'ollama']:
            raise ValueError(f"Nieznany silnik: {engine_name}")
        if engine_name == 'transformers' and self.model is None:
            self._load_transformers_model()
            self._unload_transformers_model()
        self.engine = engine_name
        return f"Silnik zmieniony na {engine_name}"

    def set_mode(self, mode_name):
        """Zmiana trybu (general/programista). W razie potrzeby przeładowuje model."""
        if mode_name not in ['general', 'programista']:
            raise ValueError(f"Nieznany tryb: {mode_name}")
        old_mode = self.mode
        self.mode = mode_name
        # Jeśli używamy transformers, musimy przeładować model (inny model dla programisty)
        if self.engine == 'transformers' and self.model is not None:
            self._unload_transformers_model()
            self._load_transformers_model()
            self._unload_transformers_model()  # zwalniamy po załadowaniu (lazy)
        # Dla Ollamy zmiana trybu nie wymaga przeładowania – system prompt zmieni się przy następnym wywołaniu
        return f"Tryb zmieniony z {old_mode} na {mode_name}"

    # ------------------------- Audio -------------------------
    def _load_whisper_model(self):
        if self.whisper_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = WhisperModel("small", device=device, compute_type="float16")

    def _unload_whisper_model(self):
        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
            torch.cuda.empty_cache()

    async def transcribe_audio(self, file_path):
        self._load_whisper_model()
        segments, info = self.whisper_model.transcribe(file_path, beam_size=5)
        transcription = ' '.join([segment.text for segment in segments])
        self._unload_whisper_model()
        return transcription

    # ------------------------- Obrazy -------------------------
    def _load_diffuser_pipe(self):
        if self.diffuser_pipe is None:
            self.diffuser_pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
            )
            if torch.cuda.is_available():
                self.diffuser_pipe = self.diffuser_pipe.to("cuda")

    def _unload_diffuser_pipe(self):
        if self.diffuser_pipe is not None:
            del self.diffuser_pipe
            self.diffuser_pipe = None
            torch.cuda.empty_cache()

    async def generate_image(self, prompt):
        self._load_diffuser_pipe()
        image = await asyncio.to_thread(self.diffuser_pipe, prompt)
        image = image.images[0]
        # Zwracamy base64 zamiast zapisywać
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        self._unload_diffuser_pipe()
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
            "active_sessions": len(self.sessions),
            "history_len": len(self.sessions.get("example", {}).get("history", []))
        }

# ------------------------- Inicjalizacja globalnej instancji -------------------------
hexai = HexAI()

# ------------------------- Endpointy FastAPI -------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Główny endpoint do konwersacji. Jeśli nie podano session_id, tworzy nową sesję."""
    session_id = request.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
    try:
        response = await hexai.chat_response(session_id, request.message)
        return ChatResponse(session_id=session_id, response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/engine")
async def set_engine(request: EngineRequest):
    """Zmiana silnika (transformers/ollama)."""
    try:
        result = hexai.switch_engine(request.engine)
        return {"message": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/mode")
async def set_mode(request: ModeRequest):
    """Zmiana trybu (general/programista)."""
    try:
        result = hexai.set_mode(request.mode)
        return {"message": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rag")
async def rag_query(request: RagRequest):
    """Zapytanie do bazy RAG."""
    try:
        response = await hexai.rag_query(request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_document(request: IngestRequest):
    """Dodanie dokumentu (URL lub ścieżka lokalna) do bazy RAG."""
    try:
        result = await hexai.ingest_document(request.source)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_image")
async def generate_image(request: ImageRequest):
    """Generowanie obrazu na podstawie promptu. Zwraca obraz jako base64."""
    try:
        img_base64 = await hexai.generate_image(request.prompt)
        return {"image_base64": img_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transkrypcja pliku audio. Przyjmuje plik w formacie obsługiwanym przez Whisper."""
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        transcription = await hexai.transcribe_audio(temp_path)
        os.remove(temp_path)
        return {"transcription": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Zwraca aktualne statystyki systemu."""
    stats_data = hexai.get_stats()
    return StatsResponse(**stats_data)

@app.get("/sessions")
async def list_sessions():
    """Zwraca listę aktywnych sesji (tylko ID)."""
    return {"sessions": list(hexai.sessions.keys())}

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Usuwa sesję (wraz z historią)."""
    if session_id in hexai.sessions:
        del hexai.sessions[session_id]
        return {"message": f"Sesja {session_id} usunięta"}
    raise HTTPException(status_code=404, detail="Session not found")

# ------------------------- Uruchomienie serwera -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
