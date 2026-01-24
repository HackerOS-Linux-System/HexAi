import os
import torch
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

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

# ASCII art logo for HexAi
HEXAI_LOGO = """
 _   _           _    _    _ 
| | | |         | |  | |  (_)
| |_| | _____  _| | _| | ___ 
|  _  |/ _ \\ \\/ / |/ / |/ / |
| | | |  __/>  <|   <|   <|_|
\\_| |_/\\___/_/\\_\\_|\\_\\_|\\_(_)
"""

class HexAI:
    def __init__(self):
        self.console = Console()
        self.accelerator = None  # Lazy init if needed

        # Inicjalizuj atrybuty jako None dla lazy loading
        self.tokenizer = None
        self.model = None
        self.ollama_model = 'llama2'  # Nazwa modelu dla ollama
        self.index = None  # Lazy for RAG
        self.whisper_model = None
        self.diffuser_pipe = None

        # ChromaDB client - można inicjalizować, bo lekki
        self.chroma_client = chromadb.Client()
        self.chroma_collection = self.chroma_client.get_or_create_collection("hexai_docs")

    def _init_accelerator(self):
        if self.accelerator is None:
            self.accelerator = Accelerator()

    def _load_transformers_model(self):
        if self.model is None:
            self._init_accelerator()
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

    def _unload_transformers_model(self):
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()

    def generate_response(self, prompt):
        self._load_transformers_model()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=150)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self._unload_transformers_model()  # Zwalniaj po użyciu
        return response

    def ollama_chat(self, prompt):
        # Ollama ładuje model przy pierwszym wywołaniu, nie potrzeba unload (zarządza sama)
        response = ollama.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']

    def _load_rag_index(self):
        if self.index is None:
            vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # Przykładowe dokumenty - załaduj z katalogu /docs/ lub stwórz pusty
            if os.path.exists("docs"):
                documents = SimpleDirectoryReader("docs").load_data()
                self.index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            else:
                self.index = VectorStoreIndex([], storage_context=storage_context)
                self.console.print("[bold yellow]No docs directory found. RAG will be empty.[/bold yellow]")

    def _unload_rag_index(self):
        if self.index is not None:
            del self.index
            self.index = None
            torch.cuda.empty_cache()  # Na wszelki wypadek, jeśli embedding na GPU

    def rag_query(self, query):
        self._load_rag_index()
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        self._unload_rag_index()  # Zwalniaj po użyciu
        return str(response)

    def _load_whisper_model(self):
        if self.whisper_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = WhisperModel("small", device=device, compute_type="float16")

    def _unload_whisper_model(self):
        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
            torch.cuda.empty_cache()

    def transcribe_audio(self, file_path):
        self._load_whisper_model()
        segments, info = self.whisper_model.transcribe(file_path, beam_size=5)
        transcription = ' '.join([segment.text for segment in segments])
        self._unload_whisper_model()  # Zwalniaj po użyciu
        return transcription

    def _load_diffuser_pipe(self):
        if self.diffuser_pipe is None:
            self.diffuser_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
            if torch.cuda.is_available():
                self.diffuser_pipe = self.diffuser_pipe.to("cuda")

    def _unload_diffuser_pipe(self):
        if self.diffuser_pipe is not None:
            del self.diffuser_pipe
            self.diffuser_pipe = None
            torch.cuda.empty_cache()

    def generate_image(self, prompt):
        self._load_diffuser_pipe()
        image = self.diffuser_pipe(prompt).images[0]
        image_path = "generated_image.png"
        image.save(image_path)
        self._unload_diffuser_pipe()  # Zwalniaj po użyciu
        return f"Image generated and saved to {image_path}"

    def run(self):
        self.console.print(Panel(HEXAI_LOGO, title="HexAi", style="bold green"))
        self.console.print("[bold blue]HexAi activated! Commands: /chat <msg>, /rag <query>, /transcribe <file>, /generate_image <prompt>, /exit[/bold blue]")
        self.console.print("[bold yellow]Note: Models are loaded lazily to save VRAM.[/bold yellow]")

        while True:
            user_input = Prompt.ask("[bold green]You[/bold green]")
            if user_input.lower() == '/exit':
                self.console.print("[bold red]Exiting HexAi...[/bold red]")
                break
            elif user_input.startswith('/chat '):
                msg = user_input[6:]
                # Używaj generate_response (transformers) lub ollama_chat
                # Dla przykładu, używaj transformers; zakomentuj jeśli chcesz ollama
                response = self.generate_response(msg)
                # response = self.ollama_chat(msg)  # Alternatywa
                md = Markdown(response)
                self.console.print(Panel(md, title="HexAi (Chat)", style="bold blue"))
            elif user_input.startswith('/rag '):
                query = user_input[5:]
                response = self.rag_query(query)
                md = Markdown(response)
                self.console.print(Panel(md, title="HexAi (RAG)", style="bold blue"))
            elif user_input.startswith('/transcribe '):
                file = user_input[12:]
                if os.path.exists(file):
                    transcription = self.transcribe_audio(file)
                    md = Markdown(transcription)
                    self.console.print(Panel(md, title="HexAi (Transcription)", style="bold blue"))
                else:
                    self.console.print("[bold red]File not found.[/bold red]")
            elif user_input.startswith('/generate_image '):
                prompt = user_input[16:]
                result = self.generate_image(prompt)
                self.console.print("[bold blue]" + result + "[/bold blue]")
            else:
                self.console.print("[bold yellow]Unknown command. Use /chat, /rag, etc.[/bold yellow]")

if __name__ == "__main__":
    hexai = HexAI()
    hexai.run()
