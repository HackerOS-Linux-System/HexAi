# HexAi v2.0.0

Lokalny asystent AI z interfejsem desktopowym, terminalowym i REST API.

## Struktura projektu

```
HexAi/
├── source-code/ # Backend Python (FastAPI)
│   ├── tui/                  # Terminal UI (GoLang + bubbletea)
│       ├── src/main.rs
│       └── Cargo.toml
│   ├── gui/                  # Desktop app (Tauri + Next.js)
│       ├── src/              # Frontend TypeScript
│       └── src-tauri/        # Tauri host (Rust)
│   ├── main.py           # Główna aplikacja API
│   └── requirements.txt  # Zależności Python
├── src/main.rs               # Builder (Rust → Python binary)
    └── Cargo.toml
```

## Szybki start

### Backend
```bash
cd source-code
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker run -d -p 6379:6379 redis:7-alpine
python main.py
```

### GUI Desktop
```bash
cd gui && npm install && npm run tauri-dev
```

### TUI
```bash
cd tui && cargo run --release
```

## Wymagania
- Python 3.11–3.12
- Redis 7+
- Rust 1.78+
- Node.js 20+
- Docker (opcjonalnie, dla sandbox)
- CUDA 12+ (opcjonalnie, dla GPU)

## Licencja
GPL-3.0
