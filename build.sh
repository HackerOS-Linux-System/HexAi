#!/usr/bin/env bash
# ⬡ HexAi v0.0.1 – skrypt budowania
#
# Użycie:
#   ./build.sh             GUI + backend release (domyślnie)
#   ./build.sh backend     tylko backend Rust
#   ./build.sh gui         tylko GUI (npm build)
#   ./build.sh debug       backend debug
#   ./build.sh cuda        backend release z CUDA GPU
#   ./build.sh opencl      backend release z OpenCL GPU
#   ./build.sh clean       wyczyść artefakty

set -euo pipefail
BOLD='\033[1m'; AMBER='\033[0;33m'; GREEN='\033[0;32m'; RED='\033[0;31m'; RESET='\033[0m'
log() { echo -e "${AMBER}${BOLD}⬡  $*${RESET}"; }
ok()  { echo -e "${GREEN}✓  $*${RESET}"; }
err() { echo -e "${RED}✗  $*${RESET}"; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

check_system_deps() {
    if [[ "$(uname)" == "Linux" ]]; then
        for pkg in pkg-config cmake; do
            command -v "$pkg" &>/dev/null || \
                echo -e "${AMBER}⚠ Brak '$pkg'. Zainstaluj: apt install cmake pkg-config build-essential libwebkit2gtk-4.1-dev libgtk-3-dev${RESET}"
        done
    fi
}

build_gui() {
    log "Buduję GUI (Next.js → gui/out)…"
    command -v node &>/dev/null || err "Node.js nie znaleziony. Zainstaluj Node.js ≥ 18."
    cd "$ROOT/source-code/gui"
    npm install --silent
    NEXT_PUBLIC_BASE_PATH=/gui npm run build
    ok "GUI → source-code/gui/out/  ($(find out -name '*.html' 2>/dev/null | wc -l) stron)"
    cd "$ROOT"
}

build_backend() {
    local flags="${1:-}"
    log "Buduję backend Rust (release${flags:+ +$flags})…"
    check_system_deps
    if [[ -n "$flags" ]]; then
        cargo build --release -p hexai --features "$flags"
    else
        cargo build --release -p hexai
    fi
    ok "Binarka → target/release/hexai  ($(du -sh target/release/hexai 2>/dev/null | cut -f1 || echo '?'))"
}

MODE="${1:-all}"
case "$MODE" in
    all|release) build_gui; build_backend ;;
    backend)     build_backend ;;
    gui)         build_gui ;;
    debug)
        log "Buduję debug…"
        cargo build -p hexai
        ok "Binarka → target/debug/hexai"
        ;;
    cuda)
        build_gui
        build_backend "cuda"
        ;;
    opencl)
        build_gui
        build_backend "opencl"
        ;;
    clean)
        log "Czyszczę…"
        cargo clean
        rm -rf source-code/gui/out source-code/gui/.next source-code/gui/node_modules
        ok "Wyczyszczono"
        exit 0
        ;;
    *)
        echo "Użycie: $0 [all|backend|gui|debug|cuda|clean]"
        exit 1
        ;;
esac

echo ""
log "⬡ HexAi v0.0.1 – gotowe!"
echo ""
echo "  ./target/release/hexai                    # TUI"
echo "  ./target/release/hexai --with-gui         # GUI (natywne okno)"
echo "  ./target/release/hexai --server           # tylko API"
echo "  ./target/release/hexai --help             # pomoc"
echo ""
echo "  Własne AI (llama.cpp):"
echo "  HEXAI_ENGINE=local HEXAI_MODEL_PATH=~/models/model.gguf ./target/release/hexai"
echo ""
