/// hexai - unified entry point
///
///   hexai              → start API server + TUI
///   hexai --server     → start API server only (headless)
///   hexai --with-gui   → start API server + launch Tauri GUI
///   hexai --help       → print usage

use std::process;
use anyhow::Result;

fn print_help() {
    println!(
        r#"
⬡  HexAi v2.0.0

USAGE:
    hexai              Start the API server and TUI (default)
    hexai --server     Start the API server only (headless/daemon)
    hexai --with-gui   Start the API server and open the desktop GUI
    hexai --help       Show this help

ENVIRONMENT:
    HEXAI_HOST         API bind host        (default: 0.0.0.0)
    HEXAI_PORT         API port             (default: 8000)
    REDIS_URL          Redis URL            (default: redis://127.0.0.1:6379)
    OLLAMA_URL         Ollama base URL      (default: http://127.0.0.1:11434)
    SERPER_API_KEY     Serper search key    (optional)
"#
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let with_gui    = args.iter().any(|a| a == "--with-gui");
    let server_only = args.iter().any(|a| a == "--server");
    let help        = args.iter().any(|a| a == "--help" || a == "-h");

    if help {
        print_help();
        return Ok(());
    }

    // ── Init tracing ─────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("hexai_backend=info".parse()?)
                .add_directive("hexai=info".parse()?),
        )
        .with_target(false)
        .init();

    // ── Start backend server in background task ──────────────────
    tokio::spawn(async {
        if let Err(e) = run_server().await {
            tracing::error!("Server error: {e}");
        }
    });

    // Give the server a moment to bind
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    if server_only {
        tracing::info!("Running in server-only mode. Press Ctrl+C to stop.");
        tokio::signal::ctrl_c().await?;
        return Ok(());
    }

    if with_gui {
        launch_gui()
    } else {
        launch_tui().await
    }
}

async fn run_server() -> Result<()> {
    use hexai_backend::{
        config::Config,
        engine::LlmEngine,
        memory::PersistentMemory,
        router::build_router,
        state::AppState,
    };
    use std::net::SocketAddr;

    let cfg = Config::default();

    let redis_mgr = redis::Client::open(cfg.redis_url.clone())
        .ok()
        .and_then(|c| {
            tokio::runtime::Handle::current().block_on(async {
                redis::aio::ConnectionManager::new(c).await.ok()
            })
        })
        .map(std::sync::Arc::new);

    if redis_mgr.is_none() {
        tracing::warn!("Redis unavailable – using in-memory session store.");
    }

    let memory = PersistentMemory::new(redis_mgr, cfg.session_ttl_secs);
    let engine = LlmEngine::new(&cfg);
    let state  = AppState::new(cfg.clone(), engine, memory);
    let router = build_router(state);

    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port).parse()?;
    tracing::info!("HexAi API listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;
    Ok(())
}

async fn launch_tui() -> Result<()> {
    use crossterm::{
        event::{DisableMouseCapture, EnableMouseCapture},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    };
    use ratatui::{backend::CrosstermBackend, Terminal};
    use std::io;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = hexai_tui::app::run(&mut terminal).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;

    if let Err(e) = result {
        eprintln!("TUI error: {e}");
    }
    println!("Do widzenia! 👋");
    Ok(())
}

fn launch_gui() -> Result<()> {
    // Try to find and launch the Tauri GUI binary co-located with `hexai`,
    // or the `hexai-gui` binary on PATH.
    let exe = std::env::current_exe()?;
    let gui_candidates = [
        exe.parent().unwrap().join("hexai-gui"),
        exe.parent().unwrap().join("hexai-gui.exe"),
    ];

    for candidate in &gui_candidates {
        if candidate.exists() {
            tracing::info!("Launching GUI: {}", candidate.display());
            process::Command::new(candidate).spawn()?.wait()?;
            return Ok(());
        }
    }

    // Fallback: try to open via Tauri CLI if in dev environment
    let tauri_dev = process::Command::new("cargo")
        .args(["tauri", "dev"])
        .current_dir(
            std::env::current_exe()?
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("gui"),
        )
        .spawn();

    match tauri_dev {
        Ok(mut child) => {
            tracing::info!("Started Tauri dev GUI");
            child.wait()?;
        }
        Err(e) => {
            eprintln!(
                "Could not launch GUI: {e}\n\
                 Make sure you built the Tauri app first:\n\
                 cd gui && npm install && npm run tauri build\n\
                 Or put the `hexai-gui` binary next to the `hexai` binary."
            );
            std::process::exit(1);
        }
    }
    Ok(())
}
