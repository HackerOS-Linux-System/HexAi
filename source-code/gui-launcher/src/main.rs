use anyhow::Result;
use std::sync::Arc;

fn print_help() {
    println!(r#"
⬡  HexAi v2.0.0

UŻYCIE:
    hexai              Serwer API + TUI (domyślnie)
    hexai --server     Tylko serwer API (headless)
    hexai --with-gui   Serwer API + natywne okno GUI
    hexai --verbose    Szczegółowe logi
    hexai --help       Ta pomoc

ŚRODOWISKO:
    HEXAI_HOST           Adres nasłuchiwania   (domyślnie: 0.0.0.0)
    HEXAI_PORT           Port API              (domyślnie: 8000)
    HEXAI_ENGINE         Silnik LLM            (ollama|openai|candle)
    REDIS_URL            URL Redis             (domyślnie: redis://127.0.0.1:6379)
    OLLAMA_URL           URL Ollama            (domyślnie: http://127.0.0.1:11434)
    OPENAI_API_KEY       Klucz OpenAI
    OPENAI_API_BASE      Base URL OpenAI       (domyślnie: https://api.openai.com)
    HEXAI_EMBED_MODEL    Model embeddings      (domyślnie: nomic-embed-text)
    SERPER_API_KEY       Klucz Serper          (opcjonalne)
    HEXAI_DB_PATH        Ścieżka SQLite        (domyślnie: ./hexai.db)
    HEXAI_AUTH           Włącz JWT auth        (0|1)
    HEXAI_JWT_SECRET     Sekret JWT
    HEXAI_ADMIN_PASS     Hasło admina          (domyślnie: hexai-admin)
    HEXAI_RATE_LIMIT_RPM Limit żądań/min       (domyślnie: 60)
    HEXAI_CORS_ORIGINS   Dozwolone origins     (domyślnie: *)
    RUST_LOG             Poziom logów          (info|debug|trace)
"#);
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let with_gui    = args.iter().any(|a| a == "--with-gui");
    let server_only = args.iter().any(|a| a == "--server");
    let verbose     = args.iter().any(|a| a == "--verbose" || a == "-v");
    let help        = args.iter().any(|a| a == "--help" || a == "-h");

    if help { print_help(); return Ok(()); }

    if verbose && std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "debug,hyper=info,reqwest=info,h2=warn");
    }
    let log_level = if verbose { "debug" } else { "info" };

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(
                    format!("hexai_backend={log_level},hexai={log_level},warn")
                )),
        )
        .with_target(verbose)
        .with_file(verbose)
        .with_line_number(verbose)
        .init();

    tracing::info!("⬡ HexAi v2.0.0 startuje…");

    if with_gui {
        // GUI tryb: WebView wymaga głównego wątku dla event loop.
        // Startujemy Tokio runtime w osobnym wątku, główny wątek = WebView.
        launch_gui_mode(verbose)
    } else {
        // TUI / server tryb: Tokio na głównym wątku
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async_main(server_only, verbose))
    }
}

// ── Async entry (TUI + server mode) ──────────────────────────────

async fn async_main(server_only: bool, _verbose: bool) -> Result<()> {
    let server_handle = tokio::spawn(run_server());

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    if server_handle.is_finished() {
        match server_handle.await {
            Ok(Err(e)) => {
                tracing::error!("Serwer nie mógł się uruchomić: {e}");
                tracing::error!("Sprawdź czy port nie jest zajęty (HEXAI_PORT=8001 ./hexai)");
                std::process::exit(1);
            }
            _ => std::process::exit(1),
        }
    }

    if server_only {
        tracing::info!("Tryb serwera. Ctrl+C aby zatrzymać.");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            res = server_handle => {
                if let Ok(Err(e)) = res {
                    tracing::error!("Serwer zakończył się błędem: {e}");
                }
            }
        }
        return Ok(());
    }

    let result = launch_tui().await;
    server_handle.abort();
    result
}

// ── GUI mode: server in thread, WebView on main thread ───────────

fn launch_gui_mode(_verbose: bool) -> Result<()> {
    use std::thread;
    use std::sync::atomic::{AtomicBool, Ordering};

    let port = std::env::var("HEXAI_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(8000);
    let url = format!("http://127.0.0.1:{port}/gui");

    // Shared ready flag
    let ready = Arc::new(AtomicBool::new(false));
    let ready_clone = Arc::clone(&ready);

    // Start Tokio + Axum in a dedicated OS thread
    thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("Tokio runtime");
        rt.block_on(async move {
            // Small delay then signal ready
            let ready2 = Arc::clone(&ready_clone);
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(600)).await;
                ready2.store(true, Ordering::SeqCst);
            });
            if let Err(e) = run_server().await {
                tracing::error!("Serwer GUI zakończył się: {e}");
            }
        });
    });

    // Wait for server to be ready (max 5s)
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !ready.load(std::sync::atomic::Ordering::SeqCst) {
        if std::time::Instant::now() > deadline {
            tracing::warn!("Timeout oczekiwania na serwer, otwieram GUI mimo to…");
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    tracing::info!("Otwieram GUI: {url}");
    create_webview_window(&url)
}

// ── WebView window (wry + tao) ────────────────────────────────────

fn create_webview_window(url: &str) -> Result<()> {
    use tao::{
        dpi::LogicalSize,
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::WindowBuilder,
    };
    use wry::WebViewBuilder;

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("⬡ HexAi v2.0.0")
        .with_inner_size(LogicalSize::new(1200u32, 800u32))
        .with_min_inner_size(LogicalSize::new(800u32, 600u32))
        .with_resizable(true)
        .build(&event_loop)?;

    let url_owned = url.to_string();
    let _webview = WebViewBuilder::new(&window)
        .with_url(&url_owned)
        .with_devtools(cfg!(debug_assertions))
        .build()?;

    tracing::info!("GUI okno otwarte");

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                tracing::info!("Zamykanie GUI…");
                *control_flow = ControlFlow::Exit;
            }
            _ => {}
        }
    });
}

// ── Serwer API ────────────────────────────────────────────────────

async fn run_server() -> Result<()> {
    use hexai_backend::{
        config::Config, engine::LlmEngine,
        memory::PersistentMemory, router::build_router, state::AppState,
    };
    use std::net::SocketAddr;

    let cfg = Config::default();

    let redis_mgr: Option<Arc<tokio::sync::Mutex<redis::aio::ConnectionManager>>> =
        match redis::Client::open(cfg.redis_url.clone()) {
            Ok(client) => match redis::aio::ConnectionManager::new(client).await {
                Ok(mgr) => {
                    tracing::info!("✓ Redis: {}", cfg.redis_url);
                    Some(Arc::new(tokio::sync::Mutex::new(mgr)))
                }
                Err(e) => { tracing::warn!("Redis niedostępny ({e}) – SQLite"); None }
            },
            Err(e) => { tracing::warn!("Redis URL błąd ({e}) – SQLite"); None }
        };

    let memory = PersistentMemory::new(
        &cfg.db_path, redis_mgr,
        cfg.session_ttl_secs, cfg.session_max_tokens,
    );
    let engine = LlmEngine::new(&cfg);
    let state  = AppState::new(cfg.clone(), engine, memory);
    let router = build_router(state);

    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port).parse()?;
    let listener = tokio::net::TcpListener::bind(&addr).await
        .map_err(|e| anyhow::anyhow!("Nie można zbindować {addr}: {e}"))?;

    tracing::info!("⬡ HexAi API → http://{addr}  [engine={}]",
        std::env::var("HEXAI_ENGINE").unwrap_or_else(|_| "ollama".into()));
    axum::serve(listener, router).await?;
    Ok(())
}

// ── TUI ───────────────────────────────────────────────────────────

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
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    let result = hexai_tui::app::run(&mut terminal).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;

    if let Err(e) = &result { tracing::error!("TUI błąd: {e}"); }
    println!("Do widzenia! 👋");
    result.map_err(Into::into)
}
