use crate::api::{self, Stats, StreamEvent};
use anyhow::Result;
use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyModifiers};
use futures::StreamExt;
use ratatui::{backend::Backend, Terminal};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

// ── Screens ──────────────────────────────────────────────────────

#[derive(Clone, PartialEq)]
pub enum Screen { Chat, Settings, Help }

// ── Message ──────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub ts: chrono::DateTime<chrono::Local>,
}

#[derive(Clone, PartialEq)]
pub enum Role { User, Assistant }

// ── App ──────────────────────────────────────────────────────────

pub struct App {
    pub screen: Screen,
    pub messages: Vec<Message>,
    pub input: String,
    pub input_cursor: usize,
    pub session_id: Option<String>,
    pub loading: bool,
    pub stream_buf: String,
    pub engine: String,
    pub mode: String,
    pub stats: Option<Stats>,
    pub stats_loaded: bool,
    pub settings_cursor: usize,
    pub status_msg: Option<(String, bool)>, // (text, is_error)
    pub scroll_offset: u16,
    pub last_stats_fetch: Instant,
    pub stream_tx: Option<mpsc::Sender<StreamEvent>>,
    pub stream_rx: Option<mpsc::Receiver<StreamEvent>>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            screen: Screen::Chat,
            messages: vec![],
            input: String::new(),
            input_cursor: 0,
            session_id: None,
            loading: false,
            stream_buf: String::new(),
            engine: "transformers".into(),
            mode: "general".into(),
            stats: None,
            stats_loaded: false,
            settings_cursor: 0,
            status_msg: None,
            scroll_offset: 0,
            last_stats_fetch: Instant::now() - Duration::from_secs(10),
            stream_tx: None,
            stream_rx: None,
        }
    }
}

impl App {
    pub fn set_status(&mut self, msg: &str, is_err: bool) {
        self.status_msg = Some((msg.to_string(), is_err));
    }

    pub fn new_session(&mut self) {
        self.messages.clear();
        self.session_id = None;
        self.stream_buf.clear();
        self.loading = false;
        self.set_status("Nowa rozmowa", false);
        self.scroll_offset = 0;
    }

    pub fn insert_char(&mut self, c: char) {
        self.input.insert(self.input_cursor, c);
        self.input_cursor += c.len_utf8();
    }

    pub fn delete_char_back(&mut self) {
        if self.input_cursor > 0 {
            let prev = self.input[..self.input_cursor]
                .char_indices()
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.input.remove(prev);
            self.input_cursor = prev;
        }
    }
}

// ── Event loop ────────────────────────────────────────────────────

pub async fn run<B: Backend>(terminal: &mut Terminal<B>) -> Result<()> {
    let mut app = App::default();
    let mut events = EventStream::new();
    let mut stats_interval = tokio::time::interval(Duration::from_secs(5));

    loop {
        // Draw
        terminal.draw(|f| crate::ui::draw(f, &mut app))?;

        // Poll stats on interval
        if app.last_stats_fetch.elapsed() >= Duration::from_secs(5) {
            app.last_stats_fetch = Instant::now();
            if let Ok(s) = api::fetch_stats().await {
                app.engine = s.engine.clone();
                app.mode   = s.mode.clone();
                app.stats  = Some(s);
                app.stats_loaded = true;
            }
        }

        // Drain stream events
        if let Some(rx) = &mut app.stream_rx {
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    StreamEvent::Token(t) => {
                        app.stream_buf.push_str(&t);
                        let buf = app.stream_buf.clone();
                        if let Some(last) = app.messages.last_mut() {
                            if last.role == Role::Assistant {
                                last.content = buf;
                            }
                        } else {
                            app.messages.push(Message {
                                role: Role::Assistant,
                                content: buf,
                                ts: chrono::Local::now(),
                            });
                        }
                        // auto-scroll
                        app.scroll_offset = u16::MAX;
                    }
                    StreamEvent::Done => {
                        app.loading = false;
                        app.stream_buf.clear();
                        app.stream_tx = None;
                        app.stream_rx = None;
                    }
                    StreamEvent::Error(e) => {
                        app.loading = false;
                        app.stream_buf.clear();
                        app.stream_tx = None;
                        app.stream_rx = None;
                        app.set_status(&format!("Błąd: {e}"), true);
                    }
                }
            }
        }

        // Input events (100ms timeout so we keep refreshing)
        let timeout = tokio::time::sleep(Duration::from_millis(100));
        tokio::select! {
            _ = timeout => {}
            maybe_event = events.next() => {
                if let Some(Ok(Event::Key(key))) = maybe_event {
                    if handle_key(&mut app, key).await? {
                        return Ok(());
                    }
                }
            }
        }
    }
}

async fn handle_key(app: &mut App, key: KeyEvent) -> Result<bool> {
    // Global shortcuts
    match (key.modifiers, key.code) {
        (KeyModifiers::CONTROL, KeyCode::Char('c')) |
        (KeyModifiers::CONTROL, KeyCode::Char('q')) => return Ok(true),

        (KeyModifiers::CONTROL, KeyCode::Char('n')) => {
            app.new_session();
            return Ok(false);
        }
        (KeyModifiers::CONTROL, KeyCode::Char('s')) => {
            app.screen = if app.screen == Screen::Settings { Screen::Chat } else { Screen::Settings };
            return Ok(false);
        }
        _ => {}
    }

    match app.screen.clone() {
        Screen::Help => {
            app.screen = Screen::Chat;
        }
        Screen::Settings => handle_settings_key(app, key).await,
        Screen::Chat     => handle_chat_key(app, key).await?,
    }
    Ok(false)
}

async fn handle_settings_key(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc => { app.screen = Screen::Chat; }
        KeyCode::Up | KeyCode::Char('k') => { if app.settings_cursor > 0 { app.settings_cursor -= 1; } }
        KeyCode::Down | KeyCode::Char('j') => { if app.settings_cursor < 3 { app.settings_cursor += 1; } }
        KeyCode::Enter | KeyCode::Char(' ') => {
            app.screen = Screen::Chat;
            match app.settings_cursor {
                0 => { app.engine = "transformers".into(); app.set_status("Silnik: GPU (Transformers)", false); let _ = api::set_engine("transformers").await; }
                1 => { app.engine = "ollama".into(); app.set_status("Silnik: CPU (Ollama)", false); let _ = api::set_engine("ollama").await; }
                2 => { app.mode = "general".into(); app.set_status("Tryb: Ogólny", false); let _ = api::set_mode("general").await; }
                3 => { app.mode = "programista".into(); app.set_status("Tryb: Programista", false); let _ = api::set_mode("programista").await; }
                _ => {}
            }
        }
        _ => {}
    }
}

async fn handle_chat_key(app: &mut App, key: KeyEvent) -> Result<()> {
    match key.code {
        KeyCode::Char('?') if key.modifiers.is_empty() && app.input.is_empty() => {
            app.screen = Screen::Help;
        }
        KeyCode::Enter if !key.modifiers.contains(KeyModifiers::SHIFT) => {
            let text = app.input.trim().to_string();
            if text.is_empty() || app.loading { return Ok(()); }
            app.input.clear();
            app.input_cursor = 0;
            app.messages.push(Message {
                role: Role::User,
                content: text.clone(),
                ts: chrono::Local::now(),
            });
            app.loading = true;
            app.scroll_offset = u16::MAX;

            let (tx, rx) = mpsc::channel(256);
            app.stream_tx = Some(tx.clone());
            app.stream_rx = Some(rx);
            let sid = app.session_id.clone();
            tokio::spawn(async move {
                api::stream_chat(sid.as_deref(), &text, tx).await;
            });
        }
        KeyCode::Char(c) => { app.insert_char(c); }
        KeyCode::Backspace => { app.delete_char_back(); }
        KeyCode::Up => {
            app.scroll_offset = app.scroll_offset.saturating_sub(1);
        }
        KeyCode::Down => {
            app.scroll_offset = app.scroll_offset.saturating_add(1);
        }
        KeyCode::PageUp   => { app.scroll_offset = app.scroll_offset.saturating_sub(10); }
        KeyCode::PageDown => { app.scroll_offset = app.scroll_offset.saturating_add(10); }
        KeyCode::Left => {
            if app.input_cursor > 0 {
                app.input_cursor -= 1;
            }
        }
        KeyCode::Right => {
            if app.input_cursor < app.input.len() {
                app.input_cursor += 1;
            }
        }
        _ => {}
    }
    Ok(())
}
