use crate::app::{App, Role, Screen};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, Gauge, List, ListItem, Paragraph, Wrap},
    Frame,
};

// ── Colour palette ────────────────────────────────────────────────
const ACCENT:     Color = Color::Rgb(217, 119, 6);
const ACCENT_DIM: Color = Color::Rgb(146, 64, 14);
const TEXT_PRI:   Color = Color::Rgb(245, 240, 232);
const TEXT_SEC:   Color = Color::Rgb(168, 152, 128);
const TEXT_MUTED: Color = Color::Rgb(107, 96, 87);
const TEXT_AMB:   Color = Color::Rgb(245, 158, 11);
const BG_SURF:    Color = Color::Rgb(33, 31, 28);
const BG_ELEV:    Color = Color::Rgb(42, 40, 37);
const C_GREEN:    Color = Color::Rgb(74, 222, 128);
const C_RED:      Color = Color::Rgb(248, 113, 113);

// ── Main draw ────────────────────────────────────────────────────

pub fn draw(f: &mut Frame, app: &mut App) {
    let size = f.area();

    // background
    f.render_widget(Block::default().style(Style::default().bg(Color::Rgb(26, 25, 22))), size);

    match app.screen {
        Screen::Help     => draw_chat_bg(f, app, size), // draw chat behind overlay
        Screen::Settings => draw_chat_bg(f, app, size),
        Screen::Chat     => { draw_chat_bg(f, app, size); return; }
    }

    // Overlay on top
    match app.screen {
        Screen::Help     => draw_help_overlay(f, app, size),
        Screen::Settings => draw_settings_overlay(f, app, size),
        _ => {}
    }
}

fn draw_chat_bg(f: &mut Frame, app: &mut App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),   // header
            Constraint::Length(1),   // divider
            Constraint::Min(4),      // viewport
            Constraint::Length(5),   // input box
            Constraint::Length(1),   // status bar
        ])
        .split(area);

    render_header(f, app, chunks[0]);
    render_divider(f, chunks[1]);
    render_messages(f, app, chunks[2]);
    render_input(f, app, chunks[3]);
    render_statusbar(f, app, chunks[4]);
}

// ── Header ────────────────────────────────────────────────────────

fn render_header(f: &mut Frame, app: &App, area: Rect) {
    let engine_str = if app.engine == "ollama" { "CPU" } else { "GPU" };
    let mode_str   = if app.mode == "programista" { "Dev" } else { "Ogólny" };
    let model_str  = if app.stats.as_ref().map(|s| s.model_loaded).unwrap_or(false) { "● model" } else { "○ offline" };

    let left  = Line::from(vec![
        Span::styled("⬡ HexAi", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled("  v2.0.0", Style::default().fg(TEXT_MUTED)),
    ]);
    let right = Line::from(vec![
        Span::styled(format!(" {engine_str} "), Style::default().fg(ACCENT).bg(Color::Rgb(45, 21, 0)).add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::styled(mode_str, Style::default().fg(TEXT_SEC)),
        Span::raw("  "),
        Span::styled(model_str, Style::default().fg(if app.stats.as_ref().map(|s| s.model_loaded).unwrap_or(false) { C_GREEN } else { TEXT_MUTED })),
    ]);

    let block = Block::default().style(Style::default().bg(BG_SURF));
    f.render_widget(block, area);

    let left_p  = Paragraph::new(left).block(Block::default().style(Style::default().bg(BG_SURF)));
    let right_p = Paragraph::new(right)
        .alignment(Alignment::Right)
        .block(Block::default().style(Style::default().bg(BG_SURF)));
    f.render_widget(left_p,  area);
    f.render_widget(right_p, area);
}

fn render_divider(f: &mut Frame, area: Rect) {
    let line = "─".repeat(area.width as usize);
    f.render_widget(Paragraph::new(line).style(Style::default().fg(Color::Rgb(45, 43, 40))), area);
}

// ── Messages ──────────────────────────────────────────────────────

fn render_messages(f: &mut Frame, app: &mut App, area: Rect) {
    if app.messages.is_empty() {
        let empty = vec![
            Line::from(""),
            Line::from(Span::styled("        ⬡  HexAi", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))),
            Line::from(""),
            Line::from(Span::styled("        Jak mogę Ci dziś pomóc?", Style::default().fg(TEXT_PRI).add_modifier(Modifier::BOLD))),
            Line::from(""),
            Line::from(Span::styled("        Wpisz wiadomość i naciśnij Enter.", Style::default().fg(TEXT_MUTED))),
            Line::from(Span::styled("        Użyj ? aby zobaczyć skróty klawiszowe.", Style::default().fg(TEXT_MUTED))),
        ];
        f.render_widget(Paragraph::new(empty), area);
        return;
    }

    let wrap_width = (area.width as usize).saturating_sub(8);
    let mut lines: Vec<Line> = vec![];

    for msg in &app.messages {
        let ts = msg.ts.format("%H:%M").to_string();
        match msg.role {
            Role::User => {
                lines.push(Line::from(vec![
                    Span::styled("  Ty", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
                    Span::styled(format!("  {ts}"), Style::default().fg(TEXT_MUTED)),
                ]));
                for line in textwrap::wrap(&msg.content, wrap_width) {
                    lines.push(Line::from(Span::styled(
                        format!("  {line}"),
                        Style::default().fg(Color::Rgb(254, 243, 199)),
                    )));
                }
            }
            Role::Assistant => {
                lines.push(Line::from(vec![
                    Span::styled("  HexAi", Style::default().fg(TEXT_AMB).add_modifier(Modifier::BOLD)),
                    Span::styled(format!("  {ts}"), Style::default().fg(TEXT_MUTED)),
                ]));
                for line in textwrap::wrap(&msg.content, wrap_width) {
                    lines.push(render_ai_line(&line));
                }
            }
        }
        lines.push(Line::from(""));
    }

    if app.loading && app.stream_buf.is_empty() {
        lines.push(Line::from(Span::styled("  HexAi", Style::default().fg(TEXT_AMB).add_modifier(Modifier::BOLD))));
        lines.push(Line::from(Span::styled("  ⣾ myślę…", Style::default().fg(ACCENT))));
        lines.push(Line::from(""));
    }

    let total_lines = lines.len() as u16;
    let visible = area.height;

    // Clamp scroll
    let max_scroll = total_lines.saturating_sub(visible);
    if app.scroll_offset == u16::MAX {
        app.scroll_offset = max_scroll;
    } else {
        app.scroll_offset = app.scroll_offset.min(max_scroll);
    }

    let p = Paragraph::new(Text::from(lines))
        .scroll((app.scroll_offset, 0));
    f.render_widget(p, area);
}

fn render_ai_line<'a>(line: &str) -> Line<'a> {
    // Highlight `code` spans
    let mut spans = vec![Span::raw("  ")];
    let mut rest = line.to_string();
    while let Some(start) = rest.find('`') {
        let before = rest[..start].to_string();
        if !before.is_empty() {
            spans.push(Span::styled(before, Style::default().fg(TEXT_PRI)));
        }
        rest = rest[start + 1..].to_string();
        if let Some(end) = rest.find('`') {
            let code = rest[..end].to_string();
            spans.push(Span::styled(code, Style::default().fg(TEXT_AMB).add_modifier(Modifier::BOLD)));
            rest = rest[end + 1..].to_string();
        } else {
            spans.push(Span::styled("`", Style::default().fg(TEXT_PRI)));
            break;
        }
    }
    if !rest.is_empty() {
        spans.push(Span::styled(rest, Style::default().fg(TEXT_PRI)));
    }
    Line::from(spans)
}

// ── Input ─────────────────────────────────────────────────────────

fn render_input(f: &mut Frame, app: &App, area: Rect) {
    let spinner_frames = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"];
    let spin_idx = (chrono::Local::now().timestamp_millis() / 120) as usize % spinner_frames.len();
    let hint = if app.loading {
        format!(" {} Generowanie…", spinner_frames[spin_idx])
    } else {
        " Enter wyślij · Shift+Enter nowa linia · Ctrl+N nowa · ? pomoc".into()
    };

    // Show cursor in input
    let before_cursor = &app.input[..app.input_cursor];
    let after_cursor  = &app.input[app.input_cursor..];
    let cursor_char   = if after_cursor.is_empty() { " " } else { &after_cursor[..after_cursor.chars().next().map(|c| c.len_utf8()).unwrap_or(1)] };
    let after_cursor_rest = if after_cursor.len() > cursor_char.len() { &after_cursor[cursor_char.len()..] } else { "" };

    let input_line = Line::from(vec![
        Span::styled(before_cursor, Style::default().fg(TEXT_PRI)),
        Span::styled(cursor_char, Style::default().fg(Color::Black).bg(ACCENT)),
        Span::styled(after_cursor_rest, Style::default().fg(TEXT_PRI)),
    ]);

    let text = Text::from(vec![
        input_line,
        Line::from(Span::styled(hint, Style::default().fg(TEXT_MUTED))),
    ]);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT_DIM))
        .style(Style::default().bg(BG_SURF));

    f.render_widget(Paragraph::new(text).block(block).wrap(Wrap { trim: false }), area);
}

// ── Status bar ────────────────────────────────────────────────────

fn render_statusbar(f: &mut Frame, app: &App, area: Rect) {
    let keys = [("Ctrl+N", "Nowa"), ("Ctrl+S", "Ustawienia"), ("?", "Pomoc"), ("Ctrl+C", "Wyjście")];
    let mut spans: Vec<Span> = vec![];
    for (k, v) in &keys {
        spans.push(Span::styled(format!(" {k} "), Style::default().fg(TEXT_SEC).bg(BG_ELEV)));
        spans.push(Span::styled(format!(" {v}  "), Style::default().fg(TEXT_MUTED).bg(BG_SURF)));
    }

    if let Some((msg, is_err)) = &app.status_msg {
        let col = if *is_err { C_RED } else { ACCENT };
        spans.push(Span::styled(format!(" {msg} "), Style::default().fg(col)));
    } else if let Some(s) = &app.stats {
        if let (Some(used), Some(total)) = (s.vram_used_gb, s.vram_total_gb) {
            let pct = (used / total * 100.0) as u64;
            spans.push(Span::styled(
                format!(" VRAM {pct}% · {} sesji ", s.active_sessions),
                Style::default().fg(TEXT_MUTED),
            ));
        }
    }

    f.render_widget(
        Paragraph::new(Line::from(spans)).style(Style::default().bg(BG_SURF)),
        area,
    );
}

// ── Settings overlay ──────────────────────────────────────────────

fn draw_settings_overlay(f: &mut Frame, app: &App, area: Rect) {
    let modal_w = 44u16;
    let modal_h = 18u16;
    let x = area.width.saturating_sub(modal_w) / 2;
    let y = area.height.saturating_sub(modal_h) / 2;
    let modal_area = Rect::new(x, y, modal_w.min(area.width), modal_h.min(area.height));

    f.render_widget(Clear, modal_area);

    let block = Block::default()
        .title(" ⚙  Ustawienia ")
        .title_style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT_DIM))
        .style(Style::default().bg(BG_SURF));

    let items: Vec<(&str, &str, &str, usize)> = vec![
        ("SILNIK", "transformers", "⚡ Transformers (GPU)", 0),
        ("SILNIK", "ollama",       "🖥  Ollama (CPU)",      1),
        ("TRYB",   "general",      "💬 Ogólny",             2),
        ("TRYB",   "programista",  "💻 Programista (Dev)",  3),
    ];

    let mut lines: Vec<Line> = vec![Line::from("")];
    let mut prev_group = "";
    for (group, val, label, idx) in &items {
        if *group != prev_group {
            lines.push(Line::from(Span::styled(*group, Style::default().fg(TEXT_MUTED))));
            prev_group = group;
        }
        let is_current = ((*group == "SILNIK") && (*val == app.engine))
            || ((*group == "TRYB") && (*val == app.mode));
        let is_cursor = app.settings_cursor == *idx;
        let prefix = if is_current { "  ● " } else { "  ○ " };

        let style = if is_current {
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
        } else if is_cursor {
            Style::default().fg(TEXT_PRI).bg(Color::Rgb(49, 47, 43))
        } else {
            Style::default().fg(TEXT_SEC)
        };
        lines.push(Line::from(Span::styled(format!("{prefix}{label}"), style)));
    }

    // VRAM bar
    if let Some(s) = &app.stats {
        if let (Some(used), Some(total)) = (s.vram_used_gb, s.vram_total_gb) {
            let pct = (used / total * 100.0) as u16;
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("  VRAM", Style::default().fg(TEXT_MUTED))));
            let bar_len = 20usize;
            let filled = (pct as usize * bar_len / 100).min(bar_len);
            let color = if pct > 80 { C_RED } else if pct > 60 { ACCENT } else { C_GREEN };
            let bar = format!("  {}{}", "█".repeat(filled), "░".repeat(bar_len - filled));
            lines.push(Line::from(vec![
                Span::styled(bar, Style::default().fg(color)),
                Span::styled(format!(" {pct}% ({used:.1}/{total:.1}GB)"), Style::default().fg(TEXT_MUTED)),
            ]));
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  ↑↓/jk Nawiguj · Enter Wybierz · Esc Zamknij",
        Style::default().fg(TEXT_MUTED),
    )));

    f.render_widget(Paragraph::new(Text::from(lines)).block(block), modal_area);
}

// ── Help overlay ──────────────────────────────────────────────────

fn draw_help_overlay(f: &mut Frame, _app: &App, area: Rect) {
    let modal_w = 50u16;
    let modal_h = 16u16;
    let x = area.width.saturating_sub(modal_w) / 2;
    let y = area.height.saturating_sub(modal_h) / 2;
    let modal_area = Rect::new(x, y, modal_w.min(area.width), modal_h.min(area.height));
    f.render_widget(Clear, modal_area);

    let block = Block::default()
        .title(" ⬡  HexAi – Pomoc ")
        .title_style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT_DIM))
        .style(Style::default().bg(BG_SURF));

    let shortcuts: &[(&str, &str)] = &[
        ("Enter",           "Wyślij wiadomość"),
        ("Shift+Enter",     "Nowa linia"),
        ("Ctrl+N",          "Nowa rozmowa"),
        ("Ctrl+S",          "Ustawienia"),
        ("↑↓ / PgUp/PgDn", "Przewijaj historię"),
        ("?",               "Ta pomoc"),
        ("Ctrl+C / Ctrl+Q", "Wyjście"),
    ];

    let mut lines: Vec<Line> = vec![Line::from("")];
    for (k, v) in shortcuts {
        lines.push(Line::from(vec![
            Span::styled(format!("  {k:<18}", k = k), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
            Span::styled(*v, Style::default().fg(TEXT_PRI)),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled("  API: http://localhost:8000", Style::default().fg(TEXT_MUTED))));
    lines.push(Line::from(Span::styled("  HexAi for HackerOS · GPL-3.0", Style::default().fg(TEXT_MUTED))));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled("  Naciśnij dowolny klawisz aby zamknąć", Style::default().fg(TEXT_MUTED))));

    f.render_widget(Paragraph::new(Text::from(lines)).block(block), modal_area);
}
