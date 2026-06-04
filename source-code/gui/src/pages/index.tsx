import { useState, useEffect, useRef, useCallback } from 'react';
import { api, Stats } from '@/lib/api';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import oneDark from 'react-syntax-highlighter/dist/cjs/styles/prism/one-dark';
import Head from 'next/head';

type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
};

type ConversationMeta = {
  id: string;
  title: string;
  createdAt: Date;
};

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversations, setConversations] = useState<ConversationMeta[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<Stats | null>(null);
  const [engine, setEngine] = useState('transformers');
  const [mode, setMode] = useState('general');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [currentTitle, setCurrentTitle] = useState('Nowa rozmowa');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Restore conversations from localStorage (not sessionId - each launch is fresh)
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const stored = localStorage.getItem('hexai_conversations');
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setConversations(parsed.map((c: any) => ({ ...c, createdAt: new Date(c.createdAt) })));
      } catch {}
    }
  }, []);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const s = await api.getStats();
        setStats(s);
        setEngine(s.engine);
        setMode(s.mode);
      } catch {}
    };
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
  }, [input]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || loading) return;
    const text = input.trim();
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: text, timestamp: new Date() };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    if (messages.length === 0) setCurrentTitle(text.slice(0, 40) + (text.length > 40 ? '…' : ''));

    try {
      const assistantMsg: Message = { id: (Date.now() + 1).toString(), role: 'assistant', content: '', timestamp: new Date() };
      setMessages(prev => [...prev, assistantMsg]);
      let full = '';
      await api.streamChat(sessionId, text, token => {
        full += token;
        setMessages(prev => prev.map(m => m.id === assistantMsg.id ? { ...m, content: full } : m));
      });
    } catch {
      setMessages(prev => [...prev, { id: Date.now().toString(), role: 'assistant', content: '❌ Błąd połączenia z serwerem.', timestamp: new Date() }]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, sessionId, messages.length]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const handleEngineChange = async (val: string) => {
    await api.setEngine(val).catch(() => {});
    setEngine(val);
  };

  const handleModeChange = async (val: string) => {
    await api.setMode(val).catch(() => {});
    setMode(val);
  };

  const handleNewSession = () => {
    if (sessionId && messages.length > 0) {
      const conv: ConversationMeta = { id: sessionId, title: currentTitle, createdAt: new Date() };
      const updated = [conv, ...conversations].slice(0, 20);
      setConversations(updated);
      if (typeof window !== 'undefined') localStorage.setItem('hexai_conversations', JSON.stringify(updated));
    }
    setSessionId(null);
    setMessages([]);
    setCurrentTitle('Nowa rozmowa');
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.type.startsWith('audio/')) {
      try { const { transcription } = await api.transcribe(file); setInput(transcription); } catch {}
    } else if (file.type.startsWith('image/')) {
      try {
        const { description } = await api.analyzeImage(file, 'Opisz ten obraz po polsku.');
        setMessages(prev => [...prev, { id: Date.now().toString(), role: 'assistant', content: `🖼️ ${description}`, timestamp: new Date() }]);
      } catch {}
    }
  };

  const fmt = (d: Date) => d.toLocaleTimeString('pl-PL', { hour: '2-digit', minute: '2-digit' });
  const vramPct = stats?.vram_total_gb && stats?.vram_used_gb ? Math.round((stats.vram_used_gb / stats.vram_total_gb) * 100) : null;

  return (
    <>
      <Head>
        <title>HexAi – Asystent AI</title>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
      </Head>

      <div className="app-shell">
        {/* Sidebar */}
        <aside className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
          <div className="sidebar-header">
            <div className="logo">
              <span className="logo-hex">⬡</span>
              <span className="logo-text">HexAi</span>
            </div>
            <button className="icon-btn" onClick={() => setSidebarOpen(false)}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M15 18l-6-6 6-6"/></svg>
            </button>
          </div>

          <button className="new-chat-btn" onClick={handleNewSession}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M12 5v14M5 12h14"/></svg>
            Nowa rozmowa
          </button>

          <div className="sidebar-section-label">Historia</div>
          <div className="conversations-list">
            {conversations.length === 0 && <div className="conv-empty">Brak poprzednich rozmów</div>}
            {conversations.map(c => (
              <div key={c.id} className="conv-item">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
                <span className="conv-title">{c.title}</span>
              </div>
            ))}
          </div>

          <div className="sidebar-bottom">
            <div className="sidebar-section-label">Konfiguracja</div>
            <div className="config-group">
              <label className="config-label">Silnik</label>
              <div className="segmented">
                <button className={engine === 'transformers' ? 'active' : ''} onClick={() => handleEngineChange('transformers')}>GPU</button>
                <button className={engine === 'ollama' ? 'active' : ''} onClick={() => handleEngineChange('ollama')}>CPU</button>
              </div>
            </div>
            <div className="config-group">
              <label className="config-label">Tryb</label>
              <div className="segmented">
                <button className={mode === 'general' ? 'active' : ''} onClick={() => handleModeChange('general')}>Ogólny</button>
                <button className={mode === 'programista' ? 'active' : ''} onClick={() => handleModeChange('programista')}>Dev</button>
              </div>
            </div>
            <div className="config-group">
              <label className="config-label">Plik (audio / obraz)</label>
              <label className="file-upload-btn">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>
                Wybierz plik
                <input type="file" accept="audio/*,image/*" onChange={handleFileUpload} style={{ display: 'none' }} />
              </label>
            </div>

            {stats && (
              <div className="stats-panel">
                <div className="stat-row"><span>Model</span><span className={`stat-badge ${stats.model_loaded ? 'loaded' : 'idle'}`}>{stats.model_loaded ? 'aktywny' : 'nieaktywny'}</span></div>
                <div className="stat-row"><span>Sesje</span><span>{stats.active_sessions}</span></div>
                {vramPct !== null && (
                  <div className="stat-vram">
                    <div className="stat-vram-label"><span>VRAM</span><span>{stats.vram_used_gb?.toFixed(1)} / {stats.vram_total_gb?.toFixed(1)} GB</span></div>
                    <div className="vram-bar"><div className="vram-fill" style={{ width: `${vramPct}%`, background: vramPct > 80 ? '#ef4444' : vramPct > 60 ? '#f59e0b' : '#d97706' }} /></div>
                  </div>
                )}
              </div>
            )}
          </div>
        </aside>

        {!sidebarOpen && (
          <button className="sidebar-toggle" onClick={() => setSidebarOpen(true)}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 18l6-6-6-6"/></svg>
          </button>
        )}

        {/* Main */}
        <main className="chat-main">
          <header className="chat-header">
            <div className="chat-title">
              <span className="chat-title-text">{currentTitle}</span>
              {mode === 'programista' && <span className="mode-chip">Dev</span>}
            </div>
            <div className="header-actions">
              <span className="engine-indicator">{engine === 'transformers' ? '⚡ GPU' : '🖥 CPU'}</span>
            </div>
          </header>

          <div className="messages-area">
            {messages.length === 0 && (
              <div className="empty-state">
                <div className="empty-hex">⬡</div>
                <h2>Jak mogę Ci pomóc?</h2>
                <p>Zapytaj o cokolwiek – kod, analizę, pytania ogólne.</p>
                <div className="suggestions">
                  {['Napisz funkcję w Rust', 'Wyjaśnij kwantowe splątanie', 'Pomóż z debugowaniem'].map(s => (
                    <button key={s} className="suggestion-chip" onClick={() => setInput(s)}>{s}</button>
                  ))}
                </div>
              </div>
            )}

            {messages.map(msg => (
              <div key={msg.id} className={`message-row ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? <div className="avatar-user">U</div> : <div className="avatar-ai">⬡</div>}
                </div>
                <div className="message-content">
                  <div className="message-meta">
                    <span className="message-author">{msg.role === 'user' ? 'Ty' : 'HexAi'}</span>
                    <span className="message-time">{fmt(msg.timestamp)}</span>
                  </div>
                  <div className={`message-bubble ${msg.role}`}>
                    <ReactMarkdown
                      components={{
                        code({ node, inline, className, children, ...props }: any) {
                          const match = /language-(\w+)/.exec(className || '');
                          return !inline && match ? (
                            <div className="code-block">
                              <div className="code-header">
                                <span className="code-lang">{match[1]}</span>
                                <button className="copy-btn" onClick={() => navigator.clipboard.writeText(String(children))}>Kopiuj</button>
                              </div>
                              <SyntaxHighlighter style={oneDark} language={match[1]} PreTag="div" customStyle={{ margin: 0, borderRadius: '0 0 8px 8px', fontSize: '13px' }} {...props}>
                                {String(children).replace(/\n$/, '')}
                              </SyntaxHighlighter>
                            </div>
                          ) : <code className="inline-code" {...props}>{children}</code>;
                        },
                        p: ({ children }) => <p className="md-p">{children}</p>,
                        ul: ({ children }) => <ul className="md-ul">{children}</ul>,
                        ol: ({ children }) => <ol className="md-ol">{children}</ol>,
                        li: ({ children }) => <li className="md-li">{children}</li>,
                        h1: ({ children }) => <h1 className="md-h1">{children}</h1>,
                        h2: ({ children }) => <h2 className="md-h2">{children}</h2>,
                        h3: ({ children }) => <h3 className="md-h3">{children}</h3>,
                        blockquote: ({ children }) => <blockquote className="md-blockquote">{children}</blockquote>,
                      }}
                    >
                      {msg.content}
                    </ReactMarkdown>
                    {msg.role === 'assistant' && !msg.content && <div className="thinking-dots"><span/><span/><span/></div>}
                  </div>
                </div>
              </div>
            ))}

            {loading && messages[messages.length - 1]?.role !== 'assistant' && (
              <div className="message-row assistant">
                <div className="message-avatar"><div className="avatar-ai">⬡</div></div>
                <div className="message-content">
                  <div className="message-bubble assistant"><div className="thinking-dots"><span/><span/><span/></div></div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-area">
            <div className="input-container">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Napisz wiadomość… (Enter – wyślij, Shift+Enter – nowa linia)"
                className="message-input"
                rows={1}
                disabled={loading}
              />
              <div className="input-actions">
                <label className="attach-btn">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/></svg>
                  <input type="file" accept="audio/*,image/*" onChange={handleFileUpload} style={{ display: 'none' }} />
                </label>
                <button onClick={handleSend} disabled={loading || !input.trim()} className="send-btn">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                </button>
              </div>
            </div>
            <div className="input-hint">HexAi może popełniać błędy. Weryfikuj ważne informacje.</div>
          </div>
        </main>
      </div>

      <style jsx global>{`
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        :root {
          --bg-base: #1a1916; --bg-surface: #211f1c; --bg-elevated: #2a2825;
          --bg-hover: #312f2b; --bg-active: #3a3835;
          --border: #3a3835; --border-subtle: #2d2b28;
          --accent: #d97706; --accent-dim: #92400e; --accent-glow: rgba(217,119,6,0.15);
          --text-primary: #f5f0e8; --text-secondary: #a89880; --text-muted: #6b6057; --text-accent: #f59e0b;
          --sidebar-w: 280px; --header-h: 56px;
          --radius: 12px; --radius-sm: 8px; --radius-xs: 6px;
          --font-sans: 'DM Sans','Inter',ui-sans-serif,system-ui,-apple-system,sans-serif;
          --font-serif: 'Instrument Serif','Palatino Linotype','Book Antiqua',Georgia,serif;
          --font-mono: 'JetBrains Mono','Fira Code','Cascadia Code','Consolas','Liberation Mono',monospace;
        }
        html, body { height: 100%; background: var(--bg-base); color: var(--text-primary); font-family: var(--font-sans); font-size: 14px; line-height: 1.6; overflow: hidden; }
        .app-shell { display: flex; height: 100vh; overflow: hidden; }

        .sidebar { width: var(--sidebar-w); min-width: var(--sidebar-w); background: var(--bg-surface); border-right: 1px solid var(--border-subtle); display: flex; flex-direction: column; transition: width .25s ease, min-width .25s ease; overflow: hidden; }
        .sidebar.closed { width: 0; min-width: 0; }
        .sidebar-header { display: flex; align-items: center; justify-content: space-between; padding: 20px 16px 16px; border-bottom: 1px solid var(--border-subtle); }
        .logo { display: flex; align-items: center; gap: 8px; }
        .logo-hex { font-size: 22px; color: var(--accent); }
        .logo-text { font-family: var(--font-serif); font-size: 20px; letter-spacing: -.02em; }
        .icon-btn { background: none; border: none; cursor: pointer; color: var(--text-muted); padding: 4px; border-radius: var(--radius-xs); display: flex; align-items: center; transition: color .15s, background .15s; }
        .icon-btn:hover { color: var(--text-primary); background: var(--bg-hover); }
        .new-chat-btn { margin: 12px 12px 4px; display: flex; align-items: center; gap: 6px; background: var(--accent-glow); border: 1px solid var(--accent-dim); color: var(--text-accent); padding: 8px 14px; border-radius: var(--radius-sm); cursor: pointer; font-size: 13px; font-weight: 500; font-family: var(--font-sans); transition: background .15s; width: calc(100% - 24px); }
        .new-chat-btn:hover { background: rgba(217,119,6,.25); border-color: var(--accent); }
        .sidebar-section-label { padding: 12px 16px 4px; font-size: 10px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--text-muted); }
        .conversations-list { flex: 1; overflow-y: auto; padding: 4px 8px; }
        .conversations-list::-webkit-scrollbar { width: 3px; }
        .conversations-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
        .conv-empty { color: var(--text-muted); font-size: 12px; padding: 8px; }
        .conv-item { display: flex; align-items: center; gap: 8px; padding: 7px 8px; border-radius: var(--radius-xs); cursor: pointer; color: var(--text-secondary); transition: background .15s, color .15s; font-size: 13px; }
        .conv-item:hover { background: var(--bg-hover); color: var(--text-primary); }
        .conv-title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .sidebar-bottom { border-top: 1px solid var(--border-subtle); padding: 8px; }
        .config-group { margin: 4px 0 8px; padding: 0 8px; }
        .config-label { display: block; font-size: 11px; color: var(--text-muted); margin-bottom: 5px; font-weight: 500; }
        .segmented { display: flex; background: var(--bg-elevated); border-radius: var(--radius-xs); padding: 2px; gap: 2px; }
        .segmented button { flex: 1; background: none; border: none; cursor: pointer; padding: 4px 8px; border-radius: 4px; font-size: 12px; color: var(--text-secondary); font-family: var(--font-sans); font-weight: 500; transition: background .15s, color .15s; }
        .segmented button.active { background: var(--bg-active); color: var(--text-primary); }
        .segmented button:hover:not(.active) { background: var(--bg-hover); color: var(--text-primary); }
        .file-upload-btn { display: inline-flex; align-items: center; gap: 6px; background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); padding: 5px 10px; border-radius: var(--radius-xs); cursor: pointer; font-size: 12px; font-family: var(--font-sans); transition: background .15s; }
        .file-upload-btn:hover { background: var(--bg-hover); color: var(--text-primary); }
        .stats-panel { background: var(--bg-elevated); border: 1px solid var(--border-subtle); border-radius: var(--radius-xs); padding: 8px 10px; margin: 8px 8px 4px; }
        .stat-row { display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: var(--text-secondary); margin-bottom: 4px; }
        .stat-badge { font-size: 10px; padding: 1px 6px; border-radius: 99px; font-weight: 600; }
        .stat-badge.loaded { background: rgba(34,197,94,.15); color: #4ade80; }
        .stat-badge.idle   { background: rgba(107,96,87,.3); color: var(--text-muted); }
        .stat-vram { margin-top: 6px; }
        .stat-vram-label { display: flex; justify-content: space-between; font-size: 10px; color: var(--text-muted); margin-bottom: 4px; }
        .vram-bar { height: 3px; background: var(--bg-hover); border-radius: 99px; overflow: hidden; }
        .vram-fill { height: 100%; border-radius: 99px; transition: width .5s ease; }

        .sidebar-toggle { position: fixed; left: 12px; top: 14px; z-index: 100; background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); width: 32px; height: 32px; border-radius: var(--radius-xs); cursor: pointer; display: flex; align-items: center; justify-content: center; transition: background .15s; }
        .sidebar-toggle:hover { background: var(--bg-hover); color: var(--text-primary); }

        .chat-main { flex: 1; display: flex; flex-direction: column; overflow: hidden; background: var(--bg-base); }
        .chat-header { height: var(--header-h); border-bottom: 1px solid var(--border-subtle); display: flex; align-items: center; justify-content: space-between; padding: 0 24px; }
        .chat-title { display: flex; align-items: center; gap: 10px; }
        .chat-title-text { font-family: var(--font-serif); font-size: 17px; }
        .mode-chip { font-size: 10px; font-weight: 600; letter-spacing: .05em; background: var(--accent-glow); border: 1px solid var(--accent-dim); color: var(--text-accent); padding: 1px 7px; border-radius: 99px; }
        .engine-indicator { font-size: 12px; color: var(--text-muted); }

        .messages-area { flex: 1; overflow-y: auto; padding: 24px 0; scroll-behavior: smooth; }
        .messages-area::-webkit-scrollbar { width: 4px; }
        .messages-area::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

        .empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; text-align: center; padding: 40px 20px; }
        .empty-hex { font-size: 48px; color: var(--accent); margin-bottom: 16px; }
        .empty-state h2 { font-family: var(--font-serif); font-size: 28px; margin-bottom: 8px; }
        .empty-state p { color: var(--text-secondary); font-size: 15px; margin-bottom: 24px; }
        .suggestions { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
        .suggestion-chip { background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); padding: 8px 16px; border-radius: 99px; cursor: pointer; font-size: 13px; font-family: var(--font-sans); transition: background .15s, color .15s; }
        .suggestion-chip:hover { background: var(--bg-hover); color: var(--text-primary); }

        .message-row { display: flex; gap: 12px; padding: 4px 24px; max-width: 860px; margin: 0 auto; width: 100%; }
        .message-row.user { flex-direction: row-reverse; }
        .message-row.user .message-content { align-items: flex-end; }
        .message-avatar { flex-shrink: 0; padding-top: 2px; }
        .avatar-user, .avatar-ai { width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 600; }
        .avatar-user { background: var(--accent-dim); color: var(--text-accent); }
        .avatar-ai   { background: var(--bg-elevated); color: var(--accent); border: 1px solid var(--border); font-size: 16px; }
        .message-content { flex: 1; display: flex; flex-direction: column; gap: 3px; min-width: 0; }
        .message-meta { display: flex; align-items: baseline; gap: 8px; }
        .message-author { font-size: 12px; font-weight: 600; color: var(--text-secondary); }
        .message-time   { font-size: 11px; color: var(--text-muted); }
        .message-bubble { padding: 12px 16px; border-radius: var(--radius); line-height: 1.65; word-break: break-word; max-width: 100%; }
        .message-bubble.user      { background: var(--accent-dim); color: #fef3c7; border-radius: var(--radius) var(--radius) 2px var(--radius); }
        .message-bubble.assistant { background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius) var(--radius) var(--radius) 2px; }

        .md-p { margin-bottom: 8px; }
        .md-p:last-child { margin-bottom: 0; }
        .md-ul, .md-ol { padding-left: 20px; margin: 6px 0; }
        .md-li { margin: 3px 0; }
        .md-h1 { font-family: var(--font-serif); font-size: 22px; margin: 12px 0 6px; }
        .md-h2 { font-family: var(--font-serif); font-size: 18px; margin: 10px 0 5px; }
        .md-h3 { font-size: 15px; font-weight: 600; margin: 8px 0 4px; }
        .md-blockquote { border-left: 3px solid var(--accent-dim); padding-left: 12px; color: var(--text-secondary); margin: 8px 0; font-style: italic; }
        .inline-code { font-family: var(--font-mono); font-size: 12px; background: var(--bg-elevated); border: 1px solid var(--border); padding: 1px 5px; border-radius: 4px; color: var(--text-accent); }
        .code-block { border-radius: var(--radius-sm); overflow: hidden; margin: 8px 0; border: 1px solid var(--border); }
        .code-header { display: flex; justify-content: space-between; align-items: center; background: #282c34; padding: 6px 12px; border-bottom: 1px solid rgba(255,255,255,.05); }
        .code-lang { font-family: var(--font-mono); font-size: 11px; color: #888; font-weight: 500; }
        .copy-btn { font-size: 11px; color: #888; background: none; border: none; cursor: pointer; font-family: var(--font-sans); padding: 2px 6px; border-radius: 4px; transition: color .15s, background .15s; }
        .copy-btn:hover { color: #fff; background: rgba(255,255,255,.1); }

        .thinking-dots { display: flex; gap: 4px; align-items: center; height: 20px; }
        .thinking-dots span { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); animation: bounce 1.2s ease-in-out infinite; }
        .thinking-dots span:nth-child(1) { animation-delay: 0s; }
        .thinking-dots span:nth-child(2) { animation-delay: .2s; }
        .thinking-dots span:nth-child(3) { animation-delay: .4s; }
        @keyframes bounce { 0%,60%,100% { transform: translateY(0); opacity: .4; } 30% { transform: translateY(-6px); opacity: 1; } }

        .input-area { border-top: 1px solid var(--border-subtle); padding: 12px 24px 16px; background: var(--bg-base); }
        .input-container { display: flex; align-items: flex-end; gap: 8px; background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 12px; transition: border-color .2s; max-width: 860px; margin: 0 auto; }
        .input-container:focus-within { border-color: var(--accent-dim); }
        .message-input { flex: 1; background: none; border: none; outline: none; color: var(--text-primary); font-size: 14px; font-family: var(--font-sans); line-height: 1.5; resize: none; min-height: 22px; max-height: 200px; overflow-y: auto; }
        .message-input::placeholder { color: var(--text-muted); }
        .message-input:disabled { opacity: .5; }
        .input-actions { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
        .attach-btn { display: flex; align-items: center; justify-content: center; width: 30px; height: 30px; border-radius: var(--radius-xs); cursor: pointer; color: var(--text-muted); transition: color .15s, background .15s; }
        .attach-btn:hover { color: var(--text-primary); background: var(--bg-hover); }
        .send-btn { display: flex; align-items: center; justify-content: center; width: 32px; height: 32px; background: var(--accent); border: none; border-radius: var(--radius-xs); color: white; cursor: pointer; transition: background .15s, transform .1s; }
        .send-btn:hover:not(:disabled) { background: #b45309; transform: scale(1.05); }
        .send-btn:disabled { background: var(--bg-active); color: var(--text-muted); cursor: not-allowed; transform: none; }
        .input-hint { text-align: center; font-size: 11px; color: var(--text-muted); max-width: 860px; margin: 8px auto 0; }
      `}</style>
    </>
  );
}
