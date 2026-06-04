// Detect if running inside Tauri and resolve API base dynamically
async function getApiBase(): Promise<string> {
  if (typeof window !== 'undefined' && (window as any).__TAURI__) {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      return await invoke<string>('get_api_base');
    } catch {}
  }
  return process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000';
}

let _apiBase: string | null = null;
async function apiBase(): Promise<string> {
  if (!_apiBase) _apiBase = await getApiBase();
  return _apiBase;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const base = await apiBase();
  const r = await fetch(`${base}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function get<T>(path: string, params?: Record<string, string>): Promise<T> {
  const base = await apiBase();
  const url = new URL(`${base}${path}`);
  if (params) Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  const r = await fetch(url.toString());
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export interface ChatResponse { session_id: string; response: string; }

export interface Stats {
  engine: string;
  mode: string;
  vram_used_gb: number | null;
  vram_total_gb: number | null;
  active_sessions: number;
  history_len: number;
  model_loaded: boolean;
  model_idle_seconds: number;
}

export const api = {
  async chat(sessionId: string | null, message: string): Promise<ChatResponse> {
    return post('/chat', { session_id: sessionId, message });
  },

  async streamChat(
    sessionId: string | null,
    message: string,
    onToken: (t: string) => void,
  ): Promise<void> {
    const base = await apiBase();
    const resp = await fetch(`${base}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message, stream: true }),
    });
    const reader = resp.body?.getReader();
    if (!reader) return;
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      onToken(decoder.decode(value, { stream: true }));
    }
  },

  async setEngine(engine: string): Promise<void> { await post('/engine', { engine }); },
  async setMode(mode: string): Promise<void>     { await post('/mode', { mode }); },

  async ragQuery(query: string): Promise<{ response: string }> {
    return post('/rag', { query });
  },

  async generateImage(prompt: string): Promise<{ image_base64: string }> {
    return post('/generate_image', { prompt });
  },

  async transcribe(file: File): Promise<{ transcription: string }> {
    const base = await apiBase();
    const fd = new FormData(); fd.append('file', file);
    const r = await fetch(`${base}/transcribe`, { method: 'POST', body: fd });
    return r.json();
  },

  async analyzeImage(file: File, prompt: string): Promise<{ description: string }> {
    const base = await apiBase();
    const fd = new FormData(); fd.append('file', file); fd.append('prompt', prompt);
    const r = await fetch(`${base}/analyze_image`, { method: 'POST', body: fd });
    return r.json();
  },

  async tts(text: string): Promise<Blob> {
    const base = await apiBase();
    const r = await fetch(`${base}/tts?text=${encodeURIComponent(text)}`);
    return r.blob();
  },

  async getStats(): Promise<Stats>               { return get('/stats'); },
  async listSessions(): Promise<{ sessions: string[] }> { return get('/sessions'); },

  async deleteSession(sessionId: string): Promise<void> {
    const base = await apiBase();
    await fetch(`${base}/session/${sessionId}`, { method: 'DELETE' });
  },
};
