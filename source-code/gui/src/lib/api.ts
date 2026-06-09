function getApiBase(): string {
  if (typeof window === 'undefined') return 'http://localhost:8000';
  // If served by Axum (same origin), use relative URLs via window.location
  const isEmbedded = !window.location.hostname.includes('localhost') ||
                     window.location.port !== '3000';
  if (isEmbedded) {
    // Same origin – use root-relative paths
    return window.location.origin;
  }
  // Dev mode – Next.js dev server on :3000, API on :8000
  return (process.env.NEXT_PUBLIC_API_BASE as string) ?? 'http://localhost:8000';
}

const API_BASE = typeof window !== 'undefined' ? getApiBase() : 'http://localhost:8000';

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, options);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export interface ChatResponse { session_id: string; response: string; }

export interface Stats {
  engine: string; mode: string;
  vram_used_gb: number | null; vram_total_gb: number | null;
  active_sessions: number; history_len: number;
  model_loaded: boolean; model_idle_seconds: number;
  rag_chunks?: number; auth_enabled?: boolean;
}

export const api = {
  async chat(sessionId: string | null, message: string): Promise<ChatResponse> {
    return apiFetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message }),
    });
  },

  async streamChat(
    sessionId: string | null,
    message: string,
    onToken: (t: string) => void,
  ): Promise<void> {
    const resp = await fetch(`${API_BASE}/chat`, {
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

  async setEngine(engine: string): Promise<void> {
    await apiFetch('/engine', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ engine }) });
  },
  async setMode(mode: string): Promise<void> {
    await apiFetch('/mode', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode }) });
  },
  async ragQuery(query: string): Promise<{ response: string }> {
    return apiFetch('/rag', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query }) });
  },
  async generateImage(prompt: string): Promise<{ image_base64: string }> {
    return apiFetch('/generate_image', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt }) });
  },
  async transcribe(file: File): Promise<{ transcription: string }> {
    const fd = new FormData(); fd.append('file', file);
    return apiFetch('/transcribe', { method: 'POST', body: fd });
  },
  async analyzeImage(file: File, prompt: string): Promise<{ description: string }> {
    const fd = new FormData(); fd.append('file', file); fd.append('prompt', prompt);
    return apiFetch('/analyze_image', { method: 'POST', body: fd });
  },
  async tts(text: string): Promise<Blob> {
    const r = await fetch(`${API_BASE}/tts?text=${encodeURIComponent(text)}`);
    return r.blob();
  },
  async getStats(): Promise<Stats>                       { return apiFetch('/stats'); },
  async listSessions(): Promise<{ sessions: string[] }> { return apiFetch('/sessions'); },
  async deleteSession(id: string): Promise<void>        { await fetch(`${API_BASE}/session/${id}`, { method: 'DELETE' }); },
};
