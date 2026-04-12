import axios from 'axios';
const API_BASE = 'http://localhost:8000';
export interface ChatResponse { session_id: string; response: string; }
export interface Stats {
  engine: string; mode: string;
  vram_used_gb: number | null; vram_total_gb: number | null;
  active_sessions: number; history_len: number;
  model_loaded: boolean; model_idle_seconds: number;
}
export const api = {
  async chat(sessionId: string | null, message: string): Promise<ChatResponse> {
    return (await axios.post<ChatResponse>(`${API_BASE}/chat`, { session_id: sessionId, message })).data;
  },
  async streamChat(sessionId: string | null, message: string, onToken: (t: string) => void): Promise<void> {
    const resp = await fetch(`${API_BASE}/chat`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
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
  async setEngine(engine: string): Promise<void> { await axios.post(`${API_BASE}/engine`, { engine }); },
  async setMode(mode: string): Promise<void> { await axios.post(`${API_BASE}/mode`, { mode }); },
  async ragQuery(query: string): Promise<{ response: string }> {
    return (await axios.post(`${API_BASE}/rag`, { query })).data;
  },
  async generateImage(prompt: string): Promise<{ image_base64: string }> {
    return (await axios.post(`${API_BASE}/generate_image`, { prompt })).data;
  },
  async transcribe(file: File): Promise<{ transcription: string }> {
    const fd = new FormData(); fd.append('file', file);
    return (await axios.post(`${API_BASE}/transcribe`, fd, { headers: { 'Content-Type': 'multipart/form-data' } })).data;
  },
  async analyzeImage(file: File, prompt: string): Promise<{ description: string }> {
    const fd = new FormData(); fd.append('file', file); fd.append('prompt', prompt);
    return (await axios.post(`${API_BASE}/analyze_image`, fd, { headers: { 'Content-Type': 'multipart/form-data' } })).data;
  },
  async tts(text: string): Promise<Blob> {
    return (await axios.get(`${API_BASE}/tts`, { params: { text }, responseType: 'blob' })).data;
  },
  async getStats(): Promise<Stats> { return (await axios.get<Stats>(`${API_BASE}/stats`)).data; },
  async listSessions(): Promise<{ sessions: string[] }> { return (await axios.get(`${API_BASE}/sessions`)).data; },
  async deleteSession(sessionId: string): Promise<void> { await axios.delete(`${API_BASE}/session/${sessionId}`); },
};
