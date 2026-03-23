import axios from 'axios';

const API_BASE = 'http://localhost:8000'; // Adres backendu

export interface ChatResponse {
  session_id: string;
  response: string;
}

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

export interface Session {
  session_id: string;
}

export const api = {
  async chat(sessionId: string | null, message: string): Promise<ChatResponse> {
    const response = await axios.post<ChatResponse>(`${API_BASE}/chat`, {
      session_id: sessionId,
      message,
    });
    return response.data;
  },

  async streamChat(sessionId: string | null, message: string, onToken: (token: string) => void): Promise<void> {
    const response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message, stream: true }),
    });
    const reader = response.body?.getReader();
    if (!reader) return;
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      onToken(chunk);
    }
  },

  async setEngine(engine: string): Promise<void> {
    await axios.post(`${API_BASE}/engine`, { engine });
  },

  async setMode(mode: string): Promise<void> {
    await axios.post(`${API_BASE}/mode`, { mode });
  },

  async ragQuery(query: string): Promise<{ response: string }> {
    const response = await axios.post(`${API_BASE}/rag`, { query });
    return response.data;
  },

  async ingestDocument(source: string, chunkSize = 500, chunkOverlap = 100): Promise<{ message: string }> {
    const response = await axios.post(`${API_BASE}/ingest`, { source, chunk_size: chunkSize, chunk_overlap: chunkOverlap });
    return response.data;
  },

  async generateImage(prompt: string): Promise<{ image_base64: string }> {
    const response = await axios.post(`${API_BASE}/generate_image`, { prompt });
    return response.data;
  },

  async transcribe(file: File): Promise<{ transcription: string }> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${API_BASE}/transcribe`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  async analyzeImage(file: File, prompt: string): Promise<{ description: string }> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('prompt', prompt);
    const response = await axios.post(`${API_BASE}/analyze_image`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  async tts(text: string): Promise<Blob> {
    const response = await axios.get(`${API_BASE}/tts`, {
      params: { text },
      responseType: 'blob',
    });
    return response.data;
  },

  async getStats(): Promise<Stats> {
    const response = await axios.get<Stats>(`${API_BASE}/stats`);
    return response.data;
  },

  async listSessions(): Promise<{ sessions: string[] }> {
    const response = await axios.get(`${API_BASE}/sessions`);
    return response.data;
  },

  async deleteSession(sessionId: string): Promise<void> {
    await axios.delete(`${API_BASE}/session/${sessionId}`);
  },
};
