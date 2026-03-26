import { useState, useEffect, useRef } from 'react';
import { api, Stats } from '@/lib/api';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import vscDarkPlus from 'react-syntax-highlighter/dist/esm/styles/prism/vsc-dark-plus';

type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
};

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<Stats | null>(null);
  const [engine, setEngine] = useState('transformers');
  const [mode, setMode] = useState('general');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Load initial session
  useEffect(() => {
    const storedSession = localStorage.getItem('hexai_session');
    if (storedSession) setSessionId(storedSession);
  }, []);

    useEffect(() => {
      if (sessionId) localStorage.setItem('hexai_session', sessionId);
    }, [sessionId]);

      useEffect(() => {
        const fetchStats = async () => {
          try {
            const s = await api.getStats();
            setStats(s);
            setEngine(s.engine);
            setMode(s.mode);
          } catch (err) {
            console.error('Failed to fetch stats', err);
          }
        };
        fetchStats();
        const interval = setInterval(fetchStats, 5000);
        return () => clearInterval(interval);
      }, []);

      const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      };

      useEffect(() => {
        scrollToBottom();
      }, [messages]);

      const handleSend = async () => {
        if (!input.trim() || loading) return;
        const userMessage: Message = { id: Date.now().toString(), role: 'user', content: input };
        setMessages((prev) => [...prev, userMessage]);
        setInput('');
        setLoading(true);

        try {
          if (true) { // streaming enabled
            let assistantContent = '';
            const assistantMessage: Message = { id: (Date.now() + 1).toString(), role: 'assistant', content: '' };
            setMessages((prev) => [...prev, assistantMessage]);

            await api.streamChat(sessionId, input, (token) => {
              assistantContent += token;
              setMessages((prev) =>
              prev.map((msg) =>
              msg.id === assistantMessage.id ? { ...msg, content: assistantContent } : msg
              )
              );
            });
          } else {
            const response = await api.chat(sessionId, input);
            if (!sessionId) setSessionId(response.session_id);
            const assistantMessage: Message = { id: Date.now().toString(), role: 'assistant', content: response.response };
            setMessages((prev) => [...prev, assistantMessage]);
          }
        } catch (err) {
          console.error(err);
          setMessages((prev) => [
            ...prev,
            { id: Date.now().toString(), role: 'assistant', content: '❌ Błąd połączenia z serwerem.' },
          ]);
        } finally {
          setLoading(false);
        }
      };

      const handleEngineChange = async (newEngine: string) => {
        try {
          await api.setEngine(newEngine);
          setEngine(newEngine);
        } catch (err) {
          console.error(err);
        }
      };

      const handleModeChange = async (newMode: string) => {
        try {
          await api.setMode(newMode);
          setMode(newMode);
        } catch (err) {
          console.error(err);
        }
      };

      const handleNewSession = () => {
        setSessionId(null);
        setMessages([]);
        localStorage.removeItem('hexai_session');
      };

      const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        // Example: transcribe if audio, analyze if image
        if (file.type.startsWith('audio/')) {
          try {
            const { transcription } = await api.transcribe(file);
            setInput(transcription);
          } catch (err) {
            console.error(err);
          }
        } else if (file.type.startsWith('image/')) {
          // Optionally analyze image
          const { description } = await api.analyzeImage(file, 'Describe this image in detail.');
          setMessages((prev) => [
            ...prev,
            { id: Date.now().toString(), role: 'assistant', content: `🖼️ Opis obrazu: ${description}` },
          ]);
        }
      };

      return (
        <div className="flex h-screen bg-gray-100">
        {/* Sidebar */}
        <div className="w-64 bg-white shadow-md flex flex-col p-4">
        <h1 className="text-xl font-bold mb-4">HexAi</h1>
        <button
        onClick={handleNewSession}
        className="bg-blue-500 text-white py-2 px-4 rounded mb-4 hover:bg-blue-600"
        >
        Nowa rozmowa
        </button>
        <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Silnik</label>
        <select
        value={engine}
        onChange={(e) => handleEngineChange(e.target.value)}
        className="w-full border rounded p-2"
        >
        <option value="transformers">Transformers (GPU)</option>
        <option value="ollama">Ollama (CPU)</option>
        </select>
        </div>
        <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Tryb</label>
        <select
        value={mode}
        onChange={(e) => handleModeChange(e.target.value)}
        className="w-full border rounded p-2"
        >
        <option value="general">Ogólny</option>
        <option value="programista">Programista</option>
        </select>
        </div>
        <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Plik</label>
        <input type="file" accept="audio/*,image/*" onChange={handleFileUpload} />
        </div>
        <div className="mt-auto text-xs text-gray-500">
        {stats && (
          <>
          <p>VRAM: {stats.vram_used_gb?.toFixed(2)} / {stats.vram_total_gb?.toFixed(2)} GB</p>
          <p>Sesje: {stats.active_sessions}</p>
          <p>Model: {stats.model_loaded ? 'załadowany' : 'nieaktywny'}</p>
          </>
        )}
        </div>
        </div>

        {/* Chat area */}
        <div className="flex-1 flex flex-col">
        <div className="flex-1 overflow-y-auto p-4">
        {messages.map((msg) => (
          <div
          key={msg.id}
          className={`mb-4 flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
          <div
          className={`max-w-3xl rounded-lg p-3 ${
            msg.role === 'user'
            ? 'bg-blue-500 text-white'
            : 'bg-white border shadow-sm'
          }`}
          >
          <ReactMarkdown
          components={{
            code({ node, inline, className, children, ...props }: any) {
              const match = /language-(\w+)/.exec(className || '');
              return !inline && match ? (
                <SyntaxHighlighter
                style={vscDarkPlus}
                language={match[1]}
                PreTag="div"
                {...props}
                >
                {String(children).replace(/\n$/, '')}
                </SyntaxHighlighter>
              ) : (
                <code className={className} {...props}>
                {children}
                </code>
              );
            },
          }}
          >
          {msg.content}
          </ReactMarkdown>
          </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
          <div className="bg-white border shadow-sm rounded-lg p-3">
          <div className="animate-pulse">...</div>
          </div>
          </div>
        )}
        <div ref={messagesEndRef} />
        </div>
        <div className="border-t p-4 bg-white">
        <div className="flex">
        <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
        placeholder="Wpisz wiadomość..."
        className="flex-1 border rounded-l-lg p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
        onClick={handleSend}
        disabled={loading}
        className="bg-blue-500 text-white px-4 rounded-r-lg hover:bg-blue-600 disabled:opacity-50"
        >
        Wyślij
        </button>
        </div>
        </div>
        </div>
        </div>
      );
}
