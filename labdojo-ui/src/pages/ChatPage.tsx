import { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { sendChatMessage, type ChatResponse } from '@/lib/api';

interface Message {
  role: 'user' | 'assistant';
  text: string;
  meta?: string;
}

const EXAMPLES = [
  'What is the role of BRCA1 in DNA damage repair?',
  'Compare PD-1 and PD-L1 inhibitors in melanoma treatment',
  'What are the latest findings on TP53 mutations in colorectal cancer?',
];

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [verbosity, setVerbosity] = useState('detailed');
  const [deterministic, setDeterministic] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (text?: string) => {
    const msg = (text || input).trim();
    if (!msg || isLoading) return;
    setInput('');

    const userMsg: Message = { role: 'user', text: msg, meta: new Date().toLocaleTimeString() };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const data: ChatResponse = await sendChatMessage(msg, verbosity, deterministic);
      let meta = `via ${data.source}`;
      if (data.grounding?.length) meta += ` | grounded in: ${data.grounding.join(', ')}`;
      if (data.latency) meta += ` | ${data.latency}s`;
      setMessages((prev) => [...prev, { role: 'assistant', text: data.response, meta }]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: `Error: ${err.message || 'Connection failed. Is the server running?'}` },
      ]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const renderMarkdown = (text: string) => {
    // Simple markdown rendering — bold, code, links, newlines
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/`(.*?)`/g, '<code class="bg-bg-tertiary px-1 py-0.5 rounded text-[13px]">$1</code>')
      .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" class="text-accent hover:underline">$1</a>')
      .replace(/\n/g, '<br/>');
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-5 border-b border-border shrink-0">
        <h2 className="text-xl font-semibold">Chat</h2>
        <p className="text-[13px] text-text-secondary mt-1">Ask research questions grounded in 20 biomedical databases</p>
      </div>

      {/* Controls */}
      <div className="px-6 py-3 border-b border-border flex gap-3 items-center flex-wrap shrink-0">
        <label className="text-xs text-text-secondary">Verbosity:</label>
        <select
          value={verbosity}
          onChange={(e) => setVerbosity(e.target.value)}
          className="text-xs bg-bg-tertiary border border-border text-text-primary px-2 py-1 rounded"
        >
          <option value="concise">Concise</option>
          <option value="detailed">Detailed</option>
          <option value="comprehensive">Comprehensive</option>
        </select>
        <label className="flex items-center gap-1.5 text-xs text-text-secondary cursor-pointer">
          <input
            type="checkbox"
            checked={deterministic}
            onChange={(e) => setDeterministic(e.target.checked)}
            className="accent-accent"
          />
          Deterministic
        </label>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {messages.length === 0 && (
          <div className="text-center py-16 text-text-secondary">
            <h3 className="text-lg text-text-primary mb-3">Welcome to Lab Dojo</h3>
            <p className="text-sm max-w-md mx-auto leading-relaxed mb-6">
              Your AI research assistant with access to 20 biomedical databases. Ask any research question to get started.
            </p>
            <div className="inline-flex flex-col gap-2 text-left">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => handleSend(ex)}
                  className="bg-bg-card border border-border rounded-lg px-4 py-2.5 text-[13px] text-text-secondary hover:border-accent hover:text-accent transition-colors text-left"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`mb-4 max-w-[85%] ${msg.role === 'user' ? 'ml-auto' : 'mr-auto'}`}>
            <div
              className={`px-4 py-3 rounded-xl text-sm leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-accent text-white rounded-br-sm'
                  : 'bg-bg-card border border-border rounded-bl-sm markdown-content'
              }`}
              dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.text) }}
            />
            {msg.meta && (
              <p className={`text-[11px] text-text-muted mt-1 px-1 ${msg.role === 'user' ? 'text-right' : ''}`}>
                {msg.meta}
              </p>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="mb-4 max-w-[85%] mr-auto">
            <div className="bg-bg-card border border-border rounded-xl rounded-bl-sm px-4 py-3">
              <div className="flex gap-1">
                <span className="w-1.5 h-1.5 bg-text-muted rounded-full animate-bounce" />
                <span className="w-1.5 h-1.5 bg-text-muted rounded-full animate-bounce [animation-delay:0.2s]" />
                <span className="w-1.5 h-1.5 bg-text-muted rounded-full animate-bounce [animation-delay:0.4s]" />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-6 py-4 border-t border-border shrink-0">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask a research question..."
            className="flex-1 bg-bg-tertiary border border-border text-text-primary px-4 py-3 rounded-lg text-sm outline-none focus:border-accent transition-colors"
          />
          <button
            onClick={() => handleSend()}
            disabled={isLoading || !input.trim()}
            className="px-5 py-3 rounded-lg bg-accent text-white font-medium text-sm hover:bg-accent-hover disabled:opacity-50 transition-colors flex items-center gap-1.5"
          >
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </div>
  );
}
