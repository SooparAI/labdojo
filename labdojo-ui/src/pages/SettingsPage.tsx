import { useState, useEffect } from 'react';
import { Loader2, Save, Trash2, Download, AlertTriangle } from 'lucide-react';
import {
  getSettings,
  saveSettings as saveSettingsApi,
  clearConversation,
  clearBadData,
  resetLearning,
  getConversationExportUrl,
} from '@/lib/api';

export default function SettingsPage() {
  const [ollamaHost, setOllamaHost] = useState('http://localhost:11434');
  const [openaiKey, setOpenaiKey] = useState('');
  const [anthropicKey, setAnthropicKey] = useState('');
  const [vastaiKey, setVastaiKey] = useState('');
  const [endpoint, setEndpoint] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [openaiConfigured, setOpenaiConfigured] = useState(false);
  const [anthropicConfigured, setAnthropicConfigured] = useState(false);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await getSettings();
        setOllamaHost(data.ollama_host || 'http://localhost:11434');
        setOpenaiConfigured(data.openai_api_key === 'configured');
        setAnthropicConfigured(data.anthropic_api_key === 'configured');
        setEndpoint(data.serverless_endpoint_id || '');
      } catch {
        // silently fail
      } finally {
        setIsLoading(false);
      }
    };
    load();
  }, []);

  const handleSave = async () => {
    setIsSaving(true);
    const updates: Record<string, string | number> = { ollama_host: ollamaHost };
    if (openaiKey.trim()) updates.openai_api_key = openaiKey.trim();
    if (anthropicKey.trim()) updates.anthropic_api_key = anthropicKey.trim();
    if (vastaiKey.trim()) updates.vastai_api_key = vastaiKey.trim();
    if (endpoint.trim()) updates.serverless_endpoint_id = parseInt(endpoint) || 0;
    await saveSettingsApi(updates as any);
    setIsSaving(false);
    alert('Settings saved.');
  };

  const handleClearChat = async () => {
    if (!confirm('Clear all chat history?')) return;
    await clearConversation();
    alert('Chat history cleared.');
  };

  const handleClearBad = async () => {
    await clearBadData();
    alert('Bad data cleaned.');
  };

  const handleReset = async () => {
    if (!confirm('This will delete ALL learned data. Continue?')) return;
    await resetLearning();
    alert('Learning data reset.');
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="w-5 h-5 animate-spin text-text-muted" />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-6 py-5 border-b border-border shrink-0">
        <h2 className="text-xl font-semibold">Settings</h2>
        <p className="text-[13px] text-text-secondary mt-1">Configure AI backends and manage data</p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        {/* AI Backends */}
        <div className="mb-8">
          <h3 className="text-[15px] font-semibold mb-3 pb-2 border-b border-border">AI Backends</h3>
          <div className="space-y-3">
            <SettingRow label="Ollama Host" value={ollamaHost} onChange={setOllamaHost} />
            <SettingRow
              label="OpenAI API Key"
              value={openaiKey}
              onChange={setOpenaiKey}
              type="password"
              placeholder={openaiConfigured ? 'Configured (enter new to change)' : 'sk-...'}
            />
            <SettingRow
              label="Anthropic API Key"
              value={anthropicKey}
              onChange={setAnthropicKey}
              type="password"
              placeholder={anthropicConfigured ? 'Configured (enter new to change)' : 'sk-ant-...'}
            />
            <SettingRow label="Vast.ai API Key" value={vastaiKey} onChange={setVastaiKey} type="password" placeholder="Optional" />
            <SettingRow label="Serverless Endpoint" value={endpoint} onChange={setEndpoint} placeholder="Endpoint ID" />
          </div>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="mt-4 flex items-center gap-1.5 px-3 py-2 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent-hover disabled:opacity-50 transition-colors"
          >
            {isSaving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
            Save
          </button>
        </div>

        {/* Data Management */}
        <div>
          <h3 className="text-[15px] font-semibold mb-3 pb-2 border-b border-border">Data Management</h3>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={handleClearChat}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-bg-card border border-border text-text-primary text-xs font-medium hover:bg-bg-tertiary transition-colors"
            >
              <Trash2 className="w-3 h-3" /> Clear Chat History
            </button>
            <button
              onClick={handleClearBad}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-bg-card border border-border text-text-primary text-xs font-medium hover:bg-bg-tertiary transition-colors"
            >
              <Trash2 className="w-3 h-3" /> Clean Bad Data
            </button>
            <button
              onClick={handleReset}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-red text-white text-xs font-medium hover:bg-red/90 transition-colors"
            >
              <AlertTriangle className="w-3 h-3" /> Reset All Learning
            </button>
            <a
              href={getConversationExportUrl()}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-bg-card border border-border text-text-primary text-xs font-medium hover:bg-bg-tertiary transition-colors"
            >
              <Download className="w-3 h-3" /> Export Conversation
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}

function SettingRow({
  label,
  value,
  onChange,
  type = 'text',
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
  placeholder?: string;
}) {
  return (
    <div className="flex items-center gap-3">
      <label className="w-40 text-[13px] text-text-secondary shrink-0">{label}:</label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="flex-1 bg-bg-tertiary border border-border text-text-primary px-3 py-2 rounded-md text-[13px] outline-none focus:border-accent transition-colors"
      />
    </div>
  );
}
