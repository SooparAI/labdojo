import { useEffect, useState } from 'react';
import { MessageCircle, FileText, Workflow, FolderOpen, Globe, Settings } from 'lucide-react';
import { getStatus } from '@/lib/api';

interface SidebarProps {
  activePage: string;
  onNavigate: (page: string) => void;
}

const navItems = [
  { id: 'chat', label: 'Chat', icon: MessageCircle },
  { id: 'papers', label: 'Papers', icon: FileText },
  { id: 'pipelines', label: 'Pipelines', icon: Workflow },
  { id: 'projects', label: 'Projects', icon: FolderOpen },
  { id: 'apis', label: 'APIs', icon: Globe },
  { id: 'settings', label: 'Settings', icon: Settings },
];

export default function Sidebar({ activePage, onNavigate }: SidebarProps) {
  const [aiStatus, setAiStatus] = useState<{ available: boolean; label: string }>({
    available: false,
    label: 'Checking AI...',
  });

  useEffect(() => {
    const check = async () => {
      try {
        const data = await getStatus();
        const ai = data.ai_backends || {};
        if (ai.ollama?.available) {
          setAiStatus({ available: true, label: `Ollama: ${ai.ollama.model || 'connected'}` });
        } else if (ai.openai?.available) {
          setAiStatus({ available: true, label: 'OpenAI connected' });
        } else if (ai.anthropic?.available) {
          setAiStatus({ available: true, label: 'Anthropic connected' });
        } else if (ai.serverless?.available) {
          setAiStatus({ available: true, label: 'Serverless connected' });
        } else {
          setAiStatus({ available: false, label: 'No AI backend' });
        }
      } catch {
        setAiStatus({ available: false, label: 'Offline' });
      }
    };
    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <aside className="w-60 bg-bg-secondary border-r border-border flex flex-col shrink-0">
      {/* Brand */}
      <div className="px-5 py-5 border-b border-border">
        <h1 className="text-lg font-bold text-accent tracking-tight">Lab Dojo</h1>
        <span className="text-[11px] text-text-muted block mt-0.5">Pathology Research Workstation</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-2 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activePage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-[13px] font-medium transition-colors mb-0.5 ${
                isActive
                  ? 'bg-accent/10 text-accent'
                  : 'text-text-secondary hover:bg-bg-tertiary hover:text-text-primary'
              }`}
            >
              <Icon className="w-[18px] h-[18px]" />
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* Status footer */}
      <div className="p-3 border-t border-border">
        <div className="flex items-center gap-1.5 px-2 py-1">
          <span
            className={`w-2 h-2 rounded-full ${aiStatus.available ? 'bg-green' : 'bg-red'}`}
          />
          <span className="text-[11px] text-text-muted">{aiStatus.label}</span>
        </div>
        <div className="flex items-center gap-1.5 px-2 py-1">
          <span className="w-2 h-2 rounded-full bg-green" />
          <span className="text-[11px] text-text-muted">20 APIs connected</span>
        </div>
      </div>
    </aside>
  );
}
