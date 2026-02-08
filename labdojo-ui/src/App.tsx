import { useState } from 'react';
import Sidebar from '@/components/Sidebar';
import ChatPage from '@/pages/ChatPage';
import PapersPage from '@/pages/PapersPage';
import PipelinesPage from '@/pages/PipelinesPage';
import ProjectsPage from '@/pages/ProjectsPage';
import ApisPage from '@/pages/ApisPage';
import SettingsPage from '@/pages/SettingsPage';

const pages: Record<string, React.ComponentType> = {
  chat: ChatPage,
  papers: PapersPage,
  pipelines: PipelinesPage,
  projects: ProjectsPage,
  apis: ApisPage,
  settings: SettingsPage,
};

export default function App() {
  const [activePage, setActivePage] = useState('chat');
  const PageComponent = pages[activePage] || ChatPage;

  return (
    <div className="flex h-screen">
      <Sidebar activePage={activePage} onNavigate={setActivePage} />
      <main className="flex-1 overflow-hidden">
        <PageComponent />
      </main>
    </div>
  );
}
