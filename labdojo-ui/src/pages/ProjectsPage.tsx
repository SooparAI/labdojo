import { useState, useEffect, useCallback } from 'react';
import { Plus, Trash2, Loader2 } from 'lucide-react';
import { getProjects, createProject, deleteProject, type Project } from '@/lib/api';

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [keyTerms, setKeyTerms] = useState('');

  const loadProjects = useCallback(async () => {
    try {
      const data = await getProjects();
      setProjects(data.projects);
    } catch {
      // silently fail
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadProjects();
  }, [loadProjects]);

  const handleCreate = async () => {
    if (!name.trim()) return;
    await createProject(name.trim(), description, keyTerms);
    setName('');
    setDescription('');
    setKeyTerms('');
    setShowForm(false);
    loadProjects();
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this project and all its data?')) return;
    await deleteProject(id);
    loadProjects();
  };

  return (
    <div className="flex flex-col h-full">
      <div className="px-6 py-5 border-b border-border shrink-0">
        <h2 className="text-xl font-semibold">Projects</h2>
        <p className="text-[13px] text-text-secondary mt-1">Persistent research context and decision logs</p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        <button
          onClick={() => setShowForm(!showForm)}
          className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent-hover transition-colors mb-4"
        >
          <Plus className="w-4 h-4" /> New Project
        </button>

        {showForm && (
          <div className="bg-bg-card border border-border rounded-lg p-4 mb-4 space-y-3">
            <div className="flex items-center gap-3">
              <label className="w-28 text-[13px] text-text-secondary shrink-0">Name:</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="flex-1 bg-bg-tertiary border border-border text-text-primary px-3 py-2 rounded-md text-[13px] outline-none focus:border-accent"
              />
            </div>
            <div className="flex items-center gap-3">
              <label className="w-28 text-[13px] text-text-secondary shrink-0">Description:</label>
              <input
                type="text"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="flex-1 bg-bg-tertiary border border-border text-text-primary px-3 py-2 rounded-md text-[13px] outline-none focus:border-accent"
              />
            </div>
            <div className="flex items-center gap-3">
              <label className="w-28 text-[13px] text-text-secondary shrink-0">Key Terms:</label>
              <input
                type="text"
                value={keyTerms}
                onChange={(e) => setKeyTerms(e.target.value)}
                placeholder="comma-separated"
                className="flex-1 bg-bg-tertiary border border-border text-text-primary px-3 py-2 rounded-md text-[13px] outline-none focus:border-accent"
              />
            </div>
            <button
              onClick={handleCreate}
              className="px-3 py-1.5 rounded-lg bg-accent text-white text-xs font-medium hover:bg-accent-hover transition-colors"
            >
              Create
            </button>
          </div>
        )}

        {isLoading && (
          <div className="text-center py-8">
            <Loader2 className="w-5 h-5 animate-spin mx-auto text-text-muted" />
          </div>
        )}

        {!isLoading && projects.length === 0 && (
          <p className="text-text-muted text-sm">No projects yet. Create one to start tracking research context.</p>
        )}

        <div className="space-y-2.5">
          {projects.map((project) => (
            <div key={project.id} className="bg-bg-card border border-border rounded-lg p-4 hover:border-accent/50 transition-colors">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-[15px] font-semibold">{project.name}</h3>
                  <p className="text-[13px] text-text-secondary mt-1">{project.description || 'No description'}</p>
                  <p className="text-[11px] text-text-muted mt-1.5">
                    Status: {project.status} | Created: {project.created_at || ''}
                  </p>
                </div>
                <button
                  onClick={() => handleDelete(project.id)}
                  className="p-1.5 rounded-md text-text-muted hover:text-red hover:bg-red/10 transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
