import { useState } from 'react';
import { Loader2, BookOpen, Dna, Pill, Route, Activity } from 'lucide-react';
import { runPipeline } from '@/lib/api';

const PIPELINE_TYPES = [
  { id: 'literature_review', label: 'Literature Review', desc: 'Search, collect, and synthesize papers', icon: BookOpen },
  { id: 'protein_analysis', label: 'Protein Analysis', desc: 'UniProt + PDB + STRING + AlphaFold', icon: Dna },
  { id: 'drug_target', label: 'Drug/Target', desc: 'ChEMBL + PubChem + FDA + Trials', icon: Pill },
  { id: 'pathway_analysis', label: 'Pathway Analysis', desc: 'Reactome + KEGG + Gene Ontology', icon: Route },
  { id: 'cancer_genomics', label: 'Cancer Genomics', desc: 'GDC + cBioPortal + Literature', icon: Activity },
];

export default function PipelinesPage() {
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [databases, setDatabases] = useState<string[]>([]);
  const [error, setError] = useState('');

  const handleRun = async () => {
    if (!query.trim() || !selectedType) return;
    setIsRunning(true);
    setError('');
    setResult(null);
    try {
      const data = await runPipeline(selectedType, query.trim());
      const r = data.result || {};
      setDatabases(r.databases_queried || []);
      setResult(r.synthesis || r.analysis || JSON.stringify(r, null, 2));
    } catch (err: any) {
      setError(err.message || 'Pipeline failed');
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="px-6 py-5 border-b border-border shrink-0">
        <h2 className="text-xl font-semibold">Pipelines</h2>
        <p className="text-[13px] text-text-secondary mt-1">Multi-step automated research workflows</p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        {/* Pipeline type cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3 mb-5">
          {PIPELINE_TYPES.map((pt) => {
            const Icon = pt.icon;
            const isActive = selectedType === pt.id;
            return (
              <button
                key={pt.id}
                onClick={() => setSelectedType(pt.id)}
                className={`bg-bg-card border rounded-lg p-4 text-center transition-colors ${
                  isActive ? 'border-accent bg-accent/5' : 'border-border hover:border-accent/50'
                }`}
              >
                <Icon className={`w-5 h-5 mx-auto mb-2 ${isActive ? 'text-accent' : 'text-text-secondary'}`} />
                <h4 className="text-sm font-semibold mb-1">{pt.label}</h4>
                <p className="text-xs text-text-secondary">{pt.desc}</p>
              </button>
            );
          })}
        </div>

        {/* Query input */}
        {selectedType && (
          <div className="flex gap-2 mb-5">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleRun()}
              placeholder={`Enter query for ${selectedType.replace('_', ' ')}...`}
              className="flex-1 bg-bg-tertiary border border-border text-text-primary px-4 py-2.5 rounded-lg text-sm outline-none focus:border-accent transition-colors"
            />
            <button
              onClick={handleRun}
              disabled={isRunning || !query.trim()}
              className="px-4 py-2.5 rounded-lg bg-accent text-white font-medium text-sm hover:bg-accent-hover disabled:opacity-50 transition-colors flex items-center gap-1.5"
            >
              {isRunning ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" /> Running...
                </>
              ) : (
                'Run Pipeline'
              )}
            </button>
          </div>
        )}

        {isRunning && (
          <div className="text-center py-8 text-text-secondary">
            <Loader2 className="w-6 h-6 animate-spin mx-auto mb-3" />
            <p className="text-sm">Running {selectedType?.replace('_', ' ')}... This may take 30-60 seconds.</p>
          </div>
        )}

        {error && <p className="text-red text-sm mb-4">{error}</p>}

        {result && (
          <div className="bg-bg-card border border-border rounded-lg p-5">
            <h3 className="text-sm font-semibold mb-1">
              Pipeline: {selectedType?.replace('_', ' ')}
            </h3>
            {databases.length > 0 && (
              <p className="text-xs text-text-secondary mb-3">Databases: {databases.join(', ')}</p>
            )}
            <div
              className="text-sm leading-relaxed text-text-secondary markdown-content"
              dangerouslySetInnerHTML={{ __html: result }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
