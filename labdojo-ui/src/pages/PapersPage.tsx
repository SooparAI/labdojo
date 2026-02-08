import { useState } from 'react';
import { Search, Loader2, ExternalLink, Download } from 'lucide-react';
import { searchPapers, getExportUrl, type Paper } from '@/lib/api';

export default function PapersPage() {
  const [query, setQuery] = useState('');
  const [papers, setPapers] = useState<Paper[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setIsLoading(true);
    setError('');
    setHasSearched(true);
    try {
      const data = await searchPapers(query.trim());
      setPapers(data.papers);
    } catch {
      setError('Search failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const pmids = papers.map((p) => p.pmid);

  return (
    <div className="flex flex-col h-full">
      <div className="px-6 py-5 border-b border-border shrink-0">
        <h2 className="text-xl font-semibold">Papers</h2>
        <p className="text-[13px] text-text-secondary mt-1">Search PubMed with citation verification</p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        {/* Search bar */}
        <div className="flex gap-2 mb-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Search PubMed..."
              className="w-full pl-9 pr-3 py-2.5 bg-bg-tertiary border border-border text-text-primary rounded-lg text-sm outline-none focus:border-accent transition-colors"
            />
          </div>
          <button
            onClick={handleSearch}
            disabled={isLoading}
            className="px-4 py-2.5 rounded-lg bg-accent text-white font-medium text-sm hover:bg-accent-hover disabled:opacity-50 transition-colors"
          >
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Search'}
          </button>
        </div>

        {/* Export buttons */}
        {papers.length > 0 && (
          <div className="flex gap-2 mb-4">
            {['bibtex', 'ris', 'markdown'].map((fmt) => (
              <a
                key={fmt}
                href={getExportUrl(fmt, pmids)}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-bg-card border border-border text-text-primary text-xs font-medium hover:bg-bg-tertiary transition-colors"
              >
                <Download className="w-3 h-3" />
                {fmt.charAt(0).toUpperCase() + fmt.slice(1)}
              </a>
            ))}
          </div>
        )}

        {/* Results */}
        {error && <p className="text-red text-sm">{error}</p>}

        {!hasSearched && !isLoading && (
          <div className="text-center py-16 text-text-secondary">
            <p className="text-sm">Search PubMed to find papers. Results include PMID, DOI, and citation export.</p>
          </div>
        )}

        {hasSearched && !isLoading && papers.length === 0 && (
          <p className="text-text-muted text-sm">No results found.</p>
        )}

        <div className="space-y-2.5">
          {papers.map((paper) => {
            const authors = (paper.authors || []).slice(0, 3).join(', ') + (paper.authors?.length > 3 ? ' et al.' : '');
            return (
              <div key={paper.pmid} className="bg-bg-card border border-border rounded-lg p-4">
                <h3 className="text-sm font-semibold leading-snug mb-1.5">{paper.title || 'Untitled'}</h3>
                <p className="text-xs text-text-secondary">
                  PMID: {paper.pmid} | {authors} | {paper.journal || ''} ({paper.pub_date || ''})
                </p>
                <div className="flex gap-3 mt-2">
                  {paper.doi_url && (
                    <a href={paper.doi_url} target="_blank" rel="noopener noreferrer" className="text-accent text-xs flex items-center gap-1 hover:underline">
                      <ExternalLink className="w-3 h-3" /> DOI
                    </a>
                  )}
                  <a
                    href={`https://pubmed.ncbi.nlm.nih.gov/${paper.pmid}/`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-accent text-xs flex items-center gap-1 hover:underline"
                  >
                    <ExternalLink className="w-3 h-3" /> PubMed
                  </a>
                </div>
                {paper.abstract && (
                  <p className="text-[13px] text-text-secondary mt-2 leading-relaxed">
                    {paper.abstract.substring(0, 300)}...
                  </p>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
