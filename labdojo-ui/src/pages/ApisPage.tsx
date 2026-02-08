import { useState, useEffect } from 'react';
import { Loader2 } from 'lucide-react';
import { getApis, type ApiInfo } from '@/lib/api';

export default function ApisPage() {
  const [apis, setApis] = useState<Record<string, ApiInfo>>({});
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await getApis();
        setApis(data.apis);
      } catch {
        // silently fail
      } finally {
        setIsLoading(false);
      }
    };
    load();
  }, []);

  return (
    <div className="flex flex-col h-full">
      <div className="px-6 py-5 border-b border-border shrink-0">
        <h2 className="text-xl font-semibold">Connected APIs</h2>
        <p className="text-[13px] text-text-secondary mt-1">20 biomedical databases, all free, no keys required</p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        {isLoading ? (
          <div className="text-center py-8">
            <Loader2 className="w-5 h-5 animate-spin mx-auto text-text-muted" />
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2.5">
            {Object.entries(apis).map(([id, api]) => (
              <div key={id} className="bg-bg-card border border-border rounded-lg p-3">
                <h4 className="text-[13px] font-semibold mb-1">{api.name}</h4>
                <p className="text-[11px] text-text-secondary mb-2">{api.description}</p>
                <span className="inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold bg-green/15 text-green">
                  {api.rate_limit}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
