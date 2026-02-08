/**
 * API client for Lab Dojo FastAPI backend.
 * All endpoints proxy through Vite dev server to http://localhost:8080.
 */

const BASE = '';

export interface ChatResponse {
  response: string;
  source: string;
  grounding: string[];
  latency?: number;
}

export interface Paper {
  pmid: string;
  title: string;
  authors: string[];
  journal: string;
  pub_date: string;
  abstract?: string;
  doi_url?: string;
}

export interface PaperSearchResult {
  papers: Paper[];
  total: number;
}

export interface PipelineResult {
  result: {
    synthesis?: string;
    analysis?: string;
    databases_queried?: string[];
    [key: string]: unknown;
  };
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  status: string;
  created_at?: string;
}

export interface ApiInfo {
  name: string;
  description: string;
  rate_limit: string;
  category: string;
}

export interface StatusResponse {
  version: string;
  ai_backends: {
    ollama?: { available: boolean; model?: string };
    openai?: { available: boolean };
    anthropic?: { available: boolean };
    serverless?: { available: boolean };
  };
  apis_connected: number;
}

export interface Settings {
  ollama_host: string;
  openai_api_key: string;
  anthropic_api_key: string;
  vastai_api_key: string;
  serverless_endpoint_id: string;
}

// ─── Chat ────────────────────────────────────────────────────────────────────

export async function sendChatMessage(
  message: string,
  verbosity: string = 'detailed',
  deterministic: boolean = false
): Promise<ChatResponse> {
  const resp = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, verbosity, deterministic }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(err.detail || 'Unknown error');
  }
  return resp.json();
}

// ─── Papers ──────────────────────────────────────────────────────────────────

export async function searchPapers(query: string, maxResults: number = 10): Promise<PaperSearchResult> {
  const resp = await fetch(`${BASE}/papers/search?query=${encodeURIComponent(query)}&max_results=${maxResults}`);
  if (!resp.ok) throw new Error('Paper search failed');
  return resp.json();
}

export function getExportUrl(format: string, pmids: string[]): string {
  return `${BASE}/export/${format}?pmids=${pmids.join(',')}`;
}

// ─── Pipelines ───────────────────────────────────────────────────────────────

export async function runPipeline(pipelineType: string, query: string): Promise<PipelineResult> {
  const resp = await fetch(`${BASE}/pipeline/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pipeline_type: pipelineType, query }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: 'Pipeline failed' }));
    throw new Error(err.detail || 'Pipeline error');
  }
  return resp.json();
}

// ─── Projects ────────────────────────────────────────────────────────────────

export async function getProjects(): Promise<{ projects: Project[] }> {
  const resp = await fetch(`${BASE}/projects`);
  if (!resp.ok) throw new Error('Failed to load projects');
  return resp.json();
}

export async function createProject(name: string, description: string, keyTerms: string): Promise<void> {
  await fetch(`${BASE}/projects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description, key_terms: keyTerms }),
  });
}

export async function deleteProject(id: string): Promise<void> {
  await fetch(`${BASE}/projects/${id}`, { method: 'DELETE' });
}

// ─── APIs ────────────────────────────────────────────────────────────────────

export async function getApis(): Promise<{ apis: Record<string, ApiInfo> }> {
  const resp = await fetch(`${BASE}/apis`);
  if (!resp.ok) throw new Error('Failed to load APIs');
  return resp.json();
}

// ─── Status ──────────────────────────────────────────────────────────────────

export async function getStatus(): Promise<StatusResponse> {
  const resp = await fetch(`${BASE}/status`);
  if (!resp.ok) throw new Error('Failed to get status');
  return resp.json();
}

// ─── Settings ────────────────────────────────────────────────────────────────

export async function getSettings(): Promise<Settings> {
  const resp = await fetch(`${BASE}/settings`);
  if (!resp.ok) throw new Error('Failed to load settings');
  return resp.json();
}

export async function saveSettings(updates: Partial<Settings>): Promise<void> {
  await fetch(`${BASE}/settings/update`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  });
}

// ─── Data Management ─────────────────────────────────────────────────────────

export async function clearConversation(): Promise<void> {
  await fetch(`${BASE}/conversation/clear`, { method: 'POST' });
}

export async function clearBadData(): Promise<void> {
  await fetch(`${BASE}/learning/clear_bad`, { method: 'POST' });
}

export async function resetLearning(): Promise<void> {
  await fetch(`${BASE}/learning/reset`, { method: 'POST' });
}

export function getConversationExportUrl(): string {
  return `${BASE}/export/conversation`;
}
