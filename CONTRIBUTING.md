# Contributing to Lab Dojo

Lab Dojo is an open-source AI research agent built for scientists. We welcome contributions from researchers, developers, and anyone who wants to make AI more useful for science.

## How to Contribute

### Reporting Issues

- Use GitHub Issues to report bugs or request features
- Include your OS, Python version, and Ollama model (if applicable)
- For bugs, include the full error traceback from the terminal

### Adding Science APIs

Lab Dojo connects to 7 science databases. To add a new one:

1. Add a new async method to the `ScienceAPIs` class
2. Add the API to the `API_CATALOG` dictionary
3. Add keyword detection in the chat endpoint's `is_science_question` logic
4. Add a UI section in the Papers or Dashboard page
5. Test with the E2E test suite

### Adding AI Backends

To add a new LLM provider:

1. Create a new client class following the pattern of `OllamaClient` or `ChatGPTClient`
2. Add it to the fallback chain in the chat endpoint
3. Add configuration fields to the `Config` dataclass
4. Add UI controls in the Settings page

### Code Style

- Python 3.11+
- Type hints on all function signatures
- Docstrings on public methods
- Thread-safe database operations (use `self._lock` for writes)
- All science API calls must use `aiohttp` for async

### Testing

Run the E2E test suite before submitting:

```bash
python3.11 labdojo.py &
sleep 5
python3.11 tests/e2e_test.py
```

All 39 tests must pass.

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-api`)
3. Make your changes
4. Run the E2E tests
5. Submit a PR with a clear description of what changed and why

## Architecture

Lab Dojo is a single-file Python application (~3,800 lines) with:

- **KnowledgeBase** — SQLite with 14 tables, WAL mode, thread-safe persistent connection
- **ScienceAPIs** — Async HTTP clients for 7 science databases with in-memory caching
- **AI Clients** — Ollama, Vast.ai Serverless, OpenAI, Anthropic with automatic fallback
- **FastAPI** — 30+ REST endpoints for chat, projects, pipelines, export, monitoring
- **Dashboard** — Single-page HTML/CSS/JS app embedded in the Python file

## License

MIT — see [LICENSE](LICENSE)
