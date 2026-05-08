# Internal Features Log

## ImpactArbiter CLI MVP (Initial Release)
* **AST Extractor (`ast_extractor.py`)**: Safely parses generated PyTorch code using Python's `ast` module. Extracts class definitions, `forward()` passes, and tensor operations into a structured `TorchMethod` dataclass.
* **Paper Parser (`paper_parser.py`)**: Uses a multimodal LLM API stub to convert academic PDFs directly into executable Pytest validation suites testing mathematical claims.
* **Sandbox Executor (`sandbox_executor.py`)**: securely executes code inside a Docker container using the `docker` SDK. Rejects invalid code and outputs exact tracebacks and execution time securely.
* **CLI Entrypoint (`cli.py`)**: Provides a `click` based interface to tie the validation pipeline together and export deterministic JSON reports.

## Remote Fetching & Correction Artifact Upgrade
* **Remote Inputs (`utils.py`)**: Added `download_arxiv_pdf` to pull papers directly from the ArXiv PDF endpoint and `clone_github_repo` to pull generated implementations directly from GitHub.
* **Correction Artifact Generation**: Upgraded `sandbox_executor.py` to parse tracebacks and generate a highly specific `llm_feedback_payload` designed to be pasted back into the LLM context window.
* **CLI Orchestration (`cli.py`)**: Updated commands to accept `--arxiv-url` and `--repo-url`. The CLI automatically parses all Python files in the target repo and displays the LLM feedback payload within explicit block delimiters (`========== COPY THIS FOR LLM ==========`) upon validation failure.

## LLM Provider Support Upgrade
* **Multi-Provider Support (`cli.py`)**: Added support for different LLM providers via `--provider` flag (`vertex`, `openai`, `anthropic`).
* **Vertex AI Integration**: Added specific initialization and options (`--project-id`, `--location`) for Google Cloud Vertex AI using `gemini-1.5-pro`.
* **API Key Support**: Added standard `--api-key` validation for non-Vertex providers.

## Agentic Refactor — EngineerAgent + AgentSupervisor Architecture
* **`EngineerAgent` class (`cli.py`)**: Stateful agent with its own `message_history`. Persona locked to Senior ML Infrastructure Engineer. Exposes `generate_implementation(feedback_payload)` — on first call injects the initial task; on retries appends the Supervisor's structured OBSERVATION payload so the full chain-of-thought is preserved across turns.
* **`AgentSupervisor` class (`cli.py`)**: Orchestrates the generate → validate → feedback loop. On `evaluate_kernel()` FAIL it constructs a strict `OBSERVATION:` payload containing the autograd traceback, injects it back into the EngineerAgent, and retries up to `--max-retries` times. Terminal output uses distinct rich colors: `[ENGINEER AGENT]` in cyan, `[IMPACT ARBITER SUPERVISOR]` in magenta/red/green.
* **`build_generate_fn` factory (`cli.py`)**: Provider-agnostic backend builder supporting `vertex`, `openai`, and `anthropic`. Swappable via CLI flags — no code changes required to switch models.
* **Trace Logger**: `AgentSupervisor._save_trace()` emits the full multi-turn conversation (initial prompt → hallucinated code → supervisor observation → fixed code → PASS/FAIL) as a timestamped JSON artifact in `traces/`.
* **Temp-file hygiene**: Uses `tempfile.NamedTemporaryFile` with `os.unlink` cleanup instead of persistent `temp_kernel_attempt_N.py` files.

