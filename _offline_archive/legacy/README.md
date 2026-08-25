# Legacy / archived scratch

Machine-local and one-off artifacts moved out of the active tree during the
headless multi-process architecture finalization.

| Path | Notes |
|------|--------|
| `assets/` | Superseded logo backups |
| `suite_scratch/` | Ad-hoc smoke scripts |
| `Modelfile.draft` | Incomplete Ollama Modelfile draft |
| `notes.txt` | Local operator scratch |

Active Meta-Broker suite demos (`lru_cache.py`, `vector_math.py`, …) remain at
the repo root so suite harnesses and pytest imports keep working. Runtime
scratch (`.dana_scratch/`, `logs/`, `.dana/`) stays gitignored.
