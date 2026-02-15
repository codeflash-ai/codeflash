# Module Reference

| Feature area | Primary module | Key files |
|-------------|----------------|-----------|
| New optimization strategy | `optimization/` | `function_optimizer.py`, `optimizer.py` |
| New test type | `verification/`, `models/` | `test_runner.py`, `pytest_plugin.py`, `test_type.py` |
| New AI service endpoint | `api/` | `aiservice.py` |
| New language support | `languages/` | Create new `languages/<lang>/support.py` |
| Context extraction change | `context/` | `code_context_extractor.py` |
| New CLI command | `cli_cmds/` | `cli.py` |
| New config option | `setup/`, `code_utils/` | `config_consts.py`, `setup/detector.py` |
| Discovery filter | `discovery/` | `functions_to_optimize.py` |
| PR/result changes | `github/`, `result/` | Relevant handlers |
