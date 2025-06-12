---
sidebar_position: 5
---

# Manual Configuration

Codeflash is installed and configured on a per-project basis.
`codeflash init` should guide you through the configuration process, but if you need to manually configure Codeflash or set advanced settings, you can do so by editing the `pyproject.toml` file in the root directory of your project.

## Configuration Options
Codeflash config looks like the following
```toml
[tool.codeflash]
module-root = "my_module"
tests-root = "tests"
test-framework = "pytest"
formatter-cmds = ["black $file"]
# optional configuration
benchmarks-root = "tests/benchmarks" # Required when running with --benchmark
ignore-paths = ["my_module/build/"]
pytest-cmd = "pytest"
disable-imports-sorting = false
disable-telemetry = false
```
All file paths are relative to the directory of the `pyproject.toml` file.

Required Options:
- `module-root`: The Python module you want Codeflash to optimize going forward. Only code under this directory will be optimized. It should also have an `__init__.py` file to make the module importable.
- `tests-root`: The directory where your tests are located. Codeflash will use this directory to discover existing tests as well as generate new tests.
- `test-framework`: The test framework you use for your project. Codeflash supports `pytest` and `unittest`.

Optional Configuration:
- `benchmarks-root`: The directory where your benchmarks are located. Codeflash will use this directory to discover existing benchmarks. Note that this option is required when running with `--benchmark`.
- `ignore-paths`: A list of paths withing the `module-root` to ignore when optimizing code. Codeflash will not optimize code in these paths. Useful for ignoring build directories or other generated code. You can also leave this empty if not needed.
- `pytest-cmd`: The command to run your tests. Defaults to `pytest`. You can specify extra commandline arguments here for pytest.
- `formatter-cmds`: The command line to run your code formatter or linter. Defaults to `["black $file"]`. In the command line `$file` refers to the current file being optimized. The assumption with using tools here is that they overwrite the same file and returns a zero exit code. You can also specify multiple tools here that run in a chain as a toml array. You can also disable code formatting by setting this to `["disabled"]`.
  - `ruff` - A recommended way to run ruff linting and formatting is `["ruff check --exit-zero --fix $file", "ruff format $file"]`. To make `ruff check --fix` return a 0 exit code please add a `--exit-zero` argument. 
- `disable-imports-sorting`: By default, codeflash uses isort to organize your imports before creating suggestions. You can disable this by setting this field to `true`. This could be useful if you don't sort your imports or while using linters like ruff that sort imports too.
- `disable-telemetry`: Disable telemetry data collection. Defaults to `false`. Set this to `true` to disable telemetry data collection. Codeflash collects anonymized telemetry data to understand how users are using Codeflash and to improve the product. Telemetry does not collect any code data.

## Example Configuration
Here's an example project with the following structure:
```text
acme-project/
|- foo_module/
|  |- __init__.py
|  |- foo.py
|  |- main.py
|- tests/
|  |- __init__.py
|  |- test_script.py
|- pyproject.toml
```

Here's a sample `pyproject.toml` file for the above project:
```toml
[tool.codeflash]
module-root = "foo_module"
tests-root = "tests"
test-framework = "pytest" # or "unittest"
ignore-paths = []
```
