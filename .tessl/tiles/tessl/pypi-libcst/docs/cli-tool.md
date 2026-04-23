# LibCST Command-Line Tool

LibCST provides a powerful command-line interface for parsing, analyzing, and transforming Python code using concrete syntax trees. The CLI tool offers several commands for different operations.

## Installation and Usage

The LibCST CLI tool is invoked as a Python module:

```bash
python -m libcst.tool --help
```

## Core Capabilities

The LibCST command-line tool provides four main commands:

- **`print`** - Print the LibCST tree representation of Python files
- **`codemod`** - Execute code transformations using codemod commands  
- **`list`** - List all available codemod commands
- **`initialize`** - Initialize a directory with default LibCST configuration

### Global Options

All commands support these global options:

```{ .api }
--version           Print current version of LibCST toolset
--help, -h         Show help message
@file               Read arguments from a file (one per line)
```

## Print Command

Parse and display the LibCST concrete syntax tree representation of Python code.

```{ .api }
python -m libcst.tool print [OPTIONS] INFILE

Arguments:
  INFILE                    File to print tree for. Use "-" for stdin

Options:
  --show-whitespace        Show whitespace nodes in printed tree
  --show-defaults          Show values that are unchanged from the default
  --show-syntax            Show values that exist only for syntax (commas, semicolons)
  --graphviz               Display the graph in .dot format, compatible with Graphviz
  --indent-string STRING   String to use for indenting levels (default: "  ")
  -p, --python-version VER Override the version string used for parsing Python files
```

### Print Usage Examples

Print the CST of a Python file:
```bash
python -m libcst.tool print my_script.py
```

Parse from stdin and show whitespace nodes:
```bash
cat my_script.py | python -m libcst.tool print --show-whitespace -
```

Generate Graphviz output for visualization:
```bash
python -m libcst.tool print --graphviz my_script.py | dot -Tpng -o tree.png
```

Parse with specific Python version:
```bash
python -m libcst.tool print --python-version 3.9 my_script.py
```

## Codemod Command

Execute code transformations using LibCST's codemod framework. Codemods are automated code transformation tools that can refactor, update, or modify Python codebases systematically.

```{ .api }
python -m libcst.tool codemod [OPTIONS] COMMAND PATH [PATH ...]

Arguments:
  COMMAND                   Codemod command to execute (e.g., strip_strings_from_types.StripStringsCommand)
  PATH                      Path(s) to transform. Can be files, directories, or "-" for stdin

Options:
  -x, --external           Interpret COMMAND as just a module/class specifier
  -j, --jobs JOBS          Number of jobs for parallel processing (default: number of cores)
  -p, --python-version VER Override Python version for parsing (default: current Python version)
  -u, --unified-diff [N]   Output unified diff instead of contents (default context: 5 lines)
  --include-generated      Process generated files (normally skipped)
  --include-stubs          Process typing stub files (.pyi)
  --no-format              Skip formatting with configured formatter
  --show-successes         Print successfully transformed files with no warnings
  --hide-generated-warnings  Don't print warnings for skipped generated files
  --hide-blacklisted-warnings  Don't print warnings for skipped blacklisted files
  --hide-progress          Don't show progress indicator
```

### Codemod Usage Examples

Remove unused imports from a file:
```bash
python -m libcst.tool codemod remove_unused_imports.RemoveUnusedImportsCommand my_script.py
```

Transform an entire directory with diff output:
```bash
python -m libcst.tool codemod --unified-diff strip_strings_from_types.StripStringsCommand src/
```

Run codemod from stdin to stdout:
```bash
cat my_script.py | python -m libcst.tool codemod noop.NOOPCommand -
```

Use external codemod with custom arguments:
```bash
python -m libcst.tool codemod -x my_custom_codemods.MyTransform --old_name foo --new_name bar src/
```

Process files in parallel:
```bash
python -m libcst.tool codemod --jobs 8 convert_format_to_fstring.ConvertFormatStringCommand src/
```

## List Command

Display all available codemod commands and their descriptions.

```{ .api }
python -m libcst.tool list
```

### List Usage Examples

Show all available codemods:
```bash
python -m libcst.tool list
```

Example output:
```
add_trailing_commas.AddTrailingCommasCommand - Add trailing commas to function calls, literals, etc.
convert_format_to_fstring.ConvertFormatStringCommand - Convert .format(...) strings to f-strings
noop.NOOPCommand - Does absolutely nothing.
rename.RenameCommand - Rename all instances of a local or imported object.
remove_unused_imports.RemoveUnusedImportsCommand - Remove unused imports from modules.
```

## Initialize Command

Create a default LibCST configuration file in a directory to configure codemod behavior.

```{ .api }
python -m libcst.tool initialize PATH

Arguments:
  PATH                     Path to initialize with default LibCST configuration
```

### Configuration File

The `initialize` command creates a `.libcst.codemod.yaml` file with these settings:

```yaml
# String that LibCST should look for in code which indicates that the module is generated code.
generated_code_marker: "@generated"

# Command line and arguments for invoking a code formatter.
formatter: ["black", "-"]

# List of regex patterns which LibCST will evaluate against filenames to determine if the module should be touched.
blacklist_patterns: []

# List of modules that contain codemods inside of them.
modules:
- libcst.codemod.commands

# Absolute or relative path of the repository root, used for providing full-repo metadata.
repo_root: "."
```

### Initialize Usage Examples

Initialize current directory:
```bash
python -m libcst.tool initialize .
```

Initialize a specific project directory:
```bash
python -m libcst.tool initialize /path/to/my/project
```

## Built-in Codemod Commands

LibCST includes several built-in codemod commands for common transformations:

### Text and String Transformations
- **`convert_format_to_fstring.ConvertFormatStringCommand`** - Convert `.format()` calls to f-strings
- **`convert_percent_format_to_fstring.ConvertPercentFormatToFStringCommand`** - Convert `%` formatting to f-strings
- **`unnecessary_format_string.UnnecessaryFormatStringCommand`** - Remove unnecessary f-string formatting

### Type System Updates
- **`convert_type_comments.ConvertTypeComments`** - Convert type comments to annotations
- **`convert_union_to_or.ConvertUnionToOrCommand`** - Convert `Union[X, Y]` to `X | Y` (Python 3.10+)
- **`strip_strings_from_types.StripStringsCommand`** - Remove string quotes from type annotations

### Import Management
- **`remove_unused_imports.RemoveUnusedImportsCommand`** - Remove imports that are not used
- **`ensure_import_present.EnsureImportPresentCommand`** - Add imports if they don't exist
- **`rename.RenameCommand`** - Rename symbols and update imports

### Code Style
- **`add_trailing_commas.AddTrailingCommasCommand`** - Add trailing commas to collections
- **`convert_namedtuple_to_dataclass.ConvertNamedTupleToDataclassCommand`** - Convert NamedTuple to dataclass

### Pyre Integration
- **`add_pyre_directive.AddPyreDirectiveCommand`** - Add Pyre type checker directives
- **`remove_pyre_directive.RemovePyreDirectiveCommand`** - Remove Pyre directives
- **`fix_pyre_directives.FixPyreDirectivesCommand`** - Fix malformed Pyre directives

### Example Codemod Arguments

Many codemods accept additional arguments. For example:

```bash
# Rename a symbol across the codebase
python -m libcst.tool codemod rename.RenameCommand \
    --old_name "mymodule.OldClass" \
    --new_name "mymodule.NewClass" \
    src/

# Ensure an import is present
python -m libcst.tool codemod ensure_import_present.EnsureImportPresentCommand \
    --module "typing" \
    --entity "List" \
    src/
```

## Configuration Management

### Environment Variables

- **`LIBCST_TOOL_REQUIRE_CONFIG`** - If set, requires a configuration file to be present
- **`LIBCST_TOOL_COMMAND_NAME`** - Override the command name in help text

### Configuration File Discovery

LibCST searches for `.libcst.codemod.yaml` configuration files by walking up the directory tree from the current working directory. This allows different parts of a repository to have different codemod configurations.

### Formatter Integration

The configuration file specifies a code formatter command that will be automatically applied after codemod transformations. The default is Black, but any formatter that accepts code via stdin and outputs formatted code via stdout can be used.

## Advanced Usage

### Custom Codemods

You can create custom codemods by subclassing from `CodemodCommand`:

```python
from libcst.codemod import CodemodCommand
import libcst as cst

class MyCustomCommand(CodemodCommand):
    DESCRIPTION = "My custom transformation"
    
    def transform_module_impl(self, tree: cst.Module) -> cst.Module:
        # Your transformation logic here
        return tree
```

### Parallel Processing

For large codebases, use the `--jobs` option to process files in parallel:

```bash
python -m libcst.tool codemod --jobs 16 my_transform.MyCommand large_codebase/
```

### Integration with CI/CD

Use `--unified-diff` to generate patches for review:

```bash
python -m libcst.tool codemod --unified-diff my_transform.Command src/ > changes.patch
```

The CLI tool returns appropriate exit codes:
- `0` - Success, no failures
- `1` - Some files failed to transform
- `2` - Interrupted by user (Ctrl+C)

## Error Handling

The codemod command provides detailed reporting:

```
Finished codemodding 150 files!
 - Transformed 145 files successfully.
 - Skipped 3 files.
 - Failed to codemod 2 files.
 - 5 warnings were generated.
```

Files may be skipped for several reasons:
- Generated files (contain the `generated_code_marker` string)
- Files matching `blacklist_patterns`
- Syntax errors or parse failures
- Permission issues

Use the various `--hide-*` flags to control warning verbosity, and `--show-successes` to see all successfully processed files.