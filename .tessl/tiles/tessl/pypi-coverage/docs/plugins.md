# Plugin System

Extensible plugin architecture for custom file tracers, configurers, and dynamic context switchers. Enables coverage measurement for non-Python files and custom execution environments.

## Capabilities

### CoveragePlugin Base Class

Base class for all coverage.py plugins providing hooks for file tracing, configuration, and dynamic context switching.

```python { .api }
class CoveragePlugin:
    """
    Base class for coverage.py plugins.
    
    Attributes set by coverage.py:
    - _coverage_plugin_name (str): Plugin name
    - _coverage_enabled (bool): Whether plugin is enabled
    """
    
    def file_tracer(self, filename: str):
        """
        Claim a file for tracing by this plugin.
        
        Parameters:
        - filename (str): The file being imported or executed
        
        Returns:
            FileTracer | None: FileTracer instance if this plugin handles the file
        """
    
    def file_reporter(self, filename: str):
        """
        Provide a FileReporter for a file handled by this plugin.
        
        Parameters:
        - filename (str): The file needing a reporter
        
        Returns:
            FileReporter | str: FileReporter instance or source filename
        """
    
    def dynamic_context(self, frame):
        """
        Determine the dynamic context for a frame.
        
        Parameters:
        - frame: Python frame object
        
        Returns:
            str | None: Context label or None to use default
        """
    
    def find_executable_files(self, src_dir: str):
        """
        Find executable files in a source directory.
        
        Parameters:
        - src_dir (str): Directory to search
        
        Returns:
            Iterable[str]: Executable file paths
        """
    
    def configure(self, config):
        """
        Configure coverage.py during startup.
        
        Parameters:
        - config: Coverage configuration object
        """
    
    def sys_info(self):
        """
        Return debugging information about this plugin.
        
        Returns:
            Iterable[tuple[str, Any]]: Key-value pairs of debug info
        """
```

Usage example:

```python
import coverage

class MyPlugin(coverage.CoveragePlugin):
    def file_tracer(self, filename):
        if filename.endswith('.myext'):
            return MyFileTracer(filename)
        return None
    
    def configure(self, config):
        # Modify configuration as needed
        config.set_option('run:source', ['src/'])
    
    def sys_info(self):
        return [
            ('my_plugin_version', '1.0.0'),
            ('my_plugin_config', self.config_info)
        ]

def coverage_init(reg, options):
    reg.add_file_tracer(MyPlugin())
```

### FileTracer Class

Base class for file tracers that handle non-Python files or custom execution environments.

```python { .api }
class FileTracer:
    """
    Base class for file tracers that track execution in non-Python files.
    """
    
    def source_filename(self) -> str:
        """
        Get the source filename for this traced file.
        
        Returns:
            str: The source filename to report coverage for
        """
    
    def has_dynamic_source_filename(self) -> bool:
        """
        Check if source filename can change dynamically.
        
        Returns:
            bool: True if source_filename can vary per frame
        """
    
    def dynamic_source_filename(self, filename: str, frame):
        """
        Get the source filename for a specific frame.
        
        Parameters:
        - filename (str): The file being traced
        - frame: Python frame object
        
        Returns:
            str | None: Source filename for this frame
        """
    
    def line_number_range(self, frame):
        """
        Get the range of line numbers for a frame.
        
        Parameters:
        - frame: Python frame object
        
        Returns:
            tuple[int, int]: (start_line, end_line) inclusive range
        """
```

Usage example:

```python
import coverage

class TemplateFileTracer(coverage.FileTracer):
    def __init__(self, template_file):
        self.template_file = template_file
        self.source_file = template_file.replace('.tmpl', '.py')
    
    def source_filename(self):
        return self.source_file
    
    def line_number_range(self, frame):
        # Map template lines to source lines
        template_line = frame.f_lineno
        source_line = self.map_template_to_source(template_line)
        return source_line, source_line
    
    def map_template_to_source(self, template_line):
        # Custom mapping logic
        return template_line * 2  # Example mapping

class TemplatePlugin(coverage.CoveragePlugin):
    def file_tracer(self, filename):
        if filename.endswith('.tmpl'):
            return TemplateFileTracer(filename)
        return None
```

### FileReporter Class

Base class for file reporters that provide analysis information for files.

```python { .api }
class FileReporter:
    """
    Base class for file reporters that analyze files for coverage reporting.
    """
    
    def __init__(self, filename: str):
        """
        Initialize the file reporter.
        
        Parameters:
        - filename (str): The file to report on
        """
    
    def relative_filename(self) -> str:
        """
        Get the relative filename for reporting.
        
        Returns:
            str: Relative path for display in reports
        """
    
    def source(self) -> str:
        """
        Get the source code of the file.
        
        Returns:
            str: Complete source code of the file
        """
    
    def lines(self) -> set[int]:
        """
        Get the set of executable line numbers.
        
        Returns:
            set[int]: Line numbers that can be executed
        """
    
    def excluded_lines(self) -> set[int]:
        """
        Get the set of excluded line numbers.
        
        Returns:
            set[int]: Line numbers excluded from coverage
        """
    
    def translate_lines(self, lines) -> set[int]:
        """
        Translate line numbers to the original file.
        
        Parameters:
        - lines (Iterable[int]): Line numbers to translate
        
        Returns:
            set[int]: Translated line numbers
        """
    
    def arcs(self) -> set[tuple[int, int]]:
        """
        Get the set of possible execution arcs.
        
        Returns:
            set[tuple[int, int]]: Possible (from_line, to_line) arcs
        """
    
    def no_branch_lines(self) -> set[int]:
        """
        Get lines that should not be considered for branch coverage.
        
        Returns:
            set[int]: Line numbers without branches
        """
    
    def translate_arcs(self, arcs) -> set[tuple[int, int]]:
        """
        Translate execution arcs to the original file.
        
        Parameters:
        - arcs (Iterable[tuple[int, int]]): Arcs to translate
        
        Returns:
            set[tuple[int, int]]: Translated arcs
        """
    
    def exit_counts(self) -> dict[int, int]:
        """
        Get exit counts for each line.
        
        Returns:
            dict[int, int]: Mapping of line numbers to exit counts
        """
    
    def missing_arc_description(self, start: int, end: int, executed_arcs=None) -> str:
        """
        Describe a missing arc for reporting.
        
        Parameters:
        - start (int): Starting line number
        - end (int): Ending line number
        - executed_arcs: Set of executed arcs for context
        
        Returns:
            str: Human-readable description of the missing arc
        """
    
    def arc_description(self, start: int, end: int) -> str:
        """
        Describe an arc for reporting.
        
        Parameters:
        - start (int): Starting line number
        - end (int): Ending line number
        
        Returns:
            str: Human-readable description of the arc
        """
    
    def source_token_lines(self):
        """
        Get tokenized source lines for syntax highlighting.
        
        Returns:
            Iterable[list[tuple[str, str]]]: Lists of (token_type, token_text) tuples
        """
    
    def code_regions(self):
        """
        Get code regions (functions, classes) in the file.
        
        Returns:
            Iterable[CodeRegion]: Code regions with metadata
        """
    
    def code_region_kinds(self):
        """
        Get the kinds of code regions this reporter recognizes.
        
        Returns:
            Iterable[tuple[str, str]]: (kind, display_name) pairs
        """
```

Usage example:

```python
import coverage
from coverage.plugin import CodeRegion

class JSONFileReporter(coverage.FileReporter):
    def __init__(self, filename):
        super().__init__(filename)
        self.filename = filename
        with open(filename) as f:
            self.json_data = json.load(f)
    
    def source(self):
        with open(self.filename) as f:
            return f.read()
    
    def lines(self):
        # Determine executable lines based on JSON structure
        return self.analyze_json_structure()
    
    def code_regions(self):
        regions = []
        for key, value in self.json_data.items():
            if isinstance(value, dict):
                regions.append(CodeRegion(
                    kind='object',
                    name=key,
                    start=self.find_key_line(key),
                    lines=self.get_object_lines(value)
                ))
        return regions
    
    def analyze_json_structure(self):
        # Custom logic to determine what constitutes "executable" JSON
        return set(range(1, self.count_lines() + 1))
```

### CodeRegion Data Class

Represents a region of code with metadata for enhanced reporting.

```python { .api }
@dataclass
class CodeRegion:
    """
    Represents a code region like a function or class.
    
    Attributes:
    - kind (str): Type of region ('function', 'class', 'method', etc.)
    - name (str): Name of the region
    - start (int): Starting line number
    - lines (set[int]): All line numbers in the region
    """
    kind: str
    name: str  
    start: int
    lines: set[int]
```

### Plugin Registration

Plugins are registered through a `coverage_init` function in the plugin module.

```python { .api }
def coverage_init(reg, options):
    """
    Plugin initialization function.
    
    Parameters:
    - reg: Plugin registry object
    - options (dict): Plugin configuration options
    """
    # Register file tracers
    reg.add_file_tracer(MyFileTracerPlugin())
    
    # Register configurers
    reg.add_configurer(MyConfigurerPlugin())
    
    # Register dynamic context providers
    reg.add_dynamic_context(MyContextPlugin())
```

Usage example:

```python
import coverage

class DatabaseQueryPlugin(coverage.CoveragePlugin):
    def __init__(self, options):
        self.connection_string = options.get('connection', 'sqlite:///:memory:')
        self.track_queries = options.get('track_queries', True)
    
    def dynamic_context(self, frame):
        # Provide context based on database operations
        if 'sqlalchemy' in frame.f_globals.get('__name__', ''):
            return f"db_query:{frame.f_code.co_name}"
        return None
    
    def configure(self, config):
        if self.track_queries:
            config.set_option('run:contexts', ['db_operations'])

def coverage_init(reg, options):
    plugin = DatabaseQueryPlugin(options)
    reg.add_dynamic_context(plugin)
    reg.add_configurer(plugin)
```

## Plugin Types

### File Tracer Plugins

Handle measurement of non-Python files by implementing `file_tracer()` and providing `FileTracer` instances.

```python
class MarkdownPlugin(coverage.CoveragePlugin):
    """Plugin to trace Markdown files with embedded Python code."""
    
    def file_tracer(self, filename):
        if filename.endswith('.md'):
            return MarkdownTracer(filename)
        return None

class MarkdownTracer(coverage.FileTracer):
    def __init__(self, filename):
        self.filename = filename
        self.python_file = self.extract_python_code()
    
    def source_filename(self):
        return self.python_file
    
    def extract_python_code(self):
        # Extract Python code blocks from Markdown
        # Return path to generated Python file
        pass
```

### Configurer Plugins

Modify coverage.py configuration during startup by implementing `configure()`.

```python
class TeamConfigPlugin(coverage.CoveragePlugin):
    """Plugin to apply team-wide configuration standards."""
    
    def configure(self, config):
        # Apply team standards
        config.set_option('run:branch', True)
        config.set_option('run:source', ['src/', 'lib/'])
        config.set_option('report:exclude_lines', [
            'pragma: no cover',
            'def __repr__',
            'raise NotImplementedError'
        ])
```

### Dynamic Context Plugins

Provide dynamic context labels by implementing `dynamic_context()`.

```python
class TestFrameworkPlugin(coverage.CoveragePlugin):
    """Plugin to provide test-specific contexts."""
    
    def dynamic_context(self, frame):
        # Detect test framework and provide context
        code_name = frame.f_code.co_name
        filename = frame.f_code.co_filename
        
        if 'test_' in code_name or '/tests/' in filename:
            return f"test:{code_name}"
        elif 'pytest' in str(frame.f_globals.get('__file__', '')):
            return f"pytest:{code_name}"
        
        return None
```

## Complete Plugin Example

Here's a comprehensive example of a plugin that handles custom template files:

```python
import coverage
from coverage.plugin import CodeRegion
import re
import os

class TemplatePlugin(coverage.CoveragePlugin):
    """Plugin for measuring coverage of custom template files."""
    
    def __init__(self, options):
        self.template_extensions = options.get('extensions', ['.tmpl', '.tpl'])
        self.output_dir = options.get('output_dir', 'generated/')
    
    def file_tracer(self, filename):
        for ext in self.template_extensions:
            if filename.endswith(ext):
                return TemplateTracer(filename, self.output_dir)
        return None
    
    def file_reporter(self, filename):
        for ext in self.template_extensions:
            if filename.endswith(ext):
                return TemplateReporter(filename)
        return None
    
    def sys_info(self):
        return [
            ('template_plugin_version', '1.0.0'),
            ('template_extensions', self.template_extensions),
        ]

class TemplateTracer(coverage.FileTracer):
    def __init__(self, template_file, output_dir):
        self.template_file = template_file
        self.output_dir = output_dir
        self.python_file = self.generate_python_file()
    
    def source_filename(self):
        return self.python_file
    
    def generate_python_file(self):
        # Convert template to Python file
        basename = os.path.basename(self.template_file)
        python_file = os.path.join(self.output_dir, basename + '.py')
        
        with open(self.template_file) as f:
            template_content = f.read()
        
        # Simple template-to-Python conversion
        python_content = self.convert_template(template_content)
        
        os.makedirs(self.output_dir, exist_ok=True)
        with open(python_file, 'w') as f:
            f.write(python_content)
        
        return python_file
    
    def convert_template(self, content):
        # Convert template syntax to Python
        # This is a simplified example
        lines = content.split('\n')
        python_lines = []
        
        for line in lines:
            if line.strip().startswith('{{ '):
                # Template variable
                var = line.strip()[3:-3].strip()
                python_lines.append(f'print({var})')
            elif line.strip().startswith('{% '):
                # Template logic
                logic = line.strip()[3:-3].strip()
                python_lines.append(logic)
            else:
                # Static content
                python_lines.append(f'print({repr(line)})')
        
        return '\n'.join(python_lines)

class TemplateReporter(coverage.FileReporter):
    def __init__(self, filename):
        super().__init__(filename)
        self.filename = filename
    
    def source(self):
        with open(self.filename) as f:
            return f.read()
    
    def lines(self):
        # All non-empty lines are considered executable
        with open(self.filename) as f:
            lines = f.readlines()
        
        executable = set()
        for i, line in enumerate(lines, 1):
            if line.strip():
                executable.add(i)
        
        return executable
    
    def code_regions(self):
        regions = []
        with open(self.filename) as f:
            content = f.read()
        
        # Find template blocks
        for match in re.finditer(r'{%\s*(\w+)', content):
            block_type = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            
            regions.append(CodeRegion(
                kind='template_block',
                name=block_type,
                start=line_num,
                lines={line_num}
            ))
        
        return regions

def coverage_init(reg, options):
    """Initialize the template plugin."""
    plugin = TemplatePlugin(options)
    reg.add_file_tracer(plugin)
```

To use this plugin, create a configuration file:

```ini
# .coveragerc
[run]
plugins = template_plugin

[template_plugin]
extensions = .tmpl, .tpl, .template
output_dir = generated/
```