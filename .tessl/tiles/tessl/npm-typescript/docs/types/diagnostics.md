# Diagnostics

Diagnostic types and utilities for error, warning, and suggestion reporting throughout the TypeScript compilation process.

## Capabilities

### Diagnostic Interface

Core diagnostic structure.

```typescript { .api }
interface Diagnostic {
  category: DiagnosticCategory;
  code: number;
  file: SourceFile | undefined;
  start: number | undefined;
  length: number | undefined;
  messageText: string | DiagnosticMessageChain;
  relatedInformation?: DiagnosticRelatedInformation[];
  source?: string;
  reportsUnnecessary?: {};
  reportsDeprecated?: {};
}

interface DiagnosticWithLocation extends Diagnostic {
  file: SourceFile;
  start: number;
  length: number;
}

enum DiagnosticCategory {
  Warning = 0,
  Error = 1,
  Suggestion = 2,
  Message = 3
}
```

### Diagnostic Message Chain

```typescript { .api }
interface DiagnosticMessageChain {
  messageText: string;
  category: DiagnosticCategory;
  code: number;
  next?: DiagnosticMessageChain[];
}

interface DiagnosticMessage {
  key: string;
  category: DiagnosticCategory;
  code: number;
  message: string;
  reportsUnnecessary?: {};
  reportsDeprecated?: {};
}
```

### Related Information

```typescript { .api }
interface DiagnosticRelatedInformation {
  category: DiagnosticCategory;
  code: number;
  file: SourceFile | undefined;
  start: number | undefined;
  length: number | undefined;
  messageText: string | DiagnosticMessageChain;
}
```

### Pre-Emit Diagnostics

Get all diagnostics before emit.

```typescript { .api }
function getPreEmitDiagnostics(
  program: Program,
  sourceFile?: SourceFile,
  cancellationToken?: CancellationToken
): readonly Diagnostic[];
```

### Formatting Diagnostics

Format diagnostics for display.

```typescript { .api }
function formatDiagnostic(
  diagnostic: Diagnostic,
  host: FormatDiagnosticsHost
): string;

function formatDiagnostics(
  diagnostics: readonly Diagnostic[],
  host: FormatDiagnosticsHost
): string;

function formatDiagnosticsWithColorAndContext(
  diagnostics: readonly Diagnostic[],
  host: FormatDiagnosticsHost
): string;

interface FormatDiagnosticsHost {
  getCurrentDirectory(): string;
  getCanonicalFileName(fileName: string): string;
  getNewLine(): string;
}
```

### Flatten Message Text

```typescript { .api }
function flattenDiagnosticMessageText(
  diag: string | DiagnosticMessageChain | undefined,
  newLine: string,
  indent?: number
): string;
```

### Creating Diagnostics

```typescript { .api }
function createCompilerDiagnostic(
  message: DiagnosticMessage,
  ...args: (string | number | undefined)[]
): Diagnostic;

function createFileDiagnostic(
  file: SourceFile,
  start: number,
  length: number,
  message: DiagnosticMessage,
  ...args: (string | number | undefined)[]
): DiagnosticWithLocation;

function createCompilerDiagnosticFromMessageChain(
  chain: DiagnosticMessageChain,
  relatedInformation?: DiagnosticRelatedInformation[]
): Diagnostic;
```

### Diagnostic Sorting

```typescript { .api }
function sortAndDeduplicateDiagnostics<T extends Diagnostic>(
  diagnostics: readonly T[]
): SortedReadonlyArray<T>;

function compareDiagnostics(d1: Diagnostic, d2: Diagnostic): number;

function compareDiagnosticsSkipRelatedInformation(
  d1: Diagnostic,
  d2: Diagnostic
): number;
```

### Getting Diagnostics from Program

```typescript { .api }
// From Program interface
interface Program {
  getSyntacticDiagnostics(
    sourceFile?: SourceFile,
    cancellationToken?: CancellationToken
  ): readonly Diagnostic[];
  
  getSemanticDiagnostics(
    sourceFile?: SourceFile,
    cancellationToken?: CancellationToken
  ): readonly Diagnostic[];
  
  getDeclarationDiagnostics(
    sourceFile?: SourceFile,
    cancellationToken?: CancellationToken
  ): readonly DiagnosticWithLocation[];
  
  getConfigFileParsingDiagnostics(): readonly Diagnostic[];
  
  getGlobalDiagnostics(): readonly Diagnostic[];
  
  getOptionsDiagnostics(
    cancellationToken?: CancellationToken
  ): readonly Diagnostic[];
}
```

### Cancellation Token

```typescript { .api }
interface CancellationToken {
  isCancellationRequested(): boolean;
  throwIfCancellationRequested(): void;
}

const nullCancellationToken: CancellationToken;
```

### Error Codes

TypeScript error codes are numbered from 1000-7999. Common ranges:

- 1000-1999: Syntactic errors
- 2000-2999: Semantic errors
- 4000-4999: Declaration file errors
- 5000-5999: Compiler options errors
- 6000-6999: Command line errors
- 7000-7999: Internal errors

## Usage Example

Getting and formatting diagnostics:

```typescript
import * as ts from 'typescript';

const code = `
const x: number = "not a number";
function test() { }
test(123); // Too many arguments
`;

const sourceFile = ts.createSourceFile('example.ts', code, ts.ScriptTarget.Latest, true);
const program = ts.createProgram(['example.ts'], {}, {
  getSourceFile: (fileName) => fileName === 'example.ts' ? sourceFile : undefined,
  writeFile: () => {},
  getCurrentDirectory: () => '',
  getDirectories: () => [],
  fileExists: () => true,
  readFile: () => '',
  getCanonicalFileName: (fileName) => fileName,
  useCaseSensitiveFileNames: () => true,
  getNewLine: () => '\n'
});

// Get all pre-emit diagnostics
const diagnostics = ts.getPreEmitDiagnostics(program);

// Format diagnostics
const formatHost: ts.FormatDiagnosticsHost = {
  getCurrentDirectory: () => process.cwd(),
  getCanonicalFileName: (fileName) => fileName,
  getNewLine: () => '\n'
};

console.log(ts.formatDiagnosticsWithColorAndContext(diagnostics, formatHost));

// Print individual diagnostics
diagnostics.forEach(diagnostic => {
  if (diagnostic.file) {
    const { line, character } = diagnostic.file.getLineAndCharacterOfPosition(diagnostic.start!);
    const message = ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n');
    console.log(`${diagnostic.file.fileName} (${line + 1},${character + 1}): error TS${diagnostic.code}: ${message}`);
  } else {
    console.log(ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n'));
  }
});
```

Filtering diagnostics by category:

```typescript
import * as ts from 'typescript';

function filterDiagnosticsByCategory(
  diagnostics: readonly ts.Diagnostic[],
  category: ts.DiagnosticCategory
): ts.Diagnostic[] {
  return diagnostics.filter(d => d.category === category);
}

const errors = filterDiagnosticsByCategory(diagnostics, ts.DiagnosticCategory.Error);
const warnings = filterDiagnosticsByCategory(diagnostics, ts.DiagnosticCategory.Warning);
const suggestions = filterDiagnosticsByCategory(diagnostics, ts.DiagnosticCategory.Suggestion);

console.log(`Errors: ${errors.length}`);
console.log(`Warnings: ${warnings.length}`);
console.log(`Suggestions: ${suggestions.length}`);
```
