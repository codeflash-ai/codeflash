# Utilities

Utility functions, constants, and the System interface for file system operations, text processing, and TypeScript compiler infrastructure.

## Capabilities

### System Interface

Platform abstraction interface for file system operations and environment interaction.

```typescript { .api }
interface System {
  args: string[];
  newLine: string;
  useCaseSensitiveFileNames: boolean;
  write(s: string): void;
  writeOutputIsTTY?(): boolean;
  getWidthOfTerminal?(): number;
  readFile(path: string, encoding?: string): string | undefined;
  getFileSize?(path: string): number;
  writeFile(path: string, data: string, writeByteOrderMark?: boolean): void;
  /**
   * @pollingInterval - this parameter is used in polling-based watchers and ignored
   * in watchers that use native OS file watching
   */
  watchFile?(
    path: string,
    callback: FileWatcherCallback,
    pollingInterval?: number,
    options?: WatchOptions
  ): FileWatcher;
  watchDirectory?(
    path: string,
    callback: DirectoryWatcherCallback,
    recursive?: boolean,
    options?: WatchOptions
  ): FileWatcher;
  resolvePath(path: string): string;
  fileExists(path: string): boolean;
  directoryExists(path: string): boolean;
  createDirectory(path: string): void;
  getExecutingFilePath(): string;
  getCurrentDirectory(): string;
  getDirectories(path: string): string[];
  readDirectory(
    path: string,
    extensions?: readonly string[],
    exclude?: readonly string[],
    include?: readonly string[],
    depth?: number
  ): string[];
  getModifiedTime?(path: string): Date | undefined;
  setModifiedTime?(path: string, time: Date): void;
  deleteFile?(path: string): void;
  /**
   * A good implementation is node.js' `crypto.createHash`.
   * (https://nodejs.org/api/crypto.html#crypto_crypto_createhash_algorithm)
   */
  createHash?(data: string): string;
  /**
   * This must be cryptographically secure. Only implement this method using
   * `crypto.createHash("sha256")`.
   */
  createSHA256Hash?(data: string): string;
  getMemoryUsage?(): number;
  exit(exitCode?: number): void;
  realpath?(path: string): string;
  setTimeout?(callback: (...args: any[]) => void, ms: number, ...args: any[]): any;
  clearTimeout?(timeoutId: any): void;
  clearScreen?(): void;
  base64decode?(input: string): string;
  base64encode?(input: string): string;
}

interface FileWatcher {
  close(): void;
}

let sys: System;
```

### Printer Interface

AST printing utilities for converting nodes back to source code.

```typescript { .api }
interface Printer {
  /**
   * Print a node and its subtree as-is, without any emit transformations.
   * @param hint A value indicating the purpose of a node. This is primarily used to
   * distinguish between an Identifier used in an expression position, versus an
   * Identifier used as an IdentifierName as part of a declaration. For most nodes you
   * should just pass Unspecified.
   * @param node The node to print. The node and its subtree are printed as-is, without any
   * emit transformations.
   * @param sourceFile A source file that provides context for the node. The source text of
   * the file is used to emit the original source content for literals and identifiers, while
   * the identifiers of the source file are used when generating unique names to avoid
   * collisions.
   */
  printNode(hint: EmitHint, node: Node, sourceFile: SourceFile): string;
  /**
   * Prints a list of nodes using the given format flags
   */
  printList<T extends Node>(
    format: ListFormat,
    list: NodeArray<T>,
    sourceFile: SourceFile
  ): string;
  /**
   * Prints a source file as-is, without any emit transformations.
   */
  printFile(sourceFile: SourceFile): string;
  /**
   * Prints a bundle of source files as-is, without any emit transformations.
   */
  printBundle(bundle: Bundle): string;
}

interface PrinterOptions {
  removeComments?: boolean;
  newLine?: NewLineKind;
  omitTrailingSemicolon?: boolean;
  noEmitHelpers?: boolean;
}

function createPrinter(
  printerOptions?: PrinterOptions,
  handlers?: PrintHandlers
): Printer;
```

### Version Constants

TypeScript version information.

```typescript { .api }
/**
 * The version of the TypeScript compiler (e.g., "5.9.3")
 */
const version: string;

/**
 * The major.minor version of TypeScript (e.g., "5.9")
 */
const versionMajorMinor: string;
```

### Text and Position Utilities

Functions for working with text positions and line/character coordinates.

```typescript { .api }
function tokenToString(t: SyntaxKind): string | undefined;

function getPositionOfLineAndCharacter(
  sourceFile: SourceFileLike,
  line: number,
  character: number
): number;

function getLineAndCharacterOfPosition(
  sourceFile: SourceFileLike,
  position: number
): LineAndCharacter;

interface LineAndCharacter {
  line: number;
  character: number;
}

function isWhiteSpaceLike(ch: number): boolean;

function isWhiteSpaceSingleLine(ch: number): boolean;

function isLineBreak(ch: number): boolean;

function isDigit(ch: number): boolean;

function isOctalDigit(ch: number): boolean;
```

### AST Visitor Utilities

Note: These visitor utilities complement the transformation visitor functions for AST manipulation.

```typescript { .api }
/**
 * Visits a node using a specified visitor function.
 */
function visitNode<T extends Node>(
  node: T | undefined,
  visitor: Visitor<T>,
  test?: (node: Node) => boolean,
  lift?: (node: readonly Node[]) => T
): T | undefined;

/**
 * Visits an array of nodes using a specified visitor function.
 */
function visitNodes<T extends Node>(
  nodes: NodeArray<T> | undefined,
  visitor: Visitor<T>,
  test?: (node: Node) => boolean,
  start?: number,
  count?: number
): NodeArray<T> | undefined;

/**
 * Visits each child of a node using a specified visitor function.
 */
function visitEachChild<T extends Node>(
  node: T,
  visitor: Visitor,
  context: TransformationContext
): T;
```

### Quick Transpilation

Fast transpilation APIs for converting TypeScript to JavaScript without full type checking.

```typescript { .api }
/**
 * Transpile a single module without creating a program.
 * Faster than full compilation when type checking is not needed.
 */
function transpileModule(input: string, transpileOptions: TranspileOptions): TranspileOutput;

/**
 * Transpile TypeScript declarations (.d.ts) without creating a program.
 */
function transpileDeclaration(input: string, transpileOptions: TranspileOptions): TranspileOutput;

/**
 * Quick transpile function for converting TypeScript to JavaScript.
 * Returns only the output text without diagnostics or source maps.
 */
function transpile(
  input: string,
  compilerOptions?: CompilerOptions,
  fileName?: string,
  diagnostics?: Diagnostic[],
  moduleName?: string
): string;

interface TranspileOptions {
  compilerOptions?: CompilerOptions;
  fileName?: string;
  reportDiagnostics?: boolean;
  moduleName?: string;
  renamedDependencies?: MapLike<string>;
  transformers?: CustomTransformers;
  jsDocParsingMode?: JSDocParsingMode;
}

interface TranspileOutput {
  outputText: string;
  diagnostics?: Diagnostic[];
  sourceMapText?: string;
}
```

### File System Utilities

Helper functions for path manipulation and file operations.

```typescript { .api }
/**
 * Get the path to the default library file (lib.d.ts or lib.*.d.ts) based on compiler options.
 * Useful when setting up custom compiler hosts or build tools.
 *
 * @param options - Compiler options determining which lib file to use
 * @returns The file path to the appropriate default library file
 */
function getDefaultLibFilePath(options: CompilerOptions): string;

function normalizePath(path: string): string;

function normalizeSlashes(path: string): string;

function getRootLength(path: string): number;

function getDirectoryPath(path: Path): Path;

function getDirectoryPath(path: string): string;

function isUrl(path: string): boolean;

function pathIsRelative(path: string): boolean;

function getBaseFileName(path: string): string;

function getBaseFileName(
  path: string,
  extensions: string | readonly string[],
  ignoreCase?: boolean
): string;

function combinePaths(path: string, ...paths: (string | undefined)[]): string;

function resolvePath(path: string, ...paths: (string | undefined)[]): string;

function getRelativePathToDirectoryOrUrl(
  directoryPathOrUrl: string,
  relativeOrAbsolutePath: string,
  currentDirectory: string,
  getCanonicalFileName: (fileName: string) => string,
  isAbsolutePathAnUrl: boolean
): string;
```

### Types

Utility types used throughout the TypeScript compiler.

```typescript { .api }
/**
 * String type branded as a file system path.
 * Used internally to distinguish paths from regular strings.
 */
type Path = string & {
  __pathBrand: any;
};

/**
 * String type branded for use as an escaped identifier name.
 * Used internally for efficient Symbol name storage.
 */
type __String = (string & {
  __escapedIdentifier: void;
}) | (void & {
  __escapedIdentifier: void;
});
```

### Identifier Utilities

Functions for working with escaped identifiers.

```typescript { .api }
/**
 * Add extra underscore to escaped identifier text content.
 * Used to escape identifiers that would otherwise conflict with reserved words.
 */
function escapeLeadingUnderscores(identifier: string): __String;

/**
 * Remove extra underscore from escaped identifier text content.
 * Converts an escaped identifier back to its original form.
 */
function unescapeLeadingUnderscores(identifier: __String): string;
```

### String Utilities

Functions for string manipulation and character classification.

```typescript { .api }
function isExternalModuleNameRelative(moduleName: string): boolean;

function sortAndDeduplicateDiagnostics<T extends Diagnostic>(
  diagnostics: readonly T[]
): SortedReadonlyArray<T>;

function getNewLineCharacter(options: CompilerOptions | PrinterOptions): string;

function createTextChangeRange(span: TextSpan, newLength: number): TextChangeRange;

interface TextChangeRange {
  span: TextSpan;
  newLength: number;
}

function collapseTextChangeRangesAcrossMultipleVersions(
  changes: readonly TextChangeRange[]
): TextChangeRange;
```

### Token Utilities

Functions for working with tokens and keywords.

```typescript { .api }
function isKeyword(token: SyntaxKind): boolean;

function isModifier(token: SyntaxKind): boolean;

function isContextualKeyword(token: SyntaxKind): boolean;

function isStringDoubleQuoted(
  str: StringLiteralLike,
  sourceFile: SourceFile
): boolean;

function getTextOfIdentifierOrLiteral(node: PropertyName): string;

function getTextOfNode(
  node: Node,
  includeTrivia?: boolean
): string;

function getNameOfDeclaration(
  declaration: Declaration | Expression | undefined
): DeclarationName | undefined;

function isDeclarationStatement(node: Node): node is DeclarationStatement;
```

### Module Name Utilities

Functions for working with module specifiers and imports.

```typescript { .api }
function getExternalModuleNameLiteral(
  factory: NodeFactory,
  importNode:
    | ImportDeclaration
    | ExportDeclaration
    | ImportTypeNode
    | ImportCall,
  sourceFile: SourceFile,
  host: EmitHost,
  resolver: EmitResolver,
  compilerOptions: CompilerOptions
): StringLiteral | undefined;

function tryGetModuleNameFromFile(
  factory: NodeFactory,
  file: SourceFile | undefined,
  host: EmitHost,
  options: CompilerOptions
): Expression | undefined;
```

## Usage Examples

### Quick Transpilation

Quickly convert TypeScript to JavaScript without full type checking:

```typescript
import * as ts from 'typescript';

// Simple transpilation (returns only output text)
const tsCode = `
  const greet = (name: string): string => {
    return \`Hello, \${name}!\`;
  };
`;

const jsCode = ts.transpile(tsCode, {
  module: ts.ModuleKind.CommonJS,
  target: ts.ScriptTarget.ES2015
});

console.log(jsCode);
// Output: const greet = (name) => { return `Hello, ${name}!`; };

// Transpile with diagnostics and source maps
const result = ts.transpileModule(tsCode, {
  compilerOptions: {
    module: ts.ModuleKind.ES2015,
    target: ts.ScriptTarget.ES2020,
    sourceMap: true
  },
  fileName: 'example.ts',
  reportDiagnostics: true
});

console.log('Output:', result.outputText);
console.log('Source map:', result.sourceMapText);
if (result.diagnostics && result.diagnostics.length > 0) {
  console.log('Diagnostics:', result.diagnostics);
}
```

### Using the System Interface

Access file system operations through the sys object:

```typescript
import * as ts from 'typescript';

// Check if a file exists
if (ts.sys.fileExists('tsconfig.json')) {
  // Read the file
  const content = ts.sys.readFile('tsconfig.json', 'utf8');
  console.log(content);
}

// Get current directory
const cwd = ts.sys.getCurrentDirectory();
console.log('Current directory:', cwd);

// List directories
const dirs = ts.sys.getDirectories('./src');
console.log('Subdirectories:', dirs);

// Read directory with filters
const tsFiles = ts.sys.readDirectory(
  './src',
  ['.ts', '.tsx'], // extensions
  ['node_modules'], // exclude
  ['**/*']          // include
);
console.log('TypeScript files:', tsFiles);

// Write a file
ts.sys.writeFile('output.js', 'console.log("Hello");');

// Get version information
console.log('TypeScript version:', ts.version);
console.log('Major.Minor:', ts.versionMajorMinor);
```

### Using the Printer

Convert AST nodes back to source code:

```typescript
import * as ts from 'typescript';

// Create a simple AST
const factory = ts.factory;
const statement = factory.createVariableStatement(
  undefined,
  factory.createVariableDeclarationList(
    [
      factory.createVariableDeclaration(
        'message',
        undefined,
        factory.createKeywordTypeNode(ts.SyntaxKind.StringKeyword),
        factory.createStringLiteral('Hello, World!')
      )
    ],
    ts.NodeFlags.Const
  )
);

// Create a printer
const printer = ts.createPrinter({
  newLine: ts.NewLineKind.LineFeed,
  removeComments: false,
  omitTrailingSemicolon: false
});

// Create a minimal source file for context
const sourceFile = ts.createSourceFile(
  'temp.ts',
  '',
  ts.ScriptTarget.Latest,
  false,
  ts.ScriptKind.TS
);

// Print the node
const output = printer.printNode(
  ts.EmitHint.Unspecified,
  statement,
  sourceFile
);

console.log(output);
// Output: const message: string = "Hello, World!";
```

### Working with Text Positions

Convert between positions and line/character coordinates:

```typescript
import * as ts from 'typescript';

const sourceCode = `function hello() {
  console.log("Hello");
}`;

const sourceFile = ts.createSourceFile(
  'example.ts',
  sourceCode,
  ts.ScriptTarget.Latest,
  true
);

// Get line and character from position
const position = 25; // Position of 'console'
const lineChar = ts.getLineAndCharacterOfPosition(sourceFile, position);
console.log(`Position ${position} is at line ${lineChar.line + 1}, character ${lineChar.character + 1}`);
// Output: Position 25 is at line 2, character 3

// Get position from line and character
const pos = ts.getPositionOfLineAndCharacter(sourceFile, 1, 2); // Line 2, char 3 (0-indexed)
console.log(`Line 2, character 3 is at position ${pos}`);
// Output: Line 2, character 3 is at position 25
```

### Path Manipulation

Work with file paths:

```typescript
import * as ts from 'typescript';

// Normalize paths
const normalized = ts.normalizePath('src\\components\\Button.tsx');
console.log(normalized); // src/components/Button.tsx

// Combine paths
const fullPath = ts.combinePaths('/project', 'src', 'index.ts');
console.log(fullPath); // /project/src/index.ts

// Get directory path
const dirPath = ts.getDirectoryPath('/project/src/index.ts');
console.log(dirPath); // /project/src

// Get base file name
const fileName = ts.getBaseFileName('/project/src/index.ts');
console.log(fileName); // index.ts

// Get base name without extension
const baseName = ts.getBaseFileName('/project/src/index.ts', '.ts');
console.log(baseName); // index

// Check if path is relative
console.log(ts.pathIsRelative('./file.ts'));    // true
console.log(ts.pathIsRelative('../file.ts'));   // true
console.log(ts.pathIsRelative('/abs/file.ts')); // false
```

### Token Classification

Check token types and keywords:

```typescript
import * as ts from 'typescript';

// Check if a syntax kind is a keyword
console.log(ts.isKeyword(ts.SyntaxKind.FunctionKeyword)); // true
console.log(ts.isKeyword(ts.SyntaxKind.Identifier));      // false

// Check if a token is a modifier
console.log(ts.isModifier(ts.SyntaxKind.PublicKeyword));  // true
console.log(ts.isModifier(ts.SyntaxKind.StaticKeyword));  // true

// Check if a token is a contextual keyword
console.log(ts.isContextualKeyword(ts.SyntaxKind.AbstractKeyword)); // true
console.log(ts.isContextualKeyword(ts.SyntaxKind.AsyncKeyword));    // true

// Convert token to string
console.log(ts.tokenToString(ts.SyntaxKind.FunctionKeyword)); // "function"
console.log(ts.tokenToString(ts.SyntaxKind.PlusToken));       // "+"
```

### Character Classification

Classify individual characters:

```typescript
import * as ts from 'typescript';

// Check whitespace
console.log(ts.isWhiteSpaceLike(' '.charCodeAt(0)));  // true
console.log(ts.isWhiteSpaceLike('\t'.charCodeAt(0))); // true
console.log(ts.isWhiteSpaceLike('\n'.charCodeAt(0))); // true

// Check line breaks
console.log(ts.isLineBreak('\n'.charCodeAt(0))); // true
console.log(ts.isLineBreak('\r'.charCodeAt(0))); // true
console.log(ts.isLineBreak(' '.charCodeAt(0)));  // false

// Check digits
console.log(ts.isDigit('5'.charCodeAt(0))); // true
console.log(ts.isDigit('a'.charCodeAt(0))); // false
```

### File Watching

Watch files and directories for changes (Node.js environment):

```typescript
import * as ts from 'typescript';

// Watch a single file
const fileWatcher = ts.sys.watchFile!(
  'src/index.ts',
  (fileName, eventKind) => {
    if (eventKind === ts.FileWatcherEventKind.Created) {
      console.log(`File created: ${fileName}`);
    } else if (eventKind === ts.FileWatcherEventKind.Changed) {
      console.log(`File changed: ${fileName}`);
    } else if (eventKind === ts.FileWatcherEventKind.Deleted) {
      console.log(`File deleted: ${fileName}`);
    }
  },
  250 // polling interval in ms
);

// Watch a directory recursively
const dirWatcher = ts.sys.watchDirectory!(
  'src',
  (fileName) => {
    console.log(`Directory change detected: ${fileName}`);
  },
  true // recursive
);

// Stop watching
// fileWatcher.close();
// dirWatcher.close();
```
