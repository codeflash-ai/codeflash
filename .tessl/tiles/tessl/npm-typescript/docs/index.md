# TypeScript Compiler API

TypeScript v5.9.3 provides a comprehensive compiler API with 1,683+ public API members for programmatically analyzing, transforming, and emitting JavaScript code.

## Quick Start

### Installation and Import

```typescript
import * as ts from 'typescript';
// or
const ts = require('typescript');
```

### Essential Pattern

```typescript { .api }
import * as ts from 'typescript';

// 1. Parse source code into AST
const sourceFile = ts.createSourceFile(
  'example.ts',
  'const x: number = 42;',
  ts.ScriptTarget.Latest,
  true
);

// 2. Create program for type checking
const program = ts.createProgram({
  rootNames: ['example.ts'],
  options: { target: ts.ScriptTarget.ES2020 }
});

// 3. Get type checker
const checker = program.getTypeChecker();

// 4. Traverse AST
function visit(node: ts.Node) {
  console.log(ts.SyntaxKind[node.kind]);
  ts.forEachChild(node, visit);
}
visit(sourceFile);

// 5. Emit JavaScript
const emitResult = program.emit();
```

## Common Tasks

### Task: Parse and Analyze Code

**Goal**: Parse TypeScript source code and analyze its structure.

**Pattern**:
1. Use `createSourceFile()` to parse code into AST
2. Use `createProgram()` for type checking
3. Use `program.getTypeChecker()` for type analysis
4. Traverse AST with `forEachChild()` or `visitNode()`

**APIs**:
- `createSourceFile()` - Parse source text into AST
- `createProgram()` - Create compilation program
- `program.getTypeChecker()` - Get type checker for semantic analysis
- `forEachChild()` - Traverse AST nodes

**See**: [Core Compilation APIs](./core/compilation.md) | [AST Nodes](./ast/nodes.md) | [Type System](./types/type-checker.md)

### Task: Transform Code

**Goal**: Modify TypeScript code programmatically.

**Pattern**:
1. Create transformer factory function
2. Use `transform()` or integrate with `program.emit()` via `CustomTransformers`
3. Use `TransformationContext.factory` to create/modify nodes
4. Use `visitEachChild()` to traverse and transform

**APIs**:
- `transform()` - Apply transformers to source files
- `TransformerFactory` - Create custom transformers
- `TransformationContext.factory` - Node factory for transformations
- `visitEachChild()` - Visit and transform child nodes

**See**: [Transformation](./transformation/transformers.md) | [Node Factory](./ast/factory.md)

### Task: Generate Code

**Goal**: Programmatically generate TypeScript code.

**Pattern**:
1. Use `factory` constant to create AST nodes
2. Build complete AST structure
3. Use `Printer` to convert AST to source code
4. Or use `program.emit()` to generate JavaScript

**APIs**:
- `factory` - Node factory (300+ methods)
- `createPrinter()` - Create printer for AST to source conversion
- `program.emit()` - Emit JavaScript output

**See**: [Node Factory](./ast/factory.md) | [Utilities](./utilities/printer.md)

### Task: Provide IDE Features

**Goal**: Implement editor features like completions, diagnostics, refactorings.

**Pattern**:
1. Create `LanguageServiceHost` implementation
2. Use `createLanguageService()` to create service
3. Call methods like `getCompletionsAtPosition()`, `getQuickInfoAtPosition()`
4. Update host when files change

**APIs**:
- `createLanguageService()` - Create language service
- `LanguageService` - 100+ methods for IDE features
- `LanguageServiceHost` - Host interface for file operations

**See**: [Language Service](./language-service/api.md)

### Task: Resolve Modules

**Goal**: Resolve module imports and type references.

**Pattern**:
1. Use `resolveModuleName()` for module resolution
2. Use `resolveTypeReferenceDirective()` for type references
3. Implement `ModuleResolutionHost` for file system operations
4. Use `ModuleResolutionCache` for performance

**APIs**:
- `resolveModuleName()` - Resolve module imports
- `resolveTypeReferenceDirective()` - Resolve type references
- `ModuleResolutionHost` - File system interface

**See**: [Module Resolution](./core/module-resolution.md)

### Task: Build Projects

**Goal**: Build TypeScript projects with incremental compilation.

**Pattern**:
1. Use `createSolutionBuilder()` for multi-project builds
2. Configure project references in tsconfig.json
3. Use incremental compilation with `BuilderProgram`
4. Handle diagnostics and emit results

**APIs**:
- `createSolutionBuilder()` - Multi-project builds
- `createIncrementalProgram()` - Incremental compilation
- `BuilderProgram` - Incremental program interface

**See**: [Solution Builder](./core/solution-builder.md) | [Core Compilation](./core/compilation.md)

## Core Architecture

TypeScript's compiler API consists of these major subsystems:

1. **Parser & Scanner**: Lexical analysis and AST construction
2. **Binder**: Symbol creation and scope analysis
3. **Type Checker**: Type inference, checking, and semantic analysis
4. **Transformer**: AST transformation pipeline
5. **Emitter**: JavaScript and declaration file generation
6. **Language Service**: Editor integration (completions, diagnostics, refactorings)
7. **Module Resolution**: Resolving imports and type references
8. **Server Protocol**: TSServer communication

## API Reference by Category

### Core Compilation
- [Compilation APIs](./core/compilation.md) - Program creation, compilation, emit
- [Module Resolution](./core/module-resolution.md) - Import and type resolution
- [Solution Builder](./core/solution-builder.md) - Multi-project builds
- [Configuration Parsing](./core/config.md) - tsconfig.json parsing

### AST Manipulation
- [AST Nodes](./ast/nodes.md) - Node types and traversal (359 syntax kinds)
- [Node Factory](./ast/factory.md) - Creating AST nodes (300+ methods)
- [Scanner and Parser](./ast/scanner-parser.md) - Lexical analysis and parsing
- [Type Guards](./ast/type-guards.md) - Type guard functions for all node types

### Type System
- [Type Checker](./types/type-checker.md) - TypeChecker API (150+ methods)
- [Types and Symbols](./types/types-symbols.md) - Type and Symbol interfaces
- [Type Utilities](./types/utilities.md) - Type analysis utilities
- [Diagnostics](./types/diagnostics.md) - Error reporting and diagnostics

### Code Transformation
- [Transformers](./transformation/transformers.md) - AST transformation pipeline
- [Transformation Context](./transformation/context.md) - Transformation context API
- [Emit Helpers](./transformation/emit-helpers.md) - Runtime helper utilities

### Editor Integration
- [Language Service](./language-service/api.md) - IDE features (100+ methods)
- [Completions](./language-service/completions.md) - Code completion APIs
- [Navigation](./language-service/navigation.md) - Go to definition, find references
- [Refactorings](./language-service/refactorings.md) - Code refactoring APIs
- [Formatting](./language-service/formatting.md) - Code formatting APIs
- [TSServer Protocol](./language-service/tsserver.md) - Language server protocol

### Utilities
- [System Interface](./utilities/system.md) - File system operations
- [Printer](./utilities/printer.md) - AST to source code conversion
- [Transpilation](./utilities/transpilation.md) - Quick TypeScript to JavaScript conversion
- [Path Utilities](./utilities/paths.md) - Path manipulation functions
- [Text Utilities](./utilities/text.md) - Text and position utilities

## Essential APIs

### Program Creation

```typescript { .api }
function createProgram(
  rootNames: readonly string[],
  options: CompilerOptions,
  host?: CompilerHost,
  oldProgram?: Program,
  configFileParsingDiagnostics?: readonly Diagnostic[]
): Program;

interface Program {
  getRootFileNames(): readonly string[];
  getSourceFiles(): readonly SourceFile[];
  getSourceFile(fileName: string): SourceFile | undefined;
  getTypeChecker(): TypeChecker;
  emit(
    targetSourceFile?: SourceFile,
    writeFile?: WriteFileCallback,
    cancellationToken?: CancellationToken,
    emitOnlyDtsFiles?: boolean,
    customTransformers?: CustomTransformers
  ): EmitResult;
  getSemanticDiagnostics(sourceFile?: SourceFile): readonly Diagnostic[];
  getSyntacticDiagnostics(sourceFile?: SourceFile): readonly Diagnostic[];
  getCompilerOptions(): CompilerOptions;
}
```

### AST Nodes and Traversal

```typescript { .api }
enum SyntaxKind {
  Unknown = 0,
  Identifier = 80,
  // ... 359 syntax kinds total
}

interface Node {
  readonly kind: SyntaxKind;
  readonly flags: NodeFlags;
  readonly parent: Node;
  getSourceFile(): SourceFile;
  getText(sourceFile?: SourceFile): string;
  forEachChild<T>(cbNode: (node: Node) => T | undefined): T | undefined;
}

function forEachChild<T>(
  node: Node,
  cbNode: (node: Node) => T | undefined
): T | undefined;
```

### Type System

```typescript { .api }
interface TypeChecker {
  getTypeAtLocation(node: Node): Type;
  getSymbolAtLocation(node: Node): Symbol | undefined;
  getTypeOfSymbolAtLocation(symbol: Symbol, node: Node): Type;
  getPropertiesOfType(type: Type): Symbol[];
  getSignaturesOfType(type: Type, kind: SignatureKind): readonly Signature[];
  typeToString(type: Type, enclosingDeclaration?: Node, flags?: TypeFormatFlags): string;
  // ... 150+ more methods
}

interface Type {
  flags: TypeFlags;
  symbol: Symbol;
  getProperties(): Symbol[];
  getCallSignatures(): readonly Signature[];
}
```

### Node Factory

```typescript { .api }
const factory: NodeFactory;

interface NodeFactory {
  createSourceFile(statements: readonly Statement[], endOfFileToken: EndOfFileToken, flags: NodeFlags): SourceFile;
  createIdentifier(text: string): Identifier;
  createVariableStatement(modifiers: readonly Modifier[] | undefined, declarationList: VariableDeclarationList): VariableStatement;
  createFunctionDeclaration(modifiers: readonly ModifierLike[] | undefined, asteriskToken: AsteriskToken | undefined, name: string | Identifier | undefined, typeParameters: readonly TypeParameterDeclaration[] | undefined, parameters: readonly ParameterDeclaration[], type: TypeNode | undefined, body: Block | undefined): FunctionDeclaration;
  // ... 300+ more factory functions
}
```

### Language Service

```typescript { .api }
function createLanguageService(
  host: LanguageServiceHost,
  documentRegistry?: DocumentRegistry,
  syntaxOnlyOrLanguageServiceMode?: boolean | LanguageServiceMode
): LanguageService;

interface LanguageService {
  getCompletionsAtPosition(fileName: string, position: number, options?: GetCompletionsAtPositionOptions): WithMetadata<CompletionInfo> | undefined;
  getQuickInfoAtPosition(fileName: string, position: number): QuickInfo | undefined;
  getDefinitionAtPosition(fileName: string, position: number): readonly DefinitionInfo[] | undefined;
  findReferences(fileName: string, position: number): ReferencedSymbol[] | undefined;
  getApplicableRefactors(fileName: string, positionOrRange: number | TextRange, preferences?: UserPreferences): ApplicableRefactorInfo[];
  // ... 100+ more methods
}
```

### Transformation

```typescript { .api }
type TransformerFactory<T extends Node> = (
  context: TransformationContext
) => Transformer<T>;

type Transformer<T extends Node> = (node: T) => T;

interface TransformationContext {
  factory: NodeFactory;
  getCompilerOptions(): CompilerOptions;
  hoistFunctionDeclaration(node: FunctionDeclaration): void;
  hoistVariableDeclaration(node: Identifier): void;
  requestEmitHelper(helper: EmitHelper): void;
  enableSubstitution(kind: SyntaxKind): void;
  enableEmitNotification(kind: SyntaxKind): void;
}

function transform<T extends Node>(
  source: T | T[],
  transformers: TransformerFactory<T>[],
  compilerOptions?: CompilerOptions
): TransformationResult<T>;
```

## Key Enumerations

- `SyntaxKind` (359 values) - All AST node types
- `TypeFlags` - Type classification flags
- `SymbolFlags` - Symbol classification flags
- `ScriptTarget` - JavaScript target versions
- `ModuleKind` - Module system types
- `ModuleResolutionKind` - Module resolution strategies
- `DiagnosticCategory` - Error, Warning, Suggestion, Message
- `NodeFlags` - Node metadata flags

## Version Information

TypeScript v5.9.3 includes:
- 64 enumerations
- 803 interfaces
- 269 type aliases
- 510 functions
- 21 constants
- 10 classes
- 6 namespaces

## Getting Help

For specific tasks:
- **Parsing code**: See [Scanner and Parser](./ast/scanner-parser.md)
- **Type checking**: See [Type System](./types/type-checker.md)
- **Code transformation**: See [Transformation](./transformation/transformers.md)
- **Building projects**: See [Solution Builder](./core/solution-builder.md)
- **IDE features**: See [Language Service](./language-service/api.md)
- **Module resolution**: See [Module Resolution](./core/module-resolution.md)

