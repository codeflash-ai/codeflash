# Transformation

AST transformation pipeline for custom transformers and code generation. The transformation system provides a powerful API for programmatically modifying TypeScript AST nodes during compilation.

## Capabilities

### Core Transformation Types

Basic types for creating custom transformers.

```typescript { .api }
/**
 * A function that is used to initialize and return a Transformer callback,
 * which in turn will be used to transform one or more nodes.
 */
type TransformerFactory<T extends Node> = (
  context: TransformationContext
) => Transformer<T>;

/**
 * A function that transforms a node.
 */
type Transformer<T extends Node> = (node: T) => T;

/**
 * A function that accepts and possibly transforms a node.
 */
type Visitor<
  TIn extends Node = Node,
  TOut extends Node | undefined = TIn | undefined
> = (node: TIn) => VisitResult<TOut>;

type VisitResult<T extends Node | undefined> = T | readonly Node[];
```

### TransformationContext Interface

Context object provided to transformers with factory access and lifecycle management.

```typescript { .api }
interface CoreTransformationContext {
  readonly factory: NodeFactory;
  /** Gets the compiler options supplied to the transformer. */
  getCompilerOptions(): CompilerOptions;
  /** Starts a new lexical environment. */
  startLexicalEnvironment(): void;
  /** Suspends the current lexical environment, usually after visiting a parameter list. */
  suspendLexicalEnvironment(): void;
  /** Resumes a suspended lexical environment, usually before visiting a function body. */
  resumeLexicalEnvironment(): void;
  /** Ends a lexical environment, returning any declarations. */
  endLexicalEnvironment(): Statement[] | undefined;
  /** Hoists a function declaration to the containing scope. */
  hoistFunctionDeclaration(node: FunctionDeclaration): void;
  /** Hoists a variable declaration to the containing scope. */
  hoistVariableDeclaration(node: Identifier): void;
}

interface TransformationContext extends CoreTransformationContext {
  /** Records a request for a non-scoped emit helper in the current context. */
  requestEmitHelper(helper: EmitHelper): void;
  /** Gets and resets the requested non-scoped emit helpers. */
  readEmitHelpers(): EmitHelper[] | undefined;
  /** Enables expression substitutions in the pretty printer for the provided SyntaxKind. */
  enableSubstitution(kind: SyntaxKind): void;
  /** Determines whether expression substitutions are enabled for the provided node. */
  isSubstitutionEnabled(node: Node): boolean;
  /**
   * Hook used by transformers to substitute expressions just before they
   * are emitted by the pretty printer.
   *
   * NOTE: Transformation hooks should only be modified during Transformer initialization,
   * before returning the NodeTransformer callback.
   */
  onSubstituteNode: (hint: EmitHint, node: Node) => Node;
  /**
   * Enables before/after emit notifications in the pretty printer for the provided
   * SyntaxKind.
   */
  enableEmitNotification(kind: SyntaxKind): void;
  /**
   * Determines whether before/after emit notifications should be raised in the pretty
   * printer when it emits a node.
   */
  isEmitNotificationEnabled(node: Node): boolean;
  /**
   * Hook used to allow transformers to capture state before or after
   * the printer emits a node.
   *
   * NOTE: Transformation hooks should only be modified during Transformer initialization,
   * before returning the NodeTransformer callback.
   */
  onEmitNode: (
    hint: EmitHint,
    node: Node,
    emitCallback: (hint: EmitHint, node: Node) => void
  ) => void;
}
```

### CustomTransformers Interface

Configuration for custom transformer pipeline stages.

```typescript { .api }
type CustomTransformerFactory = (
  context: TransformationContext
) => CustomTransformer;

interface CustomTransformer {
  transformSourceFile(node: SourceFile): SourceFile;
  transformBundle(node: Bundle): Bundle;
}

interface CustomTransformers {
  /** Custom transformers to evaluate before built-in .js transformations. */
  before?: (TransformerFactory<SourceFile> | CustomTransformerFactory)[];
  /** Custom transformers to evaluate after built-in .js transformations. */
  after?: (TransformerFactory<SourceFile> | CustomTransformerFactory)[];
  /** Custom transformers to evaluate after built-in .d.ts transformations. */
  afterDeclarations?: (
    | TransformerFactory<Bundle | SourceFile>
    | CustomTransformerFactory
  )[];
}
```

### TransformationResult Interface

Result object returned from transformation operations.

```typescript { .api }
interface TransformationResult<T extends Node> {
  /** Gets the transformed source files. */
  transformed: T[];
  /** Gets diagnostics for the transformation. */
  diagnostics?: DiagnosticWithLocation[];
  /**
   * Gets a substitute for a node, if one is available; otherwise, returns the original node.
   *
   * @param hint A hint as to the intended usage of the node.
   * @param node The node to substitute.
   */
  substituteNode(hint: EmitHint, node: Node): Node;
  /**
   * Emits a node with possible notification.
   *
   * @param hint A hint as to the intended usage of the node.
   * @param node The node to emit.
   * @param emitCallback A callback used to emit the node.
   */
  emitNodeWithNotification(
    hint: EmitHint,
    node: Node,
    emitCallback: (hint: EmitHint, node: Node) => void
  ): void;
  /**
   * Indicates if a given node needs an emit notification
   *
   * @param node The node to emit.
   */
  isEmitNotificationEnabled?(node: Node): boolean;
  /**
   * Clean up EmitNode entries on any parse-tree nodes.
   */
  dispose(): void;
}
```

### Transform Function

Main entry point for applying transformers to source files.

```typescript { .api }
/**
 * Transform one or more nodes using the supplied transformers.
 * @param source A single Node or an array of Node objects.
 * @param transformers An array of TransformerFactory callbacks used to process the transformation.
 * @param compilerOptions Optional compiler options.
 */
function transform<T extends Node>(
  source: T | T[],
  transformers: TransformerFactory<T>[],
  compilerOptions?: CompilerOptions
): TransformationResult<T>;
```

### Visitor Utilities

Functions for traversing and transforming AST nodes.

```typescript { .api }
/**
 * Visits a Node using the supplied visitor, possibly returning a new Node in its place.
 *
 * @param node The Node to visit.
 * @param visitor The callback used to visit the Node.
 * @param test A callback to execute to verify the Node is valid.
 * @param lift An optional callback to execute to lift a NodeArray into a valid Node.
 */
function visitNode<
  TIn extends Node | undefined,
  TVisited extends Node | undefined,
  TOut extends Node
>(
  node: TIn,
  visitor: Visitor<NonNullable<TIn>, TVisited>,
  test: (node: Node) => node is TOut,
  lift?: (node: readonly Node[]) => Node
): TOut | (TIn & undefined) | (TVisited & undefined);

function visitNode<
  TIn extends Node | undefined,
  TVisited extends Node | undefined
>(
  node: TIn,
  visitor: Visitor<NonNullable<TIn>, TVisited>,
  test?: (node: Node) => boolean,
  lift?: (node: readonly Node[]) => Node
): Node | (TIn & undefined) | (TVisited & undefined);

/**
 * Visits a NodeArray using the supplied visitor, possibly returning a new NodeArray in its place.
 *
 * @param nodes The NodeArray to visit.
 * @param visitor The callback used to visit a Node.
 * @param test A node test to execute for each node.
 * @param start An optional value indicating the starting offset at which to start visiting.
 * @param count An optional value indicating the maximum number of nodes to visit.
 */
function visitNodes<
  TIn extends Node,
  TInArray extends NodeArray<TIn> | undefined,
  TOut extends Node
>(
  nodes: TInArray,
  visitor: Visitor<TIn, Node | undefined>,
  test: (node: Node) => node is TOut,
  start?: number,
  count?: number
): NodeArray<TOut> | (TInArray & undefined);

function visitNodes<
  TIn extends Node,
  TInArray extends NodeArray<TIn> | undefined
>(
  nodes: TInArray,
  visitor: Visitor<TIn, Node | undefined>,
  test?: (node: Node) => boolean,
  start?: number,
  count?: number
): NodeArray<Node> | (TInArray & undefined);

/**
 * Visits each child of a Node using the supplied visitor, possibly returning a new Node
 * of the same kind in its place.
 *
 * @param node The Node whose children will be visited.
 * @param visitor The callback used to visit each child.
 * @param context A lexical environment context for the visitor.
 */
function visitEachChild<T extends Node>(
  node: T,
  visitor: Visitor,
  context: TransformationContext | undefined
): T;

function visitEachChild<T extends Node>(
  node: T | undefined,
  visitor: Visitor,
  context: TransformationContext | undefined,
  nodesVisitor?: typeof visitNodes,
  tokenVisitor?: Visitor
): T | undefined;

/**
 * Starts a new lexical environment and visits a statement list, ending the lexical
 * environment and merging hoisted declarations upon completion.
 */
function visitLexicalEnvironment(
  statements: NodeArray<Statement>,
  visitor: Visitor,
  context: TransformationContext,
  start?: number,
  ensureUseStrict?: boolean,
  nodesVisitor?: NodesVisitor
): NodeArray<Statement>;

/**
 * Starts a new lexical environment and visits a parameter list, suspending the lexical
 * environment upon completion.
 */
function visitParameterList(
  nodes: NodeArray<ParameterDeclaration>,
  visitor: Visitor,
  context: TransformationContext,
  nodesVisitor?: NodesVisitor
): NodeArray<ParameterDeclaration>;

function visitParameterList(
  nodes: NodeArray<ParameterDeclaration> | undefined,
  visitor: Visitor,
  context: TransformationContext,
  nodesVisitor?: NodesVisitor
): NodeArray<ParameterDeclaration> | undefined;

/**
 * Resumes a suspended lexical environment and visits a function body, ending the lexical
 * environment and merging hoisted declarations upon completion.
 */
function visitFunctionBody(
  node: FunctionBody,
  visitor: Visitor,
  context: TransformationContext
): FunctionBody;

function visitFunctionBody(
  node: FunctionBody | undefined,
  visitor: Visitor,
  context: TransformationContext
): FunctionBody | undefined;

function visitFunctionBody(
  node: ConciseBody,
  visitor: Visitor,
  context: TransformationContext
): ConciseBody;
```

### Emit Helpers

Utilities for requesting runtime helpers during transformation.

```typescript { .api }
interface EmitHelperBase {
  readonly name: string;
  readonly scoped: boolean;
  readonly text: string | ((node: EmitHelperUniqueNameCallback) => string);
  readonly priority?: number;
  readonly dependencies?: EmitHelper[];
}

interface ScopedEmitHelper extends EmitHelperBase {
  readonly scoped: true;
}

interface UnscopedEmitHelper extends EmitHelperBase {
  readonly scoped: false;
  readonly text: string;
}

type EmitHelper = ScopedEmitHelper | UnscopedEmitHelper;

type EmitHelperUniqueNameCallback = (name: string) => string;

function addEmitHelper<T extends Node>(node: T, helper: EmitHelper): T;

function addEmitHelpers<T extends Node>(
  node: T,
  helpers: EmitHelper[] | undefined
): T;
```

### Emit Hints and Flags

Enums for controlling emit behavior.

```typescript { .api }
enum EmitHint {
  SourceFile = 0,
  Expression = 1,
  IdentifierName = 2,
  MappedTypeParameter = 3,
  Unspecified = 4,
  EmbeddedStatement = 5,
  JsxAttributeValue = 6,
  ImportTypeNodeAttributes = 7
}

enum EmitFlags {
  None = 0,
  SingleLine = 1,
  MultiLine = 2,
  AdviseOnEmitNode = 4,
  NoSubstitution = 8,
  CapturesThis = 16,
  NoLeadingSourceMap = 32,
  NoTrailingSourceMap = 64,
  NoSourceMap = 96,
  NoNestedSourceMaps = 128,
  NoTokenLeadingSourceMaps = 256,
  NoTokenTrailingSourceMaps = 512,
  NoTokenSourceMaps = 768,
  NoLeadingComments = 1024,
  NoTrailingComments = 2048,
  NoComments = 3072,
  NoNestedComments = 4096,
  HelperName = 8192,
  ExportName = 16384,
  LocalName = 32768,
  InternalName = 65536,
  Indented = 131072,
  NoIndentation = 262144,
  AsyncFunctionBody = 524288,
  ReuseTempVariableScope = 1048576,
  CustomPrologue = 2097152,
  NoHoisting = 4194304,
  Iterator = 8388608,
  NoAsciiEscaping = 16777216
}

function setEmitFlags<T extends Node>(node: T, emitFlags: EmitFlags): T;
```

### Source Map Utilities

Functions for managing source map ranges during transformation.

```typescript { .api }
interface SourceMapRange extends TextRange {
  source?: SourceMapSource;
}

interface SourceMapSource {
  fileName: string;
  text: string;
  skipTrivia?: (pos: number) => number;
  getLineAndCharacterOfPosition(pos: number): LineAndCharacter;
}

function createSourceMapSource(
  fileName: string,
  text: string,
  skipTrivia?: (pos: number) => number
): SourceMapSource;

function getSourceMapRange(node: Node): SourceMapRange;

function setSourceMapRange<T extends Node>(
  node: T,
  range: SourceMapRange | undefined
): T;

function getTokenSourceMapRange(
  node: Node,
  token: SyntaxKind
): SourceMapRange | undefined;

function setTokenSourceMapRange<T extends Node>(
  node: T,
  token: SyntaxKind,
  range: SourceMapRange | undefined
): T;
```

## Usage Examples

### Creating a Simple Transformer

Transform all string literals to uppercase:

```typescript
import * as ts from 'typescript';

// Create a transformer factory
function uppercaseStringsTransformer(): ts.TransformerFactory<ts.SourceFile> {
  return (context: ts.TransformationContext) => {
    // Return the transformer function
    return (sourceFile: ts.SourceFile) => {
      // Visitor function for each node
      const visitor = (node: ts.Node): ts.Node => {
        // If it's a string literal, transform it
        if (ts.isStringLiteral(node)) {
          return context.factory.createStringLiteral(
            node.text.toUpperCase()
          );
        }
        // Otherwise, continue visiting child nodes
        return ts.visitEachChild(node, visitor, context);
      };

      // Start visiting from the source file
      return ts.visitNode(sourceFile, visitor) as ts.SourceFile;
    };
  };
}

// Use the transformer
const source = ts.createSourceFile(
  'test.ts',
  'const message = "hello world";',
  ts.ScriptTarget.Latest
);

const result = ts.transform(source, [uppercaseStringsTransformer()]);
const transformedSourceFile = result.transformed[0];

// Print the result
const printer = ts.createPrinter();
const output = printer.printFile(transformedSourceFile);
console.log(output); // const message = "HELLO WORLD";

// Clean up
result.dispose();
```

### Using Custom Transformers with Program

Integrate custom transformers into the compilation process:

```typescript
import * as ts from 'typescript';

const customTransformers: ts.CustomTransformers = {
  before: [uppercaseStringsTransformer()],
  after: [],
  afterDeclarations: []
};

const program = ts.createProgram({
  rootNames: ['input.ts'],
  options: { target: ts.ScriptTarget.ES2020 }
});

// Emit with custom transformers
program.emit(
  undefined, // targetSourceFile
  undefined, // writeFile
  undefined, // cancellationToken
  false,     // emitOnlyDtsFiles
  customTransformers
);
```

### Advanced Transformer with Lexical Environment

Transform async functions and track hoisted variables:

```typescript
import * as ts from 'typescript';

function asyncTransformer(): ts.TransformerFactory<ts.SourceFile> {
  return (context: ts.TransformationContext) => {
    return (sourceFile: ts.SourceFile) => {
      const visitor = (node: ts.Node): ts.Node => {
        if (ts.isFunctionDeclaration(node) && node.modifiers?.some(
          m => m.kind === ts.SyntaxKind.AsyncKeyword
        )) {
          // Start new lexical environment for the function
          context.startLexicalEnvironment();

          // Visit function body
          const visitedBody = ts.visitEachChild(node.body!, visitor, context);

          // Get hoisted declarations
          const hoisted = context.endLexicalEnvironment();

          // Create new function with hoisted declarations
          if (hoisted && hoisted.length > 0) {
            const newBody = context.factory.updateBlock(
              node.body!,
              [...hoisted, ...visitedBody.statements]
            );
            return context.factory.updateFunctionDeclaration(
              node,
              node.modifiers,
              node.asteriskToken,
              node.name,
              node.typeParameters,
              node.parameters,
              node.type,
              newBody
            );
          }
        }

        return ts.visitEachChild(node, visitor, context);
      };

      return ts.visitNode(sourceFile, visitor) as ts.SourceFile;
    };
  };
}
```

### Using Emit Helpers

Request runtime helper functions during transformation:

```typescript
import * as ts from 'typescript';

function spreadTransformer(): ts.TransformerFactory<ts.SourceFile> {
  return (context: ts.TransformationContext) => {
    return (sourceFile: ts.SourceFile) => {
      const visitor = (node: ts.Node): ts.Node => {
        if (ts.isSpreadElement(node)) {
          // Request __spread emit helper
          context.requestEmitHelper({
            name: '__spread',
            scoped: false,
            text: 'var __spread = ...'
          } as ts.UnscopedEmitHelper);
        }

        return ts.visitEachChild(node, visitor, context);
      };

      return ts.visitNode(sourceFile, visitor) as ts.SourceFile;
    };
  };
}
```
