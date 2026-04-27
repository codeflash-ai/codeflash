# Emit Helpers

Utilities for requesting runtime helpers during transformation.

## Emit Helper Interfaces

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

## Emit Hints and Flags

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

## Source Map Utilities

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

## Usage Example

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

