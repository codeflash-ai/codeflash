# Printer

AST printing utilities for converting nodes back to source code.

## Printer Interface

```typescript { .api }
interface Printer {
  printNode(hint: EmitHint, node: Node, sourceFile: SourceFile): string;
  printList<T extends Node>(
    format: ListFormat,
    list: NodeArray<T>,
    sourceFile: SourceFile
  ): string;
  printFile(sourceFile: SourceFile): string;
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

## Usage Example

```typescript
import * as ts from 'typescript';

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

const printer = ts.createPrinter({
  newLine: ts.NewLineKind.LineFeed,
  removeComments: false,
  omitTrailingSemicolon: false
});

const sourceFile = ts.createSourceFile(
  'temp.ts',
  '',
  ts.ScriptTarget.Latest,
  false,
  ts.ScriptKind.TS
);

const output = printer.printNode(
  ts.EmitHint.Unspecified,
  statement,
  sourceFile
);

console.log(output);
// Output: const message: string = "Hello, World!";
```

