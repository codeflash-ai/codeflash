# Transpilation

Fast transpilation APIs for converting TypeScript to JavaScript without full type checking.

## Transpilation APIs

```typescript { .api }
function transpileModule(input: string, transpileOptions: TranspileOptions): TranspileOutput;

function transpileDeclaration(input: string, transpileOptions: TranspileOptions): TranspileOutput;

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

## Usage Example

```typescript
import * as ts from 'typescript';

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

