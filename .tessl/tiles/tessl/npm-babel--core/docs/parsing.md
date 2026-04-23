# Code Parsing

JavaScript parsing functionality that converts source code into Abstract Syntax Trees (ASTs) using Babel's parser. Provides both synchronous and asynchronous parsing with extensive configuration options for different JavaScript dialects and syntax extensions.

## Capabilities

### String Parsing

Parse JavaScript code from strings into Babel ASTs.

```typescript { .api }
/**
 * Parse JavaScript code synchronously
 * @param code - JavaScript source code to parse
 * @param opts - Parsing options including syntax plugins and parser settings
 * @returns Babel AST (File node) or null if parsing fails
 */
function parseSync(code: string, opts?: InputOptions): ParseResult | null;

/**
 * Parse JavaScript code asynchronously
 * @param code - JavaScript source code to parse
 * @param opts - Parsing options including syntax plugins and parser settings
 * @returns Promise resolving to Babel AST (File node) or null
 */
function parseAsync(code: string, opts?: InputOptions): Promise<ParseResult | null>;

/**
 * Parse JavaScript code with callback (legacy API, deprecated in Babel 8)
 * @param code - JavaScript source code to parse
 * @param opts - Parsing options
 * @param callback - Callback function receiving error and AST result
 */
function parse(
  code: string,
  opts: InputOptions | null | undefined,
  callback: FileParseCallback
): void;
function parse(code: string, callback: FileParseCallback): void;

type ParseResult = import("@babel/types").File;
type FileParseCallback = (err: Error | null, ast: ParseResult | null) => void;
```

**Usage Examples:**

```typescript
import { parseSync, parseAsync } from "@babel/core";

// Basic parsing
const ast = parseSync(`
  function hello(name) {
    return \`Hello \${name}!\`;
  }
`, {
  sourceType: "module"
});

console.log(ast.type); // "File"
console.log(ast.program.type); // "Program"
console.log(ast.program.body[0].type); // "FunctionDeclaration"

// Parsing with TypeScript syntax
const tsAst = parseSync(`
  interface User {
    name: string;
    age: number;
  }
  
  const user: User = { name: "Alice", age: 30 };
`, {
  sourceType: "module",
  plugins: ["typescript"]
});

// Parsing with JSX syntax
const jsxAst = parseSync(`
  const Component = () => {
    return <div>Hello World</div>;
  };
`, {
  sourceType: "module", 
  plugins: ["jsx"]
});

// Asynchronous parsing
const asyncAst = await parseAsync(`
  async function getData() {
    const response = await fetch('/api/data');
    return response.json();
  }
`, {
  sourceType: "module",
  plugins: ["asyncGenerators"]
});
```

### Parser Configuration

Configure the parsing behavior with various options:

```typescript { .api }
interface ParseOptions {
  /** Source type: "script", "module", or "unambiguous" (default: "script") */
  sourceType?: "script" | "module" | "unambiguous";
  /** Filename for error reporting and source maps */
  filename?: string;
  /** Parser-specific options */
  parserOpts?: ParserOptions;
  /** Environment name for conditional parsing */
  envName?: string;
  /** Current working directory */
  cwd?: string;
  /** Root directory for config resolution */
  root?: string;
}

interface ParserOptions {
  /** Syntax plugins to enable */
  plugins?: ParserPlugin[];
  /** Source type override */
  sourceType?: "script" | "module" | "unambiguous";
  /** Allow import/export outside modules */
  allowImportExportEverywhere?: boolean;
  /** Allow return statements outside functions */
  allowReturnOutsideFunction?: boolean;
  /** Allow undeclared exports */
  allowUndeclaredExports?: boolean;
  /** Create parent references on AST nodes */
  createParenthesizedExpressions?: boolean;
  /** Track error recovery information */
  errorRecovery?: boolean;
  /** Add location information to nodes */
  ranges?: boolean;
  /** Include token list in result */
  tokens?: boolean;
  /** Strict mode parsing */
  strictMode?: boolean;
  /** Start line number (default: 1) */
  startLine?: number;
  /** Start column number (default: 0) */
  startColumn?: number;
}

type ParserPlugin = 
  | "jsx" 
  | "typescript" 
  | "flow" 
  | "decorators" 
  | "classProperties"
  | "classPrivateProperties"
  | "classPrivateMethods" 
  | "classStaticBlock"
  | "asyncGenerators"
  | "functionBind"
  | "exportDefaultFrom"
  | "exportNamespaceFrom"
  | "dynamicImport"
  | "nullishCoalescingOperator"
  | "optionalChaining"
  | "importMeta"
  | "topLevelAwait"
  | "importAssertions"
  | "importReflection"
  | "bigInt"
  | "optionalCatchBinding"
  | "throwExpressions"
  | "pipelineOperator"
  | "recordAndTuple"
  | "doExpressions"
  | "regexpUnicodeSets"
  | ["decorators", { decoratorsBeforeExport?: boolean }]
  | ["pipelineOperator", { proposal: "minimal" | "smart" | "fsharp" }]
  | ["recordAndTuple", { syntaxType: "bar" | "hash" }]
  | ["flow", { all?: boolean; enums?: boolean }]
  | ["typescript", { 
      dts?: boolean; 
      disallowAmbiguousJSXLike?: boolean;
      allowNamespaces?: boolean;
    }];
```

**Usage Examples:**

```typescript
import { parseSync } from "@babel/core";

// TypeScript with decorators
const decoratorAst = parseSync(`
  @Component({
    selector: 'app-example'
  })
  class ExampleComponent {
    @Input() value: string;
    
    @HostListener('click')
    onClick() {}
  }
`, {
  sourceType: "module",
  plugins: [
    "typescript",
    ["decorators", { decoratorsBeforeExport: true }]
  ]
});

// Flow type annotations
const flowAst = parseSync(`
  type User = {
    name: string,
    age: number
  };
  
  function greetUser(user: User): string {
    return \`Hello \${user.name}\`;
  }
`, {
  sourceType: "module",
  plugins: [["flow", { all: true }]]
});

// Modern JavaScript features
const modernAst = parseSync(`
  class APIClient {
    #baseUrl = 'https://api.example.com';
    
    async getData() {
      const response = await fetch(\`\${this.#baseUrl}/data\`);
      return response?.json() ?? null;
    }
    
    static {
      console.log('APIClient initialized');
    }
  }
`, {
  sourceType: "module",
  plugins: [
    "classPrivateProperties",
    "classPrivateMethods", 
    "classStaticBlock",
    "nullishCoalescingOperator",
    "optionalChaining",
    "topLevelAwait"
  ]
});
```

## AST Structure

The parsing result is a Babel AST with the following structure:

```typescript { .api }
interface File {
  type: "File";
  /** The program node containing all top-level statements */
  program: Program;
  /** Comments found in the source code */
  comments: Comment[];
  /** Tokens if tokens: true was specified */
  tokens?: Token[];
  /** Source location information */
  loc?: SourceLocation;
  /** Start and end positions */
  start?: number;
  end?: number;
}

interface Program {
  type: "Program";
  /** Top-level statements and declarations */
  body: Statement[];
  /** Directive nodes (like "use strict") */
  directives: Directive[];
  /** Source type that was detected/specified */
  sourceType: "script" | "module";
  /** Source location information */
  loc?: SourceLocation;
}

interface SourceLocation {
  /** Starting position */
  start: Position;
  /** Ending position */
  end: Position;
  /** Original filename */
  filename?: string;
  /** Identifier name for anonymous sources */
  identifierName?: string;
}

interface Position {
  /** Line number (1-based) */
  line: number;
  /** Column number (0-based) */
  column: number;
  /** Character index in source */
  index?: number;
}

interface Comment {
  type: "CommentBlock" | "CommentLine";
  /** Comment text content */
  value: string;
  /** Source location */
  loc?: SourceLocation;
  /** Start and end positions */
  start?: number;
  end?: number;
}
```

## Working with ASTs

Common patterns for working with parsed ASTs:

```typescript
import { parseSync } from "@babel/core";
import traverse from "@babel/traverse";
import * as t from "@babel/types";

const code = `
  function add(a, b) {
    return a + b;
  }
  
  const multiply = (x, y) => x * y;
`;

const ast = parseSync(code, { sourceType: "module" });

// Traverse the AST
traverse(ast, {
  // Visit all function declarations
  FunctionDeclaration(path) {
    console.log("Function name:", path.node.id.name);
    console.log("Parameter count:", path.node.params.length);
  },
  
  // Visit all arrow functions
  ArrowFunctionExpression(path) {
    console.log("Arrow function found");
    
    // Convert to regular function
    const params = path.node.params;
    const body = t.isExpression(path.node.body) 
      ? t.blockStatement([t.returnStatement(path.node.body)])
      : path.node.body;
      
    path.replaceWith(
      t.functionExpression(null, params, body)
    );
  }
});

// Check node types
traverse(ast, {
  enter(path) {
    if (t.isIdentifier(path.node)) {
      console.log("Identifier:", path.node.name);
    }
    if (t.isStringLiteral(path.node)) {
      console.log("String:", path.node.value);
    }
  }
});
```

## Error Handling

Parse functions may throw errors for invalid syntax:

```typescript
import { parseSync } from "@babel/core";

try {
  const ast = parseSync("const x = ;", {
    sourceType: "module"
  });
} catch (error) {
  if (error.code === "BABEL_PARSE_ERROR") {
    console.error("Parse error:", error.message);
    console.error("Location:", error.loc); // { line: 1, column: 10 }
    console.error("Position:", error.pos); // Character position
  }
}

// Handle missing plugins
try {
  const tsAst = parseSync("const x: number = 42;", {
    sourceType: "module"
    // Missing "typescript" plugin
  });
} catch (error) {
  console.error("Missing plugin:", error.message);
  // "This experimental syntax requires enabling the parser plugin: 'typescript'"
}
```

## Integration with Other Babel APIs

Parsed ASTs can be used with other Babel functions:

```typescript
import { parseSync, transformFromAstSync, traverse } from "@babel/core";

// Parse -> Modify -> Transform workflow
const code = `const greeting = name => \`Hello \${name}\`;`;

// 1. Parse to AST
const ast = parseSync(code, { 
  sourceType: "module",
  plugins: ["templateLiterals"]
});

// 2. Modify AST
traverse(ast, {
  TemplateLiteral(path) {
    // Convert template literal to concatenation
    // This is just an example - in practice use appropriate plugins
  }
});

// 3. Transform to code
const result = transformFromAstSync(ast, code, {
  presets: ["@babel/preset-env"]
});

console.log(result.code);
```