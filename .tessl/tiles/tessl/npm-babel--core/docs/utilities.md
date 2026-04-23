# Utilities and Constants

Helper functions, constants, and re-exported APIs from the Babel ecosystem. Includes version information, file extensions, environment detection, plugin resolution, and access to the complete Babel toolchain.

## Capabilities

### Version and Constants

Version information and recommended file extensions for Babel processing.

```typescript { .api }
/**
 * Current version of @babel/core package
 */
const version: string;

/**
 * Recommended set of compilable file extensions
 * Not used in @babel/core directly, but meant as an easy source for tooling
 */
const DEFAULT_EXTENSIONS: readonly [".js", ".jsx", ".es6", ".es", ".mjs", ".cjs"];
```

**Usage Examples:**

```typescript
import { version, DEFAULT_EXTENSIONS } from "@babel/core";

console.log("Babel version:", version); // "7.26.10"

console.log("Default extensions:", DEFAULT_EXTENSIONS);
// [".js", ".jsx", ".es6", ".es", ".mjs", ".cjs"]

// Use in build tools
const shouldProcess = (filename) => {
  return DEFAULT_EXTENSIONS.some(ext => filename.endsWith(ext));
};

console.log(shouldProcess("app.js")); // true
console.log(shouldProcess("styles.css")); // false
```

### Environment Detection

Detect and resolve the current Babel environment.

```typescript { .api }
/**
 * Get the current Babel environment name
 * @param defaultValue - Default environment if none specified (default: "development")
 * @returns Environment name from BABEL_ENV, NODE_ENV, or default value
 */
function getEnv(defaultValue?: string): string;
```

**Usage Examples:**

```typescript
import { getEnv } from "@babel/core";

// Without environment variables set
console.log(getEnv()); // "development"
console.log(getEnv("production")); // "production"

// With NODE_ENV=test
process.env.NODE_ENV = "test";
console.log(getEnv()); // "test"

// With BABEL_ENV=staging (takes precedence over NODE_ENV)
process.env.BABEL_ENV = "staging";
console.log(getEnv()); // "staging"

// Use in configuration
const isProd = getEnv() === "production";
const isDev = getEnv() === "development";
```

### Plugin and Preset Resolution

Resolve plugin and preset file paths (legacy APIs for backward compatibility).

```typescript { .api }
/**
 * Resolve plugin file path (legacy API)
 * @param name - Plugin name or path
 * @param dirname - Directory to resolve from
 * @returns Resolved file path
 */
function resolvePlugin(name: string, dirname: string): string;

/**
 * Resolve preset file path (legacy API)
 * @param name - Preset name or path  
 * @param dirname - Directory to resolve from
 * @returns Resolved file path
 */
function resolvePreset(name: string, dirname: string): string;
```

**Usage Examples:**

```typescript
import { resolvePlugin, resolvePreset } from "@babel/core";

// Resolve official plugins
const pluginPath = resolvePlugin("@babel/plugin-transform-arrow-functions", __dirname);
console.log(pluginPath); // "/path/to/node_modules/@babel/plugin-transform-arrow-functions/lib/index.js"

// Resolve official presets
const presetPath = resolvePreset("@babel/preset-env", process.cwd());
console.log(presetPath); // "/path/to/node_modules/@babel/preset-env/lib/index.js"

// Resolve relative paths
const localPlugin = resolvePlugin("./plugins/custom-plugin", __dirname);
console.log(localPlugin); // "/current/dir/plugins/custom-plugin.js"

// Use in plugin loading
function loadPlugin(name, dirname) {
  try {
    const pluginPath = resolvePlugin(name, dirname);
    return require(pluginPath);
  } catch (error) {
    console.error(`Failed to load plugin ${name}:`, error.message);
  }
}
```

### External Helper Generation

Generate external Babel helper functions for runtime optimization.

```typescript { .api }
/**
 * Build external helper functions as a single module
 * @param whitelist - Array of helper names to include, or undefined for all
 * @param outputType - Output format: "global", "umd", "var", or function
 * @returns Generated helper code as string
 */
function buildExternalHelpers(
  whitelist?: string[],
  outputType?: "global" | "umd" | "var" | ((name: string) => string)
): string;
```

**Usage Examples:**

```typescript
import { buildExternalHelpers } from "@babel/core";

// Generate all helpers as global variables
const allHelpers = buildExternalHelpers(undefined, "global");
console.log(allHelpers);
// Output: Global helper functions for all Babel runtime helpers

// Generate specific helpers only
const specificHelpers = buildExternalHelpers([
  "_classCallCheck",
  "_createClass", 
  "_inherits"
], "umd");
console.log(specificHelpers);
// Output: UMD module with class-related helpers only

// Generate with custom output format
const customHelpers = buildExternalHelpers(
  ["_asyncToGenerator", "_awaitAsyncGenerator"],
  (name) => `window.BabelHelpers.${name.slice(1)}`
);
console.log(customHelpers);
// Output: Helpers assigned to window.BabelHelpers

// Use in build process
const fs = require("fs");
const helpers = buildExternalHelpers(undefined, "umd");
fs.writeFileSync("dist/babel-helpers.js", helpers);
```

## Re-exported APIs

Babel Core re-exports several essential APIs from other Babel packages for convenience.

### Babel Types

Complete AST node types and utilities from @babel/types.

```typescript { .api }
/**
 * Complete @babel/types API for AST manipulation
 * Includes all node builders, validators, and utilities
 */
import * as types from "@babel/types";

// Re-exported as namespace
export * as types from "@babel/types";
```

**Usage Examples:**

```typescript
import { types as t } from "@babel/core";
// or: import * as t from "@babel/core";

// Create AST nodes
const identifier = t.identifier("myVariable");
const stringLiteral = t.stringLiteral("Hello World");
const callExpression = t.callExpression(
  t.identifier("console.log"),
  [stringLiteral]
);

// Validate node types
if (t.isIdentifier(identifier)) {
  console.log("Name:", identifier.name);
}

if (t.isCallExpression(callExpression)) {
  console.log("Callee:", callExpression.callee);
  console.log("Arguments:", callExpression.arguments);
}

// Build complex structures
const functionDeclaration = t.functionDeclaration(
  t.identifier("greet"),
  [t.identifier("name")],
  t.blockStatement([
    t.returnStatement(
      t.templateLiteral(
        [
          t.templateElement({ raw: "Hello " }, false),
          t.templateElement({ raw: "!" }, true)
        ],
        [t.identifier("name")]
      )
    )
  ])
);
```

### Babel Traverse

AST traversal utilities from @babel/traverse.

```typescript { .api }
/**
 * Default traverse function for AST traversal
 */
export { default as traverse } from "@babel/traverse";

/**
 * Node path type for AST traversal
 */
export type { NodePath } from "@babel/traverse";

/**
 * Scope information type
 */
export type { Scope } from "@babel/traverse";

/**
 * Visitor pattern type for AST traversal
 */
export type Visitor<S = unknown> = import("@babel/traverse").Visitor<S>;
```

**Usage Examples:**

```typescript
import { traverse, parseSync, types as t } from "@babel/core";

const code = `
  function calculate(a, b) {
    const result = a + b;
    return result;
  }
`;

const ast = parseSync(code, { sourceType: "module" });

// Basic traversal
traverse(ast, {
  enter(path) {
    console.log("Entering:", path.node.type);
  },
  
  exit(path) {
    console.log("Exiting:", path.node.type);
  }
});

// Specific node visitors
traverse(ast, {
  FunctionDeclaration(path) {
    console.log("Function:", path.node.id.name);
    console.log("Parameters:", path.node.params.map(p => p.name));
  },
  
  VariableDeclarator(path) {
    if (t.isIdentifier(path.node.id)) {
      console.log("Variable:", path.node.id.name);
    }
  },
  
  BinaryExpression(path) {
    console.log("Operation:", path.node.operator);
    console.log("Left:", path.node.left);
    console.log("Right:", path.node.right);
  }
});

// Path manipulation
traverse(ast, {
  Identifier(path) {
    if (path.node.name === "result") {
      path.node.name = "output";
    }
  }
});
```

### Babel Template

Template string to AST conversion from @babel/template.

```typescript { .api }
/**
 * Template function for converting template strings to AST nodes
 */
export { default as template } from "@babel/template";
```

**Usage Examples:**

```typescript
import { template, traverse, parseSync } from "@babel/core";

// Create template functions
const buildRequire = template(`
  var %%importName%% = require(%%source%%);
`);

const buildClass = template.statement(`
  class %%className%% extends %%superClass%% {
    constructor(%%params%%) {
      super(%%args%%);
      %%body%%
    }
  }
`);

// Use templates to generate AST
const requireNode = buildRequire({
  importName: t.identifier("lodash"),
  source: t.stringLiteral("lodash")
});

const classNode = buildClass({
  className: t.identifier("MyComponent"),
  superClass: t.identifier("Component"),
  params: [t.identifier("props")],
  args: [t.identifier("props")], 
  body: [
    t.expressionStatement(
      t.assignmentExpression(
        "=",
        t.memberExpression(t.thisExpression(), t.identifier("state")),
        t.objectExpression([])
      )
    )
  ]
});

// Template with expressions
const buildConditional = template.expression(`
  %%test%% ? %%consequent%% : %%alternate%%
`);

const conditional = buildConditional({
  test: t.identifier("isLoggedIn"),
  consequent: t.stringLiteral("Welcome"),
  alternate: t.stringLiteral("Please log in")
});
```

### Babel Parser Token Types

Token types from @babel/parser for advanced parsing use cases.

```typescript { .api }
/**
 * Token types from Babel parser
 */
export { tokTypes } from "@babel/parser";
```

**Usage Examples:**

```typescript
import { tokTypes, parseSync } from "@babel/core";

// Parse with tokens
const ast = parseSync("const x = 42;", {
  sourceType: "module",
  tokens: true
});

// Check token types
if (ast.tokens) {
  ast.tokens.forEach(token => {
    if (token.type === tokTypes.name) {
      console.log("Identifier token:", token.value);
    } else if (token.type === tokTypes.num) {
      console.log("Number token:", token.value);
    } else if (token.type === tokTypes._const) {
      console.log("Const keyword token");
    }
  });
}

// Token type checking
console.log("Available token types:", Object.keys(tokTypes));
// ["num", "string", "name", "_const", "_let", "_var", ...]
```

## File Context Class

The File class provides transformation context and utilities, primarily used in plugin development and advanced transformation scenarios.

```typescript { .api }
/**
 * File transformation context class
 * Provides access to transformation state, metadata, and helper functions
 */
export { default as File } from "./transformation/file/file";

class File {
  /** Transformation options merged from config and parameters */
  opts: TransformationOptions;
  /** Variable declarations at file scope */
  declarations: { [name: string]: import("@babel/types").Identifier };
  /** Root program path for AST traversal */
  path: import("@babel/traverse").NodePath<import("@babel/types").Program>;
  /** Complete file AST including program and metadata */
  ast: import("@babel/types").File;
  /** Root scope for the file */
  scope: import("@babel/traverse").Scope;
  /** Metadata collected from plugins during transformation */
  metadata: { [pluginName: string]: any };
  /** Original source code string */
  code: string;
  /** Input source map if available */
  inputMap: import("convert-source-map").SourceMapConverter | null;
  
  /** Hub interface for plugin communication and utilities */
  hub: FileHub;
  
  /** Generate a unique identifier in file scope */
  generateUid(name: string): import("@babel/types").Identifier;
  /** Check if identifier is available in file scope */
  hasIdentifier(name: string): boolean;
  /** Add import declaration to file */
  addImport(source: string, importedName: string, localName?: string): import("@babel/types").Identifier;
  /** Add helper function import */
  addHelper(name: string): import("@babel/types").Identifier;
}

interface FileHub {
  /** Reference to the current file */
  file: File;
  /** Get current transformed code */
  getCode(): string;
  /** Get file root scope */
  getScope(): import("@babel/traverse").Scope;
  /** Add Babel helper function and return its identifier */
  addHelper(name: string): import("@babel/types").Identifier;
  /** Create error with location information */
  buildError<T extends import("@babel/types").Node>(
    node: T, 
    message: string, 
    constructor?: typeof Error
  ): Error;
}

interface TransformationOptions {
  /** Source filename */
  filename?: string;
  /** Source type */
  sourceType?: "script" | "module" | "unambiguous";
  /** Plugins to apply */
  plugins?: any[];
  /** Presets to apply */
  presets?: any[];
  /** Parser options */
  parserOpts?: any;
  /** Generator options */
  generatorOpts?: any;
  /** Additional transformation options */
  [key: string]: any;
}
```

**Usage in Plugin Development:**

```typescript
import { PluginObj, PluginPass } from "@babel/core";
import * as t from "@babel/types";

function myPlugin(): PluginObj {
  return {
    visitor: {
      Program(path, state: PluginPass) {
        // Access file context
        const file = state.file;
        
        // Get transformation options
        console.log("Filename:", file.opts.filename);
        console.log("Source type:", file.opts.sourceType);
        
        // Add helper function
        const helperIdentifier = file.addHelper("classCallCheck");
        
        // Generate unique identifier
        const uniqueId = file.generateUid("temp");
        
        // Add metadata for other plugins
        file.metadata.myPlugin = {
          processedNodes: 0,
          addedHelpers: [helperIdentifier.name]
        };
        
        // Access file scope
        const hasConsole = file.scope.hasBinding("console");
        console.log("Console available:", hasConsole);
      },
      
      ClassDeclaration(path, state: PluginPass) {
        const file = state.file;
        
        // Increment counter in metadata
        if (!file.metadata.myPlugin) {
          file.metadata.myPlugin = { processedNodes: 0 };
        }
        file.metadata.myPlugin.processedNodes++;
        
        // Create error with file context
        if (path.node.id === null) {
          throw file.hub.buildError(
            path.node,
            "Anonymous classes are not supported"
          );
        }
      }
    }
  };
}
```

The File class is essential for plugin authors who need to:

1. **Access transformation context** - filename, options, and configuration
2. **Manage helper functions** - automatically import required runtime helpers
3. **Generate unique identifiers** - avoid naming conflicts in transformed code
4. **Share data between plugins** - using the metadata system
5. **Handle errors with context** - provide meaningful error messages with location info
6. **Access file-level scope information** - understand available bindings and references