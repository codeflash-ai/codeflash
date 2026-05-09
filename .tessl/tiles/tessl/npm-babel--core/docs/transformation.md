# Code Transformation

Core JavaScript transformation functionality for converting modern JavaScript code into backward-compatible versions using Babel plugins and presets. Supports string-based, file-based, and AST-based transformation workflows.

## Capabilities

### String Transformation

Transform JavaScript code from strings with full plugin and preset support.

```typescript { .api }
/**
 * Transform JavaScript code synchronously
 * @param code - JavaScript source code to transform
 * @param opts - Transformation options including plugins, presets, and parser settings
 * @returns Transformation result with code, source map, and optional AST
 */
function transformSync(code: string, opts?: InputOptions): FileResult | null;

/**
 * Transform JavaScript code asynchronously  
 * @param code - JavaScript source code to transform
 * @param opts - Transformation options including plugins, presets, and parser settings
 * @returns Promise resolving to transformation result
 */
function transformAsync(code: string, opts?: InputOptions): Promise<FileResult | null>;

/**
 * Transform JavaScript code with callback (legacy API, deprecated in Babel 8)
 * @param code - JavaScript source code to transform
 * @param opts - Transformation options
 * @param callback - Callback function receiving error and result
 */
function transform(
  code: string,
  opts: InputOptions | null | undefined,
  callback: FileResultCallback
): void;
function transform(code: string, callback: FileResultCallback): void;

type FileResultCallback = (err: Error | null, result: FileResult | null) => void;
```

**Usage Examples:**

```typescript
import { transformSync, transformAsync } from "@babel/core";

// Synchronous transformation
const result = transformSync(`
  const getMessage = () => "Hello World";
  class User {
    constructor(name) {
      this.name = name;
    }
  }
`, {
  presets: ["@babel/preset-env"],
  plugins: ["@babel/plugin-transform-arrow-functions", "@babel/plugin-transform-classes"]
});

console.log(result.code);
// Output: ES5 compatible code

// Asynchronous transformation
const asyncResult = await transformAsync(`
  import { useState } from 'react';
  export const Component = () => <div>Hello</div>;
`, {
  presets: ["@babel/preset-react", "@babel/preset-env"],
  filename: "component.jsx"
});

console.log(asyncResult.code);
```

### File Transformation

Transform JavaScript files directly from the filesystem.

```typescript { .api }
/**
 * Transform JavaScript file synchronously
 * @param filename - Path to JavaScript file to transform
 * @param opts - Transformation options (filename will be added automatically)
 * @returns Transformation result with code, source map, and optional AST
 */
function transformFileSync(filename: string, opts?: InputOptions): FileResult | null;

/**
 * Transform JavaScript file asynchronously
 * @param filename - Path to JavaScript file to transform  
 * @param opts - Transformation options (filename will be added automatically)
 * @returns Promise resolving to transformation result
 */
function transformFileAsync(filename: string, opts?: InputOptions): Promise<FileResult | null>;

/**
 * Transform JavaScript file with callback
 * @param filename - Path to JavaScript file to transform
 * @param opts - Transformation options
 * @param callback - Callback function receiving error and result
 */
function transformFile(
  filename: string,
  opts: InputOptions | null | undefined,
  callback: FileResultCallback
): void;
function transformFile(filename: string, callback: FileResultCallback): void;
```

**Usage Examples:**

```typescript
import { transformFileSync, transformFileAsync } from "@babel/core";

// Synchronous file transformation
const result = transformFileSync("./src/app.js", {
  presets: ["@babel/preset-env"],
  sourceMaps: true
});

if (result) {
  console.log("Transformed:", result.code);
  console.log("Source map:", result.map);
}

// Asynchronous file transformation
const asyncResult = await transformFileAsync("./src/component.tsx", {
  presets: ["@babel/preset-typescript", "@babel/preset-react"],
  plugins: ["@babel/plugin-transform-runtime"]
});
```

### AST Transformation

Transform JavaScript code from existing Abstract Syntax Trees (ASTs).

```typescript { .api }
/**
 * Transform from AST synchronously
 * @param ast - Babel AST (File or Program node)
 * @param code - Original source code string for source map generation
 * @param opts - Transformation options
 * @returns Transformation result with code, source map, and optional AST
 */
function transformFromAstSync(
  ast: AstRoot, 
  code: string, 
  opts?: InputOptions
): FileResult | null;

/**
 * Transform from AST asynchronously
 * @param ast - Babel AST (File or Program node)
 * @param code - Original source code string for source map generation
 * @param opts - Transformation options
 * @returns Promise resolving to transformation result
 */
function transformFromAstAsync(
  ast: AstRoot, 
  code: string, 
  opts?: InputOptions
): Promise<FileResult | null>;

/**
 * Transform from AST with callback (legacy API, deprecated in Babel 8)
 * @param ast - Babel AST (File or Program node)
 * @param code - Original source code string
 * @param opts - Transformation options
 * @param callback - Callback function receiving error and result
 */
function transformFromAst(
  ast: AstRoot,
  code: string,
  opts: InputOptions | null | undefined,
  callback: FileResultCallback
): void;
function transformFromAst(
  ast: AstRoot, 
  code: string, 
  callback: FileResultCallback
): void;

type AstRoot = import("@babel/types").File | import("@babel/types").Program;
```

**Usage Examples:**

```typescript
import { parseSync, transformFromAstSync } from "@babel/core";

// Parse then transform
const code = `const x = () => 42;`;
const ast = parseSync(code, {
  sourceType: "module",
  plugins: ["jsx"]
});

const result = transformFromAstSync(ast, code, {
  presets: ["@babel/preset-env"]
});

console.log(result.code);
// Output: Transformed code from the AST

// Modify AST before transformation
import traverse from "@babel/traverse";
import * as t from "@babel/types";

traverse(ast, {
  ArrowFunctionExpression(path) {
    // Convert arrow function to regular function
    path.replaceWith(
      t.functionExpression(null, path.node.params, 
        t.blockStatement([t.returnStatement(path.node.body)])
      )
    );
  }
});

const modifiedResult = transformFromAstSync(ast, code, {
  presets: ["@babel/preset-env"]
});
```

## Transformation Result

All transformation functions return a `FileResult` object containing the transformed code and metadata.

```typescript { .api }
interface FileResult {
  /** Transformed JavaScript code, null if transformation was skipped */
  code: string | null;
  /** Source map object for debugging, null if source maps disabled */
  map: object | null;
  /** AST object if ast: true option was provided, null otherwise */
  ast: object | null;
  /** Metadata collected during transformation including plugin information */
  metadata: {
    /** Modules that were processed during transformation */
    modules?: {
      imports: Array<{
        source: string;
        imported: string[];
        specifiers: any[];
      }>;
      exports: Array<{
        exported: string[];
        specifiers: any[];
      }>;
    };
    /** List of external helper functions that were used */
    externalHelpers?: string[];
    /** Plugin-specific metadata */
    [pluginName: string]: any;
  };
}
```

## Common Transformation Options

Key options for controlling the transformation process:

```typescript { .api }
interface TransformationOptions {
  /** Plugins to apply during transformation */
  plugins?: Array<string | [string, any] | PluginFunction | [PluginFunction, any]>;
  /** Presets to apply during transformation (applied before plugins) */
  presets?: Array<string | [string, any] | PresetFunction | [PresetFunction, any]>;
  /** Include AST in result (default: false) */
  ast?: boolean;
  /** Generate source maps: false, true, "inline", or "both" */
  sourceMaps?: boolean | "inline" | "both";
  /** Compact output: true, false, or "auto" (default: "auto") */
  compact?: boolean | "auto";
  /** Environment name for conditional configuration */
  envName?: string;
  /** Override source filename in source maps and error messages */
  filename?: string;
  /** Parser options passed to @babel/parser */
  parserOpts?: {
    sourceType?: "script" | "module" | "unambiguous";
    allowImportExportEverywhere?: boolean;
    allowReturnOutsideFunction?: boolean;
    plugins?: string[];
    strictMode?: boolean;
    ranges?: boolean;
    tokens?: boolean;
  };
  /** Generator options passed to @babel/generator */
  generatorOpts?: {
    /** Retain parentheses around expressions */
    retainLines?: boolean;
    /** Compact whitespace */
    compact?: boolean;
    /** Number of spaces for indentation */
    indent?: number;
    /** Quote style: "single" or "double" */
    quotes?: "single" | "double";
  };
}
```

## Error Handling

Transformation functions may throw errors for various reasons:

```typescript
import { transformSync } from "@babel/core";

try {
  const result = transformSync("invalid syntax {{", {
    presets: ["@babel/preset-env"]
  });
} catch (error) {
  if (error.code === "BABEL_PARSE_ERROR") {
    console.error("Parse error:", error.message);
    console.error("Location:", error.loc);
  } else if (error.code === "BABEL_TRANSFORM_ERROR") {
    console.error("Transform error:", error.message);
    console.error("Plugin:", error.plugin);
  } else {
    console.error("Other error:", error.message);
  }
}
```