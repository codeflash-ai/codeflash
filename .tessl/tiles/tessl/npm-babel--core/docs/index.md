# Babel Core

Babel Core is the core compiler for Babel, providing programmatic APIs for JavaScript code transformation, parsing, and configuration. It enables developers to transpile modern JavaScript code into backward-compatible versions, parse JavaScript into Abstract Syntax Trees (ASTs), and configure the transformation process through plugins and presets.

## Package Information

- **Package Name**: @babel/core
- **Package Type**: npm
- **Language**: TypeScript
- **Installation**: `npm install @babel/core`

## Core Imports

```typescript
import * as babel from "@babel/core";
```

For specific functions:

```typescript
import { 
  transform, 
  transformSync, 
  parse, 
  parseSync, 
  loadOptions,
  createConfigItem,
  types,
  traverse,
  template,
  type PluginPass,
  type Visitor,
  type NodePath,
  type Scope
} from "@babel/core";
```

CommonJS:

```javascript
const babel = require("@babel/core");
const { transform, parse, loadOptions } = require("@babel/core");
```

## Basic Usage

```typescript
import { transformSync, parseSync } from "@babel/core";

// Transform JavaScript code
const result = transformSync(`
  const arrow = () => console.log("Hello");
  class MyClass {
    method() { return 42; }
  }
`, {
  presets: ["@babel/preset-env"],
  plugins: ["@babel/plugin-transform-arrow-functions"]
});

console.log(result.code);
// Output: Transpiled ES5 compatible code

// Parse JavaScript to AST
const ast = parseSync(`
  function hello() {
    return "world";
  }
`, {
  sourceType: "module",
  plugins: ["jsx", "typescript"]
});

console.log(ast.type); // "File"
```

## Architecture

Babel Core is built around several key components:

- **Transformation Engine**: Core APIs (`transform`, `transformSync`, `transformAsync`) that apply plugins and presets to JavaScript code
- **Parser Interface**: Wrapper around @babel/parser (`parse`, `parseSync`, `parseAsync`) for AST generation
- **Configuration System**: Option loading and validation (`loadOptions`, `loadPartialConfig`) with support for config files
- **Plugin/Preset Management**: Configuration item creation and resolution (`createConfigItem`, `resolvePlugin`, `resolvePreset`)
- **File Processing**: File-based transformation APIs (`transformFile`, `transformFileSync`, `transformFileAsync`)
- **AST Processing**: Direct AST transformation (`transformFromAst`, `transformFromAstSync`, `transformFromAstAsync`)

## Browser Compatibility

Babel Core includes browser-compatible variants for client-side usage:

- **File system operations** are replaced with browser-compatible alternatives
- **Config file resolution** is modified for browser environments  
- **Transform file APIs** (`transformFile*`) use alternative implementations that don't rely on Node.js file system
- **Module resolution** adapts to browser module loading patterns

The package automatically uses browser-compatible versions when bundled for web environments through the `browser` field in package.json.

## Capabilities

### Code Transformation

Core JavaScript transformation functionality supporting both code strings and files, with synchronous and asynchronous variants.

```typescript { .api }
function transformSync(
  code: string, 
  opts?: InputOptions
): FileResult | null;

function transformAsync(
  code: string, 
  opts?: InputOptions
): Promise<FileResult | null>;

function transform(code: string, callback: FileResultCallback): void;
function transform(
  code: string,
  opts: InputOptions | null | undefined,
  callback: FileResultCallback
): void;

type FileResultCallback = (err: Error | null, result: FileResult | null) => void;

interface FileResult {
  code: string | null;
  map: object | null;
  ast: object | null;
  metadata: object;
}
```

[Transformation](./transformation.md)

### Code Parsing

JavaScript parsing functionality that converts source code into Abstract Syntax Trees (ASTs) using Babel's parser.

```typescript { .api }
function parseSync(
  code: string, 
  opts?: InputOptions
): ParseResult | null;

function parseAsync(
  code: string, 
  opts?: InputOptions
): Promise<ParseResult | null>;

function parse(code: string, callback: FileParseCallback): void;
function parse(
  code: string,
  opts: InputOptions | null | undefined,
  callback: FileParseCallback
): void;

type ParseResult = import("@babel/types").File;
type FileParseCallback = (err: Error | null, ast: ParseResult | null) => void;
```

[Parsing](./parsing.md)

### Configuration Management

Babel configuration loading, validation, and management system supporting various config file formats and runtime options.

```typescript { .api }
function loadOptionsSync(opts?: InputOptions): ResolvedConfig | null;
function loadOptionsAsync(opts?: InputOptions): Promise<ResolvedConfig | null>;
function loadOptions(opts: InputOptions, callback: (err: Error | null, config: ResolvedConfig | null) => void): void;
function loadOptions(callback: (err: Error | null, config: ResolvedConfig | null) => void): void;

function loadPartialConfigSync(opts?: InputOptions): PartialConfig | null;
function loadPartialConfigAsync(opts?: InputOptions): Promise<PartialConfig | null>;
function loadPartialConfig(opts: InputOptions, callback: (err: Error | null, config: PartialConfig | null) => void): void;
function loadPartialConfig(callback: (err: Error | null, config: PartialConfig | null) => void): void;

function createConfigItemSync(
  target: PluginTarget, 
  options?: any
): ConfigItem<PluginAPI> | null;
function createConfigItemAsync(
  target: PluginTarget, 
  options?: any
): Promise<ConfigItem<PluginAPI> | null>;
function createConfigItem(
  target: PluginTarget,
  options: any,
  callback: (err: Error | null, item: ConfigItem<PluginAPI> | null) => void
): void;
```

[Configuration](./configuration.md)

### Utilities and Constants  

Helper functions, constants, and re-exported APIs from the Babel ecosystem.

```typescript { .api }
const version: string;
const DEFAULT_EXTENSIONS: readonly string[];

function getEnv(defaultValue?: string): string;
function resolvePlugin(name: string, dirname: string): string;
function resolvePreset(name: string, dirname: string): string;
```

[Utilities](./utilities.md)

## Core Types

```typescript { .api }
interface InputOptions {
  /** Input source code filename for error reporting and source maps */
  filename?: string;
  /** Input source type: "script", "module", or "unambiguous" */
  sourceType?: "script" | "module" | "unambiguous";
  /** Array of plugins to apply during transformation */
  plugins?: PluginItem[];
  /** Array of presets to apply during transformation */
  presets?: PresetItem[];
  /** Parser options passed to @babel/parser */
  parserOpts?: ParserOptions;
  /** Generator options passed to @babel/generator */
  generatorOpts?: GeneratorOptions;
  /** Whether to include AST in result */
  ast?: boolean;
  /** Source map generation options */
  sourceMaps?: boolean | "inline" | "both";
  /** Code compaction options */
  compact?: boolean | "auto";
  /** Root directory for config file search */
  root?: string;
  /** Current working directory */
  cwd?: string;
  /** Environment name for conditional config */
  envName?: string;
  /** Babel configuration file path or search behavior */
  configFile?: string | false;
  /** .babelrc file search behavior */
  babelrc?: boolean;
  /** Metadata about the calling tool */
  caller?: CallerMetadata;
}

interface CallerMetadata {
  name: string;
  version?: string;
  [key: string]: any;
}

type PluginItem = string | [string, any] | PluginFunction | [PluginFunction, any];
type PresetItem = string | [string, any] | PresetFunction | [PresetFunction, any];

interface FileResult {
  /** Transformed JavaScript code */
  code: string | null;
  /** Source map for the transformation */
  map: object | null;
  /** AST if requested via ast: true option */
  ast: object | null;  
  /** Metadata from plugins and transformation process */
  metadata: {
    [key: string]: any;
  };
}

interface ResolvedConfig {
  /** Resolved and validated options */
  [key: string]: any;
}

interface PartialConfig {
  /** Partial configuration that may need further resolution */
  options: ResolvedConfig | null;
  config?: any;
  babelrc?: any;
  [key: string]: any;
}

interface ConfigItem<T = PluginAPI> {
  /** Plugin or preset value */
  value: T;
  /** Configuration options */
  options: any;
  /** Directory context */
  dirname: string;
  /** Item name */
  name?: string;
}

interface PluginAPI {
  /** Plugin target metadata */
  [key: string]: any;
}

interface PluginPass {
  /** Current transformation file context */
  file: File;
  /** Plugin key/name */
  key: string;
  /** Plugin options */
  opts: any;
  /** Current working directory */
  cwd: string;
  /** Filename being processed */
  filename?: string;
}

type Visitor<S = unknown> = {
  /** Called when entering any AST node */
  enter?(path: NodePath, state: S): void;
  /** Called when exiting any AST node */
  exit?(path: NodePath, state: S): void;
  /** Specific node type visitors (e.g., FunctionDeclaration, Identifier) */
  [NodeType: string]: 
    | ((path: NodePath, state: S) => void)
    | { enter?(path: NodePath, state: S): void; exit?(path: NodePath, state: S): void }
    | undefined;
};

interface NodePath<T = any> {
  /** The AST node this path represents */
  node: T;
  /** Parent path */
  parent: NodePath | null;
  /** Parent AST node */
  parentPath: NodePath | null;
  /** Current scope information */
  scope: Scope;
  /** Current state passed through traversal */
  state: any;
  /** Array of child paths */
  paths?: NodePath[];
  /** Key in parent node */
  key?: string | number;
  /** Index if parent is array */
  listKey?: string;
  
  /** Replace this node with a new node */
  replaceWith(node: any): void;
  /** Remove this node */
  remove(): void;
  /** Skip traversing children of this node */
  skip(): void;
  /** Stop traversal entirely */
  stop(): void;
  /** Get the source code for this node */
  getSource(): string;
  /** Check if this path represents a specific node type */
  isNodeType(type: string): boolean;
  /** Find parent path of specific type */
  findParent(callback: (path: NodePath) => boolean): NodePath | null;
  /** Get binding information for identifier */
  get(key: string): NodePath | NodePath[] | null;
}

interface Scope {
  /** Parent scope */
  parent: Scope | null;
  /** Path that created this scope */
  path: NodePath;
  /** Block that created this scope */
  block: any;
  /** All bindings in this scope */
  bindings: { [name: string]: Binding };
  /** Referenced identifiers */
  references: { [name: string]: any[] };
  /** Global scope references */
  globals: { [name: string]: any };
  
  /** Check if identifier is bound in this scope */
  hasBinding(name: string): boolean;
  /** Get binding for identifier */
  getBinding(name: string): Binding | undefined;
  /** Generate unique identifier */
  generateUid(name?: string): string;
  /** Add binding to scope */
  registerBinding(kind: string, path: NodePath): void;
}

interface Binding {
  /** Identifier name */
  identifier: any;
  /** Scope this binding belongs to */
  scope: Scope;
  /** Path that created the binding */
  path: NodePath;
  /** Kind of binding (var, let, const, function, etc.) */
  kind: string;
  /** Whether binding is referenced */
  referenced: boolean;
  /** Number of references */
  references: number;
  /** All reference paths */
  referencePaths: NodePath[];
}
```