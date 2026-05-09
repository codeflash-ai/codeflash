# @babel/preset-typescript

@babel/preset-typescript is a Babel preset that enables TypeScript compilation through Babel's plugin pipeline. It configures the necessary plugins to transform TypeScript syntax into JavaScript while maintaining compatibility with Babel's ecosystem and build tools.

## Package Information

- **Package Name**: @babel/preset-typescript
- **Package Type**: npm
- **Language**: TypeScript/JavaScript
- **Installation**: `npm install --save-dev @babel/preset-typescript`

## Core Imports

```javascript
// babel.config.js
module.exports = {
  presets: ['@babel/preset-typescript']
};
```

With options:

```javascript
// babel.config.js
module.exports = {
  presets: [
    ['@babel/preset-typescript', {
      allowNamespaces: true,
      onlyRemoveTypeImports: true,
      optimizeConstEnums: false
    }]
  ]
};
```

For programmatic usage:

```javascript
const presetTypescript = require('@babel/preset-typescript').default;
// or
import presetTypescript from '@babel/preset-typescript';
```

## Basic Usage

```javascript
// babel.config.js - Basic configuration
module.exports = {
  presets: ['@babel/preset-typescript']
};
```

```javascript
// babel.config.js - Advanced configuration
module.exports = {
  presets: [
    ['@babel/preset-typescript', {
      // Allow TypeScript namespaces (default: true)
      allowNamespaces: true,
      
      // Only remove type-only imports (default: true in Babel 8)
      onlyRemoveTypeImports: true,
      
      // Optimize const enum transformations (default: false)
      optimizeConstEnums: false,
      
      // Custom JSX pragma (default: "React")
      jsxPragma: "React",
      
      // Custom JSX fragment pragma (default: "React.Fragment")
      jsxPragmaFrag: "React.Fragment",
      
      // Rewrite TypeScript import extensions (default: false)
      rewriteImportExtensions: false
    }]
  ]
};
```

## Architecture

The preset is built around several key components:

- **Core Preset Function**: Main function that configures TypeScript transformation plugins based on options and file extensions
- **Options Normalizer**: Validates and normalizes configuration options with Babel version-specific behavior
- **Import Rewriter Plugin**: Optional plugin that rewrites TypeScript import extensions to JavaScript equivalents
- **Plugin Configuration**: Automatically configures `@babel/plugin-transform-typescript`, JSX syntax support, and CommonJS transformation based on file types

## Capabilities

### Preset Configuration

The main preset function that configures TypeScript transformation for Babel. The preset is exported as the default export wrapped in Babel's `declarePreset` utility.

```typescript { .api }
// The actual export from @babel/preset-typescript
const preset: (api: PresetAPI, options?: Options, dirname?: string) => PresetObject;

// From @babel/core
interface PresetObject {
  plugins?: PluginList;
  presets?: PresetList;
  overrides?: Array<PresetObject>;
  env?: { [envName: string]: PresetObject };
  ignore?: IgnoreList;
  only?: IgnoreList;
  test?: ConfigApplicableTest;
  include?: ConfigApplicableTest;
  exclude?: ConfigApplicableTest;
}

// The preset returns a configuration with plugins and overrides
interface PresetResult {
  plugins: Array<string | [string, any]>;
  overrides: Array<{
    test?: RegExp | ((filename?: string) => boolean);
    sourceType?: "module" | "unambiguous";
    plugins: Array<string | [string, any]>;
  }>;
}
```

### Options Configuration

Configuration options for customizing TypeScript transformation behavior.

```typescript { .api }
interface Options {
  /** Ignore file extensions when determining file type */
  ignoreExtensions?: boolean;
  
  /** Allow TypeScript declare fields (Babel 7 only) */
  allowDeclareFields?: boolean;
  
  /** Allow TypeScript namespaces (default: true) */
  allowNamespaces?: boolean;
  
  /** Disallow ambiguous JSX-like syntax */
  disallowAmbiguousJSXLike?: boolean;
  
  /** JSX pragma to use (default: "React") */
  jsxPragma?: string;
  
  /** JSX fragment pragma (default: "React.Fragment") */
  jsxPragmaFrag?: string;
  
  /** Only remove type-only imports */
  onlyRemoveTypeImports?: boolean;
  
  /** Optimize const enums transformation */
  optimizeConstEnums?: boolean;
  
  /** Rewrite TypeScript import extensions to JavaScript */
  rewriteImportExtensions?: boolean;
  
  /** Handle all file extensions (deprecated in Babel 8) */
  allExtensions?: boolean;
  
  /** Force JSX parsing (deprecated in Babel 8) */
  isTSX?: boolean;
}
```

### Option Normalization

**Internal function** that validates and normalizes preset options. This function is not exported from the main package and is only used internally by the preset.

```typescript { .api }
// Internal function - NOT exported from @babel/preset-typescript
// Located in: src/normalize-options.ts
function normalizeOptions(options?: Options): Required<Options>;
```

**Usage:** Called internally by the preset to validate and apply default values to user-provided options. This function handles Babel version-specific option validation and provides helpful error messages for deprecated options.

### Import Extension Rewriting

**Internal plugin** that rewrites TypeScript import extensions to JavaScript equivalents. This plugin is not exported from the main package and is only used internally when `rewriteImportExtensions: true` is specified.

```typescript { .api }
// Internal plugin - NOT exported from @babel/preset-typescript
// Located in: src/plugin-rewrite-ts-imports.ts
function pluginRewriteTSImports(): PluginObject;

// From @babel/core
interface PluginObject {
  name?: string;
  visitor: Visitor;
  pre?: (state: any) => void;
  post?: (state: any) => void;
  manipulateOptions?: (opts: any, parserOpts: any) => void;
}
```

**Transformation Examples:**
- `./module.ts` → `./module.js`
- `./component.tsx` → `./component.jsx` (or `.js` if JSX preservation is disabled)
- `./module.mts` → `./module.mjs`
- `./module.cts` → `./module.cjs`
- `./types.d.ts` → `./types.d.ts` (preserved)

## File Extension Handling

The preset automatically applies different plugin configurations based on file extensions:

### TypeScript Files (.ts)

Standard TypeScript files receive basic TypeScript transformation:

```javascript
// Configuration applied for .ts files
{
  plugins: [
    ['@babel/plugin-transform-typescript', { 
      allowNamespaces: true,
      disallowAmbiguousJSXLike: false,
      // ... other options
    }]
  ]
}
```

### TypeScript JSX Files (.tsx)

TypeScript files with JSX receive TypeScript transformation plus JSX syntax support:

```javascript
// Configuration applied for .tsx files
{
  plugins: [
    ['@babel/plugin-transform-typescript', { 
      isTSX: true,
      allowNamespaces: true,
      // ... other options
    }],
    '@babel/plugin-syntax-jsx'
  ]
}
```

### ES Module TypeScript (.mts)

TypeScript ES module files with strict module type checking:

```javascript
// Configuration applied for .mts files
{
  sourceType: "module",
  plugins: [
    ['@babel/plugin-transform-typescript', { 
      disallowAmbiguousJSXLike: true,
      // ... other options
    }]
  ]
}
```

### CommonJS TypeScript (.cts)

TypeScript CommonJS files with automatic CommonJS transformation:

```javascript
// Configuration applied for .cts files
{
  sourceType: "unambiguous",
  plugins: [
    ['@babel/plugin-transform-modules-commonjs', { 
      allowTopLevelThis: true 
    }],
    ['@babel/plugin-transform-typescript', { 
      disallowAmbiguousJSXLike: true,
      // ... other options
    }]
  ]
}
```

## Version Compatibility

### Babel 7 Compatibility

Babel 7 supports additional legacy options:

- `allowDeclareFields`: Controls TypeScript declare field support
- `allExtensions`: Forces handling of all file extensions
- `isTSX`: Forces JSX parsing for all files

### Babel 8 Compatibility

Babel 8 removes deprecated options and enforces stricter validation:

- Removes `allowDeclareFields`, `allExtensions`, and `isTSX` options
- Stricter option validation with helpful error messages
- Enhanced Node.js version requirements (>=20.19.0 || >=22.12.0)

## Plugin Dependencies

The preset automatically configures these Babel plugins:

- **@babel/plugin-transform-typescript**: Core TypeScript syntax transformation
- **@babel/plugin-syntax-jsx**: JSX syntax parsing support
- **@babel/plugin-transform-modules-commonjs**: CommonJS module transformation (for .cts files)

## Common Configuration Examples

### React TypeScript Project

```javascript
// babel.config.js
module.exports = {
  presets: [
    ['@babel/preset-typescript', {
      jsxPragma: 'React',
      jsxPragmaFrag: 'React.Fragment'
    }],
    '@babel/preset-react'
  ]
};
```

### Node.js TypeScript Project

```javascript
// babel.config.js
module.exports = {
  presets: [
    ['@babel/preset-typescript', {
      allowNamespaces: true,
      optimizeConstEnums: true
    }],
    ['@babel/preset-env', {
      targets: { node: 'current' }
    }]
  ]
};
```

### Strict TypeScript Configuration

```javascript
// babel.config.js
module.exports = {
  presets: [
    ['@babel/preset-typescript', {
      onlyRemoveTypeImports: true,
      disallowAmbiguousJSXLike: true,
      ignoreExtensions: true
    }]
  ]
};
```

### Import Extension Rewriting

```javascript
// babel.config.js
module.exports = {
  presets: [
    ['@babel/preset-typescript', {
      rewriteImportExtensions: true
    }]
  ]
};
```

This configuration transforms:

```typescript
// Input TypeScript
import { helper } from './utils.ts';
import Component from './Component.tsx';

// Output JavaScript
import { helper } from './utils.js';
import Component from './Component.jsx';
```