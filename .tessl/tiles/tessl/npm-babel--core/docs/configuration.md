# Configuration Management

Babel configuration loading, validation, and management system supporting various config file formats, runtime options, and plugin/preset resolution. Provides both full and partial configuration loading for different use cases.

## Capabilities

### Options Loading

Load and resolve complete Babel configuration from various sources.

```typescript { .api }
/**
 * Load complete Babel options synchronously
 * @param opts - Input options to merge with config file settings
 * @returns Resolved configuration object or null if no config found
 */
function loadOptionsSync(opts?: InputOptions): ResolvedConfig | null;

/**
 * Load complete Babel options asynchronously
 * @param opts - Input options to merge with config file settings
 * @returns Promise resolving to resolved configuration or null
 */
function loadOptionsAsync(opts?: InputOptions): Promise<ResolvedConfig | null>;

/**
 * Load complete Babel options with callback (legacy API, deprecated in Babel 8)
 * @param opts - Input options to merge with config file settings
 * @param callback - Callback function receiving error and resolved config
 */
function loadOptions(
  opts: InputOptions,
  callback: (err: Error | null, config: ResolvedConfig | null) => void
): void;
function loadOptions(
  callback: (err: Error | null, config: ResolvedConfig | null) => void
): void;

interface ResolvedConfig {
  /** Resolved plugins with their options */
  plugins: Array<ConfigItem>;
  /** Resolved presets with their options */
  presets: Array<ConfigItem>;
  /** Parser options */
  parserOpts: ParserOptions;
  /** Generator options */
  generatorOpts: GeneratorOptions;
  /** All other resolved options */
  [key: string]: any;
}
```

**Usage Examples:**

```typescript
import { loadOptionsSync, loadOptionsAsync } from "@babel/core";

// Load configuration from babel.config.js and .babelrc files
const config = loadOptionsSync({
  cwd: "/path/to/project",
  filename: "src/app.js",
  envName: "production"
});

if (config) {
  console.log("Plugins:", config.plugins.map(p => p.name));
  console.log("Presets:", config.presets.map(p => p.name));
  console.log("Parser options:", config.parserOpts);
}

// Override config file settings
const customConfig = loadOptionsSync({
  presets: ["@babel/preset-env"],
  plugins: ["@babel/plugin-transform-runtime"],
  targets: "> 0.25%, not dead"
});

// Async loading
const asyncConfig = await loadOptionsAsync({
  cwd: process.cwd(),
  configFile: "./babel.config.json",
  envName: process.env.NODE_ENV
});
```

### Partial Configuration

Load partial configuration for advanced use cases where full resolution isn't needed.

```typescript { .api }
/**
 * Load partial Babel configuration synchronously
 * @param opts - Input options for partial resolution
 * @returns Partial configuration object or null
 */
function loadPartialConfigSync(opts?: InputOptions): PartialConfig | null;

/**
 * Load partial Babel configuration asynchronously
 * @param opts - Input options for partial resolution
 * @returns Promise resolving to partial configuration or null
 */
function loadPartialConfigAsync(opts?: InputOptions): Promise<PartialConfig | null>;

/**
 * Load partial configuration with callback (legacy API, deprecated in Babel 8)
 * @param opts - Input options for partial resolution
 * @param callback - Callback function receiving error and partial config
 */
function loadPartialConfig(
  opts: InputOptions,
  callback: (err: Error | null, config: PartialConfig | null) => void
): void;
function loadPartialConfig(
  callback: (err: Error | null, config: PartialConfig | null) => void
): void;

interface PartialConfig {
  /** Resolved options (may be null if no valid config) */
  options: ResolvedConfig | null;
  /** Loaded config file information */
  config?: {
    filepath: string;
    dirname: string;
    options: any;
  };
  /** Loaded .babelrc file information */
  babelrc?: {
    filepath: string;
    dirname: string;
    options: any;
  };
  /** Whether this config ignores the file */
  hasFilesystemConfig(): boolean;
}
```

**Usage Examples:**

```typescript
import { loadPartialConfigSync } from "@babel/core";

// Check if a file should be processed
const partialConfig = loadPartialConfigSync({
  filename: "src/components/Button.tsx",
  cwd: "/path/to/project"
});

if (partialConfig) {
  if (partialConfig.hasFilesystemConfig()) {
    console.log("File has Babel config");
    
    if (partialConfig.config) {
      console.log("Config file:", partialConfig.config.filepath);
    }
    
    if (partialConfig.babelrc) {
      console.log("Babelrc file:", partialConfig.babelrc.filepath);
    }
    
    // Use resolved options
    if (partialConfig.options) {
      console.log("Resolved plugins:", partialConfig.options.plugins.length);
    }
  } else {
    console.log("No Babel config found for this file");
  }
}
```

### Configuration Items

Create and manage individual plugin and preset configuration items.

```typescript { .api }
/**
 * Create configuration item synchronously
 * @param target - Plugin/preset function, module name, or path
 * @param options - Options to pass to the plugin/preset
 * @returns Configuration item or null if invalid
 */
function createConfigItemSync(
  target: PluginTarget,
  options?: ConfigItemOptions
): ConfigItem<PluginAPI> | null;

/**
 * Create configuration item asynchronously
 * @param target - Plugin/preset function, module name, or path
 * @param options - Options to pass to the plugin/preset
 * @returns Promise resolving to configuration item or null
 */
function createConfigItemAsync(
  target: PluginTarget,
  options?: ConfigItemOptions
): Promise<ConfigItem<PluginAPI> | null>;

/**
 * Create configuration item with callback (legacy API, deprecated in Babel 8)
 * @param target - Plugin/preset function, module name, or path
 * @param options - Options to pass to the plugin/preset
 * @param callback - Callback function receiving error and config item
 */
function createConfigItem(
  target: PluginTarget,
  options: ConfigItemOptions,
  callback: (err: Error | null, item: ConfigItem<PluginAPI> | null) => void
): void;

type PluginTarget = 
  | string 
  | PluginFunction 
  | PresetFunction
  | [string, any]
  | [PluginFunction, any]
  | [PresetFunction, any];

interface ConfigItemOptions {
  /** Directory context for resolution */  
  dirname?: string;
  /** Item type: "plugin" or "preset" */
  type?: "plugin" | "preset";
}

interface ConfigItem<T = PluginAPI> {
  /** Resolved plugin/preset function */
  value: T;
  /** Configuration options passed to the plugin/preset */
  options: any;
  /** Directory where the plugin/preset was resolved */
  dirname: string;
  /** Name of the plugin/preset */
  name?: string;
  /** Full file path if resolved from file */
  file?: {
    request: string;
    resolved: string;
  };
}
```

**Usage Examples:**

```typescript
import { createConfigItemSync } from "@babel/core";

// Create plugin config item
const pluginItem = createConfigItemSync("@babel/plugin-transform-runtime", {
  dirname: "/path/to/project",
  type: "plugin"
});

if (pluginItem) {
  console.log("Plugin name:", pluginItem.name);
  console.log("Plugin options:", pluginItem.options);
  console.log("Resolved from:", pluginItem.file?.resolved);
}

// Create preset config item with options
const presetItem = createConfigItemSync(
  ["@babel/preset-env", {
    targets: "> 0.25%, not dead",
    useBuiltIns: "usage",
    corejs: 3
  }],
  {
    dirname: process.cwd(),
    type: "preset"  
  }
);

// Create from function
const customPlugin = function(babel) {
  return {
    visitor: {
      Identifier(path) {
        console.log("Found identifier:", path.node.name);
      }
    }
  };
};

const customItem = createConfigItemSync(customPlugin, {
  dirname: __dirname,
  type: "plugin"
});
```

## Configuration File Support

Babel supports various configuration file formats:

```typescript { .api }
interface ConfigFileOptions {
  /** Path to specific config file, or false to disable */
  configFile?: string | false;
  /** Enable/disable .babelrc file loading */
  babelrc?: boolean;
  /** Root directory for config file search */
  root?: string;
  /** Current working directory */
  cwd?: string;
  /** Override root mode: "root", "upward", or "upward-optional" */
  rootMode?: "root" | "upward" | "upward-optional";
}
```

**Supported Config Files:**

- `babel.config.json` - Project-wide configuration
- `babel.config.js` - Project-wide with JavaScript
- `babel.config.mjs` - Project-wide with ES modules
- `babel.config.cjs` - Project-wide with CommonJS
- `.babelrc` - File-relative configuration
- `.babelrc.json` - File-relative JSON
- `.babelrc.js` - File-relative JavaScript
- `package.json` - Babel field in package.json

**Usage Examples:**

```typescript
import { loadOptionsSync } from "@babel/core";

// Use specific config file
const config1 = loadOptionsSync({
  configFile: "./babel.production.js",
  cwd: "/path/to/project"
});

// Disable config file loading
const config2 = loadOptionsSync({
  configFile: false,
  plugins: ["@babel/plugin-transform-arrow-functions"]
});

// Disable .babelrc files
const config3 = loadOptionsSync({
  babelrc: false,
  presets: ["@babel/preset-env"]
});

// Search from different root
const config4 = loadOptionsSync({
  root: "/different/root",
  rootMode: "upward",
  filename: "src/app.js"
});
```

## Environment-based Configuration

Configure Babel behavior based on environment:

```typescript { .api }
interface EnvironmentOptions {
  /** Environment name (defaults to BABEL_ENV || NODE_ENV || "development") */
  envName?: string;
  /** Caller metadata for conditional configuration */
  caller?: CallerMetadata;
}

interface CallerMetadata {
  /** Name of the calling tool */
  name: string;
  /** Version of the calling tool */
  version?: string;
  /** Whether the caller supports ES modules */
  supportsStaticESM?: boolean;
  /** Whether the caller supports dynamic imports */
  supportsDynamicImport?: boolean;
  /** Whether the caller supports top-level await */
  supportsTopLevelAwait?: boolean;
  /** Additional caller-specific properties */
  [key: string]: any;
}
```

**Usage Examples:**

```typescript
import { loadOptionsSync } from "@babel/core";

// Load development configuration
const devConfig = loadOptionsSync({
  envName: "development",
  caller: {
    name: "webpack",
    version: "5.0.0",
    supportsStaticESM: true
  }
});

// Load production configuration
const prodConfig = loadOptionsSync({
  envName: "production",
  caller: {
    name: "rollup", 
    version: "2.0.0",
    supportsDynamicImport: true
  }
});

// Override environment
process.env.NODE_ENV = "test";
const testConfig = loadOptionsSync({
  envName: "testing", // Overrides NODE_ENV
  filename: "test/example.spec.js"
});
```

## Advanced Configuration Patterns

### Conditional Configuration

```javascript
// babel.config.js
module.exports = function(api) {
  // Cache configuration based on environment
  api.cache.using(() => process.env.NODE_ENV);
  
  const presets = ["@babel/preset-env"];
  const plugins = [];
  
  // Add plugins based on environment
  if (api.env("development")) {
    plugins.push("react-refresh/babel");
  }
  
  // Add plugins based on caller
  if (api.caller(caller => caller?.name === "webpack")) {
    plugins.push("@babel/plugin-syntax-dynamic-import");
  }
  
  return { presets, plugins };
};
```

### Programmatic Configuration

```typescript
import { loadOptionsSync, transformSync } from "@babel/core";

// Build configuration programmatically
const baseConfig = loadOptionsSync({
  presets: ["@babel/preset-env"],
  configFile: false
});

// Extend with additional plugins
const extendedConfig = {
  ...baseConfig,
  plugins: [
    ...baseConfig.plugins,
    ["@babel/plugin-transform-runtime", { corejs: 3 }]
  ]
};

// Use extended configuration
const result = transformSync(code, extendedConfig);
```

## Error Handling

Configuration functions may throw errors for invalid configurations:

```typescript
import { loadOptionsSync, createConfigItemSync } from "@babel/core";

try {
  const config = loadOptionsSync({
    plugins: ["non-existent-plugin"]
  });
} catch (error) {
  if (error.code === "BABEL_UNKNOWN_PLUGIN") {
    console.error("Unknown plugin:", error.message);
  }
}

try {
  const item = createConfigItemSync("invalid-plugin", {
    dirname: "/nonexistent"
  });
} catch (error) {
  console.error("Failed to create config item:", error.message);
}
```