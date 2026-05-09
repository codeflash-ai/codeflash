# Configuration

Jest provides a comprehensive configuration system supporting global settings, per-project configuration, and extensive customization options for all aspects of test execution.

## Capabilities

### Core Configuration Types

Jest uses a hierarchical configuration system with different types serving specific purposes.

```typescript { .api }
/**
 * Main user configuration interface (alias for Config.InitialOptions)
 * This is what users provide in jest.config.js or package.json
 */
interface Config {
  // Test discovery
  testMatch?: Array<string>;
  testRegex?: string | Array<string>;
  testPathIgnorePatterns?: Array<string>;
  
  // Test environment
  testEnvironment?: string;
  testEnvironmentOptions?: Record<string, any>;
  
  // Module resolution
  moduleDirectories?: Array<string>;
  moduleFileExtensions?: Array<string>;
  moduleNameMapper?: Record<string, string>;
  modulePaths?: Array<string>;
  
  // Transforms
  transform?: Record<string, string>;
  transformIgnorePatterns?: Array<string>;
  
  // Setup
  setupFiles?: Array<string>;
  setupFilesAfterEnv?: Array<string>;
  
  // Coverage
  collectCoverage?: boolean;
  collectCoverageFrom?: Array<string>;
  coverageDirectory?: string;
  coveragePathIgnorePatterns?: Array<string>;
  coverageProvider?: "babel" | "v8";
  coverageReporters?: Array<string>;
  coverageThreshold?: Record<string, Record<string, number>>;
  
  // Execution
  maxWorkers?: number | string;
  testTimeout?: number;
  
  // Output
  verbose?: boolean;
  silent?: boolean;
  
  // Project structure
  rootDir?: string;
  roots?: Array<string>;
  projects?: Array<string | Config>;
}

/**
 * Global configuration that applies to the entire Jest run
 */
interface Config.GlobalConfig {
  bail: number;
  changedFilesWithAncestor: boolean;
  changedSince?: string;
  ci: boolean;
  collectCoverage: boolean;
  collectCoverageFrom: Array<string>;
  collectCoverageOnlyFrom?: Record<string, boolean>;
  coverageDirectory: string;
  coveragePathIgnorePatterns?: Array<string>;
  coverageProvider: "babel" | "v8";
  coverageReporters: Array<string>;
  coverageThreshold?: Record<string, Record<string, number>>;
  detectLeaks: boolean;
  detectOpenHandles: boolean;
  errorOnDeprecated: boolean;
  expand: boolean;
  filter?: string;
  findRelatedTests: boolean;
  forceExit: boolean;
  json: boolean;
  globalSetup?: string;
  globalTeardown?: string;
  lastCommit: boolean;
  listTests: boolean;
  logHeapUsage: boolean;
  maxConcurrency: number;
  maxWorkers: number;
  noSCM?: boolean;
  noStackTrace: boolean;
  notify: boolean;
  notifyMode: string;
  onlyChanged: boolean;
  onlyFailures: boolean;
  outputFile?: string;
  passWithNoTests: boolean;
  projects: Array<string>;
  randomize?: boolean;
  rootDir: string;
  runTestsByPath: boolean;
  seed?: number;
  silent: boolean;
  skipFilter: boolean;
  testFailureExitCode: number;
  testNamePattern?: string;
  testPathPattern: string;
  testPathPatterns: Array<string>;
  testResultsProcessor?: string;
  testSequencer: string;
  testTimeout?: number;
  updateSnapshot: SnapshotUpdateState;
  useStderr: boolean;
  verbose?: boolean;
  watch: boolean;
  watchAll: boolean;
  watchPathIgnorePatterns: Array<string>;
  watchman: boolean;
}

/**
 * Per-project configuration
 */
interface Config.ProjectConfig {
  automock: boolean;
  cache: boolean;
  cacheDirectory: string;
  clearMocks: boolean;
  collectCoverageFrom: Array<string>;
  coveragePathIgnorePatterns: Array<string>;
  cwd: string;
  dependencyExtractor?: string;
  detectLeaks: boolean;
  displayName?: string;
  errorOnDeprecated: boolean;
  extensionsToTreatAsEsm: Array<string>;
  extraGlobals: Array<string>;
  forceCoverageMatch: Array<string>;
  globalSetup?: string;
  globalTeardown?: string;
  globals: Record<string, unknown>;
  haste: HasteConfig;
  injectGlobals: boolean;
  moduleDirectories: Array<string>;
  moduleFileExtensions: Array<string>;
  moduleNameMapper: Array<[string, string]>;
  modulePaths?: Array<string>;
  modulePathIgnorePatterns: Array<string>;
  preset?: string;
  prettierPath: string;
  resetMocks: boolean;
  resetModules: boolean;
  resolver?: string;
  restoreMocks: boolean;
  rootDir: string;
  roots: Array<string>;
  runner: string;
  setupFiles: Array<string>;
  setupFilesAfterEnv: Array<string>;
  skipFilter: boolean;
  skipNodeResolution?: boolean;
  slowTestThreshold: number;
  snapshotResolver?: string;
  snapshotSerializers: Array<string>;
  testEnvironment: string;
  testEnvironmentOptions: Record<string, unknown>;
  testLocationInResults: boolean;
  testMatch: Array<string>;
  testPathIgnorePatterns: Array<string>;
  testRegex: Array<string | RegExp>;
  testRunner: string;
  testURL?: string;
  timers: "real" | "fake";
  transform: Array<[string, string, Record<string, unknown>]>;
  transformIgnorePatterns: Array<string>;
  unmockedModulePathPatterns?: Array<string>;
  watchPathIgnorePatterns: Array<string>;
}

/**
 * Command line arguments interface
 */
interface Config.Argv {
  // Test execution
  all?: boolean;
  bail?: boolean | number;
  ci?: boolean;
  findRelatedTests?: boolean;
  listTests?: boolean;
  onlyChanged?: boolean;
  onlyFailures?: boolean;
  passWithNoTests?: boolean;
  runInBand?: boolean;
  runTestsByPath?: boolean;
  testNamePattern?: string;
  testPathPatterns?: Array<string>;
  testTimeout?: number;
  
  // Watch mode
  watch?: boolean;
  watchAll?: boolean;
  watchPathIgnorePatterns?: Array<string>;
  
  // Coverage
  collectCoverage?: boolean;
  coverage?: boolean;
  collectCoverageFrom?: Array<string>;
  coverageDirectory?: string;
  coveragePathIgnorePatterns?: Array<string>;
  coverageProvider?: "babel" | "v8";
  coverageReporters?: Array<string>;
  coverageThreshold?: Record<string, number>;
  
  // Output
  json?: boolean;
  outputFile?: string;
  verbose?: boolean;
  silent?: boolean;
  noStackTrace?: boolean;
  color?: boolean;
  colors?: boolean;
  
  // Configuration
  config?: string;
  rootDir?: string;
  roots?: Array<string>;
  projects?: Array<string>;
  maxWorkers?: number | string;
  cache?: boolean;
  clearCache?: boolean;
  debug?: boolean;
  updateSnapshot?: boolean;
}
```

### Configuration File Examples

**Basic Configuration (jest.config.js):**

```javascript
module.exports = {
  // Test discovery
  testMatch: ["**/__tests__/**/*.js", "**/?(*.)+(spec|test).js"],
  testPathIgnorePatterns: ["/node_modules/", "/build/"],
  
  // Test environment
  testEnvironment: "jsdom",
  
  // Module resolution
  moduleDirectories: ["node_modules", "src"],
  moduleFileExtensions: ["js", "json", "jsx", "ts", "tsx"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
    "\\.(css|less|scss)$": "identity-obj-proxy"
  },
  
  // Transforms
  transform: {
    "^.+\\.jsx?$": "babel-jest",
    "^.+\\.tsx?$": "ts-jest"
  },
  
  // Setup
  setupFilesAfterEnv: ["<rootDir>/src/setupTests.js"],
  
  // Coverage
  collectCoverage: true,
  collectCoverageFrom: [
    "src/**/*.{js,jsx,ts,tsx}",
    "!src/**/*.d.ts",
    "!src/index.js"
  ],
  coverageDirectory: "coverage",
  coverageReporters: ["text", "lcov", "html"],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  }
};
```

**TypeScript Configuration:**

```javascript
module.exports = {
  preset: "ts-jest",
  testEnvironment: "node",
  
  // TypeScript specific
  globals: {
    "ts-jest": {
      tsconfig: "tsconfig.test.json"
    }
  },
  
  // Module resolution for TypeScript
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1"
  },
  
  // File extensions
  moduleFileExtensions: ["ts", "tsx", "js", "jsx", "json"],
  
  // Transform configuration
  transform: {
    "^.+\\.tsx?$": "ts-jest"
  },
  
  // Test files
  testMatch: ["**/__tests__/**/*.ts", "**/?(*.)+(spec|test).ts"]
};
```

**Multi-Project Configuration:**

```javascript
module.exports = {
  projects: [
    {
      displayName: "client",
      testMatch: ["<rootDir>/packages/client/**/*.test.js"],
      testEnvironment: "jsdom",
      setupFilesAfterEnv: ["<rootDir>/packages/client/src/setupTests.js"]
    },
    {
      displayName: "server",
      testMatch: ["<rootDir>/packages/server/**/*.test.js"],
      testEnvironment: "node",
      setupFilesAfterEnv: ["<rootDir>/packages/server/src/setupTests.js"]
    },
    {
      displayName: "shared",
      testMatch: ["<rootDir>/packages/shared/**/*.test.js"],
      testEnvironment: "node"
    }
  ],
  
  // Global coverage settings
  collectCoverage: true,
  coverageDirectory: "coverage",
  coverageReporters: ["text-summary", "html"]
};
```

### Configuration Loading and Resolution

Jest resolves configuration in the following order:

1. `--config` CLI argument
2. `jest.config.js` file
3. `jest.config.ts` file
4. `jest` field in `package.json`
5. Default configuration

**Programmatic Configuration:**

```typescript
import { runCLI, buildArgv } from "jest";

// Override configuration programmatically
async function runWithCustomConfig() {
  const customConfig = {
    testMatch: ["**/*.custom.test.js"],
    testEnvironment: "node",
    collectCoverage: true
  };
  
  const argv = await buildArgv([
    `--config=${JSON.stringify(customConfig)}`
  ]);
  
  return runCLI(argv, [process.cwd()]);
}

// Merge with existing configuration
async function extendConfiguration(baseConfigPath: string) {
  const baseConfig = require(baseConfigPath);
  
  const extendedConfig = {
    ...baseConfig,
    collectCoverage: true,
    coverageThreshold: {
      global: {
        branches: 90,
        functions: 90,
        lines: 90,
        statements: 90
      }
    }
  };
  
  const argv = await buildArgv([
    `--config=${JSON.stringify(extendedConfig)}`
  ]);
  
  return runCLI(argv, [process.cwd()]);
}
```

### Environment-Specific Configuration

**Development Configuration:**

```javascript
// jest.dev.config.js
module.exports = {
  ...require('./jest.config.js'),
  
  // Development specific settings
  verbose: true,
  collectCoverage: false,
  watchAll: true,
  
  // Faster transforms for development
  transform: {
    "^.+\\.jsx?$": "babel-jest"
  }
};
```

**CI Configuration:**

```javascript
// jest.ci.config.js
module.exports = {
  ...require('./jest.config.js'),
  
  // CI specific settings
  ci: true,
  collectCoverage: true,
  coverageReporters: ["text", "lcov"],
  maxWorkers: "50%",
  cache: false,
  
  // Strict coverage requirements
  coverageThreshold: {
    global: {
      branches: 90,
      functions: 90,
      lines: 90,
      statements: 90
    }
  }
};
```

### Advanced Configuration Patterns

**Dynamic Configuration:**

```javascript
// jest.config.js
const isCI = process.env.CI === "true";
const isDevelopment = process.env.NODE_ENV === "development";

module.exports = {
  testMatch: ["**/__tests__/**/*.js", "**/?(*.)+(spec|test).js"],
  
  // Dynamic coverage settings
  collectCoverage: isCI,
  coverageReporters: isCI 
    ? ["text", "lcov"] 
    : ["text-summary"],
  
  // Dynamic worker settings
  maxWorkers: isCI ? "50%" : 1,
  
  // Development specific settings
  verbose: isDevelopment,
  silent: isCI,
  
  // Environment specific transforms
  transform: isDevelopment 
    ? { "^.+\\.jsx?$": "babel-jest" }
    : { "^.+\\.jsx?$": ["babel-jest", { cacheDirectory: false }] }
};
```

**Conditional Test Patterns:**

```javascript
module.exports = {
  // Base test patterns
  testMatch: ["**/__tests__/**/*.js"],
  
  // Add integration tests in CI
  ...(process.env.CI && {
    testMatch: [
      "**/__tests__/**/*.js",
      "**/integration/**/*.test.js"
    ]
  }),
  
  // Add e2e tests for full test runs
  ...(process.env.FULL_TEST_SUITE && {
    testMatch: [
      "**/__tests__/**/*.js",
      "**/integration/**/*.test.js",
      "**/e2e/**/*.test.js"
    ]
  })
};
```

Jest's configuration system provides complete control over test execution behavior, enabling optimization for different environments and use cases while maintaining consistency across development workflows.