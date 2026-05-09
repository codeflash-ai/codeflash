# Configuration

Type-safe configuration APIs for Vitest with TypeScript support.

## Configuration Functions

```typescript { .api }
import { defineConfig, defineProject } from 'vitest/config';

// Main config
function defineConfig(config: UserConfig): UserConfig;

// Workspace project config
function defineProject(config: UserWorkspaceConfig): UserWorkspaceConfig;
```

**Basic Example:**
```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './setup.ts',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html']
    }
  }
});
```

## Core Configuration Options

```typescript { .api }
interface TestProjectConfiguration {
  // Test file patterns
  include?: string[];                    // Default: ['**/*.{test,spec}.?(c|m)[jt]s?(x)']
  exclude?: string[];                    // Default: ['**/node_modules/**', ...]

  // Environment
  environment?: 'node' | 'jsdom' | 'happy-dom' | 'edge-runtime' | string;
  globals?: boolean;                     // Enable global APIs (test, expect, etc.)

  // Setup & Timeouts
  setupFiles?: string | string[];
  testTimeout?: number;                  // Default: 5000ms
  hookTimeout?: number;                  // Default: 10000ms

  // Execution
  pool?: 'threads' | 'forks' | 'vmThreads' | 'vmForks' | 'browser';
  poolOptions?: PoolOptions;
  isolate?: boolean;                     // Isolate each test file
  maxConcurrency?: number;               // Max concurrent tests
  sequence?: SequenceOptions;            // Test ordering

  // Retry & Repeat
  retry?: number;                        // Retry failed tests N times
  repeats?: number;                      // Repeat each test N times

  // Reporting
  reporters?: Array<Reporter | string>;
  outputFile?: string | Record<string, string>;

  // Coverage
  coverage?: CoverageOptions;

  // Browser Testing
  browser?: BrowserConfigOptions;

  // Typecheck
  typecheck?: TypecheckConfig;

  // Watch Mode
  watch?: boolean;
  watchExclude?: string[];

  // Mocking
  mockReset?: boolean;                   // Reset mocks before each test
  restoreMocks?: boolean;                // Restore mocks after each test
  clearMocks?: boolean;                  // Clear mock history before each test
  unstubGlobals?: boolean;               // Unstub globals after each test
  unstubEnvs?: boolean;                  // Unstub env vars after each test

  // Multi-project
  projects?: Array<UserWorkspaceConfig>;
  workspace?: string;

  // Sharding
  shard?: { index: number; count: number };

  // Snapshot
  snapshotFormat?: SnapshotStateOptions;

  // CSS
  css?: boolean | { include?: RegExp | RegExp[]; modules?: { classNameStrategy?: string } };
}
```

## Configuration Defaults

```typescript { .api }
configDefaults: Partial<UserConfig>;
defaultInclude: string[];               // ['**/*.{test,spec}.?(c|m)[jt]s?(x)']
defaultExclude: string[];               // ['**/node_modules/**', '**/dist/**', ...]
defaultBrowserPort: number;             // 63315
benchmarkConfigDefaults: BenchmarkConfig;
coverageConfigDefaults: CoverageOptions;
fakeTimersDefaults: FakeTimerInstallOpts;
```

**Usage:**
```typescript
import { defaultInclude, defaultExclude, defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: [...defaultInclude, '**/*.integration.test.ts'],
    exclude: [...defaultExclude, '**/fixtures/**']
  }
});
```

## Coverage Configuration

```typescript { .api }
interface CoverageOptions {
  provider?: 'v8' | 'istanbul';
  enabled?: boolean;
  include?: string[];
  exclude?: string[];
  reporter?: Array<'text' | 'json' | 'html' | 'lcov' | 'text-summary' | string>;
  reportsDirectory?: string;
  all?: boolean;

  // Thresholds
  lines?: number;
  functions?: number;
  branches?: number;
  statements?: number;

  clean?: boolean;
  extension?: string | string[];
}
```

**Example:**
```typescript
export default defineConfig({
  test: {
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['src/**/*.ts'],
      exclude: ['**/*.test.ts', '**/*.spec.ts'],
      lines: 80,
      functions: 80,
      branches: 80,
      statements: 80
    }
  }
});
```

## Pool Options

```typescript { .api }
interface PoolOptions {
  threads?: {
    maxThreads?: number;
    minThreads?: number;
    singleThread?: boolean;
    useAtomics?: boolean;
    isolate?: boolean;
  };

  forks?: {
    maxForks?: number;
    minForks?: number;
    singleFork?: boolean;
    isolate?: boolean;
  };

  vmThreads?: {
    maxThreads?: number;
    minThreads?: number;
    memoryLimit?: string | number;
    useAtomics?: boolean;
  };

  vmForks?: {
    maxForks?: number;
    minForks?: number;
    memoryLimit?: string | number;
  };
}
```

**Example:**
```typescript
export default defineConfig({
  test: {
    pool: 'threads',
    poolOptions: {
      threads: {
        maxThreads: 4,
        minThreads: 1
      }
    }
  }
});
```

## Multiple Projects

```typescript
// vitest.config.ts
export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: 'unit',
          environment: 'node',
          include: ['src/**/*.test.ts']
        }
      },
      {
        test: {
          name: 'integration',
          environment: 'node',
          include: ['tests/**/*.integration.test.ts'],
          testTimeout: 30000
        }
      },
      {
        test: {
          name: 'browser',
          environment: 'jsdom',
          include: ['src/**/*.browser.test.ts']
        }
      }
    ]
  }
});
```

## Browser Configuration

```typescript { .api }
interface BrowserConfigOptions {
  enabled?: boolean;
  name?: 'chrome' | 'firefox' | 'safari' | 'edge';
  provider?: 'playwright' | 'webdriverio' | 'preview';
  headless?: boolean;
  screenshotFailures?: boolean;
  viewport?: { width: number; height: number };
  providerOptions?: Record<string, any>;
  api?: { port?: number; host?: string };
  scripts?: BrowserScript[];
  isolate?: boolean;
  fileParallelism?: boolean;
}
```

**Example:**
```typescript
export default defineConfig({
  test: {
    browser: {
      enabled: true,
      name: 'chrome',
      provider: 'playwright',
      headless: true,
      viewport: {
        width: 1280,
        height: 720
      }
    }
  }
});
```

## Typecheck Configuration

```typescript { .api }
interface TypecheckConfig {
  enabled?: boolean;
  checker?: 'tsc' | 'vue-tsc';
  include?: string[];
  exclude?: string[];
  allowJs?: boolean;
  ignoreSourceErrors?: boolean;
  tsconfig?: string;
}
```

**Example:**
```typescript
export default defineConfig({
  test: {
    typecheck: {
      enabled: true,
      checker: 'tsc',
      include: ['**/*.{test,spec}-d.ts']
    }
  }
});
```

## Runtime Configuration

Control configuration at runtime within tests.

```typescript { .api }
vi.setConfig(config: RuntimeOptions): void;
vi.resetConfig(): void;

interface RuntimeOptions {
  allowOnly?: boolean;
  sequence?: { shuffle?: boolean; seed?: number };
  testTimeout?: number;
  hookTimeout?: number;
}
```

**Example:**
```typescript
test('with custom timeout', () => {
  vi.setConfig({ testTimeout: 10000 });
  // Test runs with 10s timeout
  vi.resetConfig(); // Reset to defaults
});
```

## Common Configurations

### JSDOM Environment

```typescript
export default defineConfig({
  test: {
    environment: 'jsdom',
    setupFiles: './setup.ts'
  }
});
```

### Watch Mode

```typescript
export default defineConfig({
  test: {
    watch: false,
    watchExclude: ['**/node_modules/**', '**/dist/**']
  }
});
```

### Multi-threaded

```typescript
export default defineConfig({
  test: {
    pool: 'threads',
    poolOptions: {
      threads: {
        maxThreads: 4,
        minThreads: 1
      }
    }
  }
});
```

### Global Setup/Teardown

```typescript
// setup.ts
import { beforeAll, afterAll } from 'vitest';

beforeAll(async () => {
  await startDatabase();
});

afterAll(async () => {
  await stopDatabase();
});
```

```typescript
// vitest.config.ts
export default defineConfig({
  test: {
    setupFiles: ['./setup.ts'],
    globals: true
  }
});
```

## Configuration Decision Guide

| Need | Option | Value |
|------|--------|-------|
| Browser APIs | `environment` | `'jsdom'` or `'happy-dom'` |
| Real browser | `browser.enabled` | `true` |
| Global test APIs | `globals` | `true` |
| Coverage | `coverage.provider` | `'v8'` or `'istanbul'` |
| Fast execution | `pool` | `'threads'` |
| Isolation | `pool` | `'forks'` or `'vmForks'` |
| TypeScript types | `typecheck.enabled` | `true` |
| Debug one test | `test.only` | In test file |
| CI environment | `reporters` | `['junit', 'github-actions']` |

## Environment Selection

| Environment | Use Case | APIs |
|-------------|----------|------|
| `node` | Server code, CLI | Node.js APIs |
| `jsdom` | Browser simulation | DOM, window, document |
| `happy-dom` | Faster browser simulation | DOM (lighter, faster) |
| `edge-runtime` | Edge functions | Web APIs subset |
| Browser mode | Real browser | Full browser APIs |

## Type Definitions

```typescript { .api }
interface SequenceOptions {
  shuffle?: boolean;
  seed?: number;
  hooks?: 'stack' | 'list' | 'parallel';
}

interface BenchmarkConfig {
  include?: string[];
  exclude?: string[];
  outputFile?: string;
  reporters?: Array<string | BenchmarkReporter>;
}

type Reporter = BaseReporter | BuiltinReporters;

type BuiltinReporters =
  | 'default'
  | 'verbose'
  | 'dot'
  | 'tree'
  | 'json'
  | 'junit'
  | 'tap'
  | 'github-actions';
```
