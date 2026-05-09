# Node-Side Runner APIs

Programmatic APIs for running Vitest from Node.js, useful for custom test runners and integrations. Import from `vitest/node`.

## Capabilities

### Creating Vitest Instance

```typescript { .api }
/**
 * Create a Vitest instance
 * @param mode - Run mode ('test' or 'benchmark')
 * @param options - User configuration
 * @param viteOverrides - Vite-specific overrides
 * @param vitestOptions - Additional Vitest options
 * @returns Promise resolving to Vitest instance
 */
function createVitest(
  mode: 'test' | 'benchmark',
  options: UserConfig,
  viteOverrides?: ViteUserConfig,
  vitestOptions?: VitestOptions
): Promise<Vitest>;

/**
 * Start Vitest from CLI
 * @param cliFilters - Test file filters from CLI
 * @param options - CLI options
 * @param viteOverrides - Vite overrides
 * @returns Promise resolving to Vitest instance or undefined
 */
function startVitest(
  cliFilters: string[],
  options: CliOptions,
  viteOverrides?: ViteUserConfig
): Promise<Vitest | undefined>;
```

**Usage:**

```typescript
import { createVitest } from 'vitest/node';

const vitest = await createVitest('test', {
  watch: false,
  globals: true
});

await vitest.start();
await vitest.close();
```

### Vitest Class

Main test runner class.

```typescript { .api }
class Vitest {
  /** Vitest version */
  static version: string;

  /** Start test execution */
  start(filters?: string[]): Promise<void>;

  /** Close Vitest and cleanup */
  close(): Promise<void>;

  /** Re-run tests */
  rerun(files?: string[]): Promise<void>;

  /** Get test results */
  getTestResults(): Promise<TestResult[]>;

  /** Configuration */
  config: ResolvedConfig;

  /** Test projects */
  projects: TestProject[];
}
```

### Reporter Classes

Built-in reporters for test output formatting.

```typescript { .api }
/**
 * Base reporter class
 */
abstract class BaseReporter {
  onInit(ctx: Vitest): void;
  onPathsCollected(paths?: string[]): void;
  onCollected(files?: File[]): Promise<void> | void;
  onFinished(files?: File[], errors?: unknown[]): Promise<void> | void;
  onTaskUpdate(task: [string, TaskResult | undefined][]): void;
  onTestRemoved(trigger?: string): void;
  onWatcherStart(files?: File[], errors?: unknown[]): void;
  onWatcherRerun(files: string[], trigger?: string): void;
  onServerRestart(reason?: string): void;
  onProcessTimeout(): void;
}

/**
 * Default colored console reporter
 */
class DefaultReporter extends BaseReporter {}

/**
 * Verbose reporter with detailed output
 */
class VerboseReporter extends BaseReporter {}

/**
 * Minimal dot reporter (. = pass, x = fail)
 */
class DotReporter extends BaseReporter {}

/**
 * Tree-structured output
 */
class TreeReporter extends BaseReporter {}

/**
 * JSON output reporter
 */
class JsonReporter extends BaseReporter {}

/**
 * Binary blob reporter for test results
 */
class BlobReporter extends BaseReporter {}

/**
 * TAP (Test Anything Protocol) reporter
 */
class TapReporter extends BaseReporter {}

/**
 * Flat TAP reporter
 */
class TapFlatReporter extends BaseReporter {}

/**
 * JUnit XML reporter
 */
class JUnitReporter extends BaseReporter {}

/**
 * GitHub Actions reporter
 */
class GithubActionsReporter extends BaseReporter {}

/**
 * Hanging process detector
 */
class HangingProcessReporter extends BaseReporter {}
```

**Usage:**

```typescript
import { createVitest, DefaultReporter, JsonReporter } from 'vitest/node';

const vitest = await createVitest('test', {
  reporters: [new DefaultReporter(), new JsonReporter()]
});
```

### Test Sequencers

Control test execution order.

```typescript { .api }
/**
 * Base sequencer class
 */
abstract class BaseSequencer {
  abstract sort(files: File[]): Promise<File[]>;
  abstract shard(files: File[]): Promise<File[]>;
}

/**
 * Random test sequencer
 */
class RandomSequencer extends BaseSequencer {
  sort(files: File[]): Promise<File[]>;
  shard(files: File[]): Promise<File[]>;
}
```

**Usage:**

```typescript
import { defineConfig } from 'vitest/config';
import { RandomSequencer } from 'vitest/node';

export default defineConfig({
  test: {
    sequence: {
      sequencer: RandomSequencer
    }
  }
});
```

### Test Collection Classes

```typescript { .api }
class TestProject {
  name: string;
  config: ResolvedConfig;

  globTestFiles(filters?: string[]): Promise<string[]>;
  isTestFile(file: string): boolean;
}

class TestModule {
  id: string;
  filepath: string;
  tasks: (TestCase | TestSuite)[];
}

class TestSuite {
  id: string;
  name: string;
  tasks: (TestCase | TestSuite)[];
}

class TestCase {
  id: string;
  name: string;
  result?: TestResult;
}
```

### Configuration Resolution

```typescript { .api }
/**
 * Resolve Vitest configuration
 * @param config - User configuration
 * @param mode - Run mode
 * @returns Promise resolving to resolved configuration
 */
function resolveConfig(
  config?: UserConfig,
  mode?: 'test' | 'benchmark'
): Promise<ResolvedConfig>;
```

### CLI Parsing

```typescript { .api }
/**
 * Parse CLI arguments
 * @param args - Command line arguments
 * @param options - Parse options
 * @returns Parsed options and filters
 */
function parseCLI(
  args: string[],
  options?: CliParseOptions
): { options: CliOptions; filters: string[] };
```

## Common Use Cases

### Custom Test Runner

```typescript
import { createVitest, DefaultReporter } from 'vitest/node';

async function runTests() {
  const vitest = await createVitest('test', {
    watch: false,
    reporters: [new DefaultReporter()]
  });

  await vitest.start(['**/*.test.ts']);

  const results = await vitest.getTestResults();
  const failed = results.some(r => r.state === 'fail');

  await vitest.close();

  process.exit(failed ? 1 : 0);
}

runTests();
```

### Custom Reporter

```typescript
import { BaseReporter } from 'vitest/node';
import type { File } from 'vitest';

class CustomReporter extends BaseReporter {
  onCollected(files?: File[]) {
    console.log(`Collected ${files?.length} test files`);
  }

  onFinished(files?: File[], errors?: unknown[]) {
    const total = files?.reduce((acc, file) => {
      return acc + file.tasks.length;
    }, 0) || 0;

    console.log(`Total tests: ${total}`);

    if (errors?.length) {
      console.error(`Errors: ${errors.length}`);
    }
  }
}

// Use in config
export default defineConfig({
  test: {
    reporters: [new CustomReporter()]
  }
});
```

### Programmatic Test Execution with Filtering

```typescript
import { createVitest } from 'vitest/node';

async function runSpecificTests() {
  const vitest = await createVitest('test', {});

  // Run only tests matching pattern
  await vitest.start(['src/utils/*.test.ts']);

  // Or rerun specific files
  await vitest.rerun(['src/utils/math.test.ts']);

  await vitest.close();
}
```

## Type Definitions

```typescript { .api }
interface Vitest {
  version: string;
  config: ResolvedConfig;
  projects: TestProject[];
  start(filters?: string[]): Promise<void>;
  close(): Promise<void>;
  rerun(files?: string[]): Promise<void>;
  getTestResults(): Promise<TestResult[]>;
}

interface VitestOptions {
  stdin?: NodeJS.ReadStream;
  stdout?: NodeJS.WriteStream;
  stderr?: NodeJS.WriteStream;
}

interface CliOptions extends UserConfig {
  run?: boolean;
  watch?: boolean;
  reporter?: string[];
  outputFile?: string;
}

type TestResult = TestResultPassed | TestResultFailed | TestResultSkipped;

interface TestResultPassed {
  state: 'pass';
  duration: number;
}

interface TestResultFailed {
  state: 'fail';
  error: Error;
  duration: number;
}

interface TestResultSkipped {
  state: 'skip';
}
```
