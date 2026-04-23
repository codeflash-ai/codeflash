# Reporters

Vitest reporters format and output test results. Import from `vitest/reporters`.

## Built-in Reporters

11 built-in reporter classes are available:

### DefaultReporter

Default colored console output with test hierarchy.

```typescript { .api }
class DefaultReporter extends BaseReporter {
  // Colored output with test results
  // Shows: pass (✓), fail (✕), skip (-)
}
```

**Usage:**

```typescript
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    reporters: ['default'] // or omit for default
  }
});
```

### VerboseReporter

Extended test information with full output.

```typescript { .api }
class VerboseReporter extends BaseReporter {
  // Detailed output including test names, durations, and full error stacks
}
```

**Usage:**

```typescript
export default defineConfig({
  test: {
    reporters: ['verbose']
  }
});
```

### DotReporter

Minimal dot-based output.

```typescript { .api }
class DotReporter extends BaseReporter {
  // Outputs: . (pass), x (fail), - (skip)
}
```

### TreeReporter

Tree-structured test hierarchy output.

```typescript { .api }
class TreeReporter extends BaseReporter {
  // Tree view of test suites and tests
}
```

### JsonReporter

JSON-formatted test results.

```typescript { .api }
class JsonReporter extends BaseReporter {
  // Outputs JSON test results to file
}

interface JsonTestResults {
  numTotalTests: number;
  numPassedTests: number;
  numFailedTests: number;
  numPendingTests: number;
  testResults: JsonTestResult[];
}

interface JsonTestResult {
  name: string;
  status: 'passed' | 'failed' | 'skipped';
  duration: number;
  assertionResults: JsonAssertionResult[];
}
```

**Usage:**

```typescript
export default defineConfig({
  test: {
    reporters: ['json'],
    outputFile: './test-results.json'
  }
});
```

### JUnitReporter

JUnit XML format output.

```typescript { .api }
class JUnitReporter extends BaseReporter {
  // Outputs JUnit XML format compatible with CI systems
}
```

**Usage:**

```typescript
export default defineConfig({
  test: {
    reporters: ['junit'],
    outputFile: './junit.xml'
  }
});
```

### TapReporter

TAP (Test Anything Protocol) format.

```typescript { .api }
class TapReporter extends BaseReporter {
  // TAP format output
}
```

### TapFlatReporter

Flat TAP format output.

```typescript { .api }
class TapFlatReporter extends BaseReporter {
  // Flattened TAP output
}
```

### GithubActionsReporter

GitHub Actions format for CI integration.

```typescript { .api }
class GithubActionsReporter extends BaseReporter {
  // Formats errors for GitHub Actions annotations
}
```

**Usage:**

```typescript
export default defineConfig({
  test: {
    reporters: process.env.GITHUB_ACTIONS
      ? ['github-actions']
      : ['default']
  }
});
```

### HangingProcessReporter

Detects and reports hanging processes after tests complete.

```typescript { .api }
class HangingProcessReporter extends BaseReporter {
  // Detects processes preventing test exit
}
```

### BlobReporter

Binary format for test results.

```typescript { .api }
class BlobReporter extends BaseReporter {
  // Outputs binary blob format
}
```

## Custom Reporters

Create custom reporters by extending `BaseReporter`.

```typescript { .api }
abstract class BaseReporter {
  /**
   * Called when Vitest is initialized
   */
  onInit(ctx: Vitest): void;

  /**
   * Called when test paths are collected
   */
  onPathsCollected(paths?: string[]): void;

  /**
   * Called when tests are collected from files
   */
  onCollected(files?: File[]): Promise<void> | void;

  /**
   * Called when all tests finish
   */
  onFinished(
    files?: File[],
    errors?: unknown[]
  ): Promise<void> | void;

  /**
   * Called when a task updates
   */
  onTaskUpdate(task: [string, TaskResult | undefined][]): void;

  /**
   * Called when a test is removed
   */
  onTestRemoved(trigger?: string): void;

  /**
   * Called when watcher starts
   */
  onWatcherStart(files?: File[], errors?: unknown[]): void;

  /**
   * Called when watcher reruns tests
   */
  onWatcherRerun(files: string[], trigger?: string): void;

  /**
   * Called when server restarts
   */
  onServerRestart(reason?: string): void;

  /**
   * Called when process times out
   */
  onProcessTimeout(): void;
}
```

**Usage:**

```typescript
import { BaseReporter } from 'vitest/reporters';
import type { File, Vitest } from 'vitest';

class MyCustomReporter extends BaseReporter {
  onInit(ctx: Vitest) {
    console.log('Tests starting...');
  }

  onCollected(files?: File[]) {
    console.log(`Collected ${files?.length} test files`);
  }

  onFinished(files?: File[], errors?: unknown[]) {
    if (errors?.length) {
      console.error(`Failed with ${errors.length} errors`);
    } else {
      console.log('All tests passed!');
    }
  }
}

export default defineConfig({
  test: {
    reporters: [new MyCustomReporter()]
  }
});
```

## Multiple Reporters

Use multiple reporters simultaneously.

```typescript
export default defineConfig({
  test: {
    reporters: ['default', 'json', 'junit'],
    outputFile: {
      json: './results.json',
      junit: './junit.xml'
    }
  }
});
```

## Reporter Registry

```typescript { .api }
/**
 * Registry of built-in reporters
 */
const ReportersMap: Record<BuiltinReporters, typeof BaseReporter>;

type BuiltinReporters =
  | 'default'
  | 'verbose'
  | 'dot'
  | 'tree'
  | 'json'
  | 'blob'
  | 'tap'
  | 'tap-flat'
  | 'junit'
  | 'github-actions'
  | 'hanging-process';
```

## Benchmark Reporters

Specialized reporters for benchmarking.

```typescript { .api }
class BenchmarkReporter {
  onInit(ctx: Vitest): void;
  onFinished(files?: File[], errors?: unknown[]): Promise<void> | void;
}

class VerboseBenchmarkReporter extends BenchmarkReporter {
  // Detailed benchmark results with statistics
}

type BenchmarkBuiltinReporters = 'default' | 'verbose';
```

**Usage:**

```typescript
export default defineConfig({
  test: {
    benchmark: {
      reporters: ['verbose']
    }
  }
});
```

## Common Patterns

### CI-Specific Reporters

```typescript
export default defineConfig({
  test: {
    reporters: process.env.CI
      ? ['junit', 'github-actions']
      : ['default']
  }
});
```

### Custom Metrics Reporter

```typescript
import { BaseReporter } from 'vitest/reporters';

class MetricsReporter extends BaseReporter {
  private startTime: number = 0;

  onInit() {
    this.startTime = Date.now();
  }

  onFinished(files?: File[]) {
    const duration = Date.now() - this.startTime;
    const total = files?.reduce((acc, f) => acc + f.tasks.length, 0) || 0;

    const passed = files?.reduce((acc, f) => {
      return acc + f.tasks.filter(t => t.result?.state === 'pass').length;
    }, 0) || 0;

    console.log({
      duration,
      total,
      passed,
      passRate: (passed / total * 100).toFixed(2) + '%'
    });
  }
}
```

### Conditional Output

```typescript
import { BaseReporter } from 'vitest/reporters';

class QuietReporter extends BaseReporter {
  onFinished(files?: File[], errors?: unknown[]) {
    // Only output if there are errors
    if (errors?.length) {
      console.error(`Tests failed with ${errors.length} errors`);
      errors.forEach(e => console.error(e));
    }
  }
}
```

## Type Definitions

```typescript { .api }
interface File {
  filepath: string;
  tasks: Task[];
  result?: TaskResult;
}

interface Task {
  id: string;
  name: string;
  mode: 'run' | 'skip' | 'only' | 'todo';
  result?: TaskResult;
}

interface TaskResult {
  state: 'pass' | 'fail' | 'skip';
  duration?: number;
  error?: Error;
}

interface Reporter {
  onInit?(ctx: Vitest): void;
  onPathsCollected?(paths?: string[]): void;
  onCollected?(files?: File[]): Promise<void> | void;
  onFinished?(files?: File[], errors?: unknown[]): Promise<void> | void;
  onTaskUpdate?(task: [string, TaskResult | undefined][]): void;
}
```
