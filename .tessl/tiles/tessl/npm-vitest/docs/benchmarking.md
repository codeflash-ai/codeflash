# Benchmarking

Vitest provides built-in benchmarking capabilities for measuring and comparing code performance with statistical analysis.

## Capabilities

### Benchmark Definition

Define benchmarks similar to tests using the `bench` function.

```typescript { .api }
/**
 * Define a benchmark
 * @param name - Benchmark name/description
 * @param fn - Benchmark function to measure
 * @param options - Optional benchmark options
 */
function bench(
  name: string,
  fn: BenchFunction,
  options?: BenchOptions
): void;

interface BenchFunction {
  /**
   * Benchmark implementation
   * @param options - Iteration options
   */
  (options: { count: number }): void | Promise<void>;
}

interface BenchOptions {
  /** Number of times to run the benchmark */
  time?: number;

  /** Number of iterations */
  iterations?: number;

  /** Warmup time in ms */
  warmupTime?: number;

  /** Warmup iterations */
  warmupIterations?: number;
}
```

**Usage:**

```typescript
import { bench, describe } from 'vitest';

describe('array operations', () => {
  bench('push', () => {
    const arr: number[] = [];
    for (let i = 0; i < 1000; i++) {
      arr.push(i);
    }
  });

  bench('spread', () => {
    let arr: number[] = [];
    for (let i = 0; i < 1000; i++) {
      arr = [...arr, i];
    }
  });

  bench('concat', () => {
    let arr: number[] = [];
    for (let i = 0; i < 1000; i++) {
      arr = arr.concat(i);
    }
  });
});
```

### Benchmark Modifiers

Benchmark functions support modifiers similar to tests.

```typescript { .api }
interface BenchmarkAPI {
  /** Run only this benchmark */
  only: BenchmarkAPI;

  /** Skip this benchmark */
  skip: BenchmarkAPI;

  /** Mark benchmark as todo */
  todo: BenchmarkAPI;

  /** Skip benchmark if condition is true */
  skipIf(condition: boolean): BenchmarkAPI;

  /** Run benchmark only if condition is true */
  runIf(condition: boolean): BenchmarkAPI;
}
```

**Usage:**

```typescript
import { bench } from 'vitest';

bench.only('focused benchmark', () => {
  // Only this runs
});

bench.skip('skipped benchmark', () => {
  // This is skipped
});

bench.todo('implement later');

bench.skipIf(process.platform === 'win32')('Unix only', () => {
  // Runs only on non-Windows
});
```

### Async Benchmarks

Benchmark async operations.

```typescript
import { bench } from 'vitest';

bench('async operation', async () => {
  await fetchData();
});

bench('promise chain', async () => {
  await fetch('/api/data')
    .then(res => res.json())
    .then(data => processData(data));
});
```

### Benchmark with Setup/Teardown

Use lifecycle hooks for setup and teardown.

```typescript
import { bench, describe, beforeEach, afterEach } from 'vitest';

describe('database operations', () => {
  let db;

  beforeEach(async () => {
    db = await connectToDatabase();
    await db.seed();
  });

  afterEach(async () => {
    await db.close();
  });

  bench('query users', async () => {
    await db.query('SELECT * FROM users');
  });

  bench('insert user', async () => {
    await db.insert('users', { name: 'John' });
  });
});
```

### Benchmark Options

Control benchmark execution with options.

```typescript
import { bench } from 'vitest';

bench('custom options', () => {
  // benchmark code
}, {
  time: 5000,           // Run for 5 seconds
  iterations: 100,      // Run 100 iterations
  warmupTime: 1000,     // Warmup for 1 second
  warmupIterations: 10  // 10 warmup iterations
});
```

## Benchmark Utilities

### Get Benchmark Context

Access benchmark function and options from suite utilities.

```typescript { .api }
/**
 * Get the benchmark function
 * @returns Benchmark function
 */
function getBenchFn(): BenchFunction;

/**
 * Get the benchmark options
 * @returns Benchmark options
 */
function getBenchOptions(): BenchOptions;
```

**Usage:**

```typescript
import { getBenchFn, getBenchOptions } from 'vitest/suite';

const benchFn = getBenchFn();
const options = getBenchOptions();
```

## Running Benchmarks

Run benchmarks with the `--mode benchmark` flag:

```bash
vitest bench
# or
vitest --mode benchmark
```

## Benchmark Configuration

Configure benchmarking in vitest.config.ts:

```typescript
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    benchmark: {
      include: ['**/*.bench.ts'],
      exclude: ['node_modules'],
      outputFile: './bench-results.json',
      reporters: ['default', 'verbose']
    }
  }
});
```

## Benchmark Reporters

Vitest provides specialized benchmark reporters:

```typescript { .api }
class BenchmarkReporter {
  onInit(ctx: Vitest): void;
  onFinished(files?: File[], errors?: unknown[]): Promise<void> | void;
}

class VerboseBenchmarkReporter extends BenchmarkReporter {
  // Detailed benchmark output
}
```

## Benchmark Results

Benchmark results include statistical analysis:

```typescript { .api }
interface BenchmarkResult {
  /** Benchmark name */
  name: string;

  /** Number of operations per second */
  hz: number;

  /** Minimum execution time */
  min: number;

  /** Maximum execution time */
  max: number;

  /** Mean execution time */
  mean: number;

  /** Median execution time */
  median: number;

  /** Standard deviation */
  sd: number;

  /** Margin of error */
  moe: number;

  /** Relative margin of error */
  rme: number;

  /** Sample size */
  samples: number[];
}
```

## Common Patterns

### Comparing Implementations

```typescript
import { bench, describe } from 'vitest';

describe('string concatenation', () => {
  const parts = ['a', 'b', 'c', 'd', 'e'];

  bench('plus operator', () => {
    let result = '';
    for (const part of parts) {
      result += part;
    }
  });

  bench('array join', () => {
    parts.join('');
  });

  bench('template literal', () => {
    `${parts[0]}${parts[1]}${parts[2]}${parts[3]}${parts[4]}`;
  });
});
```

### Testing Different Data Sizes

```typescript
import { bench, describe } from 'vitest';

for (const size of [10, 100, 1000, 10000]) {
  describe(`array operations (size: ${size})`, () => {
    const data = Array.from({ length: size }, (_, i) => i);

    bench('map', () => {
      data.map(x => x * 2);
    });

    bench('forEach', () => {
      const result: number[] = [];
      data.forEach(x => result.push(x * 2));
    });

    bench('for loop', () => {
      const result: number[] = [];
      for (let i = 0; i < data.length; i++) {
        result.push(data[i] * 2);
      }
    });
  });
}
```

## Type Definitions

```typescript { .api }
type BenchFunction = (options: { count: number }) => void | Promise<void>;

interface BenchOptions {
  time?: number;
  iterations?: number;
  warmupTime?: number;
  warmupIterations?: number;
}

interface BenchTask {
  name: string;
  fn: BenchFunction;
  options: BenchOptions;
  result?: BenchmarkResult;
}

interface Benchmark extends BenchTask {
  meta: Record<string, any>;
}
```
