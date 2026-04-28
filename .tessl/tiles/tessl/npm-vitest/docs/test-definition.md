# Test Definition and Lifecycle

API for defining tests, organizing them into suites, and managing test lifecycle with hooks.

## Test Definition

```typescript { .api }
function test(name: string, fn: TestFunction, options?: TestOptions): void;
function it(name: string, fn: TestFunction, options?: TestOptions): void; // Alias

interface TestFunction {
  (context: TestContext): void | Promise<void>;
}

interface TestOptions {
  timeout?: number;      // Test timeout in ms
  retry?: number;        // Retries on failure
  concurrent?: boolean;  // Run concurrently
  repeats?: number;      // Repeat test N times
}

interface TestContext {
  meta: Record<string, any>;  // Metadata storage
  task: RunnerTestCase;        // Test task object
  expect: ExpectStatic;        // Expect function
}
```

**Example:**
```typescript
test('basic test', () => {
  expect(1 + 2).toBe(3);
});

test('async test', async () => {
  const data = await fetchData();
  expect(data).toBeDefined();
});

test('with timeout', async () => {
  await longOperation();
}, { timeout: 10000 });

test('with retry', () => {
  // Flaky test code
}, { retry: 3 });
```

## Suite Definition

```typescript { .api }
function describe(name: string, fn: () => void, options?: TestOptions): void;
function suite(name: string, fn: () => void, options?: TestOptions): void; // Alias
```

**Example:**
```typescript
describe('Math operations', () => {
  test('addition', () => expect(1 + 2).toBe(3));
  test('subtraction', () => expect(5 - 2).toBe(3));

  describe('multiplication', () => {
    test('positive', () => expect(2 * 3).toBe(6));
    test('negative', () => expect(-2 * -3).toBe(6));
  });
});
```

## Test Modifiers

| Modifier | Purpose | Example |
|----------|---------|---------|
| `.only` | Run only this test/suite | `test.only('runs', ...)` |
| `.skip` | Skip this test/suite | `test.skip('skipped', ...)` |
| `.todo` | Mark as not implemented | `test.todo('implement later')` |
| `.skipIf(cond)` | Skip if condition true | `test.skipIf(isWindows)` |
| `.runIf(cond)` | Run only if condition true | `test.runIf(hasEnv)` |
| `.concurrent` | Run concurrently | `test.concurrent('parallel', ...)` |
| `.sequential` | Run sequentially | `test.sequential('serial', ...)` |
| `.fails` | Expect test to fail | `test.fails('expected fail', ...)` |
| `.each(data)` | Parameterized test | `test.each([...])('test %i', ...)` |

**Examples:**
```typescript
// Run only specific tests
test.only('focused test', () => { /* runs */ });
test('other test', () => { /* skipped */ });

// Conditional execution
test.skipIf(process.platform === 'win32')('Unix only', () => { ... });
test.runIf(process.env.INTEGRATION)('integration test', () => { ... });

// Concurrent execution
describe.concurrent('parallel suite', () => {
  test('test 1', async () => { /* runs in parallel */ });
  test('test 2', async () => { /* runs in parallel */ });
});

// Expected failures
test.fails('expected to fail', () => {
  expect(1).toBe(2); // Test passes because it fails as expected
});

// Parameterized tests
test.each([
  [1, 1, 2],
  [2, 2, 4],
  [3, 3, 6]
])('adds %i + %i = %i', (a, b, expected) => {
  expect(a + b).toBe(expected);
});
```

## Lifecycle Hooks

```typescript { .api }
function beforeAll(fn: () => void | Promise<void>, timeout?: number): void;
function afterAll(fn: () => void | Promise<void>, timeout?: number): void;
function beforeEach(fn: (context: TestContext) => void | Promise<void>, timeout?: number): void;
function afterEach(fn: (context: TestContext) => void | Promise<void>, timeout?: number): void;
```

| Hook | Runs | Use Case |
|------|------|----------|
| `beforeAll` | Once before all tests | Expensive setup (DB connection) |
| `afterAll` | Once after all tests | Cleanup (close connections) |
| `beforeEach` | Before each test | Reset state, create instances |
| `afterEach` | After each test | Cleanup, restore mocks |

**Example:**
```typescript
describe('Database tests', () => {
  let db;

  beforeAll(async () => {
    db = await connectToDatabase();
    await db.migrate();
  });

  afterAll(async () => {
    await db.close();
  });

  beforeEach(async () => {
    await db.clear();
    await db.seed();
  });

  afterEach(async () => {
    await db.rollback();
  });

  test('creates user', async () => {
    const user = await db.createUser({ name: 'John' });
    expect(user.id).toBeDefined();
  });

  test('finds user', async () => {
    await db.createUser({ name: 'John' });
    const user = await db.findUser('John');
    expect(user).toBeDefined();
  });
});
```

## Test Event Handlers

```typescript { .api }
function onTestFailed(handler: (result: RunnerTestCase, error: unknown) => void | Promise<void>): void;
function onTestFinished(handler: (result: RunnerTestCase) => void | Promise<void>): void;
```

**Example:**
```typescript
test('with event handlers', () => {
  onTestFailed((result, error) => {
    console.log(`Test "${result.name}" failed:`, error);
  });

  onTestFinished((result) => {
    console.log(`Test "${result.name}" finished: ${result.result?.state}`);
  });

  expect(1).toBe(1);
});
```

## Test Modifiers API

```typescript { .api }
interface TestAPI {
  only: TestAPI;
  skip: TestAPI;
  todo: TestAPI;
  skipIf(condition: boolean): TestAPI;
  runIf(condition: boolean): TestAPI;
  concurrent: TestAPI;
  sequential: TestAPI;
  fails: TestAPI;
  each<T>(cases: T[]): (name: string, fn: (...args: T[]) => void) => void;
}

interface SuiteAPI {
  only: SuiteAPI;
  skip: SuiteAPI;
  todo: SuiteAPI;
  skipIf(condition: boolean): SuiteAPI;
  runIf(condition: boolean): SuiteAPI;
  concurrent: SuiteAPI;
  sequential: SuiteAPI;
}
```

## Execution Control Patterns

### Focus on Specific Tests

```typescript
describe('feature', () => {
  test.only('this runs', () => { /* only this */ });
  test('skipped', () => { /* skipped */ });
});

describe.only('focused suite', () => {
  test('runs', () => { /* runs because suite focused */ });
});
```

### Skip Tests

```typescript
test.skip('not ready', () => { /* skipped */ });

describe.skip('disabled suite', () => {
  test('also skipped', () => { /* skipped */ });
});
```

### Parallel vs Sequential

```typescript
// Parallel (for independent tests)
describe.concurrent('parallel', () => {
  test('test 1', async () => { await delay(100); });
  test('test 2', async () => { await delay(100); });
  // Both run simultaneously
});

// Sequential (default, for tests with shared state)
describe.sequential('sequential', () => {
  test('test 1', () => { /* runs first */ });
  test('test 2', () => { /* runs second */ });
});
```

## Type Definitions

```typescript { .api }
interface RunnerTestCase {
  id: string;
  name: string;
  mode: 'run' | 'skip' | 'only' | 'todo';
  suite?: RunnerTestSuite;
  result?: RunnerTaskResult;
  meta: Record<string, any>;
}

interface RunnerTestSuite {
  id: string;
  name: string;
  mode: 'run' | 'skip' | 'only' | 'todo';
  tasks: (RunnerTestCase | RunnerTestSuite)[];
  meta: Record<string, any>;
}

interface RunnerTaskResult {
  state: 'pass' | 'fail' | 'skip';
  duration?: number;
  error?: unknown;
  retryCount?: number;
}

interface TaskCustomOptions {
  timeout?: number;
  retry?: number;
  repeats?: number;
  concurrent?: boolean;
}
```

## Common Patterns

### Parameterized Tests

```typescript
// Array of inputs
test.each([
  { input: 'hello', expected: 5 },
  { input: 'world', expected: 5 },
  { input: 'foo', expected: 3 }
])('length of "$input" is $expected', ({ input, expected }) => {
  expect(input.length).toBe(expected);
});

// Multiple arguments
test.each([
  [1, 1, 2],
  [1, 2, 3],
  [2, 1, 3]
])('.add(%i, %i) = %i', (a, b, expected) => {
  expect(a + b).toBe(expected);
});
```

### Conditional Tests

```typescript
const isCI = !!process.env.CI;
const isWindows = process.platform === 'win32';

test.skipIf(isWindows)('Unix-specific feature', () => { ... });
test.runIf(isCI)('CI-only test', () => { ... });
test.skipIf(!process.env.API_KEY)('needs API key', () => { ... });
```

### Shared Setup with Context

```typescript
describe('user operations', () => {
  let user;

  beforeEach(() => {
    user = createUser();
  });

  test('updates name', () => {
    user.name = 'John';
    expect(user.name).toBe('John');
  });

  test('updates email', () => {
    user.email = 'john@example.com';
    expect(user.email).toBe('john@example.com');
  });
});
```

## Best Practices

1. **Use `beforeEach` for test isolation** - Each test gets fresh state
2. **Use `beforeAll` for expensive setup** - Database connections, file reads
3. **Always clean up in `afterEach/afterAll`** - Prevent test pollution
4. **Use `.only` during debugging** - Focus on specific failing tests
5. **Use `.concurrent` for independent tests** - Faster test execution
6. **Use `.sequential` for shared state** - Tests that modify shared resources
7. **Use `.each` for similar tests** - Reduce code duplication

## Lifecycle Order

```
beforeAll (suite)
├─ beforeEach (test 1)
│  ├─ test 1
│  └─ afterEach (test 1)
├─ beforeEach (test 2)
│  ├─ test 2
│  └─ afterEach (test 2)
└─ afterAll (suite)
```

Nested suites:
```
beforeAll (outer)
├─ beforeEach (outer)
│  ├─ beforeAll (inner)
│  │  ├─ beforeEach (inner)
│  │  │  ├─ test
│  │  │  └─ afterEach (inner)
│  │  └─ afterAll (inner)
│  └─ afterEach (outer)
└─ afterAll (outer)
```
