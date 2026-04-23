# Mocking and Spying

Comprehensive mocking through the `vi` namespace (alias: `vitest`) for module mocking, function spying, and test isolation.

## Mock Functions

```typescript { .api }
vi.fn<Args extends any[], R>(implementation?: (...args: Args) => R): Mock<Args, R>;
vi.isMockFunction(fn: any): fn is Mock;

interface Mock<Args extends any[], R> {
  (...args: Args): R;

  mock: {
    calls: Args[];
    results: MockResult<R>[];
    instances: any[];
    invocationCallOrder: number[];
    lastCall?: Args;
  };

  mockImplementation(fn: (...args: Args) => R): this;
  mockImplementationOnce(fn: (...args: Args) => R): this;
  mockReturnValue(value: R): this;
  mockReturnValueOnce(value: R): this;
  mockResolvedValue(value: Awaited<R>): this;
  mockResolvedValueOnce(value: Awaited<R>): this;
  mockRejectedValue(value: any): this;
  mockRejectedValueOnce(value: any): this;
  mockClear(): this;
  mockReset(): this;
  mockRestore(): this;
  getMockName(): string;
  mockName(name: string): this;
}
```

**Examples:**
```typescript
// Basic mock
const mockFn = vi.fn((x: number) => x * 2);
mockFn(5); // Returns 10
expect(mockFn).toHaveBeenCalledWith(5);

// Return values
const fn = vi.fn();
fn.mockReturnValue(42);
fn(); // 42

fn.mockReturnValueOnce(100).mockReturnValueOnce(200);
fn(); // 100
fn(); // 200
fn(); // 42 (default)

// Async values
const asyncFn = vi.fn();
asyncFn.mockResolvedValue('success');
await asyncFn(); // 'success'

asyncFn.mockRejectedValue(new Error('failed'));
await asyncFn(); // throws Error

// Implementation
const addFn = vi.fn();
addFn.mockImplementation((a, b) => a + b);
addFn(2, 3); // 5

addFn.mockImplementationOnce((a, b) => a * b);
addFn(2, 3); // 6
addFn(2, 3); // 5 (back to default)
```

## Spying

```typescript { .api }
vi.spyOn<T, K extends keyof T>(
  object: T,
  method: K,
  accessType?: 'get' | 'set'
): MockInstance;
```

**Examples:**
```typescript
// Spy on method (preserves original behavior)
const calculator = {
  add: (a: number, b: number) => a + b,
};

const spy = vi.spyOn(calculator, 'add');
calculator.add(2, 3); // Returns 5 (original behavior)
expect(spy).toHaveBeenCalledWith(2, 3);
spy.mockRestore(); // Restore original

// Spy on getter
const obj = {
  get value() { return 42; }
};

const spy = vi.spyOn(obj, 'value', 'get');
obj.value; // 42
expect(spy).toHaveBeenCalled();

spy.mockReturnValue(100);
obj.value; // 100

// Spy on setter
let _value = 0;
const obj = {
  set value(v: number) { _value = v; }
};

const spy = vi.spyOn(obj, 'value', 'set');
obj.value = 42;
expect(spy).toHaveBeenCalledWith(42);
```

## Module Mocking

```typescript { .api }
// Hoisted mocking (runs before imports)
vi.mock(path: string, factory?: () => any): void;
vi.unmock(path: string): void;

// Non-hoisted mocking (for conditional mocks)
vi.doMock(path: string, factory?: () => any): void;
vi.doUnmock(path: string): void;

// Import helpers
vi.importActual<T>(path: string): Promise<T>;
vi.importMock<T>(path: string): Promise<T>;
vi.hoisted<T>(factory: () => T): T;
```

**Examples:**
```typescript
// Mock entire module
vi.mock('./api', () => ({
  fetchUser: vi.fn(() => Promise.resolve({ id: 1, name: 'John' })),
  deleteUser: vi.fn(() => Promise.resolve(true)),
}));

test('mocked module', async () => {
  const { fetchUser } = await import('./api');
  const user = await fetchUser(1);
  expect(user).toEqual({ id: 1, name: 'John' });
});

// Partial mock (keep some real exports)
vi.mock('./utils', async () => {
  const actual = await vi.importActual<typeof import('./utils')>('./utils');
  return {
    ...actual,
    fetchData: vi.fn(() => 'mocked data'),
  };
});

// Conditional mocking
test('conditional mock', async () => {
  vi.doMock('./config', () => ({ apiUrl: 'https://test.api' }));
  const { apiUrl } = await import('./config');
  expect(apiUrl).toBe('https://test.api');
  vi.doUnmock('./config');
});

// Hoisted factory (share values across mocks)
const { mockUsers } = vi.hoisted(() => ({
  mockUsers: [{ id: 1, name: 'John' }],
}));

vi.mock('./database', () => ({
  getUsers: vi.fn(() => Promise.resolve(mockUsers)),
}));
```

## Mock Object

Create deep mock of objects.

```typescript { .api }
vi.mockObject<T>(value: T, options?: { spy?: boolean }): MaybeMockedDeep<T>;
```

**Examples:**
```typescript
// Mock object (methods return undefined)
const api = {
  fetchUser: (id: number) => Promise.resolve({ id, name: 'John' }),
  deleteUser: (id: number) => Promise.resolve(true),
};

const mockedApi = vi.mockObject(api);
expect(vi.isMockFunction(mockedApi.fetchUser)).toBe(true);
await expect(mockedApi.fetchUser(1)).resolves.toBeUndefined();

// Configure mock behavior
mockedApi.fetchUser.mockResolvedValue({ id: 1, name: 'Test' });
await expect(mockedApi.fetchUser(1)).resolves.toEqual({ id: 1, name: 'Test' });

// Spy on object (preserves behavior)
const api2 = {
  add: (a: number, b: number) => a + b,
  multiply: (a: number, b: number) => a * b,
};

const spiedApi = vi.mockObject(api2, { spy: true });
expect(spiedApi.add(2, 3)).toBe(5); // Original behavior
expect(spiedApi.add).toHaveBeenCalledWith(2, 3); // But tracked
```

## Typed Mocks

```typescript { .api }
vi.mocked<T>(item: T, deep?: false): Mocked<T>;
vi.mocked<T>(item: T, deep: true): MaybeMockedDeep<T>;
```

**Example:**
```typescript
import { fetchUser } from './api';

vi.mock('./api');

test('typed mock', () => {
  const mockFetchUser = vi.mocked(fetchUser);
  mockFetchUser.mockResolvedValue({ id: 1, name: 'John' });
  // TypeScript knows about mockResolvedValue
});
```

## Mock State Management

```typescript { .api }
vi.clearAllMocks()   // Clear call history only
vi.resetAllMocks()   // Clear history + reset implementation
vi.restoreAllMocks() // Restore original implementations (spies)
```

| Method | Call History | Implementation | Restore Original |
|--------|--------------|----------------|------------------|
| `mockClear()` | ✓ Clear | Keep | No |
| `mockReset()` | ✓ Clear | ✓ Reset | No |
| `mockRestore()` | ✓ Clear | ✓ Reset | ✓ Yes (spies only) |

**Example:**
```typescript
beforeEach(() => {
  vi.clearAllMocks(); // Clear all mock call history
});

afterEach(() => {
  vi.restoreAllMocks(); // Restore all spies
});

test('mock cleanup', () => {
  const mockFn = vi.fn(() => 42);
  mockFn();
  mockFn();
  expect(mockFn).toHaveBeenCalledTimes(2);

  vi.clearAllMocks(); // Clear history, keep implementation
  expect(mockFn).toHaveBeenCalledTimes(0);
  expect(mockFn()).toBe(42); // Still returns 42

  vi.resetAllMocks(); // Clear history, reset implementation
  expect(mockFn()).toBeUndefined();
});
```

## Global Stubs

```typescript { .api }
vi.stubGlobal(name: string, value: any): VitestUtils;
vi.unstubAllGlobals(): VitestUtils;
vi.stubEnv(name: string, value: string): VitestUtils;
vi.unstubAllEnvs(): VitestUtils;
```

**Examples:**
```typescript
afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

test('stub fetch', () => {
  const mockFetch = vi.fn(() => Promise.resolve({ ok: true }));
  vi.stubGlobal('fetch', mockFetch);

  // fetch is now mocked
  expect(globalThis.fetch).toBe(mockFetch);
});

test('stub env vars', () => {
  vi.stubEnv('NODE_ENV', 'test');
  vi.stubEnv('API_KEY', 'test-key');

  expect(process.env.NODE_ENV).toBe('test');
  expect(process.env.API_KEY).toBe('test-key');
});
```

## Async Wait Utilities

```typescript { .api }
vi.waitFor<T>(
  callback: () => T | Promise<T>,
  options?: { timeout?: number; interval?: number }
): Promise<T>;

vi.waitUntil<T>(
  callback: () => T | Promise<T>,
  options?: { timeout?: number; interval?: number }
): Promise<T>;
```

**Difference:**
- `waitFor`: Retries until callback **doesn't throw**
- `waitUntil`: Retries until callback returns **truthy value** (fails if throws)

**Examples:**
```typescript
test('wait for condition', async () => {
  let ready = false;
  setTimeout(() => { ready = true }, 500);

  await vi.waitUntil(
    () => ready,
    { timeout: 1000, interval: 50 }
  );

  expect(ready).toBe(true);
});

test('wait for no errors', async () => {
  const server = createServer();

  await vi.waitFor(
    () => {
      if (!server.isReady) throw new Error('Not ready');
    },
    { timeout: 2000, interval: 100 }
  );

  expect(server.isReady).toBe(true);
});

test('wait for element', async () => {
  const element = await vi.waitUntil(
    () => document.querySelector('.loaded'),
    { timeout: 1000 }
  );

  expect(element).toBeTruthy();
});
```

## Module Cache

```typescript { .api }
vi.resetModules(): VitestUtils;
vi.dynamicImportSettled(): Promise<void>;
```

**Examples:**
```typescript
test('reset modules', async () => {
  const { value: value1 } = await import('./module');

  vi.resetModules(); // Clear cache

  const { value: value2 } = await import('./module');
  // Fresh module instance
});

test('wait for dynamic imports', async () => {
  const promises = [
    import('./module1'),
    import('./module2'),
    import('./module3'),
  ];

  await vi.dynamicImportSettled();
  // All imports completed
});
```

## Common Mock Patterns

### Mock Fetch

```typescript
test('mock fetch', async () => {
  const mockFetch = vi.fn(() =>
    Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ data: 'test' }),
    })
  );

  vi.stubGlobal('fetch', mockFetch);

  const response = await fetch('/api/data');
  const data = await response.json();

  expect(data).toEqual({ data: 'test' });
  expect(mockFetch).toHaveBeenCalledWith('/api/data');

  vi.unstubAllGlobals();
});
```

### Mock Date

```typescript
test('mock date', () => {
  const mockDate = new Date('2024-01-01T00:00:00Z');
  vi.setSystemTime(mockDate);

  expect(new Date().toISOString()).toBe('2024-01-01T00:00:00.000Z');

  vi.useRealTimers(); // Restore
});
```

### Partial Module Mock

```typescript
vi.mock('./api', async () => {
  const actual = await vi.importActual<typeof import('./api')>('./api');
  return {
    ...actual,
    fetchUser: vi.fn(() => Promise.resolve({ id: 1, name: 'Test' })),
  };
});

test('partial mock', async () => {
  const { fetchUser, otherFunction } = await import('./api');

  expect(vi.isMockFunction(fetchUser)).toBe(true); // Mocked
  expect(vi.isMockFunction(otherFunction)).toBe(false); // Real
});
```

### Automatic Cleanup

```typescript
beforeEach(() => {
  // Setup mocks
});

afterEach(() => {
  vi.clearAllMocks();     // Clear call history
  vi.resetAllMocks();     // Reset implementations
  vi.restoreAllMocks();   // Restore spies
  vi.unstubAllGlobals();  // Restore globals
  vi.unstubAllEnvs();     // Restore env vars
});
```

## Type Definitions

```typescript { .api }
interface MockContext<Args extends any[], R> {
  calls: Args[];
  results: MockResult<R>[];
  instances: any[];
  invocationCallOrder: number[];
  lastCall?: Args;
}

interface MockResult<T> {
  type: 'return' | 'throw';
  value: T;
}

type Mocked<T> = T extends (...args: any[]) => any
  ? MockedFunction<T>
  : T extends object
  ? MockedObject<T>
  : T;

type MockedFunction<T extends (...args: any[]) => any> = Mock<
  Parameters<T>,
  ReturnType<T>
> & T;

type MockedObject<T> = {
  [K in keyof T]: T[K] extends (...args: any[]) => any
    ? MockedFunction<T[K]>
    : T[K];
} & T;
```

## Mock Strategy Guide

| Scenario | Use | Reason |
|----------|-----|--------|
| Track function calls | `vi.fn()` | Full control over implementation |
| Keep original behavior | `vi.spyOn()` | Monitor calls without changing behavior |
| Replace entire module | `vi.mock()` | Isolate from dependencies |
| Partial module mock | `vi.mock() + importActual()` | Mock some exports, keep others |
| Mock global APIs | `vi.stubGlobal()` | Replace built-in globals (fetch, etc) |
| Conditional mock | `vi.doMock()` | Mock only in specific tests |
| Share mock state | `vi.hoisted()` | Access variables in factory |
