# Timer Control

Vitest provides fake timer APIs for controlling time in tests, enabling deterministic testing of time-dependent code such as timeouts, intervals, and animations.

## Capabilities

### Fake Timers

Enable and configure fake timers to control time manually.

```typescript { .api }
interface VitestUtils {
  /**
   * Enable fake timers
   * @param config - Optional timer configuration
   * @returns VitestUtils for chaining
   */
  useFakeTimers(config?: FakeTimerInstallOpts): VitestUtils;

  /**
   * Disable fake timers and restore real timers
   * @returns VitestUtils for chaining
   */
  useRealTimers(): VitestUtils;

  /**
   * Check if fake timers are currently enabled
   * @returns true if fake timers are active
   */
  isFakeTimers(): boolean;
}

interface FakeTimerInstallOpts {
  /**
   * Current system time (default: Date.now())
   */
  now?: number | Date;

  /**
   * Maximum number of timers to run (prevents infinite loops)
   * Default: 100_000
   */
  loopLimit?: number;

  /**
   * Whether to mock Date.now(), performance.now(), etc.
   * Default: true
   */
  shouldAdvanceTime?: boolean;

  /**
   * Which timer APIs to mock
   * Default: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'setImmediate', 'clearImmediate', 'Date']
   */
  toFake?: Array<
    | 'setTimeout'
    | 'clearTimeout'
    | 'setInterval'
    | 'clearInterval'
    | 'setImmediate'
    | 'clearImmediate'
    | 'Date'
    | 'performance'
    | 'requestAnimationFrame'
    | 'cancelAnimationFrame'
    | 'requestIdleCallback'
    | 'cancelIdleCallback'
  >;
}
```

**Usage:**

```typescript
import { test, expect, vi, beforeEach, afterEach } from 'vitest';

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

test('fake timers', () => {
  const callback = vi.fn();

  setTimeout(callback, 1000);

  // Time hasn't advanced yet
  expect(callback).not.toHaveBeenCalled();

  vi.advanceTimersByTime(1000);

  // Now callback has been called
  expect(callback).toHaveBeenCalled();
});

test('fake timers with config', () => {
  vi.useFakeTimers({
    now: new Date('2024-01-01T00:00:00Z'),
    toFake: ['setTimeout', 'Date']
  });

  expect(Date.now()).toBe(new Date('2024-01-01T00:00:00Z').getTime());

  vi.useRealTimers();
});
```

### Running Timers

Execute pending timers synchronously or asynchronously.

```typescript { .api }
interface VitestUtils {
  /**
   * Run only currently pending timers (not new ones)
   * @returns VitestUtils for chaining
   */
  runOnlyPendingTimers(): VitestUtils;

  /**
   * Async version of runOnlyPendingTimers
   * @returns Promise resolving to VitestUtils
   */
  runOnlyPendingTimersAsync(): Promise<VitestUtils>;

  /**
   * Run all pending timers (including new ones created by handlers)
   * @returns VitestUtils for chaining
   */
  runAllTimers(): VitestUtils;

  /**
   * Async version of runAllTimers
   * @returns Promise resolving to VitestUtils
   */
  runAllTimersAsync(): Promise<VitestUtils>;

  /**
   * Run all pending microtasks (process.nextTick, Promise callbacks)
   * @returns VitestUtils for chaining
   */
  runAllTicks(): VitestUtils;
}
```

**Usage:**

```typescript
import { test, expect, vi } from 'vitest';

test('run only pending timers', () => {
  vi.useFakeTimers();

  const callback1 = vi.fn();
  const callback2 = vi.fn(() => {
    // This creates a new timer during execution
    setTimeout(callback3, 100);
  });
  const callback3 = vi.fn();

  setTimeout(callback1, 100);
  setTimeout(callback2, 100);

  vi.runOnlyPendingTimers(); // Runs callback1 and callback2

  expect(callback1).toHaveBeenCalled();
  expect(callback2).toHaveBeenCalled();
  expect(callback3).not.toHaveBeenCalled(); // Not run (created during execution)

  vi.runOnlyPendingTimers(); // Now runs callback3

  expect(callback3).toHaveBeenCalled();

  vi.useRealTimers();
});

test('run all timers', () => {
  vi.useFakeTimers();

  const callback = vi.fn(() => {
    if (callback.mock.calls.length < 3) {
      setTimeout(callback, 100); // Recursive timer
    }
  });

  setTimeout(callback, 100);

  vi.runAllTimers(); // Runs all timers including recursive ones

  expect(callback).toHaveBeenCalledTimes(3);

  vi.useRealTimers();
});

test('run all ticks', async () => {
  vi.useFakeTimers();

  const callback = vi.fn();

  process.nextTick(callback);
  Promise.resolve().then(callback);

  vi.runAllTicks(); // Run all microtasks

  expect(callback).toHaveBeenCalledTimes(2);

  vi.useRealTimers();
});

test('async timer execution', async () => {
  vi.useFakeTimers();

  const callback = vi.fn();

  setTimeout(async () => {
    await Promise.resolve();
    callback();
  }, 100);

  await vi.runAllTimersAsync(); // Wait for async operations

  expect(callback).toHaveBeenCalled();

  vi.useRealTimers();
});
```

### Advancing Time

Manually advance time by specific amounts or to specific points.

```typescript { .api }
interface VitestUtils {
  /**
   * Advance timers by specified milliseconds
   * @param ms - Milliseconds to advance
   * @returns VitestUtils for chaining
   */
  advanceTimersByTime(ms: number): VitestUtils;

  /**
   * Async version of advanceTimersByTime
   * @param ms - Milliseconds to advance
   * @returns Promise resolving to VitestUtils
   */
  advanceTimersByTimeAsync(ms: number): Promise<VitestUtils>;

  /**
   * Advance to the next timer
   * @returns VitestUtils for chaining
   */
  advanceTimersToNextTimer(): VitestUtils;

  /**
   * Async version of advanceTimersToNextTimer
   * @returns Promise resolving to VitestUtils
   */
  advanceTimersToNextTimerAsync(): Promise<VitestUtils>;

  /**
   * Advance to the next animation frame (requestAnimationFrame)
   * @returns VitestUtils for chaining
   */
  advanceTimersToNextFrame(): VitestUtils;
}
```

**Usage:**

```typescript
import { test, expect, vi } from 'vitest';

test('advance by time', () => {
  vi.useFakeTimers();

  const callback = vi.fn();

  setTimeout(callback, 1000);

  vi.advanceTimersByTime(500);
  expect(callback).not.toHaveBeenCalled();

  vi.advanceTimersByTime(500);
  expect(callback).toHaveBeenCalled();

  vi.useRealTimers();
});

test('advance to next timer', () => {
  vi.useFakeTimers();

  const callback1 = vi.fn();
  const callback2 = vi.fn();
  const callback3 = vi.fn();

  setTimeout(callback1, 100);
  setTimeout(callback2, 200);
  setTimeout(callback3, 300);

  vi.advanceTimersToNextTimer(); // Advances to 100ms
  expect(callback1).toHaveBeenCalled();
  expect(callback2).not.toHaveBeenCalled();

  vi.advanceTimersToNextTimer(); // Advances to 200ms
  expect(callback2).toHaveBeenCalled();
  expect(callback3).not.toHaveBeenCalled();

  vi.advanceTimersToNextTimer(); // Advances to 300ms
  expect(callback3).toHaveBeenCalled();

  vi.useRealTimers();
});

test('advance to next frame', () => {
  vi.useFakeTimers();

  const callback = vi.fn();

  requestAnimationFrame(callback);

  expect(callback).not.toHaveBeenCalled();

  vi.advanceTimersToNextFrame(); // Advance one frame (typically ~16ms)

  expect(callback).toHaveBeenCalled();

  vi.useRealTimers();
});

test('async advance', async () => {
  vi.useFakeTimers();

  const callback = vi.fn();

  setTimeout(async () => {
    await Promise.resolve();
    callback();
  }, 1000);

  await vi.advanceTimersByTimeAsync(1000);

  expect(callback).toHaveBeenCalled();

  vi.useRealTimers();
});
```

### System Time

Control the system time (Date.now(), new Date(), etc.).

```typescript { .api }
interface VitestUtils {
  /**
   * Set the current system time
   * @param time - Time to set (number, Date, or date string)
   * @returns VitestUtils for chaining
   */
  setSystemTime(time: number | string | Date): VitestUtils;

  /**
   * Get the current mocked system time
   * @returns Mocked time as Date, or null if not using fake timers
   */
  getMockedSystemTime(): Date | null;

  /**
   * Get the real system time (even when using fake timers)
   * @returns Real current time in milliseconds
   */
  getRealSystemTime(): number;
}
```

**Usage:**

```typescript
import { test, expect, vi } from 'vitest';

test('set system time', () => {
  vi.useFakeTimers();

  const mockDate = new Date('2024-01-01T00:00:00Z');
  vi.setSystemTime(mockDate);

  expect(Date.now()).toBe(mockDate.getTime());
  expect(new Date().toISOString()).toBe('2024-01-01T00:00:00.000Z');

  // Advance time
  vi.advanceTimersByTime(1000);

  expect(Date.now()).toBe(mockDate.getTime() + 1000);

  vi.useRealTimers();
});

test('get mocked time', () => {
  expect(vi.getMockedSystemTime()).toBeNull(); // Not using fake timers

  vi.useFakeTimers();
  vi.setSystemTime(new Date('2024-01-01'));

  const mockedTime = vi.getMockedSystemTime();
  expect(mockedTime?.toISOString()).toBe('2024-01-01T00:00:00.000Z');

  vi.useRealTimers();
});

test('get real time', () => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date('2024-01-01'));

  const realTime = vi.getRealSystemTime();
  expect(realTime).toBeGreaterThan(new Date('2024-01-01').getTime());

  vi.useRealTimers();
});
```

### Timer Status

Query information about pending timers.

```typescript { .api }
interface VitestUtils {
  /**
   * Get the number of pending timers
   * @returns Number of timers waiting to be executed
   */
  getTimerCount(): number;

  /**
   * Clear all pending timers
   * @returns VitestUtils for chaining
   */
  clearAllTimers(): VitestUtils;
}
```

**Usage:**

```typescript
import { test, expect, vi } from 'vitest';

test('timer count', () => {
  vi.useFakeTimers();

  expect(vi.getTimerCount()).toBe(0);

  setTimeout(() => {}, 100);
  setTimeout(() => {}, 200);
  setInterval(() => {}, 300);

  expect(vi.getTimerCount()).toBe(3);

  vi.advanceTimersByTime(100);

  expect(vi.getTimerCount()).toBe(2); // One timeout executed

  vi.clearAllTimers();

  expect(vi.getTimerCount()).toBe(0);

  vi.useRealTimers();
});
```

## Common Patterns

### Testing Debounce

```typescript
import { test, expect, vi } from 'vitest';

function debounce<T extends (...args: any[]) => any>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}

test('debounce', () => {
  vi.useFakeTimers();

  const callback = vi.fn();
  const debounced = debounce(callback, 300);

  debounced('a');
  debounced('b');
  debounced('c');

  // Callback not called yet
  expect(callback).not.toHaveBeenCalled();

  vi.advanceTimersByTime(299);
  expect(callback).not.toHaveBeenCalled();

  vi.advanceTimersByTime(1);
  expect(callback).toHaveBeenCalledWith('c');
  expect(callback).toHaveBeenCalledTimes(1);

  vi.useRealTimers();
});
```

### Testing Throttle

```typescript
import { test, expect, vi } from 'vitest';

function throttle<T extends (...args: any[]) => any>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let lastCall = 0;
  return (...args: Parameters<T>) => {
    const now = Date.now();
    if (now - lastCall >= delay) {
      lastCall = now;
      fn(...args);
    }
  };
}

test('throttle', () => {
  vi.useFakeTimers();

  const callback = vi.fn();
  const throttled = throttle(callback, 100);

  throttled('a');
  throttled('b');
  throttled('c');

  // Only first call executes
  expect(callback).toHaveBeenCalledTimes(1);
  expect(callback).toHaveBeenCalledWith('a');

  vi.advanceTimersByTime(100);

  throttled('d');

  expect(callback).toHaveBeenCalledTimes(2);
  expect(callback).toHaveBeenCalledWith('d');

  vi.useRealTimers();
});
```

### Testing Intervals

```typescript
import { test, expect, vi } from 'vitest';

test('interval', () => {
  vi.useFakeTimers();

  const callback = vi.fn();

  const intervalId = setInterval(callback, 100);

  // No calls yet
  expect(callback).not.toHaveBeenCalled();

  vi.advanceTimersByTime(100);
  expect(callback).toHaveBeenCalledTimes(1);

  vi.advanceTimersByTime(100);
  expect(callback).toHaveBeenCalledTimes(2);

  vi.advanceTimersByTime(100);
  expect(callback).toHaveBeenCalledTimes(3);

  clearInterval(intervalId);

  vi.advanceTimersByTime(100);
  expect(callback).toHaveBeenCalledTimes(3); // No new calls

  vi.useRealTimers();
});
```

### Testing Animations

```typescript
import { test, expect, vi } from 'vitest';

test('animation', () => {
  vi.useFakeTimers();

  const callback = vi.fn();

  requestAnimationFrame(callback);
  requestAnimationFrame(callback);
  requestAnimationFrame(callback);

  expect(callback).not.toHaveBeenCalled();

  // Advance one frame
  vi.advanceTimersToNextFrame();

  expect(callback).toHaveBeenCalledTimes(3);

  vi.useRealTimers();
});
```

### Testing Retries with Backoff

```typescript
import { test, expect, vi } from 'vitest';

async function retryWithBackoff(
  fn: () => Promise<any>,
  maxRetries: number = 3,
  initialDelay: number = 100
): Promise<any> {
  let lastError;

  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (i < maxRetries - 1) {
        await new Promise(resolve =>
          setTimeout(resolve, initialDelay * Math.pow(2, i))
        );
      }
    }
  }

  throw lastError;
}

test('retry with backoff', async () => {
  vi.useFakeTimers();

  let attempts = 0;
  const mockFn = vi.fn(async () => {
    attempts++;
    if (attempts < 3) {
      throw new Error('Failed');
    }
    return 'Success';
  });

  const promise = retryWithBackoff(mockFn);

  // First attempt fails immediately
  await vi.runAllTicksAsync();
  expect(mockFn).toHaveBeenCalledTimes(1);

  // Wait 100ms for second attempt
  await vi.advanceTimersByTimeAsync(100);
  expect(mockFn).toHaveBeenCalledTimes(2);

  // Wait 200ms for third attempt
  await vi.advanceTimersByTimeAsync(200);
  expect(mockFn).toHaveBeenCalledTimes(3);

  const result = await promise;
  expect(result).toBe('Success');

  vi.useRealTimers();
});
```

## Best Practices

1. **Always restore real timers**: Use `afterEach(() => vi.useRealTimers())` to avoid affecting other tests

2. **Use async versions for async code**: When testing code with async operations, use `runAllTimersAsync()` or `advanceTimersByTimeAsync()`

3. **Be careful with infinite loops**: Fake timers have a loop limit to prevent infinite loops. Increase with `loopLimit` option if needed

4. **Clear timers in cleanup**: Use `vi.clearAllTimers()` in `afterEach` to prevent timers from affecting other tests

5. **Mock specific timer APIs**: Use `toFake` option to mock only the timer APIs you need

6. **Test deterministically**: Fake timers make time-dependent code deterministic and fast
