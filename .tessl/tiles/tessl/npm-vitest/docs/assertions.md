# Assertions and Expectations

Jest-compatible assertion APIs powered by Chai with extensive matchers and custom matcher support.

## Expect API

```typescript { .api }
function expect<T>(actual: T): Assertion<T>;

interface ExpectStatic {
  <T>(actual: T): Assertion<T>;
  extend(matchers: Record<string, CustomMatcher>): void;
  addEqualityTesters(testers: Array<Tester>): void;
  getState(): MatcherState;
  setState(state: Partial<MatcherState>): void;
  unreachable(message?: string): never;
  hasAssertions(): void;
  assertions(count: number): void;
  soft<T>(actual: T, message?: string): Assertion<T>;
  poll<T>(fn: () => T | Promise<T>, options?: ExpectPollOptions): Promise<Assertion<T>>;
}

interface ExpectPollOptions {
  interval?: number; // default: 50ms
  timeout?: number;  // default: 1000ms
  message?: string;
}
```

## Core Matchers

### Equality

| Matcher | Purpose | Example |
|---------|---------|---------|
| `.toBe(expected)` | Strict equality (===) | `expect(x).toBe(5)` |
| `.toEqual(expected)` | Deep equality | `expect({a:1}).toEqual({a:1})` |
| `.toStrictEqual(expected)` | Strict deep equality | `expect({a:1,b:undefined}).toStrictEqual({a:1,b:undefined})` |

### Truthiness

```typescript { .api }
.toBeTruthy()    // Truthy value
.toBeFalsy()     // Falsy value
.toBeDefined()   // Not undefined
.toBeUndefined() // Is undefined
.toBeNull()      // Is null
.toBeNaN()       // Is NaN
```

### Numbers

```typescript { .api }
.toBeGreaterThan(n)
.toBeGreaterThanOrEqual(n)
.toBeLessThan(n)
.toBeLessThanOrEqual(n)
.toBeCloseTo(n, digits?) // Floating point comparison
```

**Example:**
```typescript
expect(10).toBeGreaterThan(5);
expect(0.1 + 0.2).toBeCloseTo(0.3); // Handles floating point
```

### Strings

```typescript { .api }
.toMatch(regexp | string)     // Matches pattern
.toContain(substring)          // Contains substring
```

### Arrays/Iterables

```typescript { .api }
.toContain(item)              // Array/string contains item
.toContainEqual(item)         // Deep equality check for item
.toHaveLength(n)              // Length equals n
```

**Example:**
```typescript
expect([1, 2, 3]).toContain(2);
expect([{id: 1}, {id: 2}]).toContainEqual({id: 1});
expect([1, 2, 3]).toHaveLength(3);
```

### Objects

```typescript { .api }
.toHaveProperty(keyPath, value?)   // Has property at path
.toMatchObject(subset)             // Partial object match
```

**Example:**
```typescript
expect({a: 1, b: {c: 2}}).toHaveProperty('b.c', 2);
expect({a: 1, b: 2, c: 3}).toMatchObject({a: 1, b: 2});
```

### Functions/Errors

```typescript { .api }
.toThrow(expected?)           // Throws error
.toThrowError(expected?)      // Alias for toThrow
```

**Example:**
```typescript
expect(() => { throw new Error('fail') }).toThrow('fail');
expect(() => { throw new Error('fail') }).toThrow(/fail/);
```

### Promises

```typescript { .api }
.resolves   // Access resolved promise value
.rejects    // Access rejected promise error
```

**Example:**
```typescript
await expect(Promise.resolve('success')).resolves.toBe('success');
await expect(Promise.reject(new Error('fail'))).rejects.toThrow('fail');
```

### Modifiers

```typescript { .api }
.not   // Negate assertion
```

**Example:**
```typescript
expect(1).not.toBe(2);
expect('hello').not.toMatch(/world/);
```

## Mock Function Matchers

```typescript { .api }
// Call assertions
.toHaveBeenCalled()                      // Called at least once
.toBeCalled()                            // Alias
.toHaveBeenCalledTimes(n)               // Called exactly n times
.toHaveBeenCalledWith(...args)          // Called with specific args
.toHaveBeenLastCalledWith(...args)      // Last call had specific args
.toHaveBeenNthCalledWith(n, ...args)    // Nth call had specific args

// Return assertions
.toHaveReturned()                        // Returned successfully
.toHaveReturnedTimes(n)                 // Returned n times
.toHaveReturnedWith(value)              // Returned specific value
.toHaveLastReturnedWith(value)          // Last return was specific value
.toHaveNthReturnedWith(n, value)        // Nth return was specific value
```

**Example:**
```typescript
const mockFn = vi.fn((x) => x * 2);
mockFn(1); mockFn(2); mockFn(3);

expect(mockFn).toHaveBeenCalledTimes(3);
expect(mockFn).toHaveBeenCalledWith(2);
expect(mockFn).toHaveBeenNthCalledWith(2, 2);
expect(mockFn).toHaveReturnedWith(4);
expect(mockFn).toHaveLastReturnedWith(6);
```

## Snapshot Testing

```typescript { .api }
.toMatchSnapshot(message?)                         // Compare to saved snapshot
.toMatchInlineSnapshot(snapshot?, message?)        // Inline snapshot in code
.toMatchFileSnapshot(filepath, message?)           // Compare to file
.toThrowErrorMatchingSnapshot(message?)            // Error snapshot
.toThrowErrorMatchingInlineSnapshot(snapshot?)     // Inline error snapshot
```

**Example:**
```typescript
expect({id: 1, name: 'John'}).toMatchSnapshot();

expect({success: true, count: 5}).toMatchInlineSnapshot(`
  {
    "success": true,
    "count": 5
  }
`);

await expect('<div>Hello</div>').toMatchFileSnapshot('./snapshots/output.html');
```

## Advanced Features

### Soft Assertions

Continue test execution even when assertions fail.

```typescript
expect.soft(1 + 1).toBe(2);
expect.soft(2 + 2).toBe(5); // Fails but continues
expect.soft(3 + 3).toBe(6);
// Test reports all failures at end
```

### Polling Assertions

Retry assertions until condition is met.

```typescript
let value = 0;
setTimeout(() => { value = 10 }, 100);

await expect.poll(() => value, {
  interval: 20,
  timeout: 200
}).toBe(10);

// Async functions
await expect.poll(
  async () => {
    const res = await fetch('/api/status');
    return res.json();
  },
  { timeout: 5000 }
).toMatchObject({ status: 'ready' });
```

### Custom Matchers

```typescript { .api }
expect.extend(matchers: Record<string, CustomMatcher>): void;

interface CustomMatcher {
  (this: MatcherContext, actual: any, ...expected: any[]): MatcherResult;
}

interface MatcherResult {
  pass: boolean;
  message: () => string;
  actual?: any;
  expected?: any;
}
```

**Example:**
```typescript
expect.extend({
  toBeWithinRange(received: number, floor: number, ceiling: number) {
    const pass = received >= floor && received <= ceiling;
    return {
      pass,
      message: () => pass
        ? `expected ${received} not to be within ${floor}-${ceiling}`
        : `expected ${received} to be within ${floor}-${ceiling}`,
    };
  },
});

expect(15).toBeWithinRange(10, 20);
expect(25).not.toBeWithinRange(10, 20);
```

### Assertion Count

```typescript
// Require exact number of assertions
test('exact count', () => {
  expect.assertions(2);
  expect(1).toBe(1);
  expect(2).toBe(2);
  // Fails if != 2 assertions
});

// Require at least one assertion
test('at least one', async () => {
  expect.hasAssertions();
  const data = await fetchData();
  if (data) expect(data).toBeDefined();
  // Fails if no assertions run
});
```

### Create Custom Expect

```typescript { .api }
function createExpect(test?: TaskPopulated): ExpectStatic;
```

**Example:**
```typescript
const customExpect = createExpect();
customExpect.extend({
  toBeDivisibleBy(received: number, divisor: number) {
    return {
      pass: received % divisor === 0,
      message: () => `expected ${received} to be divisible by ${divisor}`,
    };
  },
});

customExpect(10).toBeDivisibleBy(5);
```

## Asymmetric Matchers

Partial matching for complex objects.

```typescript { .api }
expect.any(constructor)              // Match any instance of type
expect.anything()                    // Match anything except null/undefined
expect.arrayContaining(items)        // Array contains items
expect.objectContaining(obj)         // Object contains properties
expect.stringContaining(str)         // String contains substring
expect.stringMatching(regexp)        // String matches pattern
expect.closeTo(n, precision?)        // Number close to value
```

**Example:**
```typescript
expect({id: 1, name: 'John', age: 30}).toEqual({
  id: expect.any(Number),
  name: expect.any(String),
  age: expect.any(Number)
});

expect([1, 2, 3, 4]).toEqual(expect.arrayContaining([2, 3]));
expect({a: 1, b: 2, c: 3}).toEqual(expect.objectContaining({a: 1}));
expect('hello world').toEqual(expect.stringContaining('world'));
expect('test123').toEqual(expect.stringMatching(/test\d+/));
```

## Chai API

Vitest re-exports Chai for alternative assertion styles.

```typescript { .api }
const assert: Chai.AssertStatic;
const chai: Chai.ChaiStatic;
const should: Chai.Should;
```

**Example:**
```typescript
import { assert, should } from 'vitest';

// Node.js assert style
assert.equal(1 + 1, 2);
assert.isTrue(true);
assert.deepEqual({a: 1}, {a: 1});

// Chai should style
should();
(1 + 1).should.equal(2);
'hello'.should.be.a('string');
[1, 2, 3].should.have.lengthOf(3);
```

## Type Definitions

```typescript { .api }
interface Assertion<T> {
  not: Assertion<T>;
  resolves: Assertion<Awaited<T>>;
  rejects: Assertion<any>;

  toBe(expected: any): void;
  toEqual(expected: any): void;
  toStrictEqual(expected: any): void;
  toBeTruthy(): void;
  toBeFalsy(): void;
  toBeDefined(): void;
  toBeUndefined(): void;
  toBeNull(): void;
  toBeNaN(): void;
  toBeTypeOf(expected: string): void;
  toBeInstanceOf(expected: any): void;
  toBeGreaterThan(expected: number | bigint): void;
  toBeGreaterThanOrEqual(expected: number | bigint): void;
  toBeLessThan(expected: number | bigint): void;
  toBeLessThanOrEqual(expected: number | bigint): void;
  toBeCloseTo(expected: number, numDigits?: number): void;
  toMatch(expected: string | RegExp): void;
  toContain(expected: any): void;
  toContainEqual(expected: any): void;
  toHaveLength(expected: number): void;
  toHaveProperty(keyPath: string | string[], value?: any): void;
  toMatchObject(expected: object): void;
  toThrow(expected?: string | RegExp | Error): void;
  toSatisfy(predicate: (value: any) => boolean): void;
  toMatchSnapshot(message?: string): void;
  toMatchInlineSnapshot(snapshot?: string, message?: string): void;
  toMatchFileSnapshot(filepath: string, message?: string): Promise<void>;
}

interface MatcherState {
  assertionCalls: number;
  isExpectingAssertions: boolean;
  expectedAssertionsNumber: number | null;
  environment: Environment;
  testPath?: string;
  currentTestName?: string;
}
```

## Matcher Selection Guide

| Need | Use | Example |
|------|-----|---------|
| Primitive equality | `.toBe()` | `expect(5).toBe(5)` |
| Object equality | `.toEqual()` | `expect({a:1}).toEqual({a:1})` |
| Partial object | `.toMatchObject()` | `expect(obj).toMatchObject({a:1})` |
| Type check | `.any()` | `expect.any(String)` |
| Range check | Custom matcher | `toBe WithinRange()` |
| Async success | `.resolves` | `expect(p).resolves.toBe(x)` |
| Async failure | `.rejects` | `expect(p).rejects.toThrow()` |
| Function called | `.toHaveBeenCalled()` | Mock assertions |
| Contains item | `.toContain()` | Array/string search |
| Has property | `.toHaveProperty()` | Object property |
| Pattern match | `.toMatch()` | String/regex |
