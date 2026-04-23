# Type Testing

Vitest provides type-level testing capabilities for ensuring TypeScript types are correct without runtime execution.

## Capabilities

### ExpectTypeOf

Type-level assertions using the `expectTypeOf` function from the `expect-type` library.

```typescript { .api }
/**
 * Create type-level expectation
 * @param value - Value to check types for
 * @returns Type-level assertion object
 */
function expectTypeOf<T>(value: T): ExpectTypeOf<T>;
function expectTypeOf<T>(): ExpectTypeOf<T>;

interface ExpectTypeOf<T> {
  /** Assert type is exactly equal to Expected */
  toEqualTypeOf<Expected>(): void;

  /** Assert type matches Expected structure */
  toMatchTypeOf<Expected>(): void;

  /** Assert type is any */
  toBeAny(): void;

  /** Assert type is unknown */
  toBeUnknown(): void;

  /** Assert type is never */
  toBeNever(): void;

  /** Assert type is a function */
  toBeFunction(): void;

  /** Assert type is an object */
  toBeObject(): void;

  /** Assert type is an array */
  toBeArray(): void;

  /** Assert type is a string */
  toBeString(): void;

  /** Assert type is a number */
  toBeNumber(): void;

  /** Assert type is a boolean */
  toBeBoolean(): void;

  /** Assert type is void */
  toBeVoid(): void;

  /** Assert type is null */
  toBeNull(): void;

  /** Assert type is undefined */
  toBeUndefined(): void;

  /** Assert type is nullable (null or undefined) */
  toBeNullable(): void;

  /** Assert type has callable signature */
  toBeCallableWith(...args: any[]): void;

  /** Assert type can be constructed */
  toBeConstructibleWith(...args: any[]): void;

  /** Assert type has property */
  toHaveProperty<K extends keyof T>(key: K): ExpectTypeOf<T[K]>;

  /** Negate assertion */
  not: ExpectTypeOf<T>;

  /** Extract type from Promise */
  resolves: T extends Promise<infer U> ? ExpectTypeOf<U> : never;

  /** Extract parameters from function */
  parameters: T extends (...args: infer P) => any ? ExpectTypeOf<P> : never;

  /** Extract return type from function */
  returns: T extends (...args: any[]) => infer R ? ExpectTypeOf<R> : never;

  /** Extract type from array */
  items: T extends (infer Item)[] ? ExpectTypeOf<Item> : never;

  /** Extract branded type */
  branded: ExpectTypeOf<T>;
}
```

**Usage:**

```typescript
import { test, expectTypeOf } from 'vitest';

test('type assertions', () => {
  // Exact type equality
  expectTypeOf<string>().toEqualTypeOf<string>();
  expectTypeOf('hello').toEqualTypeOf<string>();

  // Structural matching
  expectTypeOf<{ a: number }>().toMatchTypeOf<{ a: number }>();

  // Primitive types
  expectTypeOf('').toBeString();
  expectTypeOf(123).toBeNumber();
  expectTypeOf(true).toBeBoolean();
  expectTypeOf(null).toBeNull();
  expectTypeOf(undefined).toBeUndefined();
  expectTypeOf({}).toBeObject();
  expectTypeOf([]).toBeArray();
  expectTypeOf(() => {}).toBeFunction();

  // Never and any
  expectTypeOf<never>().toBeNever();
  expectTypeOf<any>().toBeAny();
  expectTypeOf<unknown>().toBeUnknown();

  // Nullable
  expectTypeOf<string | null>().toBeNullable();
  expectTypeOf<string | undefined>().toBeNullable();

  // Negation
  expectTypeOf<number>().not.toBeString();
  expectTypeOf<string>().not.toBeNumber();
});
```

### Function Type Testing

Test function signatures, parameters, and return types.

```typescript
import { test, expectTypeOf } from 'vitest';

test('function types', () => {
  const add = (a: number, b: number): number => a + b;

  // Check if callable
  expectTypeOf(add).toBeCallableWith(1, 2);

  // Check parameters
  expectTypeOf(add).parameters.toEqualTypeOf<[number, number]>();

  // Check return type
  expectTypeOf(add).returns.toEqualTypeOf<number>();
});

test('constructor types', () => {
  class User {
    constructor(public name: string, public age: number) {}
  }

  expectTypeOf(User).toBeConstructibleWith('John', 30);
  expectTypeOf(User).instance.toHaveProperty('name');
  expectTypeOf(User).instance.toHaveProperty('age');
});
```

### Object Type Testing

Test object structure and properties.

```typescript
import { test, expectTypeOf } from 'vitest';

test('object types', () => {
  type User = {
    id: number;
    name: string;
    email?: string;
  };

  // Exact type match
  expectTypeOf<User>().toEqualTypeOf<{
    id: number;
    name: string;
    email?: string;
  }>();

  // Structural match (allows extra properties)
  expectTypeOf<User>().toMatchTypeOf<{
    id: number;
    name: string;
  }>();

  // Property types
  expectTypeOf<User>().toHaveProperty('id').toBeNumber();
  expectTypeOf<User>().toHaveProperty('name').toBeString();
  expectTypeOf<User>().toHaveProperty('email').toEqualTypeOf<string | undefined>();
});
```

### Generic Type Testing

Test generic types and type parameters.

```typescript
import { test, expectTypeOf } from 'vitest';

test('generic types', () => {
  // Array item types
  expectTypeOf<string[]>().items.toBeString();
  expectTypeOf<number[]>().items.toBeNumber();

  // Promise resolution types
  expectTypeOf<Promise<string>>().resolves.toBeString();
  expectTypeOf<Promise<number>>().resolves.toBeNumber();

  // Generic function return types
  function identity<T>(value: T): T {
    return value;
  }

  expectTypeOf(identity<string>).returns.toBeString();
  expectTypeOf(identity<number>).returns.toBeNumber();
});
```

### Union and Intersection Types

```typescript
import { test, expectTypeOf } from 'vitest';

test('union types', () => {
  type StringOrNumber = string | number;

  expectTypeOf<StringOrNumber>().toMatchTypeOf<string>();
  expectTypeOf<StringOrNumber>().toMatchTypeOf<number>();
  expectTypeOf<string>().toMatchTypeOf<StringOrNumber>();
});

test('intersection types', () => {
  type Named = { name: string };
  type Aged = { age: number };
  type Person = Named & Aged;

  expectTypeOf<Person>().toHaveProperty('name').toBeString();
  expectTypeOf<Person>().toHaveProperty('age').toBeNumber();
  expectTypeOf<Person>().toMatchTypeOf<Named>();
  expectTypeOf<Person>().toMatchTypeOf<Aged>();
});
```

### AssertType

Runtime-free type assertions for checking type compatibility.

```typescript { .api }
/**
 * Assert that value matches type T
 * Causes TypeScript error if types don't match
 * @param value - Value to type check
 */
function assertType<T>(value: T): void;
```

**Usage:**

```typescript
import { test, assertType } from 'vitest';

test('assert types', () => {
  const value: string = 'hello';

  assertType<string>(value); // OK
  // assertType<number>(value); // TypeScript error

  const obj = { id: 1, name: 'John' };

  assertType<{ id: number; name: string }>(obj); // OK

  const promise = Promise.resolve(42);

  assertType<Promise<number>>(promise); // OK
});
```

### Complex Type Scenarios

```typescript
import { test, expectTypeOf } from 'vitest';

test('conditional types', () => {
  type IsString<T> = T extends string ? true : false;

  expectTypeOf<IsString<string>>().toEqualTypeOf<true>();
  expectTypeOf<IsString<number>>().toEqualTypeOf<false>();
});

test('mapped types', () => {
  type Readonly<T> = {
    readonly [K in keyof T]: T[K];
  };

  type User = { name: string; age: number };
  type ReadonlyUser = Readonly<User>;

  expectTypeOf<ReadonlyUser>().toEqualTypeOf<{
    readonly name: string;
    readonly age: number;
  }>();
});

test('utility types', () => {
  type User = { id: number; name: string; email: string };

  expectTypeOf<Partial<User>>().toEqualTypeOf<{
    id?: number;
    name?: string;
    email?: string;
  }>();

  expectTypeOf<Required<User>>().toEqualTypeOf<{
    id: number;
    name: string;
    email: string;
  }>();

  expectTypeOf<Pick<User, 'id' | 'name'>>().toEqualTypeOf<{
    id: number;
    name: string;
  }>();

  expectTypeOf<Omit<User, 'email'>>().toEqualTypeOf<{
    id: number;
    name: string;
  }>();
});
```

## Typecheck Configuration

Configure type checking in vitest.config.ts:

```typescript
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    typecheck: {
      enabled: true,
      checker: 'tsc', // or 'vue-tsc'
      include: ['**/*.{test,spec}-d.ts'],
      exclude: ['node_modules'],
      tsconfig: './tsconfig.json'
    }
  }
});
```

## Running Type Tests

Type tests are checked during regular test runs, or can be run separately:

```bash
# Run all tests including type tests
vitest

# Run only type tests
vitest typecheck
```

## Common Patterns

### Testing API Return Types

```typescript
import { test, expectTypeOf } from 'vitest';

test('API types', () => {
  async function fetchUser(id: number): Promise<{
    id: number;
    name: string;
    email: string;
  }> {
    // implementation
  }

  expectTypeOf(fetchUser).parameter(0).toBeNumber();
  expectTypeOf(fetchUser).returns.resolves.toMatchTypeOf<{
    id: number;
    name: string;
    email: string;
  }>();
});
```

### Testing Type Narrowing

```typescript
import { test, expectTypeOf, assertType } from 'vitest';

test('type narrowing', () => {
  function process(value: string | number) {
    if (typeof value === 'string') {
      assertType<string>(value);
      expectTypeOf(value).toBeString();
    } else {
      assertType<number>(value);
      expectTypeOf(value).toBeNumber();
    }
  }
});
```

### Testing Generic Constraints

```typescript
import { test, expectTypeOf } from 'vitest';

test('generic constraints', () => {
  function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
    return obj[key];
  }

  type User = { name: string; age: number };
  const user: User = { name: 'John', age: 30 };

  expectTypeOf(getProperty(user, 'name')).toBeString();
  expectTypeOf(getProperty(user, 'age')).toBeNumber();
});
```

## Type Definitions

```typescript { .api }
type ExpectTypeOf<T> = {
  toEqualTypeOf: <Expected>() => void;
  toMatchTypeOf: <Expected>() => void;
  toBeAny: () => void;
  toBeUnknown: () => void;
  toBeNever: () => void;
  toBeFunction: () => void;
  toBeObject: () => void;
  toBeArray: () => void;
  toBeString: () => void;
  toBeNumber: () => void;
  toBeBoolean: () => void;
  toBeVoid: () => void;
  toBeNull: () => void;
  toBeUndefined: () => void;
  toBeNullable: () => void;
  toBeCallableWith: (...args: any[]) => void;
  toBeConstructibleWith: (...args: any[]) => void;
  toHaveProperty: <K extends keyof T>(key: K) => ExpectTypeOf<T[K]>;
  not: ExpectTypeOf<T>;
  resolves: T extends Promise<infer U> ? ExpectTypeOf<U> : never;
  parameters: T extends (...args: infer P) => any ? ExpectTypeOf<P> : never;
  returns: T extends (...args: any[]) => infer R ? ExpectTypeOf<R> : never;
  items: T extends (infer Item)[] ? ExpectTypeOf<Item> : never;
};

type AssertType = <T>(value: T) => void;
```
