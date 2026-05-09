# Type Utilities

Type analysis utilities and helper functions.

## Type Utilities

Type utilities are primarily accessed through the TypeChecker interface. See [Type Checker](./type-checker.md) for the main API.

Additional utility functions for working with types:

```typescript { .api }
// Type classification helpers (via TypeChecker)
interface TypeChecker {
  isArrayType(type: Type): boolean;
  isTupleType(type: Type): boolean;
  isArrayLikeType(type: Type): boolean;
  // ... see type-checker.md for full API
}
```

See [Type Checker](./type-checker.md) for complete type analysis APIs and [Types and Symbols](./types-symbols.md) for type and symbol interfaces.

