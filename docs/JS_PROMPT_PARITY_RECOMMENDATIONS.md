# JavaScript Test Generation Prompt - Parity Recommendations

This document outlines gaps between Python and JavaScript test generation prompts and provides recommendations for improving the JavaScript prompts.

## Current State Comparison

### Python Prompt (76 lines) - Comprehensive sections:
1. **PRESERVE ORIGINAL FUNCTION** - Don't modify function being tested
2. **USE REAL CLASSES - NO STUBS OR FAKES** - Import actual domain classes
3. **HANDLING INSTANCE METHODS** - Proper instantiation patterns
4. **USE CONFTEST.PY FIXTURES** - Leverage existing test fixtures
5. **DO NOT MOCK THE FUNCTION UNDER TEST** - Critical rule
6. **IMPORT CLASSES FROM THEIR REAL MODULES** - Proper import sources
7. **IMPORT EVERYTHING YOU USE** - No missing imports
8. **ONLY IMPORT WHAT YOU USE** - No unused imports
9. **USE CORRECT IMPORT SOURCES** - Match context provided
10. **DO NOT USE MOCK OBJECTS FOR DOMAIN CLASSES** - Real instances
11. **USE CORRECT CONSTRUCTOR SIGNATURES** - Proper instantiation
12. **VALID PYTHON STRING LITERALS** - Escape sequences, raw strings

### JavaScript Prompt (44 lines) - Current sections:
1. Basic/Edge/Large Scale test structure ✓
2. DO NOT MOCK THE FUNCTION UNDER TEST ✓
3. IMPORT FROM REAL MODULES ✓
4. HANDLE ASYNC PROPERLY ✓
5. IMPORT PATH RULES (no extensions) ✓
6. MOCKING RULES (Jest vs Vitest) ✓

## Gap Analysis

| Python Section | JS Status | Priority |
|----------------|-----------|----------|
| PRESERVE ORIGINAL FUNCTION | Missing | High |
| USE REAL CLASSES - NO STUBS | Missing | High |
| HANDLING INSTANCE METHODS | Missing | High |
| USE CONFTEST.PY FIXTURES | N/A (JS uses different patterns) | - |
| IMPORT EVERYTHING YOU USE | Missing | Medium |
| ONLY IMPORT WHAT YOU USE | Missing | Medium |
| USE CORRECT IMPORT SOURCES | Missing | High |
| USE CORRECT CONSTRUCTOR SIGNATURES | Missing | High |
| VALID STRING LITERALS | Missing | Medium |

## Recommended Additions to JavaScript Prompt

### Core Sections (Port from Python)

```markdown
**CRITICAL: PRESERVE ORIGINAL FUNCTION**:
- Do NOT modify or rewrite the function being tested.
- Your task is ONLY to write tests for the function as given.

**CRITICAL: USE REAL CLASSES - NO STUBS OR FAKES**:
- When the function uses custom classes/types, import and use the REAL class.
- **WRONG**: Creating inline stub classes or interfaces
- **CORRECT**: `import { UserProfile } from '../models/UserProfile'`

**CRITICAL: HANDLING CLASS METHODS**:
- For instance methods, properly instantiate the class first.
- Use the constructor signature shown in the context.
- Example:
  ```javascript
  const processor = new DataProcessor(config);
  const result = processor.processData(input);
  ```

**CRITICAL: IMPORT EVERYTHING YOU USE**:
- Every class, type, function, or constant used in tests MUST be imported.
- Do not assume anything is globally available.

**CRITICAL: ONLY IMPORT WHAT YOU USE**:
- Do not add unused imports - they cause TypeScript/linting errors.

**CRITICAL: USE CORRECT IMPORT SOURCES**:
- Import from the exact module paths shown in the provided context.
- Do not guess or infer import paths.

**CRITICAL: USE CORRECT CONSTRUCTOR SIGNATURES**:
- Check the class definition for required constructor parameters.
- Do not omit required parameters or add non-existent ones.

**CRITICAL: VALID STRING LITERALS**:
- Use proper escaping for special characters in strings.
- For multiline strings, use template literals (`backticks`).
- Escape backslashes properly: `\\n` for literal `\n`.
```

### JS/TS-Specific Sections (New)

```markdown
**CRITICAL: HANDLE TYPESCRIPT TYPES**:
- If testing TypeScript code, ensure test file uses `.test.ts` patterns.
- Respect type annotations - don't pass wrong types to functions.
- Use type assertions (`as Type`) only when necessary.

**CRITICAL: HANDLE PROMISES AND CALLBACKS**:
- For callback-based APIs, use promisify or done() callback.
- Never leave floating promises - always await or return.
- Use `expect.assertions(n)` for async error testing.

**CRITICAL: ES MODULES VS COMMONJS**:
- Check if project uses ESM (`import/export`) or CJS (`require/module.exports`).
- Match the import style to the project configuration.

**CRITICAL: HANDLE THIS BINDING**:
- Arrow functions don't have their own `this` context.
- For methods that use `this`, test with proper binding:
  ```javascript
  // Correct - preserves this context
  const instance = new MyClass();
  expect(instance.method()).toBe(expected);

  // Wrong - loses this context
  const { method } = new MyClass();
  expect(method()).toBe(expected); // May fail!
  ```

**CRITICAL: NULL VS UNDEFINED**:
- JavaScript distinguishes between `null` and `undefined`.
- Test both cases when function handles optional values.
- Use `toBe(null)` vs `toBeUndefined()` appropriately.
```

## Implementation Priority

1. **Phase 1 (High Priority)**: Add core sections ported from Python
   - PRESERVE ORIGINAL FUNCTION
   - USE REAL CLASSES
   - HANDLING CLASS METHODS
   - USE CORRECT IMPORT SOURCES
   - USE CORRECT CONSTRUCTOR SIGNATURES

2. **Phase 2 (Medium Priority)**: Add import management
   - IMPORT EVERYTHING YOU USE
   - ONLY IMPORT WHAT YOU USE
   - VALID STRING LITERALS

3. **Phase 3 (JS-Specific)**: Add JavaScript-specific guidance
   - HANDLE TYPESCRIPT TYPES
   - HANDLE PROMISES AND CALLBACKS
   - ES MODULES VS COMMONJS
   - HANDLE THIS BINDING
   - NULL VS UNDEFINED

## Metrics to Track

After implementing these changes, track:
- Test generation success rate (tests that compile)
- Test execution pass rate (tests that run without errors)
- Import error frequency
- Type error frequency (TypeScript projects)
- Mock-related failures