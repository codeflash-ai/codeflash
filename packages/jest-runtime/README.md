# @codeflash/jest-runtime

Jest runtime helpers for test instrumentation and behavior verification in Codeflash.

## Installation

```bash
npm install @codeflash/jest-runtime
# or
yarn add @codeflash/jest-runtime
# or
pnpm add @codeflash/jest-runtime
```

## Usage

### CommonJS

```javascript
const { capture, capturePerf } = require('@codeflash/jest-runtime');

// Capture function behavior (writes to SQLite)
const result = capture('fibonacci', '42', fibonacci, 10);

// Capture performance only (stdout timing markers)
const result = capturePerf('fibonacci', '42', fibonacci, 10);
```

### ES Modules

```javascript
import { capture, capturePerf } from '@codeflash/jest-runtime';

// Same API as CommonJS
const result = capture('fibonacci', '42', fibonacci, 10);
```

### TypeScript

```typescript
import { capture, capturePerf, ComparatorOptions } from '@codeflash/jest-runtime';

const result = capture<number>('fibonacci', '42', fibonacci, 10);
```

## API

### Instrumentation

- `capture(funcName, lineId, fn, ...args)` - Capture function return value and write to database
- `capturePerf(funcName, lineId, fn, ...args)` - Capture timing only (stdout markers)
- `captureMultiple(funcName, lineId, invocations)` - Capture multiple calls

### Serialization

- `serialize(value)` - Serialize any JavaScript value to Buffer
- `deserialize(buffer)` - Deserialize Buffer back to value
- `getSerializerType()` - Returns 'v8' or 'msgpack'

### Comparison

- `comparator(original, newValue, options?)` - Deep equality comparison
- `isClose(a, b, rtol?, atol?)` - Floating point tolerance comparison
- `strictComparator` - No tolerance comparator
- `looseComparator` - Larger tolerance comparator

### Result Management

- `writeResults()` - Flush results to disk
- `clearResults()` - Clear buffered results
- `getResults()` - Get buffered results
- `initDatabase()` - Initialize SQLite database

## Environment Variables

- `CODEFLASH_OUTPUT_FILE` - SQLite database path (default: `/tmp/codeflash_results.sqlite`)
- `CODEFLASH_LOOP_INDEX` - Current benchmark loop (default: `1`)
- `CODEFLASH_TEST_ITERATION` - Test iteration number (default: `0`)
- `CODEFLASH_TEST_MODULE` - Test module path

## CLI

Compare test results between two databases:

```bash
npx codeflash-compare original.sqlite candidate.sqlite
```

## License

MIT
