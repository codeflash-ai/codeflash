# User-Defined Functions

Extension system for creating custom SQL functions, aggregate functions, and virtual tables to extend SQLite functionality.

## Capabilities

### User-Defined Functions

Create custom SQL functions that can be called from within SQL queries.

```javascript { .api }
/**
 * Define user-defined function
 * @param {string} name - Function name (used in SQL queries)
 * @param {Object} [options] - Function configuration options
 * @param {Function} fn - JavaScript function implementation
 * @returns {Database} Database instance for chaining
 */
function(name, options, fn);

interface FunctionOptions {
  safeIntegers?: boolean; // Use safe integers for this function (default: database setting)
  deterministic?: boolean; // Function is deterministic (default: false)
  directOnly?: boolean;   // Function can only be called directly, not in triggers/views (default: false) 
  varargs?: boolean;      // Function accepts variable arguments (default: false)
}
```

**Usage Examples:**

```javascript
// Simple string function
db.function('reverse', (str) => {
  return typeof str === 'string' ? str.split('').reverse().join('') : null;
});

// Use in SQL query
const result = db.prepare('SELECT reverse(name) as reversed_name FROM users').all();
console.log(result); // [{ reversed_name: 'ecilA' }, { reversed_name: 'boB' }, ...]

// Math function with multiple parameters
db.function('pythagoras', (a, b) => {
  if (typeof a !== 'number' || typeof b !== 'number') return null;
  return Math.sqrt(a * a + b * b);
});

const distance = db.prepare('SELECT pythagoras(3, 4) as distance').get();
console.log(distance.distance); // 5

// Function with options
db.function('random_int', {
  deterministic: false,  // Result can vary between calls
  directOnly: true      // Can't be used in triggers or views
}, (min, max) => {
  if (typeof min !== 'number' || typeof max !== 'number') return null;
  return Math.floor(Math.random() * (max - min + 1)) + min;
});

// Variable arguments function
db.function('concat_with_separator', {
  varargs: true
}, (separator, ...args) => {
  if (typeof separator !== 'string') return null;
  return args.filter(arg => arg != null).join(separator);
});

const concatenated = db.prepare("SELECT concat_with_separator(' | ', name, email, city) as info FROM users").all();
```

### Aggregate Functions

Create custom aggregate functions for GROUP BY operations and window functions.

```javascript { .api }
/**
 * Define user-defined aggregate function
 * @param {string} name - Aggregate function name
 * @param {Object} options - Aggregate configuration (required)
 * @returns {Database} Database instance for chaining
 */
aggregate(name, options);

interface AggregateOptions {
  start?: any;            // Initial accumulator value (default: null)
  step: Function;         // Step function called for each row (required)
  inverse?: Function;     // Inverse function for window functions (optional)
  result?: Function;      // Final result transformation function (optional)
  safeIntegers?: boolean; // Use safe integers (default: database setting)
  deterministic?: boolean; // Function is deterministic (default: false)
  directOnly?: boolean;   // Function can only be called directly (default: false)
  varargs?: boolean;      // Function accepts variable arguments (default: false)
}
```

**Usage Examples:**

```javascript
// String concatenation aggregate
db.aggregate('group_concat_custom', {
  start: '',
  step: (accumulator, value) => {
    if (value == null) return accumulator;
    return accumulator === '' ? String(value) : accumulator + ', ' + String(value);
  },
  result: (accumulator) => accumulator || null
});

const groupedNames = db.prepare('SELECT group_concat_custom(name) as names FROM users').get();
console.log(groupedNames.names); // "Alice, Bob, Charlie"

// Mathematical aggregate - geometric mean
db.aggregate('geometric_mean', {
  start: { product: 1, count: 0 },
  step: (accumulator, value) => {
    if (typeof value !== 'number' || value <= 0) return accumulator;
    return {
      product: accumulator.product * value,
      count: accumulator.count + 1
    };
  },
  result: (accumulator) => {
    if (accumulator.count === 0) return null;
    return Math.pow(accumulator.product, 1 / accumulator.count);
  }
});

const geometricMean = db.prepare('SELECT geometric_mean(value) as geo_mean FROM measurements').get();

// Window function with inverse (for sliding windows)
db.aggregate('running_average', {
  start: { sum: 0, count: 0 },
  step: (accumulator, value) => {
    if (typeof value !== 'number') return accumulator;
    return {
      sum: accumulator.sum + value,
      count: accumulator.count + 1
    };
  },
  inverse: (accumulator, value) => {
    if (typeof value !== 'number') return accumulator;
    return {
      sum: accumulator.sum - value,
      count: accumulator.count - 1
    };
  },
  result: (accumulator) => {
    return accumulator.count > 0 ? accumulator.sum / accumulator.count : null;
  }
});

// Use as window function
const runningAverage = db.prepare(`
  SELECT value, 
         running_average(value) OVER (ORDER BY id ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as avg_3
  FROM measurements 
  ORDER BY id
`).all();

// Complex aggregate with multiple parameters
db.aggregate('weighted_average', {
  start: { weightedSum: 0, totalWeight: 0 },
  step: (accumulator, value, weight) => {
    if (typeof value !== 'number' || typeof weight !== 'number' || weight <= 0) {
      return accumulator;
    }
    return {
      weightedSum: accumulator.weightedSum + (value * weight),
      totalWeight: accumulator.totalWeight + weight
    };
  },
  result: (accumulator) => {
    return accumulator.totalWeight > 0 ? accumulator.weightedSum / accumulator.totalWeight : null;
  }
});
```

### Virtual Tables

Create virtual tables that generate data dynamically or interface with external data sources.

```javascript { .api }
/**
 * Define virtual table module
 * @param {string} name - Module name
 * @param {Function|Object} factory - Factory function or eponymous table definition
 * @returns {Database} Database instance for chaining
 */
table(name, factory);

// Factory function signature
interface TableFactory {
  (moduleName: string, databaseName: string, tableName: string, ...args): TableDefinition;
}

// Table definition object
interface TableDefinition {
  columns: string[];           // Column names (required)
  rows: GeneratorFunction;     // Generator function yielding rows (required)
  parameters?: string[];       // Parameter names (optional, inferred from rows.length)
  safeIntegers?: boolean;     // Use safe integers (default: database setting)
  directOnly?: boolean;       // Table can only be accessed directly (default: false)
}
```

**Usage Examples:**

```javascript
// Simple eponymous-only virtual table (no factory)
db.table('fibonacci', {
  columns: ['value', 'position'],
  *rows(limit = 10) {
    let a = 0, b = 1, position = 0;
    while (position < limit) {
      yield [a, position];
      [a, b] = [b, a + b];
      position++;
    }
  }
});

// Use the virtual table
const fibNumbers = db.prepare('SELECT * FROM fibonacci(15)').all();
console.log(fibNumbers); // First 15 Fibonacci numbers

// Factory-based virtual table for flexibility
db.table('sequence_generator', function(moduleName, databaseName, tableName, type) {
  return {
    columns: ['value', 'index'],
    parameters: ['start', 'end', 'step'],
    *rows(start, end, step = 1) {
      let index = 0;
      for (let value = start; value <= end; value += step) {
        if (type === 'even' && value % 2 !== 0) continue;
        if (type === 'odd' && value % 2 === 0) continue;
        yield { value, index: index++ };
      }
    }
  };
});

// Create specific table instances
db.exec('CREATE VIRTUAL TABLE even_numbers USING sequence_generator(even)');
db.exec('CREATE VIRTUAL TABLE odd_numbers USING sequence_generator(odd)');

// Query the virtual tables
const evenNumbers = db.prepare('SELECT * FROM even_numbers(2, 20, 2)').all();
const oddNumbers = db.prepare('SELECT * FROM odd_numbers(1, 19, 2)').all();

// Complex virtual table with external data
db.table('file_system', {
  columns: ['name', 'size', 'type', 'modified'],
  parameters: ['directory'],
  *rows(directory = '.') {
    const fs = require('fs');
    const path = require('path');
    
    try {
      const entries = fs.readdirSync(directory, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(directory, entry.name);
        const stats = fs.statSync(fullPath);
        
        yield {
          name: entry.name,
          size: stats.size,
          type: entry.isDirectory() ? 'directory' : 'file',
          modified: stats.mtime.toISOString()
        };
      }
    } catch (error) {
      // Handle directory access errors gracefully
      console.warn(`Cannot read directory ${directory}:`, error.message);
    }
  }
});

// Query file system
const files = db.prepare(`
  SELECT name, size, type 
  FROM file_system('/usr/local/bin') 
  WHERE type = 'file' AND size > 1000 
  ORDER BY size DESC
`).all();

// Virtual table with JSON data processing
db.table('json_parser', {
  columns: ['key', 'value', 'type'],
  parameters: ['json_string', 'path'],
  *rows(jsonString, path = '$') {
    try {
      const data = JSON.parse(jsonString);
      
      function* processObject(obj, currentPath) {
        for (const [key, value] of Object.entries(obj)) {
          const fullPath = currentPath === '$' ? `$.${key}` : `${currentPath}.${key}`;
          
          if (path === '$' || fullPath.startsWith(path)) {
            yield {
              key: fullPath,
              value: typeof value === 'object' ? JSON.stringify(value) : String(value),
              type: Array.isArray(value) ? 'array' : typeof value
            };
          }
          
          if (typeof value === 'object' && value !== null) {
            yield* processObject(value, fullPath);
          }
        }
      }
      
      yield* processObject(data, '$');
    } catch (error) {
      yield { key: 'error', value: error.message, type: 'error' };
    }
  }
});

// Parse JSON data
const jsonData = '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}';
const parsedData = db.prepare(`
  SELECT key, value, type 
  FROM json_parser(?, '$.users') 
  WHERE type != 'object'
`).all(jsonData);
```

### Function and Table Management

Utilities for managing user-defined functions and virtual tables.

**Function Replacement:**

```javascript
// Functions can be replaced by redefining
db.function('my_function', () => 'version 1');
let result1 = db.prepare('SELECT my_function() as result').get();
console.log(result1.result); // "version 1"

db.function('my_function', () => 'version 2');
let result2 = db.prepare('SELECT my_function() as result').get();
console.log(result2.result); // "version 2"

// Remove function by providing null
db.function('my_function', null);
// Now my_function() is no longer available
```

**Error Handling in Functions:**

```javascript
// Handle errors gracefully in user-defined functions
db.function('safe_divide', (a, b) => {
  if (typeof a !== 'number' || typeof b !== 'number') return null;
  if (b === 0) return null; // Return null for division by zero
  return a / b;
});

// Function that throws errors
db.function('strict_divide', (a, b) => {
  if (typeof a !== 'number' || typeof b !== 'number') {
    throw new Error('Arguments must be numbers');
  }
  if (b === 0) {
    throw new Error('Division by zero');
  }
  return a / b;
});

// Errors in functions become SQL errors
try {
  db.prepare('SELECT strict_divide(10, 0)').get();
} catch (error) {
  console.log(error.message); // "Division by zero"
}
```

**Performance Considerations:**

```javascript
// Use deterministic flag for cacheable functions
db.function('expensive_calculation', {
  deterministic: true // SQLite can cache results
}, (input) => {
  // Expensive computation that always returns same result for same input
  return heavyMathOperation(input);
});

// Use directOnly for security-sensitive functions
db.function('get_secret', {
  directOnly: true // Prevents use in triggers, views, or stored procedures
}, () => {
  return process.env.SECRET_KEY;
});

// Optimize virtual tables for large datasets
db.table('large_dataset', {
  columns: ['id', 'data'],
  *rows(limit = 1000000) {
    for (let i = 0; i < limit; i++) {
      // Yield data incrementally to avoid memory issues
      yield { id: i, data: `record_${i}` };
      
      // Optional: yield control periodically for very large datasets
      if (i % 10000 === 0) {
        setImmediate(() => {}); // Allow event loop to process other tasks
      }
    }
  }
});
```