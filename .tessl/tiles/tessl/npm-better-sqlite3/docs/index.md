# better-sqlite3

better-sqlite3 is the fastest and simplest library for SQLite3 in Node.js, providing a synchronous API for high-performance database operations. Unlike the standard node-sqlite3 module which uses callbacks, better-sqlite3 offers direct return values and eliminates callback complexity while maintaining full SQLite feature support.

## Package Information

- **Package Name**: better-sqlite3
- **Package Type**: npm
- **Language**: JavaScript (with TypeScript definitions)
- **Installation**: `npm install better-sqlite3`

## Core Imports

```javascript
const Database = require('better-sqlite3');
```

For TypeScript:

```typescript
import Database from 'better-sqlite3';
// Or for named imports
import Database, { SqliteError } from 'better-sqlite3';
```

## Basic Usage

```javascript
const Database = require('better-sqlite3');

// Create/open database
const db = new Database('mydb.sqlite');

// Create table
db.exec('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)');

// Prepare and execute statements
const insert = db.prepare('INSERT INTO users (name, email) VALUES (?, ?)');
const insertResult = insert.run('John Doe', 'john@example.com');
console.log(insertResult.lastInsertRowid); // 1

// Query data
const getUser = db.prepare('SELECT * FROM users WHERE id = ?');
const user = getUser.get(1);
console.log(user); // { id: 1, name: 'John Doe', email: 'john@example.com' }

// Close database
db.close();
```

## Architecture

better-sqlite3 is built around several key components:

- **Database Class**: Main interface for database connections and management
- **Statement Objects**: Prepared statements for efficient query execution
- **Transaction System**: Built-in transaction management with nested transaction support
- **Type System**: Automatic JavaScript-SQLite type conversions with safe integer support
- **Extension System**: Support for user-defined functions, aggregates, and virtual tables
- **Synchronous API**: Direct return values eliminating callback complexity

## Capabilities

### Database Management

Core database connection and lifecycle management functionality for creating, configuring, and controlling SQLite database connections.

```javascript { .api }
/**
 * Creates a new database connection
 * @param {string|Buffer} filename - Database file path, ":memory:" for in-memory, or Buffer for serialized database
 * @param {Object} [options] - Database configuration options
 * @returns {Database} Database instance
 */
function Database(filename, options);

interface DatabaseOptions {
  readonly?: boolean;        // Open in readonly mode (default: false)
  fileMustExist?: boolean;   // Throw error if file doesn't exist (default: false)
  timeout?: number;          // Timeout in ms for locked database (default: 5000)
  verbose?: Function;        // Function called with every SQL execution
  nativeBinding?: string | Object; // Path to native binding or binding object
}
```

[Database Management](./database-management.md)

### Statement Execution

Prepared statement functionality for efficient SQL query execution with parameter binding and result handling.

```javascript { .api }
/**
 * Creates a prepared statement from SQL string
 * @param {string} sql - SQL query string
 * @returns {Statement} Prepared statement object
 */  
prepare(sql);

interface Statement {
  run(...params): RunResult;       // Execute statement, return info
  get(...params): Object;          // Execute statement, return first row
  all(...params): Object[];        // Execute statement, return all rows
  iterate(...params): Iterator;    // Execute statement, return iterator
  bind(...params): Statement;      // Permanently bind parameters
  readonly reader: boolean;        // Whether statement returns data
}

interface RunResult {
  changes: number;        // Number of rows changed
  lastInsertRowid: number; // ID of last inserted row
}
```

[Statement Execution](./statement-execution.md)

### Transaction Management

Transaction system providing ACID compliance with support for nested transactions using savepoints and multiple transaction types.

```javascript { .api }
/**
 * Creates a transaction function wrapper
 * @param {Function} fn - Function to execute within transaction
 * @returns {Function} Transaction function with transaction type variants
 */
transaction(fn);

interface TransactionFunction {
  (...args): any;              // Default transaction (BEGIN)
  deferred(...args): any;      // Deferred transaction (BEGIN DEFERRED)
  immediate(...args): any;     // Immediate transaction (BEGIN IMMEDIATE)
  exclusive(...args): any;     // Exclusive transaction (BEGIN EXCLUSIVE)
  readonly database: Database; // Associated database instance
}
```

[Transaction Management](./transaction-management.md)

### Database Utilities

Utility functions for database introspection, backup, serialization, and configuration management.

```javascript { .api }
/**
 * Execute PRAGMA commands for database configuration and introspection
 * @param {string} source - PRAGMA command string
 * @param {Object} [options] - Execution options
 * @returns {Array|any} Results array or single value if simple mode
 */
pragma(source, options);

/**
 * Backup database to file
 * @param {string} filename - Destination file path
 * @param {Object} [options] - Backup options
 * @returns {Promise} Promise resolving to backup progress info
 */
backup(filename, options);

/**
 * Serialize database to Buffer
 * @param {Object} [options] - Serialization options
 * @returns {Buffer} Serialized database buffer
 */
serialize(options);
```

[Database Utilities](./database-utilities.md)

### User-Defined Functions

Extension system for creating custom SQL functions, aggregate functions, and virtual tables to extend SQLite functionality.

```javascript { .api }
/**
 * Define user-defined function
 * @param {string} name - Function name
 * @param {Object} [options] - Function options
 * @param {Function} fn - JavaScript function implementation
 * @returns {Database} Database instance for chaining
 */
function(name, options, fn);

/**
 * Define user-defined aggregate function
 * @param {string} name - Aggregate function name
 * @param {Object} options - Aggregate configuration
 * @returns {Database} Database instance for chaining
 */
aggregate(name, options);

/**
 * Define virtual table module
 * @param {string} name - Module name
 * @param {Function|Object} factory - Factory function or table definition
 * @returns {Database} Database instance for chaining
 */
table(name, factory);
```

[User-Defined Functions](./user-defined-functions.md)

## Types

```javascript { .api }
class Database {
  constructor(filename, options);
  
  // Core methods
  prepare(sql): Statement;
  transaction(fn): TransactionFunction;
  exec(sql): Database;
  close(): Database;
  
  // Database utilities
  pragma(source, options): Array | any;
  backup(filename, options): Promise;
  serialize(options): Buffer;
  
  // Extensions
  function(name, options, fn): Database;
  aggregate(name, options): Database;
  table(name, factory): Database;
  loadExtension(path, entrypoint): Database;
  
  // Configuration
  defaultSafeIntegers(enabled): Database;
  unsafeMode(enabled): Database;
  
  // Properties (read-only)
  readonly name: string;          // Database filename
  readonly open: boolean;         // Connection status
  readonly inTransaction: boolean; // Transaction status
  readonly readonly: boolean;     // Readonly mode status
  readonly memory: boolean;       // In-memory database status
}

class Statement {
  // Execution methods
  run(...params): RunResult;
  get(...params): Object | undefined;
  all(...params): Object[];
  iterate(...params): Iterator;
  
  // Configuration
  bind(...params): Statement;
  pluck(enabled): Statement;
  expand(enabled): Statement;  
  raw(enabled): Statement;
  safeIntegers(enabled): Statement;
  columns(): ColumnDescriptor[];
  
  // Properties (read-only)
  readonly reader: boolean;       // Whether statement returns data
  readonly busy: boolean;         // Whether statement is executing
  readonly source: string;        // Original SQL string used to create statement
  readonly database: Database;    // Associated database instance
}

class SqliteError extends Error {
  constructor(message, code);
  readonly name: string;    // Always "SqliteError"
  readonly code: string;    // SQLite error code
  readonly message: string; // Error message
}

interface RunResult {
  changes: number;        // Number of rows changed
  lastInsertRowid: number; // ID of last inserted row
}

interface ColumnDescriptor {
  name: string;           // Column name in result set
  column: string | null;  // Original column name
  table: string | null;   // Table name
  database: string | null; // Database name
  type: string | null;    // Column data type
}
```