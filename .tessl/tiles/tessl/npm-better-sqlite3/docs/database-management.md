# Database Management

Core database connection and lifecycle management functionality for creating, configuring, and controlling SQLite database connections.

## Capabilities

### Database Constructor

Creates a new database connection with comprehensive configuration options.

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

**Usage Examples:**

```javascript
const Database = require('better-sqlite3');

// Create/open a file database
const db = new Database('myapp.db');

// Create in-memory database
const memDb = new Database(':memory:');

// Create temporary database (deleted when closed)
const tempDb = new Database('');

// Open readonly database
const readOnlyDb = new Database('data.db', { readonly: true });

// Database with timeout and verbose logging
const verboseDb = new Database('app.db', {
  timeout: 10000,
  verbose: console.log
});

// Database that must exist (throws if file missing)
const existingDb = new Database('existing.db', { fileMustExist: true });

// Create from serialized buffer
const serializedBuffer = fs.readFileSync('backup.db');
const restoredDb = new Database(serializedBuffer);
```

### Direct SQL Execution

Execute SQL statements directly without prepared statements.

```javascript { .api }
/**
 * Execute SQL string directly (no prepared statement)
 * @param {string} sql - SQL query string (can contain multiple statements)
 * @returns {Database} Database instance for chaining
 */
exec(sql);
```

**Usage Examples:**

```javascript
// Create tables and initial data
db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE
  );
  
  CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
  
  INSERT OR IGNORE INTO users (name, email) VALUES 
    ('Admin', 'admin@example.com'),
    ('Guest', 'guest@example.com');
`);

// Drop and recreate table
db.exec('DROP TABLE IF EXISTS temp_data; CREATE TABLE temp_data (value TEXT);');
```

### Database Connection Management

Close database connections and manage connection lifecycle.

```javascript { .api }
/**
 * Close database connection
 * @returns {Database} Database instance for chaining
 */
close();
```

**Usage Examples:**

```javascript
// Proper database cleanup
try {
  const db = new Database('myapp.db');
  
  // Use database...
  
} finally {
  // Always close the database
  db.close();
}

// Check if database is still open
console.log(db.open); // false after closing
```

### Extension Loading

Load SQLite extensions to add functionality.

```javascript { .api }
/**
 * Load SQLite extension
 * @param {string} path - Path to extension file (.dll, .so, .dylib)
 * @param {string} [entrypoint] - Entry point function name (optional)
 * @returns {Database} Database instance for chaining
 */
loadExtension(path, entrypoint);
```

**Usage Examples:**

```javascript
// Load extension with automatic entry point
db.loadExtension('./extensions/json1.so');

// Load extension with specific entry point
db.loadExtension('./extensions/fts5.so', 'sqlite3_fts5_init');

// Load multiple extensions
db.loadExtension('./extensions/rtree.so')
  .loadExtension('./extensions/soundex.so');
```

### Database Configuration

Configure database-wide settings for integer handling and safety modes.

```javascript { .api }
/**
 * Set default safe integer handling for new statements
 * @param {boolean} enabled - Enable safe integers by default
 * @returns {Database} Database instance for chaining
 */
defaultSafeIntegers(enabled);

/**
 * Enable/disable unsafe mode (disables certain safety checks)
 * @param {boolean} enabled - Enable unsafe mode
 * @returns {Database} Database instance for chaining
 */
unsafeMode(enabled);
```

**Usage Examples:**

```javascript
// Enable safe integers for large numbers
db.defaultSafeIntegers(true);

// All new prepared statements will use safe integers
const stmt = db.prepare('SELECT very_large_number FROM table');
const result = stmt.get(); // Returns BigInt for large integers

// Enable unsafe mode for maximum performance (use with caution)
db.unsafeMode(true);
```

### Database Properties

Read-only properties providing database connection information.

```javascript { .api }
interface DatabaseProperties {
  readonly name: string;          // Database filename or ":memory:" 
  readonly open: boolean;         // Whether connection is open
  readonly inTransaction: boolean; // Whether currently in transaction
  readonly readonly: boolean;     // Whether database is readonly
  readonly memory: boolean;       // Whether database is in-memory
}
```

**Usage Examples:**

```javascript
const db = new Database('myapp.db', { readonly: true });

console.log(db.name);          // "myapp.db"
console.log(db.open);          // true
console.log(db.readonly);      // true
console.log(db.memory);        // false
console.log(db.inTransaction); // false

// Properties update automatically
const transaction = db.transaction(() => {
  console.log(db.inTransaction); // true during transaction
});

transaction();
console.log(db.inTransaction); // false after transaction

db.close();
console.log(db.open); // false
```

## Error Handling

```javascript { .api }
class SqliteError extends Error {
  constructor(message, code);
  readonly name: string;    // Always "SqliteError"
  readonly code: string;    // SQLite error code (e.g., "SQLITE_CONSTRAINT")
  readonly message: string; // Descriptive error message
}
```

**Common Error Scenarios:**

```javascript
try {
  const db = new Database('nonexistent.db', { fileMustExist: true });
} catch (error) {
  if (error instanceof Database.SqliteError) {
    console.log(error.code);    // "SQLITE_CANTOPEN"
    console.log(error.message); // "Cannot open database..."
  }
}

try {
  db.exec('INVALID SQL SYNTAX');
} catch (error) {
  console.log(error.code);    // "SQLITE_ERROR"
  console.log(error.message); // "near \"INVALID\": syntax error"
}
```