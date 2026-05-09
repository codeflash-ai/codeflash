# Statement Execution

Prepared statement functionality for efficient SQL query execution with parameter binding and result handling.

## Capabilities

### Statement Preparation

Create prepared statements from SQL strings for efficient repeated execution.

```javascript { .api }
/**
 * Creates a prepared statement from SQL string
 * @param {string} sql - SQL query string with optional parameter placeholders
 * @returns {Statement} Prepared statement object
 */
prepare(sql);
```

**Usage Examples:**

```javascript
// Prepare INSERT statement
const insertUser = db.prepare('INSERT INTO users (name, email) VALUES (?, ?)');

// Prepare SELECT statement with named parameters
const getUserByEmail = db.prepare('SELECT * FROM users WHERE email = @email');

// Prepare UPDATE statement with mixed parameters
const updateUser = db.prepare('UPDATE users SET name = ? WHERE id = $id');

// Prepare complex query
const getOrdersWithDetails = db.prepare(`
  SELECT o.id, o.total, u.name as customer_name, 
         COUNT(oi.id) as item_count
  FROM orders o
  JOIN users u ON o.user_id = u.id
  LEFT JOIN order_items oi ON o.id = oi.order_id
  WHERE o.created_at > ?
  GROUP BY o.id
  ORDER BY o.created_at DESC
`);
```

### Statement Execution Methods

Execute prepared statements and retrieve results in different formats.

```javascript { .api }
/**
 * Execute statement for data modification, return info about changes
 * @param {...any} params - Parameters to bind to statement
 * @returns {RunResult} Object with changes and lastInsertRowid
 */
run(...params);

/**
 * Execute statement and return first row
 * @param {...any} params - Parameters to bind to statement  
 * @returns {Object|undefined} First row object or undefined if no results
 */
get(...params);

/**
 * Execute statement and return all rows
 * @param {...any} params - Parameters to bind to statement
 * @returns {Object[]} Array of row objects
 */
all(...params);

/**
 * Execute statement and return iterator for memory-efficient processing
 * @param {...any} params - Parameters to bind to statement
 * @returns {Iterator<Object>} Iterator yielding row objects
 */
iterate(...params);

interface RunResult {
  changes: number;        // Number of rows changed by the operation
  lastInsertRowid: number; // Row ID of last inserted row (0 if none)
}
```

**Usage Examples:**

```javascript
// Using run() for data modification
const insertUser = db.prepare('INSERT INTO users (name, email) VALUES (?, ?)');
const result = insertUser.run('Alice Smith', 'alice@example.com');
console.log(result.changes);        // 1
console.log(result.lastInsertRowid); // 1 (the new user's ID)

// Using get() for single row retrieval
const getUser = db.prepare('SELECT * FROM users WHERE id = ?');
const user = getUser.get(1);
console.log(user); // { id: 1, name: 'Alice Smith', email: 'alice@example.com' }

// Using all() for multiple rows
const getAllUsers = db.prepare('SELECT * FROM users ORDER BY name');
const users = getAllUsers.all();
console.log(users.length); // Number of users
users.forEach(user => console.log(user.name));

// Using iterate() for memory-efficient processing of large result sets
const getActiveOrders = db.prepare('SELECT * FROM orders WHERE status = ?');
for (const order of getActiveOrders.iterate('active')) {
  console.log(`Order ${order.id}: $${order.total}`);
  // Process one order at a time without loading all into memory
}
```

### Parameter Binding

Bind parameters to prepared statements for secure and efficient execution.

```javascript { .api }
/**
 * Permanently bind parameters to statement
 * @param {...any} params - Parameters to bind (positional or named)
 * @returns {Statement} Statement instance for chaining
 */
bind(...params);
```

**Parameter Binding Styles:**

```javascript
// Positional parameters with ?
const stmt1 = db.prepare('SELECT * FROM users WHERE age > ? AND city = ?');
stmt1.run(25, 'New York'); // Temporary binding
stmt1.bind(25, 'New York'); // Permanent binding

// Named parameters with @name, :name, or $name
const stmt2 = db.prepare('SELECT * FROM users WHERE age > @minAge AND city = @city');
stmt2.run({ minAge: 25, city: 'New York' });

// Object binding for named parameters
const stmt3 = db.prepare('INSERT INTO users (name, email, age) VALUES (@name, @email, @age)');
stmt3.run({
  name: 'Bob Wilson',
  email: 'bob@example.com', 
  age: 30
});

// Array binding for positional parameters
const stmt4 = db.prepare('INSERT INTO users (name, email, age) VALUES (?, ?, ?)');
stmt4.run(['Charlie Brown', 'charlie@example.com', 28]);

// Permanent binding prevents further parameter passing
const boundStmt = db.prepare('SELECT * FROM users WHERE status = ?');
boundStmt.bind('active');
// boundStmt.run('inactive'); // Would throw TypeError
```

### Statement Configuration

Configure statement behavior for result formatting and data handling.

```javascript { .api }
/**
 * Enable/disable column plucking (return only first column value)
 * @param {boolean} enabled - Enable plucking mode
 * @returns {Statement} Statement instance for chaining
 */
pluck(enabled);

/**
 * Enable/disable row expansion (return nested objects by table)
 * @param {boolean} enabled - Enable expansion mode
 * @returns {Statement} Statement instance for chaining
 */
expand(enabled);

/**
 * Enable/disable raw mode (return arrays instead of objects)
 * @param {boolean} enabled - Enable raw mode
 * @returns {Statement} Statement instance for chaining
 */
raw(enabled);

/**
 * Enable/disable safe integer mode for this statement
 * @param {boolean} enabled - Enable safe integers (return BigInt for large numbers)
 * @returns {Statement} Statement instance for chaining
 */
safeIntegers(enabled);
```

**Usage Examples:**

```javascript
// Pluck mode - return only first column value
const getName = db.prepare('SELECT name FROM users WHERE id = ?').pluck();
const name = getName.get(1); // Returns "Alice Smith" instead of { name: "Alice Smith" }

// Raw mode - return arrays instead of objects
const getUserRaw = db.prepare('SELECT id, name, email FROM users WHERE id = ?').raw();
const userData = getUserRaw.get(1); // Returns [1, "Alice Smith", "alice@example.com"]

// Expand mode - nested objects by table
const getOrderWithUser = db.prepare(`
  SELECT o.id, o.total, u.name, u.email 
  FROM orders o 
  JOIN users u ON o.user_id = u.id 
  WHERE o.id = ?
`).expand();
const result = getOrderWithUser.get(1);
// Returns: { orders: { id: 1, total: 99.99 }, users: { name: "Alice", email: "alice@example.com" } }

// Safe integers for handling large numbers
const getBigNumber = db.prepare('SELECT big_integer_column FROM table WHERE id = ?').safeIntegers();
const bigNum = getBigNumber.get(1); // Returns BigInt instead of potentially unsafe number

// Chain configuration methods
const configuredStmt = db.prepare('SELECT COUNT(*) as count FROM users')
  .pluck()        // Only return the count value
  .safeIntegers(true); // Use BigInt for large counts
const userCount = configuredStmt.get(); // Returns BigInt directly
```

### Statement Metadata

Retrieve metadata information about prepared statements.

```javascript { .api }
/**
 * Get column information for SELECT statements
 * @returns {ColumnDescriptor[]} Array of column descriptors
 */
columns();

interface ColumnDescriptor {
  name: string;           // Column name in result set
  column: string | null;  // Original column name (null for expressions)
  table: string | null;   // Source table name (null for expressions)
  database: string | null; // Database name (typically "main")
  type: string | null;    // Declared column type (null for expressions)
}
```

**Usage Examples:**

```javascript
// Get column metadata
const stmt = db.prepare('SELECT u.id, u.name, COUNT(*) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id');
const columns = stmt.columns();

columns.forEach(col => {
  console.log(`Column: ${col.name}`);
  console.log(`  Original: ${col.column}`);
  console.log(`  Table: ${col.table}`);
  console.log(`  Type: ${col.type}`);
});

// Output:
// Column: id
//   Original: id
//   Table: users
//   Type: INTEGER
// Column: name  
//   Original: name
//   Table: users
//   Type: TEXT
// Column: order_count
//   Original: null
//   Table: null
//   Type: null
```

### Statement Properties

Read-only properties providing statement information.

```javascript { .api }
interface StatementProperties {
  readonly reader: boolean;       // Whether statement returns data (SELECT vs INSERT/UPDATE/etc)
  readonly busy: boolean;         // Whether statement is currently executing
  readonly database: Database;    // Associated database instance
}
```

**Usage Examples:**

```javascript
const selectStmt = db.prepare('SELECT * FROM users');
const insertStmt = db.prepare('INSERT INTO users (name) VALUES (?)');

console.log(selectStmt.reader); // true (SELECT statement)
console.log(insertStmt.reader); // false (INSERT statement)

// Can't use get/all/iterate on non-reader statements
try {
  insertStmt.get(); // Throws TypeError
} catch (error) {
  console.log(error.message); // "This statement does not return data"
}

// Can't use columns() on non-reader statements  
try {
  insertStmt.columns(); // Throws TypeError
} catch (error) {
  console.log(error.message); // "This statement does not return data"
}

console.log(selectStmt.database === db); // true
console.log(selectStmt.busy); // false (when not executing)
```

## Data Type Conversions

better-sqlite3 automatically converts between JavaScript and SQLite data types:

**JavaScript to SQLite:**
- `null`, `undefined` → NULL
- `string` → TEXT  
- `number` → INTEGER or REAL
- `bigint` → INTEGER (when safe integers enabled)
- `boolean` → INTEGER (0 or 1)
- `Buffer` → BLOB
- `Date` → TEXT (ISO string) or INTEGER (timestamp)

**SQLite to JavaScript:**
- NULL → `null`
- INTEGER → `number` or `bigint` (when safe integers enabled)
- REAL → `number`
- TEXT → `string`
- BLOB → `Buffer`

**Usage Examples:**

```javascript
// Inserting different data types
const insertData = db.prepare('INSERT INTO mixed_data (text_col, int_col, real_col, blob_col, bool_col) VALUES (?, ?, ?, ?, ?)');
insertData.run(
  'Hello World',              // TEXT
  42,                         // INTEGER  
  3.14159,                    // REAL
  Buffer.from('binary data'), // BLOB
  true                        // INTEGER (1)
);

// Date handling
const insertDate = db.prepare('INSERT INTO events (name, created_at) VALUES (?, ?)');
insertDate.run('Event Name', new Date()); // Date becomes TEXT ISO string

// Large integer handling with safe integers
db.defaultSafeIntegers(true);
const insertBigInt = db.prepare('INSERT INTO big_numbers (value) VALUES (?)');
insertBigInt.run(9007199254740992n); // BigInt preserved as INTEGER

const getBigInt = db.prepare('SELECT value FROM big_numbers WHERE id = ?');
const result = getBigInt.get(1);
console.log(typeof result.value); // "bigint"
```