# Transaction Management

Transaction system providing ACID compliance with support for nested transactions using savepoints and multiple transaction types.

## Capabilities

### Transaction Creation

Create transaction wrapper functions that automatically handle BEGIN/COMMIT/ROLLBACK operations.

```javascript { .api }
/**
 * Creates a transaction function wrapper
 * @param {Function} fn - Function to execute within transaction
 * @returns {TransactionFunction} Transaction function with transaction type variants
 */
transaction(fn);

interface TransactionFunction {
  (...args): any;              // Default transaction using BEGIN
  deferred(...args): any;      // Deferred transaction using BEGIN DEFERRED  
  immediate(...args): any;     // Immediate transaction using BEGIN IMMEDIATE
  exclusive(...args): any;     // Exclusive transaction using BEGIN EXCLUSIVE
  readonly database: Database; // Associated database instance
}
```

**Usage Examples:**

```javascript
// Create transaction function
const insertMany = db.transaction((users) => {
  const insert = db.prepare('INSERT INTO users (name, email) VALUES (@name, @email)');
  for (const user of users) {
    insert.run(user);
  }
});

// Execute transaction
insertMany([
  { name: 'Alice', email: 'alice@example.com' },
  { name: 'Bob', email: 'bob@example.com' },
  { name: 'Charlie', email: 'charlie@example.com' }
]);

// All users inserted atomically, or none if any fails
```

### Transaction Types

Different transaction types provide various levels of locking and concurrency control.

```javascript { .api }
// Transaction type variants
transactionFunction();              // BEGIN (same as deferred)
transactionFunction.deferred();     // BEGIN DEFERRED (default SQLite behavior)
transactionFunction.immediate();    // BEGIN IMMEDIATE (acquire reserved lock immediately)
transactionFunction.exclusive();    // BEGIN EXCLUSIVE (acquire exclusive lock immediately)
```

**Transaction Type Behaviors:**

```javascript
const updateInventory = db.transaction((items) => {
  const update = db.prepare('UPDATE inventory SET quantity = quantity - ? WHERE item_id = ?');
  for (const item of items) {
    update.run(item.quantity, item.id);
  }
});

// Deferred transaction - acquire locks as needed (default)
updateInventory.deferred(items);

// Immediate transaction - acquire reserved lock immediately 
// (prevents other writers from starting)
updateInventory.immediate(items);

// Exclusive transaction - acquire exclusive lock immediately
// (prevents all other connections from reading or writing)
updateInventory.exclusive(items);
```

### Nested Transactions

Transactions can be nested using SQLite savepoints for complex error handling scenarios.

```javascript { .api }
// Nested transactions automatically use savepoints
// Inner transaction failures roll back to savepoint, not beginning
```

**Usage Examples:**

```javascript
const insertUser = db.prepare('INSERT INTO users (name, email) VALUES (?, ?)');
const insertProfile = db.prepare('INSERT INTO profiles (user_id, bio) VALUES (?, ?)');
const insertPreferences = db.prepare('INSERT INTO preferences (user_id, theme) VALUES (?, ?)');

// Outer transaction
const createUserWithProfile = db.transaction((userData) => {
  // Insert user
  const userResult = insertUser.run(userData.name, userData.email);
  const userId = userResult.lastInsertRowid;
  
  // Inner transaction for profile creation
  const createProfile = db.transaction((profileData) => {
    insertProfile.run(userId, profileData.bio);
    
    if (profileData.preferences) {
      // Nested inner transaction for preferences
      const setPreferences = db.transaction((prefs) => {
        insertPreferences.run(userId, prefs.theme);
        if (prefs.theme === 'invalid') {
          throw new Error('Invalid theme'); // This will rollback only preferences
        }
      });
      
      setPreferences(profileData.preferences);
    }
  });
  
  try {
    createProfile(userData.profile);
  } catch (error) {
    console.log('Profile creation failed, but user was created');
    // User insert is preserved, only profile operations rolled back
  }
});

// Execute the nested transaction
createUserWithProfile({
  name: 'John Doe',
  email: 'john@example.com',
  profile: {
    bio: 'Software developer',
    preferences: { theme: 'dark' }
  }
});
```

### Transaction Error Handling

Transactions automatically handle errors with proper rollback behavior.

```javascript { .api }
// Automatic rollback on any thrown error
// Error propagates normally after rollback
// Nested transactions use savepoints for partial rollback
```

**Usage Examples:**

```javascript
const transferFunds = db.transaction((fromAccount, toAccount, amount) => {
  const debit = db.prepare('UPDATE accounts SET balance = balance - ? WHERE id = ?');
  const credit = db.prepare('UPDATE accounts SET balance = balance + ? WHERE id = ?');
  const getBalance = db.prepare('SELECT balance FROM accounts WHERE id = ?');
  
  // Check sufficient funds
  const fromBalance = getBalance.get(fromAccount).balance;
  if (fromBalance < amount) {
    throw new Error('Insufficient funds'); // Transaction will rollback
  }
  
  // Perform transfer
  debit.run(amount, fromAccount);
  credit.run(amount, toAccount);
  
  // Verify the transfer
  const newFromBalance = getBalance.get(fromAccount).balance;
  const newToBalance = getBalance.get(toAccount).balance;
  
  return {
    fromBalance: newFromBalance,
    toBalance: newToBalance,
    transferred: amount
  };
});

try {
  const result = transferFunds(1, 2, 100);
  console.log('Transfer completed:', result);
} catch (error) {
  console.log('Transfer failed:', error.message);
  // Database state unchanged due to automatic rollback
}
```

### Transaction Function Properties

Transaction functions expose properties for introspection and database access.

```javascript { .api }
interface TransactionFunction {
  readonly database: Database; // Associated database instance
  // All transaction type variants (deferred, immediate, exclusive) have same properties
}
```

**Usage Examples:**

```javascript
const batchInsert = db.transaction((data) => {
  // Function implementation
});

// Access the associated database
console.log(batchInsert.database === db); // true

// All transaction variants share the same database reference
console.log(batchInsert.deferred.database === db); // true
console.log(batchInsert.immediate.database === db); // true
console.log(batchInsert.exclusive.database === db); // true

// Check if currently in transaction during execution
const checkTransactionStatus = db.transaction(() => {
  console.log(db.inTransaction); // true during transaction execution
});

checkTransactionStatus();
console.log(db.inTransaction); // false after transaction completes
```

### Transaction Best Practices

Guidelines for effective transaction usage and performance optimization.

**Performance Considerations:**

```javascript
// Good: Batch multiple operations in a single transaction
const batchUserInsert = db.transaction((users) => {
  const insert = db.prepare('INSERT INTO users (name, email) VALUES (?, ?)');
  for (const user of users) {
    insert.run(user.name, user.email);
  }
});

// Bad: Individual transactions for each operation (slow)
function insertUsersSlow(users) {
  const insert = db.prepare('INSERT INTO users (name, email) VALUES (?, ?)');
  for (const user of users) {
    const singleInsert = db.transaction(() => {
      insert.run(user.name, user.email);
    });
    singleInsert(); // Creates transaction overhead for each insert
  }
}

// Good: Short-lived transactions
const quickUpdate = db.transaction((id, newValue) => {
  db.prepare('UPDATE table SET value = ? WHERE id = ?').run(newValue, id);
});

// Avoid: Long-running transactions that hold locks
const avoidLongTransaction = db.transaction(() => {
  // Don't do this - holds database locks too long
  for (let i = 0; i < 1000000; i++) {
    // Long-running computation
    heavyComputation();
  }
  db.prepare('INSERT INTO results VALUES (?)').run(result);
});
```

**Error Handling Patterns:**

```javascript
// Pattern: Specific error handling with graceful degradation
const safeUserCreation = db.transaction((userData) => {
  const insertUser = db.prepare('INSERT INTO users (name, email) VALUES (?, ?)');
  const insertAuditLog = db.prepare('INSERT INTO audit_log (action, details) VALUES (?, ?)');
  
  try {
    const result = insertUser.run(userData.name, userData.email);
    
    // Nested transaction for audit log (can fail without affecting user creation)
    const logAudit = db.transaction(() => {
      insertAuditLog.run('user_created', JSON.stringify(userData));
    });
    
    try {
      logAudit();
    } catch (auditError) {
      console.warn('Audit logging failed:', auditError.message);
      // User creation still succeeds
    }
    
    return result.lastInsertRowid;
  } catch (error) {
    console.error('User creation failed:', error.message);
    throw error; // Re-throw to trigger rollback
  }
});
```

### Manual Transaction Control

While transaction functions are recommended, manual transaction control is also possible.

```javascript
// Manual transaction control (not recommended with transaction functions)
db.exec('BEGIN');
try {
  db.prepare('INSERT INTO users (name) VALUES (?)').run('John');
  db.prepare('INSERT INTO profiles (user_id) VALUES (?)').run(1);
  db.exec('COMMIT');
} catch (error) {
  db.exec('ROLLBACK');
  throw error;
}

// Warning: Don't mix manual control with transaction functions
const mixedApproach = db.transaction(() => {
  // Don't use raw COMMIT/ROLLBACK inside transaction functions
  // db.exec('COMMIT'); // This will cause issues
});
```