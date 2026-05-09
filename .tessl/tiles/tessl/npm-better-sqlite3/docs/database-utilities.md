# Database Utilities

Utility functions for database introspection, backup, serialization, and configuration management.

## Capabilities

### PRAGMA Commands

Execute PRAGMA commands for database configuration and introspection.

```javascript { .api }
/**
 * Execute PRAGMA commands for database configuration and introspection
 * @param {string} source - PRAGMA command string (without "PRAGMA" prefix)
 * @param {Object} [options] - Execution options
 * @returns {Array|any} Results array or single value if simple mode
 */
pragma(source, options);

interface PragmaOptions {
  simple?: boolean; // Return single value instead of array (default: false)
}
```

**Usage Examples:**

```javascript
// Get database information
const userVersion = db.pragma('user_version', { simple: true });
console.log(userVersion); // Returns number directly

// Get table information
const tableInfo = db.pragma('table_info(users)');
console.log(tableInfo);
// Returns array of column descriptors:
// [
//   { cid: 0, name: 'id', type: 'INTEGER', notnull: 0, dflt_value: null, pk: 1 },
//   { cid: 1, name: 'name', type: 'TEXT', notnull: 1, dflt_value: null, pk: 0 },
//   ...
// ]

// Get foreign key information
const foreignKeys = db.pragma('foreign_key_list(orders)');
foreignKeys.forEach(fk => {
  console.log(`${fk.from} references ${fk.table}.${fk.to}`);
});

// Database configuration
db.pragma('journal_mode = WAL'); // Enable WAL mode
db.pragma('synchronous = NORMAL'); // Set synchronous mode
db.pragma('cache_size = 10000'); // Set cache size

// Get current settings
const journalMode = db.pragma('journal_mode', { simple: true });
const pageSize = db.pragma('page_size', { simple: true });
const cacheSize = db.pragma('cache_size', { simple: true });

console.log(`Journal mode: ${journalMode}, Page size: ${pageSize}, Cache size: ${cacheSize}`);
```

### Database Backup

Create backups of the database to files with progress monitoring.

```javascript { .api }
/**
 * Backup database to file (async operation)
 * @param {string} filename - Destination file path
 * @param {Object} [options] - Backup options
 * @returns {Promise<BackupProgress>} Promise resolving to backup completion info
 */
backup(filename, options);

interface BackupOptions {
  attached?: string;   // Database name to backup (default: "main")
  progress?: Function; // Progress callback function
}

interface BackupProgress {
  totalPages: number;     // Total pages in database
  remainingPages: number; // Pages remaining to backup (0 when complete)
}
```

**Usage Examples:**

```javascript
// Simple backup
await db.backup('backup.db');
console.log('Backup completed');

// Backup with progress monitoring
await db.backup('backup-with-progress.db', {
  progress(info) {
    const percent = ((info.totalPages - info.remainingPages) / info.totalPages * 100).toFixed(1);
    console.log(`Backup progress: ${percent}% (${info.remainingPages} pages remaining)`);
    
    // Return custom page transfer rate (optional)
    // return 100; // Transfer 100 pages at a time
  }
});

// Backup attached database
db.exec("ATTACH DATABASE 'other.db' AS other");
await db.backup('other-backup.db', { attached: 'other' });

// Backup with error handling
try {
  await db.backup('/invalid/path/backup.db');
} catch (error) {
  console.error('Backup failed:', error.message);
}

// Throttled backup for large databases
await db.backup('large-backup.db', {
  progress(info) {
    if (info.remainingPages > 0) {
      // Transfer fewer pages at a time to avoid blocking
      return 10;
    }
  }
});
```

### Database Serialization

Serialize database to Buffer for embedding or transmission.

```javascript { .api }
/**
 * Serialize database to Buffer
 * @param {Object} [options] - Serialization options
 * @returns {Buffer} Buffer containing complete serialized database
 */
serialize(options);

interface SerializeOptions {
  attached?: string; // Database name to serialize (default: "main")
}
```

**Usage Examples:**

```javascript
// Serialize entire database to buffer
const serialized = db.serialize();
console.log(`Database serialized to ${serialized.length} bytes`);

// Save serialized database to file
const fs = require('fs');
fs.writeFileSync('serialized-db.buf', serialized);

// Restore database from serialized buffer
const restoredDb = new Database(serialized);

// Serialize attached database
db.exec("ATTACH DATABASE 'temp.db' AS temp");
const tempSerialized = db.serialize({ attached: 'temp' });

// Use serialization for database cloning
function cloneDatabase(sourceDb) {
  const serialized = sourceDb.serialize();
  return new Database(serialized);
}

const clonedDb = cloneDatabase(db);

// Serialization with compression (using external library)
const zlib = require('zlib');
const serialized = db.serialize();
const compressed = zlib.gzipSync(serialized);
console.log(`Compressed from ${serialized.length} to ${compressed.length} bytes`);

// Restore from compressed
const decompressed = zlib.gunzipSync(compressed);
const restoredFromCompressed = new Database(decompressed);
```

### Database Analysis and Introspection

Use PRAGMA commands for comprehensive database analysis.

```javascript { .api }
// Common introspection patterns using pragma()
```

**Schema Information:**

```javascript
// Get all table names
const tables = db.pragma('table_list');
const tableNames = tables.map(table => table.name);
console.log('Tables:', tableNames);

// Get detailed table information
function getTableSchema(tableName) {
  const columns = db.pragma(`table_info(${tableName})`);
  const indexes = db.pragma(`index_list(${tableName})`);
  const foreignKeys = db.pragma(`foreign_key_list(${tableName})`);
  
  return {
    columns: columns.map(col => ({
      name: col.name,
      type: col.type,
      nullable: !col.notnull,
      defaultValue: col.dflt_value,
      primaryKey: !!col.pk
    })),
    indexes: indexes.map(idx => ({
      name: idx.name,
      unique: !!idx.unique,
      partial: !!idx.partial
    })),
    foreignKeys: foreignKeys.map(fk => ({
      column: fk.from,
      referencesTable: fk.table,
      referencesColumn: fk.to,
      onUpdate: fk.on_update,
      onDelete: fk.on_delete
    }))
  };
}

const userSchema = getTableSchema('users');
console.log(userSchema);
```

**Database Statistics:**

```javascript
// Get database size and page information
const pageCount = db.pragma('page_count', { simple: true });
const pageSize = db.pragma('page_size', { simple: true });
const freePages = db.pragma('freelist_count', { simple: true });

const totalSize = pageCount * pageSize;
const freeSize = freePages * pageSize;
const usedSize = totalSize - freeSize;

console.log(`Database size: ${totalSize} bytes`);
console.log(`Used: ${usedSize} bytes (${(usedSize/totalSize*100).toFixed(1)}%)`);
console.log(`Free: ${freeSize} bytes (${(freeSize/totalSize*100).toFixed(1)}%)`);

// Get compilation options
const compileOptions = db.pragma('compile_options');
console.log('SQLite compile options:', compileOptions);

// Check integrity
const integrityCheck = db.pragma('integrity_check');
if (integrityCheck.length === 1 && integrityCheck[0] === 'ok') {
  console.log('Database integrity OK');
} else {
  console.warn('Database integrity issues:', integrityCheck);
}
```

**Performance Analysis:**

```javascript
// Analyze query performance
function analyzeQuery(sql) {
  const queryPlan = db.pragma(`query_plan(${sql})`);
  console.log('Query execution plan:');
  queryPlan.forEach(step => {
    console.log(`${step.id}: ${step.detail}`);
  });
}

analyzeQuery('SELECT * FROM users WHERE email = "test@example.com"');

// Get database statistics
const stats = db.pragma('stats');
stats.forEach(stat => {
  console.log(`${stat.table}: ${stat.index} (${stat.cells} cells)`);
});
```

### Database Optimization

Utility functions for database maintenance and optimization.

**Vacuum and Analyze:**

```javascript
// Optimize database storage
function optimizeDatabase() {
  console.log('Starting database optimization...');
  
  // Analyze all tables for query planner statistics
  db.exec('ANALYZE');
  
  // Rebuild database to reclaim space
  db.exec('VACUUM');
  
  console.log('Database optimization completed');
}

// Incremental vacuum (for WAL mode)
function incrementalVacuum(pages = 1000) {
  db.pragma(`incremental_vacuum(${pages})`);
}

// Auto-vacuum configuration
db.pragma('auto_vacuum = INCREMENTAL');
```

**Checkpoint Management (WAL Mode):**

```javascript
// Checkpoint WAL file
function checkpoint(mode = 'PASSIVE') {
  const result = db.pragma(`wal_checkpoint(${mode})`, { simple: false });
  return {
    busy: result[0],
    log: result[1],
    checkpointed: result[2]
  };
}

// Different checkpoint modes
const passiveResult = checkpoint('PASSIVE'); // Non-blocking
const fullResult = checkpoint('FULL');       // Block until complete
const restartResult = checkpoint('RESTART'); // Reset WAL file

console.log('Checkpoint results:', fullResult);
```

### Configuration Management

Manage database-wide configuration settings.

```javascript
// Performance tuning
function configureForPerformance() {
  db.pragma('journal_mode = WAL');      // Enable WAL mode for better concurrency
  db.pragma('synchronous = NORMAL');    // Balance safety and performance
  db.pragma('cache_size = 10000');      // Increase cache size
  db.pragma('temp_store = MEMORY');     // Use memory for temporary tables
  db.pragma('mmap_size = 134217728');   // Enable memory mapping (128MB)
}

// Safety-first configuration
function configureForSafety() {
  db.pragma('journal_mode = DELETE');   // Traditional rollback journal
  db.pragma('synchronous = FULL');      // Maximum durability
  db.pragma('foreign_keys = ON');       // Enforce foreign key constraints
}

// Get current configuration
function getCurrentConfig() {
  return {
    journalMode: db.pragma('journal_mode', { simple: true }),
    synchronous: db.pragma('synchronous', { simple: true }),
    cacheSize: db.pragma('cache_size', { simple: true }),
    foreignKeys: db.pragma('foreign_keys', { simple: true }),
    autoVacuum: db.pragma('auto_vacuum', { simple: true }),
    mmapSize: db.pragma('mmap_size', { simple: true })
  };
}

console.log('Current configuration:', getCurrentConfig());
```