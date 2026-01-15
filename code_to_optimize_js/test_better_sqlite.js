try {
    const Database = require('better-sqlite3');
    console.log('better-sqlite3 loaded successfully');
    const db = new Database('/tmp/test_better_sqlite.db');
    db.exec('CREATE TABLE test (id INTEGER)');
    db.exec('INSERT INTO test VALUES (1)');
    const row = db.prepare('SELECT * FROM test').get();
    console.log('Row:', row);
    db.close();
    console.log('Database test passed');
} catch (e) {
    console.error('Error:', e.message);
}
