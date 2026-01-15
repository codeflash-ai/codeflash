const codeflash = require('./codeflash-jest-helper');
const { reverseString } = require('./string_utils');

// Manually set test context
process.env.CODEFLASH_OUTPUT_FILE = '/tmp/test_codeflash.sqlite';
process.env.CODEFLASH_LOOP_INDEX = '1';
process.env.CODEFLASH_TEST_MODULE = 'test_module';

// Mock beforeEach/afterAll for non-Jest environment
global.expect = { getState: () => ({ currentTestName: 'manual_test' }) };

// Initialize database
codeflash.initDatabase();
codeflash.setTestName('manual_test');

// Capture a function call
const result = codeflash.capture('reverseString', reverseString, 'hello');
console.log('Result:', result);

// Write results
codeflash.writeResults();
console.log('Done');
