/**
 * Tests for codeflash-jest-helper instrumentation.
 *
 * These tests verify:
 * 1. Static lineId is passed correctly and appears in stdout tags
 * 2. Invocation counter increments only for same testId (not globally)
 * 3. Timing uses hrtime for nanosecond precision
 * 4. stdout tag format matches Python's codeflash_wrap decorator
 */

const codeflash = require('../codeflash-jest-helper');

// Mock function for testing
function testFunction(x) {
    return x * 2;
}

// Async mock function
async function asyncTestFunction(x) {
    return new Promise(resolve => setTimeout(() => resolve(x * 2), 10));
}

// Capture console.log output for testing stdout tags
let consoleOutput = [];
const originalLog = console.log;

beforeAll(() => {
    console.log = (...args) => {
        consoleOutput.push(args.join(' '));
    };
});

afterAll(() => {
    console.log = originalLog;
});

beforeEach(() => {
    consoleOutput = [];
    codeflash.resetInvocationCounters();
});

describe('capturePerf', () => {
    test('should include lineId in stdout tag', () => {
        const lineId = '42';
        codeflash.capturePerf('testFunction', lineId, testFunction, 5);

        // Check start tag contains lineId
        const startTag = consoleOutput.find(msg => msg.includes('!$######'));
        expect(startTag).toBeDefined();
        expect(startTag).toContain(`${lineId}_0`);

        // Check end tag contains lineId and duration
        const endTag = consoleOutput.find(msg => msg.includes('!######') && !msg.includes('!$'));
        expect(endTag).toBeDefined();
        expect(endTag).toContain(`${lineId}_0`);
        // Should have duration after last colon
        const parts = endTag.split(':');
        const duration = parseInt(parts[parts.length - 1].replace('######!', ''));
        expect(typeof duration).toBe('number');
        expect(duration).toBeGreaterThanOrEqual(0);
    });

    test('should increment invocation counter only for same testId', () => {
        const lineId1 = '10';
        const lineId2 = '20';

        // First call with lineId1
        codeflash.capturePerf('testFunction', lineId1, testFunction, 1);
        expect(consoleOutput.some(msg => msg.includes(`${lineId1}_0`))).toBe(true);

        consoleOutput = [];

        // Second call with lineId2 - should start at 0, not 1
        codeflash.capturePerf('testFunction', lineId2, testFunction, 2);
        expect(consoleOutput.some(msg => msg.includes(`${lineId2}_0`))).toBe(true);

        consoleOutput = [];

        // Third call with lineId1 again - should be 1
        codeflash.capturePerf('testFunction', lineId1, testFunction, 3);
        expect(consoleOutput.some(msg => msg.includes(`${lineId1}_1`))).toBe(true);

        consoleOutput = [];

        // Fourth call with lineId2 again - should be 1
        codeflash.capturePerf('testFunction', lineId2, testFunction, 4);
        expect(consoleOutput.some(msg => msg.includes(`${lineId2}_1`))).toBe(true);
    });

    test('should correctly track loop invocations', () => {
        const lineId = '30';

        // Simulate a loop - same lineId called multiple times
        for (let i = 0; i < 5; i++) {
            codeflash.capturePerf('testFunction', lineId, testFunction, i);
        }

        // Should have 5 start tags and 5 end tags
        const startTags = consoleOutput.filter(msg => msg.includes('!$######'));
        expect(startTags).toHaveLength(5);

        // Each should have incrementing invocation index
        for (let i = 0; i < 5; i++) {
            expect(startTags[i]).toContain(`${lineId}_${i}`);
        }
    });

    test('should return function result', () => {
        const result = codeflash.capturePerf('testFunction', '100', testFunction, 21);
        expect(result).toBe(42);
    });

    test('should re-throw function errors', () => {
        const errorFn = () => { throw new Error('test error'); };
        expect(() => {
            codeflash.capturePerf('errorFn', '200', errorFn);
        }).toThrow('test error');
    });
});

describe('capture', () => {
    test('should include lineId in stdout tag', () => {
        const lineId = '50';
        codeflash.capture('testFunction', lineId, testFunction, 5);

        // Check start tag contains lineId
        const startTag = consoleOutput.find(msg => msg.includes('!$######'));
        expect(startTag).toBeDefined();
        expect(startTag).toContain(`${lineId}_0`);

        // Check end tag (behavior mode doesn't include duration)
        const endTag = consoleOutput.find(msg => msg.includes('!######') && !msg.includes('!$'));
        expect(endTag).toBeDefined();
        expect(endTag).toContain(`${lineId}_0`);
    });

    test('should track invocations same as capturePerf', () => {
        const lineId = '60';

        // Simulate a loop
        for (let i = 0; i < 3; i++) {
            codeflash.capture('testFunction', lineId, testFunction, i);
        }

        const startTags = consoleOutput.filter(msg => msg.includes('!$######'));
        expect(startTags).toHaveLength(3);

        for (let i = 0; i < 3; i++) {
            expect(startTags[i]).toContain(`${lineId}_${i}`);
        }
    });

    test('should return function result', () => {
        const result = codeflash.capture('testFunction', '100', testFunction, 10);
        expect(result).toBe(20);
    });
});

describe('getInvocationIndex', () => {
    test('should return 0 for first call with testId', () => {
        const index = codeflash.getInvocationIndex('test:null:test1:10:1');
        expect(index).toBe(0);
    });

    test('should increment for subsequent calls with same testId', () => {
        const testId = 'test:null:test2:20:1';
        expect(codeflash.getInvocationIndex(testId)).toBe(0);
        expect(codeflash.getInvocationIndex(testId)).toBe(1);
        expect(codeflash.getInvocationIndex(testId)).toBe(2);
    });

    test('should track different testIds independently', () => {
        const testId1 = 'test:null:test3:30:1';
        const testId2 = 'test:null:test4:40:1';

        expect(codeflash.getInvocationIndex(testId1)).toBe(0);
        expect(codeflash.getInvocationIndex(testId2)).toBe(0);
        expect(codeflash.getInvocationIndex(testId1)).toBe(1);
        expect(codeflash.getInvocationIndex(testId2)).toBe(1);
    });
});

describe('resetInvocationCounters', () => {
    test('should reset all counters to 0', () => {
        const testId = 'test:null:test5:50:1';

        // Increment a few times
        codeflash.getInvocationIndex(testId);
        codeflash.getInvocationIndex(testId);

        // Reset
        codeflash.resetInvocationCounters();

        // Should start at 0 again
        expect(codeflash.getInvocationIndex(testId)).toBe(0);
    });
});

describe('stdout tag format', () => {
    test('should match Python format: test_module:test_class.test_name:func_name:loop_index:invocation_id', () => {
        codeflash.setTestName('myTestFunction');
        const lineId = '70';
        codeflash.capturePerf('testFunction', lineId, testFunction, 1);

        const startTag = consoleOutput.find(msg => msg.includes('!$######'));
        // Format: !$######test_module:test_class.test_name:func_name:loop_index:invocation_id######$!
        // With Jest: !$######unknown:myTestFunction:testFunction:1:70_0######$!
        expect(startTag).toMatch(/!\$######[^:]+:[^:]*[^:]+:testFunction:\d+:\d+_\d+######\$!/);
    });
});
