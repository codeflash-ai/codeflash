# Test Scheduling

Jest's test scheduling system manages test execution coordination, multi-process scheduling, reporter management, and comprehensive result aggregation for optimal performance and reporting.

## Capabilities

### TestScheduler Class

The TestScheduler manages test execution scheduling and coordinates with reporters for comprehensive test result processing.

```typescript { .api }
/**
 * Manages test execution scheduling and reporting
 */
class TestScheduler {
  constructor(globalConfig: Config.GlobalConfig, context: TestSchedulerContext);
  
  /**
   * Adds a reporter to the dispatcher
   * @param reporter - Reporter instance to add
   */
  addReporter(reporter: Reporter): void;
  
  /**
   * Removes a reporter from the dispatcher
   * @param reporterConstructor - Constructor of reporter to remove
   */
  removeReporter(reporterConstructor: ReporterConstructor): void;
  
  /**
   * Schedules and executes the given tests
   * @param tests - Array of test files to execute
   * @param watcher - Test watcher for watch mode integration
   * @returns Promise resolving to aggregated test results
   */
  scheduleTests(tests: Array<Test>, watcher: TestWatcher): Promise<AggregatedResult>;
}

/**
 * Factory function that creates a TestScheduler and sets up reporters
 * @param globalConfig - Global Jest configuration
 * @param context - Scheduler context including reporter and test runner contexts
 * @returns Promise resolving to configured test scheduler
 */
function createTestScheduler(
  globalConfig: Config.GlobalConfig,
  context: TestSchedulerContext
): Promise<TestScheduler>;

type TestSchedulerContext = ReporterContext & TestRunnerContext;

type ReporterConstructor = new (
  globalConfig: Config.GlobalConfig,
  reporterConfig: Record<string, unknown>,
  reporterContext: ReporterContext,
) => JestReporter;
```

**Usage Examples:**

```typescript
import { createTestScheduler, SearchSource } from "jest";

// Create and configure test scheduler
async function setupTestScheduler() {
  const scheduler = await createTestScheduler(globalConfig, {
    ...reporterContext,
    ...testRunnerContext
  });
  
  // Add custom reporter
  scheduler.addReporter(new CustomReporter(globalConfig, {}, reporterContext));
  
  return scheduler;
}

// Execute tests with scheduler
async function executeTests(testFiles: Array<Test>) {
  const scheduler = await setupTestScheduler();
  
  const results = await scheduler.scheduleTests(
    testFiles,
    new TestWatcher({ isWatchMode: false })
  );
  
  return results;
}
```

### Test Execution Coordination

The TestScheduler coordinates test execution across multiple processes and manages the complete test lifecycle.

**Basic Test Scheduling:**

```typescript
import { createTestScheduler, SearchSource } from "jest";

async function coordinateTestExecution() {
  // 1. Discover tests
  const searchSource = new SearchSource(testContext);
  const searchResult = await searchSource.getTestPaths(
    globalConfig,
    projectConfig
  );
  
  // 2. Create scheduler
  const scheduler = await createTestScheduler(globalConfig, schedulerContext);
  
  // 3. Schedule and execute tests
  const results = await scheduler.scheduleTests(
    searchResult.tests,
    new TestWatcher({ isWatchMode: false })
  );
  
  console.log(`Executed ${results.numTotalTests} tests`);
  console.log(`Passed: ${results.numPassedTests}`);
  console.log(`Failed: ${results.numFailedTests}`);
  
  return results;
}
```

### Reporter Management

Manage test result reporting through dynamic reporter configuration.

```typescript
import { createTestScheduler } from "jest";

// Custom reporter for specialized output
class CustomReporter {
  constructor(
    private globalConfig: Config.GlobalConfig,
    private options: Record<string, unknown>,
    private context: ReporterContext
  ) {}
  
  onRunStart(results: AggregatedResult, options: ReporterOnStartOptions) {
    console.log("Starting test run...");
  }
  
  onTestResult(test: Test, testResult: TestResult, results: AggregatedResult) {
    if (testResult.testResults.some(result => result.status === "failed")) {
      console.log(`❌ ${test.path}`);
    } else {
      console.log(`✅ ${test.path}`);
    }
  }
  
  onRunComplete(contexts: Set<TestContext>, results: AggregatedResult) {
    console.log(`Test run completed: ${results.success ? "PASSED" : "FAILED"}`);
  }
}

// Configure scheduler with custom reporters
async function setupCustomReporting() {
  const scheduler = await createTestScheduler(globalConfig, schedulerContext);
  
  // Add multiple reporters
  scheduler.addReporter(new CustomReporter(globalConfig, {}, reporterContext));
  scheduler.addReporter(new JSONReporter(globalConfig, { outputFile: "results.json" }, reporterContext));
  
  // Remove default reporter if needed
  scheduler.removeReporter(DefaultReporter);
  
  return scheduler;
}
```

### Multi-Process Coordination

Handle test execution across multiple worker processes for optimal performance.

```typescript
import { createTestScheduler } from "jest";

async function scheduleTestsWithWorkers(maxWorkers: number) {
  const globalConfig = {
    ...baseGlobalConfig,
    maxWorkers,
    runInBand: maxWorkers === 1
  };
  
  const scheduler = await createTestScheduler(globalConfig, schedulerContext);
  
  // Configure for multi-process execution
  const results = await scheduler.scheduleTests(
    testFiles,
    new TestWatcher({ 
      isWatchMode: false,
      // Additional options for worker coordination
    })
  );
  
  return results;
}

// Adaptive worker configuration
async function adaptiveTestScheduling(testCount: number) {
  // Determine optimal worker count based on test count and system resources
  const maxWorkers = Math.min(
    Math.max(1, Math.floor(testCount / 10)), // At least 10 tests per worker
    require("os").cpus().length, // Don't exceed CPU count
    8 // Cap at 8 workers
  );
  
  return scheduleTestsWithWorkers(maxWorkers);
}
```

### Watch Mode Integration

Integrate with Jest's watch mode for automatic test re-execution.

```typescript
import { createTestScheduler } from "jest";
import { TestWatcher } from "jest-watcher";

async function scheduleTestsInWatchMode() {
  const scheduler = await createTestScheduler(
    { ...globalConfig, watch: true },
    schedulerContext
  );
  
  const watcher = new TestWatcher({ isWatchMode: true });
  
  // Watch mode provides automatic re-scheduling
  const results = await scheduler.scheduleTests(testFiles, watcher);
  
  // In watch mode, this promise typically never resolves
  // as Jest continues watching for file changes
  return results;
}

// Custom watch mode logic
class CustomTestWatcher extends TestWatcher {
  constructor(options: { isWatchMode: boolean }) {
    super(options);
  }
  
  async onChange(changedFiles: Set<string>) {
    console.log(`Files changed: ${Array.from(changedFiles).join(", ")}`);
    
    // Custom logic for determining which tests to re-run
    const searchSource = new SearchSource(testContext);
    const relatedTests = await searchSource.findRelatedTests(changedFiles, false);
    
    // Re-schedule only related tests
    if (relatedTests.tests.length > 0) {
      await this.scheduler.scheduleTests(relatedTests.tests, this);
    }
  }
}
```

### Result Aggregation and Processing

Process and aggregate test results for comprehensive reporting.

```typescript
import { createTestScheduler } from "jest";

interface TestExecutionMetrics {
  totalDuration: number;
  averageTestDuration: number;
  slowestTests: Array<{ path: string; duration: number }>;
  fastestTests: Array<{ path: string; duration: number }>;
  failureRate: number;
  coveragePercentage?: number;
}

async function executeWithMetrics(tests: Array<Test>): Promise<{
  results: AggregatedResult;
  metrics: TestExecutionMetrics;
}> {
  const scheduler = await createTestScheduler(globalConfig, schedulerContext);
  
  const startTime = Date.now();
  const results = await scheduler.scheduleTests(
    tests,
    new TestWatcher({ isWatchMode: false })
  );
  const totalDuration = Date.now() - startTime;
  
  // Calculate metrics
  const testDurations = results.testResults
    .flatMap(suite => suite.testResults)
    .map(test => ({ path: test.title, duration: test.duration || 0 }))
    .filter(test => test.duration > 0);
  
  const averageTestDuration = testDurations.length > 0
    ? testDurations.reduce((sum, test) => sum + test.duration, 0) / testDurations.length
    : 0;
  
  const sortedByDuration = testDurations.sort((a, b) => b.duration - a.duration);
  
  const metrics: TestExecutionMetrics = {
    totalDuration,
    averageTestDuration,
    slowestTests: sortedByDuration.slice(0, 5),
    fastestTests: sortedByDuration.slice(-5).reverse(),
    failureRate: results.numTotalTests > 0 
      ? results.numFailedTests / results.numTotalTests 
      : 0,
    coveragePercentage: results.coverageMap 
      ? calculateCoveragePercentage(results.coverageMap)
      : undefined
  };
  
  return { results, metrics };
}

function calculateCoveragePercentage(coverageMap: any): number {
  // Implementation would depend on coverage map structure
  // This is a simplified example
  const summary = coverageMap.getCoverageSummary?.();
  return summary?.lines?.pct || 0;
}
```

### Error Handling and Recovery

Implement robust error handling for test scheduling failures.

```typescript
import { createTestScheduler } from "jest";

async function robustTestScheduling(tests: Array<Test>) {
  let scheduler: TestScheduler;
  
  try {
    scheduler = await createTestScheduler(globalConfig, schedulerContext);
  } catch (error) {
    console.error("Failed to create test scheduler:", error);
    throw new Error("Test scheduler initialization failed");
  }
  
  // Add error reporter
  scheduler.addReporter(new ErrorTrackingReporter());
  
  try {
    const results = await scheduler.scheduleTests(
      tests,
      new TestWatcher({ isWatchMode: false })
    );
    
    // Check for critical failures
    if (results.numRuntimeErrorTestSuites > 0) {
      console.warn(`${results.numRuntimeErrorTestSuites} test suites had runtime errors`);
    }
    
    // Handle open handles
    if (results.openHandles && results.openHandles.length > 0) {
      console.warn(`${results.openHandles.length} open handles detected`);
      
      // Optionally force exit
      if (globalConfig.forceExit) {
        process.exit(results.success ? 0 : 1);
      }
    }
    
    return results;
    
  } catch (error) {
    console.error("Test execution failed:", error);
    
    // Attempt recovery or cleanup
    if (error.message.includes("worker")) {
      console.log("Retrying with single worker...");
      const fallbackConfig = { ...globalConfig, maxWorkers: 1, runInBand: true };
      const fallbackScheduler = await createTestScheduler(fallbackConfig, schedulerContext);
      return fallbackScheduler.scheduleTests(tests, new TestWatcher({ isWatchMode: false }));
    }
    
    throw error;
  }
}

class ErrorTrackingReporter {
  private errors: Array<{ test: string; error: any }> = [];
  
  onTestResult(test: Test, testResult: TestResult) {
    testResult.testResults.forEach(result => {
      if (result.status === "failed") {
        this.errors.push({
          test: `${test.path} > ${result.title}`,
          error: result.failureMessages
        });
      }
    });
  }
  
  onRunComplete() {
    if (this.errors.length > 0) {
      console.log("\n=== Test Failures Summary ===");
      this.errors.forEach(({ test, error }) => {
        console.log(`\n❌ ${test}`);
        console.log(error.join("\n"));
      });
    }
  }
}
```

### Performance Optimization

Optimize test scheduling for different scenarios and constraints.

```typescript
import { createTestScheduler } from "jest";

interface SchedulingStrategy {
  name: string;
  configure: (config: Config.GlobalConfig) => Config.GlobalConfig;
}

const schedulingStrategies: Record<string, SchedulingStrategy> = {
  fast: {
    name: "Fast Execution",
    configure: (config) => ({
      ...config,
      maxWorkers: "100%",
      cache: true,
      bail: 1 // Stop on first failure
    })
  },
  
  thorough: {
    name: "Thorough Testing",
    configure: (config) => ({
      ...config,
      maxWorkers: "50%",
      collectCoverage: true,
      bail: false
    })
  },
  
  debug: {
    name: "Debug Mode",
    configure: (config) => ({
      ...config,
      maxWorkers: 1,
      runInBand: true,
      verbose: true,
      detectOpenHandles: true
    })
  },
  
  ci: {
    name: "CI Optimized",
    configure: (config) => ({
      ...config,
      ci: true,
      maxWorkers: "50%",
      cache: false,
      collectCoverage: true,
      coverageReporters: ["text", "lcov"]
    })
  }
};

async function scheduleWithStrategy(
  tests: Array<Test>,
  strategyName: keyof typeof schedulingStrategies
) {
  const strategy = schedulingStrategies[strategyName];
  const optimizedConfig = strategy.configure(globalConfig);
  
  console.log(`Using ${strategy.name} strategy`);
  
  const scheduler = await createTestScheduler(optimizedConfig, schedulerContext);
  return scheduler.scheduleTests(
    tests,
    new TestWatcher({ isWatchMode: false })
  );
}

// Auto-select strategy based on environment
async function smartScheduling(tests: Array<Test>) {
  const isCI = process.env.CI === "true";
  const isDebug = process.env.DEBUG === "true";
  const testCount = tests.length;
  
  let strategy: keyof typeof schedulingStrategies;
  
  if (isDebug) {
    strategy = "debug";
  } else if (isCI) {
    strategy = "ci";
  } else if (testCount < 10) {
    strategy = "fast";
  } else {
    strategy = "thorough";
  }
  
  return scheduleWithStrategy(tests, strategy);
}
```

Jest's test scheduling system provides complete control over test execution coordination, enabling optimized performance, comprehensive reporting, and reliable test execution across different environments and use cases.