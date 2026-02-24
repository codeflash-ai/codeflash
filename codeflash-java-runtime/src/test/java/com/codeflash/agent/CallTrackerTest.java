package com.codeflash.agent;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.*;

class CallTrackerTest {

    private CallTracker tracker;

    @BeforeEach
    void setUp() {
        tracker = CallTracker.getInstance();
        tracker.reset();
    }

    @Nested
    class BasicEnterExitTests {

        @Test
        void singleEnterExitRecordsStats() {
            tracker.enter("com.example.Foo", "bar", "Foo.java", 10, new Object[]{"arg1"});
            tracker.exit();

            ConcurrentHashMap<MethodKey, MethodStats> timings = tracker.getTimings();
            assertEquals(1, timings.size());

            MethodKey key = new MethodKey("Foo.java", 10, "bar", "com.example.Foo");
            MethodStats stats = timings.get(key);
            assertNotNull(stats);
            assertEquals(1, stats.getCallCount());
            assertTrue(stats.getTotalTimeNs() >= 0);
            assertTrue(stats.getCumulativeTimeNs() >= 0);
        }

        @Test
        void multipleCallsAccumulate() {
            for (int i = 0; i < 5; i++) {
                tracker.enter("com.example.Foo", "bar", "Foo.java", 10, null);
                tracker.exit();
            }

            MethodKey key = new MethodKey("Foo.java", 10, "bar", "com.example.Foo");
            MethodStats stats = tracker.getTimings().get(key);
            assertNotNull(stats);
            assertEquals(5, stats.getCallCount());
        }

        @Test
        void exitOnEmptyStackDoesNotThrow() {
            assertDoesNotThrow(() -> tracker.exit());
        }

        @Test
        void nullArgsDoNotCrash() {
            tracker.enter("com.example.Foo", "bar", "Foo.java", 10, null);
            tracker.exit();

            MethodKey key = new MethodKey("Foo.java", 10, "bar", "com.example.Foo");
            MethodStats stats = tracker.getTimings().get(key);
            assertNotNull(stats);
            assertEquals(1, stats.getCallCount());
        }
    }

    @Nested
    class CallerTrackingTests {

        @Test
        void callerIsTracked() {
            // Simulate: A calls B
            tracker.enter("com.example.A", "methodA", "A.java", 1, null);
            tracker.enter("com.example.B", "methodB", "B.java", 1, null);
            tracker.exit(); // exit B
            tracker.exit(); // exit A

            MethodKey bKey = new MethodKey("B.java", 1, "methodB", "com.example.B");
            MethodKey aKey = new MethodKey("A.java", 1, "methodA", "com.example.A");

            MethodStats bStats = tracker.getTimings().get(bKey);
            assertNotNull(bStats);
            ConcurrentHashMap<MethodKey, AtomicLong> callers = bStats.getCallers();
            assertTrue(callers.containsKey(aKey), "B should have A as a caller");
            assertEquals(1, callers.get(aKey).get());
        }

        @Test
        void multipleCallersTracked() {
            // A calls C, then B calls C
            tracker.enter("com.example.A", "methodA", "A.java", 1, null);
            tracker.enter("com.example.C", "methodC", "C.java", 1, null);
            tracker.exit(); // exit C
            tracker.exit(); // exit A

            tracker.enter("com.example.B", "methodB", "B.java", 1, null);
            tracker.enter("com.example.C", "methodC", "C.java", 1, null);
            tracker.exit(); // exit C
            tracker.exit(); // exit B

            MethodKey cKey = new MethodKey("C.java", 1, "methodC", "com.example.C");
            MethodStats cStats = tracker.getTimings().get(cKey);
            assertNotNull(cStats);
            assertEquals(2, cStats.getCallCount());
            assertEquals(2, cStats.getCallers().size());
        }

        @Test
        void callChainACallsBCallsC() {
            tracker.enter("com.example.A", "a", "A.java", 1, null);
            tracker.enter("com.example.B", "b", "B.java", 1, null);
            tracker.enter("com.example.C", "c", "C.java", 1, null);
            tracker.exit(); // exit C
            tracker.exit(); // exit B
            tracker.exit(); // exit A

            MethodKey aKey = new MethodKey("A.java", 1, "a", "com.example.A");
            MethodKey bKey = new MethodKey("B.java", 1, "b", "com.example.B");
            MethodKey cKey = new MethodKey("C.java", 1, "c", "com.example.C");

            // C's caller should be B
            assertTrue(tracker.getTimings().get(cKey).getCallers().containsKey(bKey));
            // B's caller should be A
            assertTrue(tracker.getTimings().get(bKey).getCallers().containsKey(aKey));
        }
    }

    @Nested
    class RecursionTests {

        @Test
        void directRecursionHandled() {
            // Simulate: foo calls foo (direct recursion, depth 3)
            tracker.enter("com.example.R", "foo", "R.java", 1, null);
            tracker.enter("com.example.R", "foo", "R.java", 1, null);
            tracker.enter("com.example.R", "foo", "R.java", 1, null);
            tracker.exit(); // exit innermost foo
            tracker.exit(); // exit middle foo
            tracker.exit(); // exit outermost foo

            MethodKey key = new MethodKey("R.java", 1, "foo", "com.example.R");
            MethodStats stats = tracker.getTimings().get(key);
            assertNotNull(stats);
            // Only the outermost call should count as non-recursive
            assertEquals(1, stats.getCallCount());
            assertTrue(stats.getTotalTimeNs() >= 0);
            assertTrue(stats.getCumulativeTimeNs() >= 0);
        }

        @Test
        void indirectRecursionHandled() {
            // A -> B -> A (indirect recursion)
            tracker.enter("com.example.X", "a", "X.java", 1, null);
            tracker.enter("com.example.X", "b", "X.java", 10, null);
            tracker.enter("com.example.X", "a", "X.java", 1, null); // recursive
            tracker.exit(); // exit inner a
            tracker.exit(); // exit b
            tracker.exit(); // exit outer a

            MethodKey aKey = new MethodKey("X.java", 1, "a", "com.example.X");
            MethodStats aStats = tracker.getTimings().get(aKey);
            assertNotNull(aStats);
            // a was called twice but the second is recursive, so cc=1
            assertEquals(1, aStats.getCallCount());
        }

        @Test
        void nestedCountReturnedToZeroAfterRecursion() {
            MethodKey key = new MethodKey("R.java", 1, "foo", "com.example.R");

            tracker.enter("com.example.R", "foo", "R.java", 1, null);
            tracker.enter("com.example.R", "foo", "R.java", 1, null);
            tracker.exit();
            tracker.exit();

            MethodStats stats = tracker.getTimings().get(key);
            assertNotNull(stats);
            assertFalse(stats.isRecursive(), "After all returns, nested count should be 0");
        }
    }

    @Nested
    class MaxFunctionCountTests {

        @Test
        void respectsMaxFunctionCount() {
            tracker.setMaxFunctionCount(3);

            for (int i = 0; i < 10; i++) {
                tracker.enter("com.example.Foo", "bar", "Foo.java", 10, new Object[]{i});
                tracker.exit();
            }

            MethodKey key = new MethodKey("Foo.java", 10, "bar", "com.example.Foo");
            List<byte[]> captured = tracker.getCapturedArgs().get(key);

            // Should capture at most 3 argument sets
            assertNotNull(captured);
            assertTrue(captured.size() <= 3, "Should respect maxFunctionCount limit, got " + captured.size());
        }

        @Test
        void timingStillRecordedBeyondMaxCount() {
            tracker.setMaxFunctionCount(2);

            for (int i = 0; i < 5; i++) {
                tracker.enter("com.example.Foo", "bar", "Foo.java", 10, new Object[]{i});
                tracker.exit();
            }

            MethodKey key = new MethodKey("Foo.java", 10, "bar", "com.example.Foo");
            MethodStats stats = tracker.getTimings().get(key);
            assertNotNull(stats);
            assertEquals(5, stats.getCallCount(), "All calls should be counted even beyond maxFunctionCount");
        }
    }

    @Nested
    class ArgumentCaptureTests {

        @Test
        void argumentsCaptured() {
            tracker.enter("com.example.Foo", "bar", "Foo.java", 10, new Object[]{"hello", 42});
            tracker.exit();

            MethodKey key = new MethodKey("Foo.java", 10, "bar", "com.example.Foo");
            List<byte[]> captured = tracker.getCapturedArgs().get(key);
            assertNotNull(captured);
            assertEquals(1, captured.size());
            assertTrue(captured.get(0).length > 0, "Serialized args should be non-empty");
        }

        @Test
        void nullArgsNotCaptured() {
            tracker.enter("com.example.Foo", "bar", "Foo.java", 10, null);
            tracker.exit();

            MethodKey key = new MethodKey("Foo.java", 10, "bar", "com.example.Foo");
            assertNull(tracker.getCapturedArgs().get(key), "Null args should not produce captured entries");
        }
    }

    @Nested
    class TotalTimeTests {

        @Test
        void totalTimeIsPositiveAfterExecution() throws InterruptedException {
            tracker.markStart();
            tracker.enter("com.example.Foo", "bar", "Foo.java", 10, null);
            Thread.sleep(10); // Sleep to get measurable time
            tracker.exit();
            tracker.markEnd();

            long totalTime = tracker.getTotalTimeNs();
            assertTrue(totalTime > 0, "Total time should be positive");
            assertTrue(totalTime >= 10_000_000, "Total time should be at least 10ms");
        }

        @Test
        void totalTimeZeroBeforeStart() {
            assertEquals(0, tracker.getTotalTimeNs());
        }
    }

    @Nested
    class ResetTests {

        @Test
        void resetClearsAllData() {
            tracker.enter("com.example.Foo", "bar", "Foo.java", 10, new Object[]{"arg"});
            tracker.exit();

            assertFalse(tracker.getTimings().isEmpty());
            assertFalse(tracker.getCapturedArgs().isEmpty());

            tracker.reset();

            assertTrue(tracker.getTimings().isEmpty());
            assertTrue(tracker.getCapturedArgs().isEmpty());
            assertEquals(0, tracker.getTotalTimeNs());
        }
    }

    @Nested
    class ThreadSafetyTests {

        @Test
        void concurrentAccessDoesNotThrow() throws InterruptedException {
            int threadCount = 4;
            int callsPerThread = 100;
            ExecutorService executor = Executors.newFixedThreadPool(threadCount);
            CountDownLatch latch = new CountDownLatch(threadCount);

            for (int t = 0; t < threadCount; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    try {
                        for (int i = 0; i < callsPerThread; i++) {
                            tracker.enter(
                                "com.example.T" + threadId, "method" + i,
                                "T" + threadId + ".java", i, new Object[]{i}
                            );
                            tracker.exit();
                        }
                    } finally {
                        latch.countDown();
                    }
                });
            }

            assertTrue(latch.await(10, TimeUnit.SECONDS), "All threads should complete");
            executor.shutdown();

            // Verify no data corruption - total call counts should add up
            long totalCalls = 0;
            for (MethodStats stats : tracker.getTimings().values()) {
                totalCalls += stats.getCallCount();
            }
            assertEquals(threadCount * callsPerThread, totalCalls,
                "Total calls across all threads should match");
        }

        @Test
        void nestedCallsAcrossThreadsAreIsolated() throws InterruptedException {
            CountDownLatch latch = new CountDownLatch(2);
            ExecutorService executor = Executors.newFixedThreadPool(2);

            // Thread 1: A -> B -> exit B -> exit A
            executor.submit(() -> {
                try {
                    tracker.enter("com.example.T", "a", "T.java", 1, null);
                    Thread.sleep(5);
                    tracker.enter("com.example.T", "b", "T.java", 10, null);
                    Thread.sleep(5);
                    tracker.exit();
                    tracker.exit();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    latch.countDown();
                }
            });

            // Thread 2: C -> exit C
            executor.submit(() -> {
                try {
                    tracker.enter("com.example.T", "c", "T.java", 20, null);
                    Thread.sleep(5);
                    tracker.exit();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    latch.countDown();
                }
            });

            assertTrue(latch.await(10, TimeUnit.SECONDS));
            executor.shutdown();

            // All methods should have been tracked
            assertEquals(3, tracker.getTimings().size());
        }
    }
}
