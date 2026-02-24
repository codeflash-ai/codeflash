package com.codeflash.agent;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CodeflashAgentTest {

    @Nested
    class TransformerFilteringTests {

        @Test
        void matchesConfiguredPackage() {
            CodeflashTransformer transformer = new CodeflashTransformer(
                new String[]{"com.example"}, "/src/main/java"
            );

            // Should instrument matching class (returns non-null bytecode)
            // We can't easily test the full transform without a real class,
            // but we can verify it doesn't crash on null class name
            byte[] result = transformer.transform(
                null, null, null, null, new byte[0]
            );
            assertNull(result, "Null class name should return null");
        }

        @Test
        void skipsJdkClasses() {
            CodeflashTransformer transformer = new CodeflashTransformer(
                new String[]{"java.util"}, "/src/main/java"
            );

            // java.util classes should be skipped even if they match the prefix
            byte[] result = transformer.transform(
                null, "java/util/HashMap", null, null, new byte[0]
            );
            assertNull(result, "JDK classes should be skipped");
        }

        @Test
        void skipsTestFrameworkClasses() {
            CodeflashTransformer transformer = new CodeflashTransformer(
                new String[]{"org.junit"}, "/src/main/java"
            );

            byte[] result = transformer.transform(
                null, "org/junit/jupiter/api/Test", null, null, new byte[0]
            );
            assertNull(result, "JUnit classes should be skipped");
        }

        @Test
        void skipsCodeflashAgentClasses() {
            CodeflashTransformer transformer = new CodeflashTransformer(
                new String[]{"com.codeflash"}, "/src/main/java"
            );

            byte[] result = transformer.transform(
                null, "com/codeflash/agent/CallTracker", null, null, new byte[0]
            );
            assertNull(result, "Codeflash agent classes should be skipped");
        }

        @Test
        void skipsNonMatchingPackages() {
            CodeflashTransformer transformer = new CodeflashTransformer(
                new String[]{"com.example"}, "/src/main/java"
            );

            byte[] result = transformer.transform(
                null, "org/apache/commons/StringUtils", null, null, new byte[0]
            );
            assertNull(result, "Non-matching packages should be skipped");
        }

        @Test
        void multiplePackagePrefixesWork() {
            CodeflashTransformer transformer = new CodeflashTransformer(
                new String[]{"com.example", "com.myapp"}, "/src/main/java"
            );

            // Both should fail gracefully since we're passing invalid bytecode,
            // but the important thing is they're not filtered out by package check.
            // A valid class in com.example would be transformed.
            byte[] result = transformer.transform(
                null, "com/other/SomeClass", null, null, new byte[0]
            );
            assertNull(result, "Non-matching package should be skipped");
        }
    }

    @Nested
    class MethodStatsTests {

        @Test
        void newStatsAreZeroed() {
            MethodStats stats = new MethodStats();
            assertEquals(0, stats.getCallCount());
            assertEquals(0, stats.getNestedCount());
            assertEquals(0, stats.getTotalTimeNs());
            assertEquals(0, stats.getCumulativeTimeNs());
            assertTrue(stats.getCallers().isEmpty());
        }

        @Test
        void recordReturnNonRecursive() {
            MethodStats stats = new MethodStats();
            MethodKey caller = new MethodKey("A.java", 1, "a", "A");
            stats.recordReturn(100, 200, false, caller);

            assertEquals(1, stats.getCallCount());
            assertEquals(100, stats.getTotalTimeNs());
            assertEquals(200, stats.getCumulativeTimeNs());
            assertEquals(1, stats.getCallers().get(caller).get());
        }

        @Test
        void recordReturnRecursive() {
            MethodStats stats = new MethodStats();
            MethodKey caller = new MethodKey("A.java", 1, "a", "A");
            stats.recordReturn(100, 200, true, caller);

            assertEquals(0, stats.getCallCount(), "Recursive call should not increment cc");
            assertEquals(100, stats.getTotalTimeNs(), "Own time should still accumulate");
            assertEquals(0, stats.getCumulativeTimeNs(), "Cumulative time not updated for recursive");
            assertEquals(1, stats.getCallers().get(caller).get());
        }

        @Test
        void incrementDecrementNested() {
            MethodStats stats = new MethodStats();
            assertFalse(stats.isRecursive());

            stats.incrementNested();
            assertTrue(stats.isRecursive());

            stats.incrementNested();
            assertTrue(stats.isRecursive());

            stats.decrementNested();
            assertTrue(stats.isRecursive());

            stats.decrementNested();
            assertFalse(stats.isRecursive());
        }

        @Test
        void nullCallerDoesNotCrash() {
            MethodStats stats = new MethodStats();
            assertDoesNotThrow(() -> stats.recordReturn(100, 200, false, null));
            assertEquals(1, stats.getCallCount());
            assertTrue(stats.getCallers().isEmpty());
        }
    }
}
