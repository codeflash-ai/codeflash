package com.codeflash;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the Blackhole class.
 */
@DisplayName("Blackhole Tests")
class BlackholeTest {

    @Test
    @DisplayName("should consume int without throwing")
    void testConsumeInt() {
        assertDoesNotThrow(() -> Blackhole.consume(42));
    }

    @Test
    @DisplayName("should consume long without throwing")
    void testConsumeLong() {
        assertDoesNotThrow(() -> Blackhole.consume(Long.MAX_VALUE));
    }

    @Test
    @DisplayName("should consume double without throwing")
    void testConsumeDouble() {
        assertDoesNotThrow(() -> Blackhole.consume(3.14159));
    }

    @Test
    @DisplayName("should consume float without throwing")
    void testConsumeFloat() {
        assertDoesNotThrow(() -> Blackhole.consume(3.14f));
    }

    @Test
    @DisplayName("should consume boolean without throwing")
    void testConsumeBoolean() {
        assertDoesNotThrow(() -> Blackhole.consume(true));
        assertDoesNotThrow(() -> Blackhole.consume(false));
    }

    @Test
    @DisplayName("should consume byte without throwing")
    void testConsumeByte() {
        assertDoesNotThrow(() -> Blackhole.consume((byte) 127));
    }

    @Test
    @DisplayName("should consume short without throwing")
    void testConsumeShort() {
        assertDoesNotThrow(() -> Blackhole.consume((short) 32000));
    }

    @Test
    @DisplayName("should consume char without throwing")
    void testConsumeChar() {
        assertDoesNotThrow(() -> Blackhole.consume('x'));
    }

    @Test
    @DisplayName("should consume Object without throwing")
    void testConsumeObject() {
        assertDoesNotThrow(() -> Blackhole.consume("hello"));
        assertDoesNotThrow(() -> Blackhole.consume(Arrays.asList(1, 2, 3)));
        assertDoesNotThrow(() -> Blackhole.consume((Object) null));
    }

    @Test
    @DisplayName("should consume int array without throwing")
    void testConsumeIntArray() {
        assertDoesNotThrow(() -> Blackhole.consume(new int[]{1, 2, 3}));
        assertDoesNotThrow(() -> Blackhole.consume((int[]) null));
        assertDoesNotThrow(() -> Blackhole.consume(new int[]{}));
    }

    @Test
    @DisplayName("should consume long array without throwing")
    void testConsumeLongArray() {
        assertDoesNotThrow(() -> Blackhole.consume(new long[]{1L, 2L, 3L}));
        assertDoesNotThrow(() -> Blackhole.consume((long[]) null));
    }

    @Test
    @DisplayName("should consume double array without throwing")
    void testConsumeDoubleArray() {
        assertDoesNotThrow(() -> Blackhole.consume(new double[]{1.0, 2.0, 3.0}));
        assertDoesNotThrow(() -> Blackhole.consume((double[]) null));
    }

    @Test
    @DisplayName("should prevent dead code elimination in loop")
    void testPreventDeadCodeInLoop() {
        // This test verifies that consuming values allows the loop to run
        // without the JIT potentially eliminating it
        int sum = 0;
        for (int i = 0; i < 1000; i++) {
            sum += i;
            Blackhole.consume(sum);
        }
        // The loop should have run - this is more of a smoke test
        assertTrue(sum > 0);
    }
}
