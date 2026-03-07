package com.codeflash;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class AgentDispatcherTest {

    @Test
    void profilerModeWhenConfigPresent() {
        assertTrue(AgentDispatcher.isProfilerMode("config=/tmp/config.json"));
    }

    @Test
    void profilerModeWithMultipleArgs() {
        assertTrue(AgentDispatcher.isProfilerMode("output=results,config=/tmp/config.json"));
    }

    @Test
    void jacocoModeWhenDestfilePresent() {
        assertFalse(AgentDispatcher.isProfilerMode("destfile=/tmp/jacoco.exec"));
    }

    @Test
    void jacocoModeWhenPathContainsConfigSubstring() {
        assertFalse(AgentDispatcher.isProfilerMode("destfile=/home/config=/jacoco.exec"));
    }

    @Test
    void jacocoModeWhenNullArgs() {
        assertFalse(AgentDispatcher.isProfilerMode(null));
    }

    @Test
    void jacocoModeWhenEmptyArgs() {
        assertFalse(AgentDispatcher.isProfilerMode(""));
    }
}
