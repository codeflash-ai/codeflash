package com.codeflash.tracer;

import com.google.gson.Gson;
import com.google.gson.annotations.SerializedName;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;

public final class TracerConfig {

    @SerializedName("dbPath")
    private String dbPath = "codeflash_trace.db";

    @SerializedName("packages")
    private List<String> packages = Collections.emptyList();

    @SerializedName("excludePackages")
    private List<String> excludePackages = Collections.emptyList();

    @SerializedName("maxFunctionCount")
    private int maxFunctionCount = 256;

    @SerializedName("timeout")
    private int timeout = 0;

    @SerializedName("projectRoot")
    private String projectRoot = "";

    @SerializedName("inMemoryDb")
    private boolean inMemoryDb = false;

    private static final Gson GSON = new Gson();

    public static TracerConfig parse(String agentArgs) {
        if (agentArgs == null || agentArgs.isEmpty()) {
            return new TracerConfig();
        }

        String configPath = null;
        for (String part : agentArgs.split(",")) {
            String trimmed = part.trim();
            if (trimmed.startsWith("trace=")) {
                configPath = trimmed.substring("trace=".length());
            }
        }

        if (configPath == null) {
            System.err.println("[codeflash-tracer] No trace= in agent args: " + agentArgs);
            return new TracerConfig();
        }

        try {
            String json = new String(Files.readAllBytes(Paths.get(configPath)), StandardCharsets.UTF_8);
            TracerConfig config = GSON.fromJson(json, TracerConfig.class);
            if (config == null) {
                return new TracerConfig();
            }
            if (config.packages == null) config.packages = Collections.emptyList();
            if (config.excludePackages == null) config.excludePackages = Collections.emptyList();
            return config;
        } catch (IOException e) {
            System.err.println("[codeflash-tracer] Failed to read config: " + e.getMessage());
            return new TracerConfig();
        }
    }

    public String getDbPath() {
        return dbPath;
    }

    public List<String> getPackages() {
        return packages;
    }

    public List<String> getExcludePackages() {
        return excludePackages;
    }

    public int getMaxFunctionCount() {
        return maxFunctionCount;
    }

    public int getTimeout() {
        return timeout;
    }

    public String getProjectRoot() {
        return projectRoot;
    }

    public boolean isInMemoryDb() {
        return inMemoryDb;
    }

    public boolean shouldInstrumentClass(String internalClassName) {
        String dotName = internalClassName.replace('/', '.');

        for (String excluded : excludePackages) {
            if (dotName.startsWith(excluded)) {
                return false;
            }
        }

        if (packages.isEmpty()) {
            return true;
        }

        for (String pkg : packages) {
            if (dotName.startsWith(pkg)) {
                return true;
            }
        }

        return false;
    }
}
