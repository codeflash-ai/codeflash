# JaCoCo Agent

JaCoCo Agent provides programmatic access to the JaCoCo runtime agent JAR file for Java code coverage analysis. This module serves as a wrapper and resource provider for the jacocoagent.jar file, offering APIs to extract, access, and deploy the coverage agent in various Java environments.

## Package Information

- **Package Name**: org.jacoco.agent
- **Package Type**: maven
- **Language**: Java
- **Installation**: 
  ```xml
  <dependency>
    <groupId>org.jacoco</groupId>
    <artifactId>org.jacoco.agent</artifactId>
    <version>0.8.13</version>
  </dependency>
  ```

## Core Imports

```java
import org.jacoco.agent.AgentJar;
import java.io.File;
import java.io.InputStream;
import java.io.IOException;
import java.net.URL;
```

## Basic Usage

```java
import org.jacoco.agent.AgentJar;
import java.io.File;
import java.io.IOException;

// Get the agent JAR as a URL
URL agentUrl = AgentJar.getResource();

// Get the agent JAR as an InputStream
InputStream agentStream = AgentJar.getResourceAsStream();

// Extract agent to a temporary location
File tempAgent = AgentJar.extractToTempLocation();

// Extract agent to a specific location
File specificLocation = new File("/path/to/jacocoagent.jar");
AgentJar.extractTo(specificLocation);
```

## Capabilities

### Agent Resource Access

Access the embedded JaCoCo agent JAR file as a resource.

```java { .api }
/**
 * Returns a URL pointing to the JAR file.
 * @return URL of the JAR file
 */
public static URL getResource();

/**
 * Returns the content of the JAR file as a stream.
 * @return content of the JAR file
 */
public static InputStream getResourceAsStream();
```

**Usage Examples:**

```java
// Access via URL
URL agentUrl = AgentJar.getResource();
InputStream stream = agentUrl.openStream();

// Direct stream access with proper resource management
try (InputStream agentStream = AgentJar.getResourceAsStream()) {
    // Use the stream for processing
    // Stream is automatically closed when try block exits
}
```

### Agent Extraction

Extract the embedded agent JAR to file system locations.

```java { .api }
/**
 * Extract the JaCoCo agent JAR and put it into a temporary location. This
 * file should be deleted on exit, but may not if the VM is terminated
 * @return Location of the Agent Jar file in the local file system. The file
 *         should exist and be readable.
 * @throws IOException Unable to unpack agent jar
 */
public static File extractToTempLocation() throws IOException;

/**
 * Extract the JaCoCo agent JAR and put it into the specified location.
 * @param destination Location to write JaCoCo Agent Jar to. Must be writeable
 * @throws IOException Unable to unpack agent jar
 */
public static void extractTo(File destination) throws IOException;
```

**Usage Examples:**

```java
// Extract to temporary location (automatically deleted on JVM exit)
File tempAgentFile = AgentJar.extractToTempLocation();
System.out.println("Agent extracted to: " + tempAgentFile.getAbsolutePath());

// Extract to specific location
File agentFile = new File("./jacocoagent.jar");
try {
    AgentJar.extractTo(agentFile);
    System.out.println("Agent extracted to: " + agentFile.getAbsolutePath());
} catch (IOException e) {
    System.err.println("Failed to extract agent: " + e.getMessage());
}
```

## Types

```java { .api }
public final class AgentJar {
    // Private constructor - cannot be instantiated
    private AgentJar();
    
    // All methods are static
}
```

## Error Handling

The JaCoCo Agent API uses two main types of exceptions:

- **AssertionError**: Thrown when the embedded `/jacocoagent.jar` resource is not found. This typically indicates a build or packaging issue. The error includes a reference to `/org.jacoco.agent/README.TXT` for troubleshooting details.
- **IOException**: Thrown by extraction methods for I/O related failures, such as:
  - Destination file is not writable
  - Destination path does not exist
  - Insufficient disk space
  - File system permissions issues

**Error Handling Example:**

```java
try {
    // Resource access - may throw AssertionError
    URL agentUrl = AgentJar.getResource();
    
    // File extraction - may throw IOException  
    File agentFile = new File("./jacocoagent.jar");
    AgentJar.extractTo(agentFile);
    
} catch (AssertionError e) {
    System.err.println("Agent resource not found. Check build configuration.");
} catch (IOException e) {
    System.err.println("Failed to extract agent: " + e.getMessage());
}
```

## Key Characteristics

- **Utility Class**: AgentJar is a final class with only static methods and private constructor (cannot be instantiated)
- **Resource Provider**: Acts as a wrapper around the embedded `/jacocoagent.jar` resource within the JAR file
- **Thread Safety**: All methods are static and thread-safe
- **Self-Contained**: Includes the complete agent JAR as an embedded resource at runtime
- **Build Integration**: The agent JAR is created and embedded during the Maven build process
- **No External Dependencies**: Pure Java implementation using only standard library classes
- **Safe Resource Handling**: Internal implementation uses safe stream closing to prevent resource leaks

## Integration Patterns

Common usage patterns for integrating JaCoCo Agent in applications:

**Testing Framework Integration:**
```java
// Extract agent for use with JVM arguments
File agent = AgentJar.extractToTempLocation();
String javaagentArg = "-javaagent:" + agent.getAbsolutePath();
// Use javaagentArg when launching test JVMs
```

**Build Tool Integration:**
```java
// Extract to build directory for distribution
File buildDir = new File("target/jacoco");
buildDir.mkdirs();
File agentJar = new File(buildDir, "jacocoagent.jar");
AgentJar.extractTo(agentJar);
```

**Runtime Deployment:**
```java
// Use the built-in extraction method for deployment
File deploymentFile = new File("/path/to/deployment/jacocoagent.jar");
AgentJar.extractTo(deploymentFile);
// The extractTo method handles stream management and proper copying
```