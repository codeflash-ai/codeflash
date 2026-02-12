# Codeflash Java Runtime

This module provides the runtime library required for Java code optimization with Codeflash.

## Components

- **Serializer**: Handles serialization of test results for behavioral verification
- **Helper utilities**: Supporting classes for instrumented test execution

## Building

```bash
mvn clean package -DskipTests
```

This generates `target/codeflash-runtime-1.0.0.jar`.

## Installation

After making changes to the runtime, you must install it to your local Maven repository for projects to pick up the updates:

```bash
mvn clean install -DskipTests
```

This installs the JAR to `~/.m2/repository/com/codeflash/codeflash-runtime/1.0.0/`.

## Troubleshooting

### Maven Compilation Errors After Runtime Updates

**Symptom:** After updating the runtime code, Maven compilation fails with method signature mismatches like:

```
method serialize in class com.codeflash.Serializer cannot be applied to given types;
  required: [old signature]
  found: [new signature]
```

**Cause:** Maven is using a cached version of the runtime JAR from `~/.m2/repository/` that doesn't match the updated source code.

**Solution:** Reinstall the runtime JAR to your local Maven repository:

```bash
cd codeflash-java-runtime
mvn clean install -DskipTests
```

### Verifying Installation

Check that the JAR timestamp matches your recent build:

```bash
ls -lh ~/.m2/repository/com/codeflash/codeflash-runtime/1.0.0/codeflash-runtime-1.0.0.jar
```

## Usage in Projects

Projects using the Codeflash runtime should have this dependency in their `pom.xml`:

```xml
<dependency>
    <groupId>com.codeflash</groupId>
    <artifactId>codeflash-runtime</artifactId>
    <version>1.0.0</version>
    <scope>system</scope>
    <systemPath>${project.basedir}/codeflash-runtime-1.0.0.jar</systemPath>
</dependency>
```

**Important:** After updating the runtime, copy the new JAR to your project directory:

```bash
cp codeflash-java-runtime/target/codeflash-runtime-1.0.0.jar your-project/
```
