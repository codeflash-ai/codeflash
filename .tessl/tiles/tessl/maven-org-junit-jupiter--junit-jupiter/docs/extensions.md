# Extensions and Lifecycle

JUnit Jupiter's powerful extension model allows customizing test behavior, dependency injection, and integration with external frameworks. Extensions provide hooks into the test lifecycle and enable sophisticated test customizations.

## Imports

```java
import org.junit.jupiter.api.extension.*;
import static org.junit.jupiter.api.Assertions.*;
```

## Capabilities

### Extension Registration

Register extensions declaratively or programmatically.

```java { .api }
/**
 * Register extensions declaratively on test classes and methods
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@Repeatable(ExtendWith.List.class)
@interface ExtendWith {
    /**
     * Extension classes to register
     */
    Class<? extends Extension>[] value();
    
    @Target({ElementType.TYPE, ElementType.METHOD})
    @Retention(RetentionPolicy.RUNTIME)
    @interface List {
        ExtendWith[] value();
    }
}

/**
 * Register extension instance via static field
 */
@Target(ElementType.FIELD)
@Retention(RetentionPolicy.RUNTIME)
@interface RegisterExtension {
}

/**
 * Base extension interface - marker interface for all extensions
 */
interface Extension {
}
```

**Usage Examples:**

```java
// Declarative registration
@ExtendWith(DatabaseExtension.class)
@ExtendWith(MockitoExtension.class)
class MyTest {
    
    @Test
    void testWithExtensions() {
        // Extensions are active
    }
}

// Programmatic registration
class MyTest {
    
    @RegisterExtension
    static DatabaseExtension database = new DatabaseExtension("testdb");
    
    @RegisterExtension
    MockServerExtension mockServer = new MockServerExtension(8080);
    
    @Test
    void testWithProgrammaticExtensions() {
        // Extensions configured and active
    }
}
```

### Lifecycle Callback Extensions

Hook into test lifecycle events.

```java { .api }
/**
 * Callback before all tests in container
 */
interface BeforeAllCallback extends Extension {
    void beforeAll(ExtensionContext context) throws Exception;
}

/**
 * Callback before each test method
 */
interface BeforeEachCallback extends Extension {
    void beforeEach(ExtensionContext context) throws Exception;
}

/**
 * Callback before test method execution (after @BeforeEach)
 */
interface BeforeTestExecutionCallback extends Extension {
    void beforeTestExecution(ExtensionContext context) throws Exception;
}

/**
 * Callback after test method execution (before @AfterEach)
 */
interface AfterTestExecutionCallback extends Extension {
    void afterTestExecution(ExtensionContext context) throws Exception;
}

/**
 * Callback after each test method
 */
interface AfterEachCallback extends Extension {
    void afterEach(ExtensionContext context) throws Exception;
}

/**
 * Callback after all tests in container
 */
interface AfterAllCallback extends Extension {
    void afterAll(ExtensionContext context) throws Exception;
}
```

**Usage Example:**

```java
class TimingExtension implements BeforeTestExecutionCallback, AfterTestExecutionCallback {
    
    private static final String START_TIME = "start time";
    
    @Override
    public void beforeTestExecution(ExtensionContext context) throws Exception {
        getStore(context).put(START_TIME, System.currentTimeMillis());
    }
    
    @Override
    public void afterTestExecution(ExtensionContext context) throws Exception {
        Method testMethod = context.getRequiredTestMethod();
        long startTime = getStore(context).remove(START_TIME, long.class);
        long duration = System.currentTimeMillis() - startTime;
        
        System.out.printf("Method [%s] took %s ms.%n", testMethod.getName(), duration);
    }
    
    private ExtensionContext.Store getStore(ExtensionContext context) {
        return context.getStore(ExtensionContext.Namespace.create(getClass(), context.getRequiredTestMethod()));
    }
}

@ExtendWith(TimingExtension.class)
class TimedTest {
    
    @Test
    void testThatTakesTime() throws InterruptedException {
        Thread.sleep(100);
        assertTrue(true);
    }
}
```

### Parameter Resolution Extensions

Inject custom parameters into test methods and constructors.

```java { .api }
/**
 * Resolve parameters for test methods and constructors
 */
interface ParameterResolver extends Extension {
    /**
     * Check if this resolver supports the parameter
     */
    boolean supportsParameter(ParameterContext parameterContext, ExtensionContext extensionContext) 
        throws ParameterResolutionException;
    
    /**
     * Resolve the parameter value
     */
    Object resolveParameter(ParameterContext parameterContext, ExtensionContext extensionContext) 
        throws ParameterResolutionException;
}

/**
 * Parameter context information
 */
interface ParameterContext {
    Parameter getParameter();
    int getIndex();
    Optional<Object> getTarget();
    boolean isAnnotated(Class<? extends Annotation> annotationType);
    <A extends Annotation> Optional<A> findAnnotation(Class<A> annotationType);
    <A extends Annotation> List<A> findRepeatableAnnotations(Class<A> annotationType);
}

/**
 * Type-based parameter resolver base class
 */
abstract class TypeBasedParameterResolver<T> implements ParameterResolver {
    private final Class<T> parameterType;
    
    protected TypeBasedParameterResolver(Class<T> parameterType) {
        this.parameterType = parameterType;
    }
    
    @Override
    public final boolean supportsParameter(ParameterContext parameterContext, ExtensionContext extensionContext) {
        return parameterType.equals(parameterContext.getParameter().getType());
    }
    
    @Override
    public final Object resolveParameter(ParameterContext parameterContext, ExtensionContext extensionContext) {
        return resolveParameter(extensionContext);
    }
    
    /**
     * Resolve parameter of the supported type
     */
    public abstract T resolveParameter(ExtensionContext extensionContext);
}
```

**Usage Examples:**

```java
class DatabaseConnectionResolver implements ParameterResolver {
    
    @Override
    public boolean supportsParameter(ParameterContext parameterContext, ExtensionContext extensionContext) {
        return parameterContext.getParameter().getType() == Connection.class;
    }
    
    @Override
    public Object resolveParameter(ParameterContext parameterContext, ExtensionContext extensionContext) {
        return createDatabaseConnection();
    }
    
    private Connection createDatabaseConnection() {
        // Create and return database connection
        return DriverManager.getConnection("jdbc:h2:mem:test");
    }
}

class TempDirectoryResolver extends TypeBasedParameterResolver<Path> {
    
    public TempDirectoryResolver() {
        super(Path.class);
    }
    
    @Override
    public Path resolveParameter(ExtensionContext extensionContext) {
        return Files.createTempDirectory("junit-test");
    }
}

@ExtendWith({DatabaseConnectionResolver.class, TempDirectoryResolver.class})
class DatabaseTest {
    
    @Test
    void testWithInjectedParameters(Connection connection, Path tempDir) {
        assertNotNull(connection);
        assertNotNull(tempDir);
        assertTrue(Files.exists(tempDir));
    }
}
```

### Conditional Execution Extensions

Control when tests should be executed.

```java { .api }
/**
 * Determine whether test should be executed
 */
interface ExecutionCondition extends Extension {
    /**
     * Evaluate execution condition
     */
    ConditionEvaluationResult evaluateExecutionCondition(ExtensionContext context);
}

/**
 * Result of condition evaluation
 */
class ConditionEvaluationResult {
    /**
     * Create enabled result
     */
    static ConditionEvaluationResult enabled(String reason);
    
    /**
     * Create disabled result
     */
    static ConditionEvaluationResult disabled(String reason);
    
    /**
     * Check if execution is disabled
     */
    boolean isDisabled();
    
    /**
     * Get reason for the result
     */
    Optional<String> getReason();
}
```

**Usage Example:**

```java
class SystemPropertyCondition implements ExecutionCondition {
    
    @Override
    public ConditionEvaluationResult evaluateExecutionCondition(ExtensionContext context) {
        Optional<SystemProperty> annotation = context.getElement()
            .map(element -> element.getAnnotation(SystemProperty.class));
            
        if (annotation.isPresent()) {
            SystemProperty systemProperty = annotation.get();
            String actualValue = System.getProperty(systemProperty.name());
            
            if (systemProperty.value().equals(actualValue)) {
                return ConditionEvaluationResult.enabled("System property matches");
            } else {
                return ConditionEvaluationResult.disabled(
                    String.format("System property [%s] does not match expected value [%s]", 
                        systemProperty.name(), systemProperty.value()));
            }
        }
        
        return ConditionEvaluationResult.enabled("No system property condition");
    }
}

@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith(SystemPropertyCondition.class)
@interface SystemProperty {
    String name();
    String value();
}

class ConditionalTest {
    
    @Test
    @SystemProperty(name = "env", value = "test")
    void testOnlyInTestEnvironment() {
        // Only runs when system property env=test
        assertTrue(true);
    }
}
```

### Test Instance Extensions

Control test instance creation and lifecycle.

```java { .api }
/**
 * Create test instances
 */
interface TestInstanceFactory extends Extension {
    /**
     * Create test instance
     */
    Object createTestInstance(TestInstanceFactoryContext factoryContext, ExtensionContext extensionContext) 
        throws TestInstantiationException;
}

/**
 * Test instance factory context
 */
interface TestInstanceFactoryContext {
    Class<?> getTestClass();
    Optional<Object> getOuterInstance();
}

/**
 * Callback before test instance construction
 */
interface TestInstancePreConstructCallback extends Extension {
    void preConstructTestInstance(TestInstancePreConstructContext context, ExtensionContext extensionContext);
}

/**
 * Process test instance after construction
 */
interface TestInstancePostProcessor extends Extension {
    void postProcessTestInstance(Object testInstance, ExtensionContext context) throws Exception;
}

/**
 * Callback before test instance destruction
 */
interface TestInstancePreDestroyCallback extends Extension {
    void preDestroyTestInstance(ExtensionContext context) throws Exception;
}
```

**Usage Example:**

```java
class DependencyInjectionExtension implements TestInstancePostProcessor {
    
    @Override
    public void postProcessTestInstance(Object testInstance, ExtensionContext context) throws Exception {
        Class<?> testClass = testInstance.getClass();
        
        for (Field field : testClass.getDeclaredFields()) {
            if (field.isAnnotationPresent(Inject.class)) {
                field.setAccessible(true);
                field.set(testInstance, createDependency(field.getType()));
            }
        }
    }
    
    private Object createDependency(Class<?> type) {
        // Create dependency instance
        if (type == UserService.class) {
            return new UserService();
        }
        return null;
    }
}

@Target(ElementType.FIELD)
@Retention(RetentionPolicy.RUNTIME)
@interface Inject {
}

@ExtendWith(DependencyInjectionExtension.class)
class ServiceTest {
    
    @Inject
    private UserService userService;
    
    @Test
    void testWithInjectedService() {
        assertNotNull(userService);
        // Use injected service
    }
}
```

### Exception Handling Extensions

Handle test execution exceptions.

```java { .api }
/**
 * Handle exceptions thrown during test execution
 */
interface TestExecutionExceptionHandler extends Extension {
    /**
     * Handle test execution exception
     */
    void handleTestExecutionException(ExtensionContext context, Throwable throwable) throws Throwable;
}

/**
 * Callback before test interruption
 */
interface PreInterruptCallback extends Extension {
    /**
     * Called before test thread is interrupted
     */
    void preInterrupt(PreInterruptContext context, ExtensionContext extensionContext) throws Exception;
}

/**
 * Pre-interrupt context information
 */
interface PreInterruptContext {
    Thread getThreadToInterrupt();
    Optional<String> getReason();
}

/**
 * Watch test execution and results
 */
interface TestWatcher extends Extension {
    /**
     * Called when test is disabled
     */
    default void testDisabled(ExtensionContext context, Optional<String> reason) {}
    
    /**
     * Called when test succeeds
     */
    default void testSuccessful(ExtensionContext context) {}
    
    /**
     * Called when test is aborted
     */
    default void testAborted(ExtensionContext context, Throwable cause) {}
    
    /**
     * Called when test fails
     */
    default void testFailed(ExtensionContext context, Throwable cause) {}
}

/**
 * Intercept method invocations
 */
interface InvocationInterceptor extends Extension {
    /**
     * Intercept test method invocation
     */
    default void interceptTestMethod(Invocation<Void> invocation, 
                                   ReflectiveInvocationContext<Method> invocationContext, 
                                   ExtensionContext extensionContext) throws Throwable {
        invocation.proceed();
    }
    
    /**
     * Intercept test class constructor invocation
     */
    default <T> T interceptTestClassConstructor(Invocation<T> invocation,
                                              ReflectiveInvocationContext<Constructor<T>> invocationContext,
                                              ExtensionContext extensionContext) throws Throwable {
        return invocation.proceed();
    }
    
    /**
     * Intercept BeforeAll method invocation
     */
    default void interceptBeforeAllMethod(Invocation<Void> invocation,
                                         ReflectiveInvocationContext<Method> invocationContext,
                                         ExtensionContext extensionContext) throws Throwable {
        invocation.proceed();
    }
    
    /**
     * Intercept BeforeEach method invocation
     */
    default void interceptBeforeEachMethod(Invocation<Void> invocation,
                                          ReflectiveInvocationContext<Method> invocationContext,
                                          ExtensionContext extensionContext) throws Throwable {
        invocation.proceed();
    }
    
    /**
     * Intercept AfterEach method invocation
     */
    default void interceptAfterEachMethod(Invocation<Void> invocation,
                                         ReflectiveInvocationContext<Method> invocationContext,
                                         ExtensionContext extensionContext) throws Throwable {
        invocation.proceed();
    }
    
    /**
     * Intercept AfterAll method invocation
     */
    default void interceptAfterAllMethod(Invocation<Void> invocation,
                                        ReflectiveInvocationContext<Method> invocationContext,
                                        ExtensionContext extensionContext) throws Throwable {
        invocation.proceed();
    }
}
```

**Usage Example:**

```java
class RetryExtension implements TestExecutionExceptionHandler {
    
    @Override
    public void handleTestExecutionException(ExtensionContext context, Throwable throwable) throws Throwable {
        Optional<Retry> retryAnnotation = context.getElement()
            .map(element -> element.getAnnotation(Retry.class));
            
        if (retryAnnotation.isPresent()) {
            int maxRetries = retryAnnotation.get().value();
            ExtensionContext.Store store = getStore(context);
            int retryCount = store.getOrComputeIfAbsent("retryCount", key -> 0, Integer.class);
            
            if (retryCount < maxRetries) {
                store.put("retryCount", retryCount + 1);
                // Retry the test by not re-throwing the exception
                return;
            }
        }
        
        // Re-throw if no retry or max retries reached
        throw throwable;
    }
    
    private ExtensionContext.Store getStore(ExtensionContext context) {
        return context.getStore(ExtensionContext.Namespace.create(getClass(), context.getRequiredTestMethod()));
    }
}

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith(RetryExtension.class)
@interface Retry {
    int value() default 3;
}

class FlakeyTest {
    
    @Test
    @Retry(5)
    void testThatMightFail() {
        if (Math.random() < 0.7) {
            throw new RuntimeException("Random failure");
        }
        assertTrue(true);
    }
}
```

### Extension Context and Store

Access test context and store data across extension callbacks.

```java { .api }
/**
 * Extension execution context
 */
interface ExtensionContext {
    /**
     * Get parent context
     */
    Optional<ExtensionContext> getParent();
    
    /**
     * Get root context
     */
    ExtensionContext getRoot();
    
    /**
     * Get unique ID
     */
    String getUniqueId();
    
    /**
     * Get display name
     */
    String getDisplayName();
    
    /**
     * Get all tags
     */
    Set<String> getTags();
    
    /**
     * Get annotated element (class or method)
     */
    Optional<AnnotatedElement> getElement();
    
    /**
     * Get test class
     */
    Optional<Class<?>> getTestClass();
    
    /**
     * Get required test class (throws if not present)
     */
    Class<?> getRequiredTestClass();
    
    /**
     * Get test instance lifecycle
     */
    Optional<TestInstance.Lifecycle> getTestInstanceLifecycle();
    
    /**
     * Get test instance (may be null for static methods)
     */
    Optional<Object> getTestInstance();
    
    /**
     * Get all test instances for nested tests
     */
    Optional<TestInstances> getTestInstances();
    
    /**
     * Get test method
     */
    Optional<Method> getTestMethod();
    
    /**
     * Get required test method (throws if not present)
     */
    Method getRequiredTestMethod();
    
    /**
     * Get execution exception if test failed
     */
    Optional<Throwable> getExecutionException();
    
    /**
     * Get configuration parameter
     */
    Optional<String> getConfigurationParameter(String key);
    
    /**
     * Get store for sharing data
     */
    Store getStore(Namespace namespace);
    
    /**
     * Publish entry to test report
     */
    void publishReportEntry(Map<String, String> map);
    void publishReportEntry(String key, String value);
    
    /**
     * Store namespace for organizing data
     */
    class Namespace {
        static Namespace create(Object... parts);
        static final Namespace GLOBAL;
    }
    
    /**
     * Key-value store for extension data
     */
    interface Store {
        Object get(Object key);
        <V> V get(Object key, Class<V> requiredType);
        <K, V> Object getOrComputeIfAbsent(K key, Function<K, V> defaultCreator);
        <K, V> V getOrComputeIfAbsent(K key, Function<K, V> defaultCreator, Class<V> requiredType);
        void put(Object key, Object value);
        Object remove(Object key);
        <V> V remove(Object key, Class<V> requiredType);
        <K, V> V getOrDefault(K key, Class<V> requiredType, V defaultValue);
        void clear();
        
        interface CloseableResource {
            void close() throws Throwable;
        }
    }
}
```

**Usage Example:**

```java
class DataSharingExtension implements BeforeAllCallback, AfterAllCallback, BeforeEachCallback {
    
    @Override
    public void beforeAll(ExtensionContext context) throws Exception {
        // Store shared data for all tests
        ExtensionContext.Store store = context.getStore(ExtensionContext.Namespace.GLOBAL);
        store.put("sharedData", new SharedTestData());
    }
    
    @Override
    public void beforeEach(ExtensionContext context) throws Exception {
        // Access test-specific information
        String testName = context.getDisplayName();
        Set<String> tags = context.getTags();
        
        // Store per-test data
        ExtensionContext.Store store = getStore(context);
        store.put("testStartTime", System.currentTimeMillis());
        
        System.out.printf("Starting test: %s with tags: %s%n", testName, tags);
    }
    
    @Override
    public void afterAll(ExtensionContext context) throws Exception {
        // Cleanup shared data
        ExtensionContext.Store store = context.getStore(ExtensionContext.Namespace.GLOBAL);
        store.remove("sharedData");
    }
    
    private ExtensionContext.Store getStore(ExtensionContext context) {
        return context.getStore(ExtensionContext.Namespace.create(getClass(), context.getRequiredTestMethod()));
    }
}
```

## Additional Types

### Invocation and Context Types

Types used with InvocationInterceptor and other advanced extension features.

```java { .api }
/**
 * Represents an invocation that can be proceeded
 */
interface Invocation<T> {
    /**
     * Proceed with the invocation
     */
    T proceed() throws Throwable;
    
    /**
     * Skip the invocation
     */
    void skip();
}

/**
 * Context for reflective invocations
 */
interface ReflectiveInvocationContext<T> {
    /**
     * Get the executable being invoked (Constructor or Method)
     */
    T getExecutable();
    
    /**
     * Get the arguments for the invocation
     */
    List<Object> getArguments();
    
    /**
     * Get the target instance (null for static methods/constructors)
     */
    Optional<Object> getTarget();
}

/**
 * Test instances hierarchy for nested tests
 */
interface TestInstances {
    /**
     * Get the innermost (most nested) test instance
     */
    Object getInnermostInstance();
    
    /**
     * Get all test instances from outermost to innermost
     */
    List<Object> getEnclosingInstances();
    
    /**
     * Get all test instances (enclosing + innermost)
     */
    List<Object> getAllInstances();
    
    /**
     * Find test instance of specific type
     */
    <T> Optional<T> findInstance(Class<T> requiredType);
}
```