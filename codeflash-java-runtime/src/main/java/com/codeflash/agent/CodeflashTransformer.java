package com.codeflash.agent;

import org.objectweb.asm.*;
import org.objectweb.asm.commons.AdviceAdapter;

import java.lang.instrument.ClassFileTransformer;
import java.security.ProtectionDomain;
import java.util.Set;

/**
 * ASM-based ClassFileTransformer that instruments methods to call
 * CallTracker.enter() / CallTracker.exit() for profiling.
 *
 * Only classes whose package matches configured prefixes are instrumented.
 */
public final class CodeflashTransformer implements ClassFileTransformer {

    private static final String CALL_TRACKER_INTERNAL = "com/codeflash/agent/CallTracker";

    private final String[] packagePrefixes;
    private final String sourceRoot;

    // Packages to always skip
    private static final Set<String> SKIP_PACKAGES = Set.of(
        "com.codeflash.agent.",
        "com.codeflash.Serializer",
        "com.codeflash.CodeFlash",
        "com.codeflash.Comparator",
        "com.codeflash.ResultWriter",
        "com.codeflash.BenchmarkContext",
        "com.codeflash.BenchmarkResult",
        "com.codeflash.Blackhole",
        "com.codeflash.KryoPlaceholder",
        "org.junit.",
        "org.hamcrest.",
        "org.mockito.",
        "org.assertj.",
        "sun.",
        "jdk.",
        "java.",
        "javax."
    );

    public CodeflashTransformer(String[] packagePrefixes, String sourceRoot) {
        this.packagePrefixes = packagePrefixes;
        this.sourceRoot = sourceRoot;
    }

    @Override
    public byte[] transform(ClassLoader loader, String classInternalName,
                            Class<?> classBeingRedefined, ProtectionDomain protectionDomain,
                            byte[] classfileBuffer) {
        if (classInternalName == null) {
            return null;
        }

        String className = classInternalName.replace('/', '.');

        // Skip codeflash agent classes, JDK, and test framework classes
        for (String skip : SKIP_PACKAGES) {
            if (className.startsWith(skip)) {
                return null;
            }
        }

        // Check if class matches any configured package prefix
        boolean matches = false;
        for (String prefix : packagePrefixes) {
            if (className.startsWith(prefix)) {
                matches = true;
                break;
            }
        }
        if (!matches) {
            return null;
        }

        try {
            ClassReader cr = new ClassReader(classfileBuffer);
            ClassWriter cw = new ClassWriter(cr, ClassWriter.COMPUTE_FRAMES | ClassWriter.COMPUTE_MAXS);
            InstrumentingClassVisitor cv = new InstrumentingClassVisitor(cw, className, sourceRoot);
            cr.accept(cv, ClassReader.EXPAND_FRAMES);
            return cw.toByteArray();
        } catch (Exception e) {
            System.err.println("[codeflash-agent] Failed to instrument " + className + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * ClassVisitor that captures the source file name and delegates to InstrumentingMethodVisitor.
     */
    private static final class InstrumentingClassVisitor extends ClassVisitor {

        private final String className;
        private final String sourceRoot;
        private String sourceFileName;

        InstrumentingClassVisitor(ClassWriter cw, String className, String sourceRoot) {
            super(Opcodes.ASM9, cw);
            this.className = className;
            this.sourceRoot = sourceRoot;
        }

        @Override
        public void visitSource(String source, String debug) {
            this.sourceFileName = source;
            super.visitSource(source, debug);
        }

        @Override
        public MethodVisitor visitMethod(int access, String name, String descriptor,
                                         String signature, String[] exceptions) {
            MethodVisitor mv = super.visitMethod(access, name, descriptor, signature, exceptions);

            // Skip methods that should not be instrumented
            if (shouldSkipMethod(access, name)) {
                return mv;
            }

            String resolvedFileName = resolveSourcePath();

            return new InstrumentingMethodVisitor(
                Opcodes.ASM9, mv, access, name, descriptor,
                className, resolvedFileName
            );
        }

        private boolean shouldSkipMethod(int access, String name) {
            // Skip static initializers
            if ("<clinit>".equals(name)) return true;
            // Skip synthetic methods (compiler-generated)
            if ((access & Opcodes.ACC_SYNTHETIC) != 0) return true;
            // Skip bridge methods
            if ((access & Opcodes.ACC_BRIDGE) != 0) return true;
            // Skip native methods
            if ((access & Opcodes.ACC_NATIVE) != 0) return true;
            // Skip abstract methods
            if ((access & Opcodes.ACC_ABSTRACT) != 0) return true;
            // Skip lambda methods
            if (name.startsWith("lambda$")) return true;
            return false;
        }

        private String resolveSourcePath() {
            if (sourceFileName == null) {
                return className.replace('.', '/') + ".java";
            }
            // Convert class name to path: com.example.Foo -> com/example/
            String packagePath = className.contains(".")
                ? className.substring(0, className.lastIndexOf('.')).replace('.', '/')
                : "";
            String relativePath = packagePath.isEmpty()
                ? sourceFileName
                : packagePath + "/" + sourceFileName;

            if (sourceRoot != null && !sourceRoot.isEmpty()) {
                String root = sourceRoot.endsWith("/") ? sourceRoot : sourceRoot + "/";
                return root + relativePath;
            }
            return relativePath;
        }
    }

    /**
     * MethodVisitor that injects CallTracker.enter() at method entry and
     * CallTracker.exit() in a try-finally block wrapping the method body.
     */
    private static final class InstrumentingMethodVisitor extends AdviceAdapter {

        private final String className;
        private final String fileName;
        private int firstLineNumber = -1;

        InstrumentingMethodVisitor(int api, MethodVisitor mv, int access,
                                   String name, String descriptor,
                                   String className, String fileName) {
            super(api, mv, access, name, descriptor);
            this.className = className;
            this.fileName = fileName;
        }

        @Override
        public void visitLineNumber(int line, Label start) {
            if (firstLineNumber < 0) {
                firstLineNumber = line;
            }
            super.visitLineNumber(line, start);
        }

        @Override
        protected void onMethodEnter() {
            // CallTracker.getInstance()
            mv.visitMethodInsn(
                Opcodes.INVOKESTATIC,
                CALL_TRACKER_INTERNAL,
                "getInstance",
                "()L" + CALL_TRACKER_INTERNAL + ";",
                false
            );

            // Push className
            mv.visitLdcInsn(className);
            // Push methodName
            mv.visitLdcInsn(getName());
            // Push fileName
            mv.visitLdcInsn(fileName);
            // Push lineNumber
            mv.visitLdcInsn(Math.max(firstLineNumber, 0));

            // Push args array: box method parameters into Object[]
            loadArgsAsObjectArray();

            // CallTracker.enter(className, methodName, fileName, lineNumber, args)
            mv.visitMethodInsn(
                Opcodes.INVOKEVIRTUAL,
                CALL_TRACKER_INTERNAL,
                "enter",
                "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;I[Ljava/lang/Object;)V",
                false
            );
        }

        @Override
        protected void onMethodExit(int opcode) {
            // Insert exit on both normal return and exception to keep the call stack balanced
            insertExit();
        }

        /**
         * Insert a call to CallTracker.getInstance().exit().
         */
        private void insertExit() {
            mv.visitMethodInsn(
                Opcodes.INVOKESTATIC,
                CALL_TRACKER_INTERNAL,
                "getInstance",
                "()L" + CALL_TRACKER_INTERNAL + ";",
                false
            );
            mv.visitMethodInsn(
                Opcodes.INVOKEVIRTUAL,
                CALL_TRACKER_INTERNAL,
                "exit",
                "()V",
                false
            );
        }

        /**
         * Load method parameters into an Object[] on the stack.
         * Static methods start at index 0, instance methods at index 1 (skip 'this').
         */
        private void loadArgsAsObjectArray() {
            Type[] argTypes = Type.getArgumentTypes(methodDesc);

            mv.visitLdcInsn(argTypes.length);
            mv.visitTypeInsn(Opcodes.ANEWARRAY, "java/lang/Object");

            int localVarIndex = (methodAccess & Opcodes.ACC_STATIC) != 0 ? 0 : 1;

            for (int i = 0; i < argTypes.length; i++) {
                mv.visitInsn(Opcodes.DUP);
                mv.visitLdcInsn(i);

                Type argType = argTypes[i];
                mv.visitVarInsn(argType.getOpcode(Opcodes.ILOAD), localVarIndex);
                boxIfPrimitive(argType);

                mv.visitInsn(Opcodes.AASTORE);
                localVarIndex += argType.getSize();
            }
        }

        /**
         * Box a primitive value on the stack into its wrapper type.
         */
        private void boxIfPrimitive(Type type) {
            switch (type.getSort()) {
                case Type.BOOLEAN:
                    mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Boolean", "valueOf", "(Z)Ljava/lang/Boolean;", false);
                    break;
                case Type.BYTE:
                    mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Byte", "valueOf", "(B)Ljava/lang/Byte;", false);
                    break;
                case Type.CHAR:
                    mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Character", "valueOf", "(C)Ljava/lang/Character;", false);
                    break;
                case Type.SHORT:
                    mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Short", "valueOf", "(S)Ljava/lang/Short;", false);
                    break;
                case Type.INT:
                    mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Integer", "valueOf", "(I)Ljava/lang/Integer;", false);
                    break;
                case Type.LONG:
                    mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Long", "valueOf", "(J)Ljava/lang/Long;", false);
                    break;
                case Type.FLOAT:
                    mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Float", "valueOf", "(F)Ljava/lang/Float;", false);
                    break;
                case Type.DOUBLE:
                    mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Double", "valueOf", "(D)Ljava/lang/Double;", false);
                    break;
                default:
                    // Object type, no boxing needed
                    break;
            }
        }
    }
}
