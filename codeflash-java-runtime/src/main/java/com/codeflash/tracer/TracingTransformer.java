package com.codeflash.tracer;

import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

import java.lang.instrument.ClassFileTransformer;
import java.security.ProtectionDomain;
import java.util.HashMap;
import java.util.Map;

public class TracingTransformer implements ClassFileTransformer {

    private final TracerConfig config;

    public TracingTransformer(TracerConfig config) {
        this.config = config;
    }

    @Override
    public byte[] transform(ClassLoader loader, String className,
                            Class<?> classBeingRedefined, ProtectionDomain protectionDomain,
                            byte[] classfileBuffer) {
        if (className == null || !config.shouldInstrumentClass(className)) {
            return null;
        }

        // Skip internal JDK, framework, and synthetic classes
        if (className.startsWith("java/")
                || className.startsWith("javax/")
                || className.startsWith("jdk/")
                || className.startsWith("sun/")
                || className.startsWith("com/sun/")
                || className.startsWith("com/codeflash/")
                || className.contains("ConstructorAccess")
                || className.contains("FieldAccess")
                || className.contains("$$")) {
            return null;
        }

        try {
            return instrumentClass(className, classfileBuffer);
        } catch (Throwable e) {
            System.err.println("[codeflash-tracer] Failed to instrument " + className + ": "
                    + e.getClass().getName() + ": " + e.getMessage());
            return null;
        }
    }

    private byte[] instrumentClass(String internalClassName, byte[] bytecode) {
        ClassReader cr = new ClassReader(bytecode);

        // Pre-scan: collect the first source line number for each method.
        // ASM's visitMethod() doesn't provide line info — it arrives later via visitLineNumber().
        // We do a lightweight read pass first so the instrumentation pass has accurate line numbers.
        Map<String, Integer> methodLineNumbers = new HashMap<>();
        cr.accept(new ClassVisitor(Opcodes.ASM9) {
            @Override
            public MethodVisitor visitMethod(int access, String name, String descriptor,
                                             String signature, String[] exceptions) {
                String key = name + descriptor;
                return new MethodVisitor(Opcodes.ASM9) {
                    private boolean captured = false;

                    @Override
                    public void visitLineNumber(int line, Label start) {
                        if (!captured) {
                            methodLineNumbers.put(key, line);
                            captured = true;
                        }
                    }
                };
            }
        }, ClassReader.SKIP_FRAMES);

        // Use COMPUTE_MAXS only (not COMPUTE_FRAMES) to preserve original stack map frames.
        // COMPUTE_FRAMES recomputes all frames and calls getCommonSuperClass() which either
        // triggers classloader deadlocks or produces incorrect frames when returning "java/lang/Object".
        // With COMPUTE_MAXS + ClassReader passed to constructor, ASM copies original frames and
        // adjusts offsets for injected code. Our AdviceAdapter only injects at method entry
        // (before any branch points), so existing frames remain valid.
        ClassWriter cw = new ClassWriter(cr, ClassWriter.COMPUTE_MAXS);
        TracingClassVisitor cv = new TracingClassVisitor(cw, internalClassName, methodLineNumbers);
        cr.accept(cv, ClassReader.EXPAND_FRAMES);
        return cw.toByteArray();
    }
}
