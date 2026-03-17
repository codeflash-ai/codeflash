package com.codeflash.profiler;

import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassWriter;

import java.lang.instrument.ClassFileTransformer;
import java.security.ProtectionDomain;

/**
 * {@link ClassFileTransformer} that instruments target classes with line profiling.
 *
 * <p>When a class matches the profiler configuration, it is run through ASM
 * to inject {@link ProfilerData#hit(int)} calls at each line number.
 */
public class LineProfilingTransformer implements ClassFileTransformer {

    private final ProfilerConfig config;

    public LineProfilingTransformer(ProfilerConfig config) {
        this.config = config;
    }

    @Override
    public byte[] transform(ClassLoader loader, String className,
                            Class<?> classBeingRedefined, ProtectionDomain protectionDomain,
                            byte[] classfileBuffer) {
        if (className == null || !config.shouldInstrumentClass(className)) {
            return null; // null = don't transform
        }

        try {
            return instrumentClass(className, classfileBuffer);
        } catch (Exception e) {
            System.err.println("[codeflash-profiler] Failed to instrument " + className + ": " + e.getMessage());
            return null;
        }
    }

    private byte[] instrumentClass(String internalClassName, byte[] bytecode) {
        ClassReader cr = new ClassReader(bytecode);
        ClassWriter cw = new ClassWriter(cr, ClassWriter.COMPUTE_FRAMES | ClassWriter.COMPUTE_MAXS);
        LineProfilingClassVisitor cv = new LineProfilingClassVisitor(cw, internalClassName, config);
        cr.accept(cv, ClassReader.EXPAND_FRAMES);
        return cw.toByteArray();
    }
}
