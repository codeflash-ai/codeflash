package com.codeflash.profiler;

import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/**
 * ASM ClassVisitor that filters methods and wraps target methods with
 * {@link LineProfilingMethodVisitor} for line-level profiling.
 */
public class LineProfilingClassVisitor extends ClassVisitor {

    private final String internalClassName;
    private final ProfilerConfig config;
    private String sourceFile;

    public LineProfilingClassVisitor(ClassVisitor classVisitor, String internalClassName, ProfilerConfig config) {
        super(Opcodes.ASM9, classVisitor);
        this.internalClassName = internalClassName;
        this.config = config;
    }

    @Override
    public void visitSource(String source, String debug) {
        super.visitSource(source, debug);
        // Resolve the absolute source file path from the config
        this.sourceFile = config.resolveSourceFile(internalClassName);
    }

    @Override
    public MethodVisitor visitMethod(int access, String name, String descriptor,
                                     String signature, String[] exceptions) {
        MethodVisitor mv = super.visitMethod(access, name, descriptor, signature, exceptions);

        if (config.shouldInstrumentMethod(internalClassName, name)) {
            return new LineProfilingMethodVisitor(mv, access, name, descriptor,
                    internalClassName, sourceFile);
        }
        return mv;
    }
}
