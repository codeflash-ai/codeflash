package com.codeflash.profiler;

import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.commons.AdviceAdapter;

/**
 * ASM MethodVisitor that injects line-level profiling probes.
 *
 * <p>At each {@code LineNumber} table entry within the target method:
 * <ol>
 *   <li>Registers the line with {@link ProfilerRegistry} (happens once at class-load time)</li>
 *   <li>Injects bytecode: {@code LDC globalId; INVOKESTATIC ProfilerData.hit(I)V}</li>
 * </ol>
 *
 * <p>At method entry (first line): injects {@code ProfilerData.enterMethod(entryLineId)}.
 * <p>At method exit (every RETURN/ATHROW): injects {@code ProfilerData.exitMethod()}.
 */
public class LineProfilingMethodVisitor extends AdviceAdapter {

    private static final String PROFILER_DATA = "com/codeflash/profiler/ProfilerData";

    private final String internalClassName;
    private final String sourceFile;
    private final String methodName;
    private boolean firstLineVisited = false;

    protected LineProfilingMethodVisitor(
            MethodVisitor mv, int access, String name, String descriptor,
            String internalClassName, String sourceFile) {
        super(Opcodes.ASM9, mv, access, name, descriptor);
        this.internalClassName = internalClassName;
        this.sourceFile = sourceFile;
        this.methodName = name;
    }

    @Override
    public void visitLineNumber(int line, Label start) {
        super.visitLineNumber(line, start);

        // Register this line and get its global ID (happens once at class-load time)
        String dotClassName = internalClassName.replace('/', '.');
        int globalId = ProfilerRegistry.register(sourceFile, dotClassName, methodName, line);

        if (!firstLineVisited) {
            firstLineVisited = true;
            // Inject enterMethod call at the first line of the method
            mv.visitLdcInsn(globalId);
            mv.visitMethodInsn(INVOKESTATIC, PROFILER_DATA, "enterMethod", "(I)V", false);
        }

        // Inject: ProfilerData.hit(globalId)
        mv.visitLdcInsn(globalId);
        mv.visitMethodInsn(INVOKESTATIC, PROFILER_DATA, "hit", "(I)V", false);
    }

    @Override
    protected void onMethodExit(int opcode) {
        // Before every RETURN or ATHROW, flush timing for the last line
        // This fixes the "last line always shows 0ms" bug
        mv.visitMethodInsn(INVOKESTATIC, PROFILER_DATA, "exitMethod", "()V", false);
    }
}
