/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.io;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.util.Constants;

import java.io.IOException;
import java.lang.invoke.MethodHandle;
import java.lang.ref.Cleaner;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Array;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Optional;

/**
 * The libc calls and off-heap memory the io_uring backend needs, bound through the FFM {@code Linker}. Bound
 * reflectively like {@code PrefetchHelper.DirectMadvise}, because the plugin compiles for Java 21, where
 * {@code java.lang.foreign} is a preview API. Pointers and sizes are passed as 64-bit integers (LP64, Linux x86_64 and
 * aarch64). Rings and buffers are accessed through direct {@link ByteBuffer} views, so callers need no FFM types.
 */
@Log4j2
final class NativeIo {
    static final int PAGE = 4096;

    /** Null if the bindings are available, otherwise why not. */
    static final String UNAVAILABLE;

    private static final MethodHandle SYSCALL; // (long nr, long a1..a6) -> long
    private static final MethodHandle MMAP; // (long addr, long len, int prot, int flags, int fd, long off) -> long
    private static final MethodHandle MUNMAP; // (long addr, long len) -> int
    private static final MethodHandle OPEN; // (long path, int flags) -> int
    private static final MethodHandle CLOSE; // (int fd) -> int
    private static final MethodHandle OF_ADDRESS; // (long) -> MemorySegment
    private static final Method REINTERPRET; // MemorySegment.reinterpret(long), caller sensitive so not a method handle
    private static final MethodHandle AS_SLICE; // (MemorySegment, long, long) -> MemorySegment
    private static final MethodHandle AS_BYTE_BUFFER; // (MemorySegment) -> ByteBuffer
    private static final MethodHandle SEGMENT_ADDRESS; // (MemorySegment) -> long
    private static final MethodHandle OF_AUTO; // () -> Arena
    private static final MethodHandle ALLOCATE; // (Arena, long size, long align) -> MemorySegment

    static final int O_DIRECT = switch (System.getProperty("os.arch")) {
        case "amd64", "x86_64" -> 0x4000;
        case "aarch64" -> 0x10000;
        default -> 0;
    };
    private static final int O_CLOEXEC = 0x80000;

    static {
        String unavailable = null;
        MethodHandle syscall = null, mmap = null, munmap = null, open = null, close = null, ofAddress = null, asSlice = null,
            asByteBuffer = null, segmentAddress = null, ofAuto = null, allocate = null;
        Method reinterpret = null;
        if (Constants.LINUX == false || Constants.JRE_IS_64BIT == false || O_DIRECT == 0) {
            unavailable = "needs 64-bit Linux on x86_64 or aarch64";
        } else {
            try {
                final Class<?> segmentClass = Class.forName("java.lang.foreign.MemorySegment");
                final Class<?> layoutClass = Class.forName("java.lang.foreign.MemoryLayout");
                final Class<?> descriptorClass = Class.forName("java.lang.foreign.FunctionDescriptor");
                final Class<?> linkerClass = Class.forName("java.lang.foreign.Linker");
                final Class<?> optionClass = Class.forName("java.lang.foreign.Linker$Option");
                final Class<?> arenaClass = Class.forName("java.lang.foreign.Arena");
                final Class<?> valueLayoutClass = Class.forName("java.lang.foreign.ValueLayout");
                final Object javaInt = valueLayoutClass.getField("JAVA_INT").get(null);
                final Object javaLong = valueLayoutClass.getField("JAVA_LONG").get(null);
                final Object linker = linkerClass.getMethod("nativeLinker").invoke(null);
                final Object libc = linkerClass.getMethod("defaultLookup").invoke(linker);
                final Method find = Class.forName("java.lang.foreign.SymbolLookup").getMethod("find", String.class);
                final Method descriptorOf = descriptorClass.getMethod("of", layoutClass, layoutClass.arrayType());
                final Method downcall = linkerClass.getMethod("downcallHandle", segmentClass, descriptorClass, optionClass.arrayType());
                final Object noOptions = Array.newInstance(optionClass, 0);
                final Object variadicAfterNumber = Array.newInstance(optionClass, 1);
                Array.set(variadicAfterNumber, 0, optionClass.getMethod("firstVariadicArg", int.class).invoke(null, 1));

                syscall = (MethodHandle) downcall.invoke(
                    linker,
                    ((Optional<?>) find.invoke(libc, "syscall")).orElseThrow(),
                    descriptorOf.invoke(
                        null,
                        javaLong,
                        layouts(layoutClass, javaLong, javaLong, javaLong, javaLong, javaLong, javaLong, javaLong)
                    ),
                    variadicAfterNumber
                );
                mmap = (MethodHandle) downcall.invoke(
                    linker,
                    ((Optional<?>) find.invoke(libc, "mmap")).orElseThrow(),
                    descriptorOf.invoke(null, javaLong, layouts(layoutClass, javaLong, javaLong, javaInt, javaInt, javaInt, javaLong)),
                    noOptions
                );
                munmap = (MethodHandle) downcall.invoke(
                    linker,
                    ((Optional<?>) find.invoke(libc, "munmap")).orElseThrow(),
                    descriptorOf.invoke(null, javaInt, layouts(layoutClass, javaLong, javaLong)),
                    noOptions
                );
                open = (MethodHandle) downcall.invoke(
                    linker,
                    ((Optional<?>) find.invoke(libc, "open")).orElseThrow(),
                    descriptorOf.invoke(null, javaInt, layouts(layoutClass, javaLong, javaInt)),
                    noOptions
                );
                close = (MethodHandle) downcall.invoke(
                    linker,
                    ((Optional<?>) find.invoke(libc, "close")).orElseThrow(),
                    descriptorOf.invoke(null, javaInt, layouts(layoutClass, javaInt)),
                    noOptions
                );

                final MethodHandles.Lookup lookup = MethodHandles.publicLookup();
                ofAddress = lookup.findStatic(segmentClass, "ofAddress", MethodType.methodType(segmentClass, long.class))
                    .asType(MethodType.methodType(Object.class, long.class));
                reinterpret = segmentClass.getMethod("reinterpret", long.class);
                asSlice = lookup.findVirtual(segmentClass, "asSlice", MethodType.methodType(segmentClass, long.class, long.class))
                    .asType(MethodType.methodType(Object.class, Object.class, long.class, long.class));
                asByteBuffer = lookup.findVirtual(segmentClass, "asByteBuffer", MethodType.methodType(ByteBuffer.class))
                    .asType(MethodType.methodType(ByteBuffer.class, Object.class));
                segmentAddress = lookup.findVirtual(segmentClass, "address", MethodType.methodType(long.class))
                    .asType(MethodType.methodType(long.class, Object.class));
                ofAuto = lookup.findStatic(arenaClass, "ofAuto", MethodType.methodType(arenaClass))
                    .asType(MethodType.methodType(Object.class));
                allocate = lookup.findVirtual(arenaClass, "allocate", MethodType.methodType(segmentClass, long.class, long.class))
                    .asType(MethodType.methodType(Object.class, Object.class, long.class, long.class));
            } catch (Throwable t) {
                // e.g. no FFM, or native access denied (--illegal-native-access=deny)
                unavailable = t.toString();
            }
        }
        UNAVAILABLE = unavailable;
        SYSCALL = syscall;
        MMAP = mmap;
        MUNMAP = munmap;
        OPEN = open;
        CLOSE = close;
        OF_ADDRESS = ofAddress;
        REINTERPRET = reinterpret;
        AS_SLICE = asSlice;
        AS_BYTE_BUFFER = asByteBuffer;
        SEGMENT_ADDRESS = segmentAddress;
        OF_AUTO = ofAuto;
        ALLOCATE = allocate;
    }

    private NativeIo() {}

    private static Object layouts(final Class<?> layoutClass, final Object... layouts) {
        final Object array = Array.newInstance(layoutClass, layouts.length);
        for (int i = 0; i < layouts.length; i++) {
            Array.set(array, i, layouts[i]);
        }
        return array;
    }

    static long syscall(final long nr, final long a1, final long a2, final long a3, final long a4, final long a5, final long a6) {
        try {
            return (long) SYSCALL.invokeExact(nr, a1, a2, a3, a4, a5, a6);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** @return the mapping's address, or a negative value on failure */
    static long mmap(final long length, final int prot, final int flags, final int fd, final long offset) {
        try {
            return (long) MMAP.invokeExact(0L, length, prot, flags, fd, offset);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Opens {@code path} read-only with O_DIRECT; returns the fd, or a negative value on failure. */
    static int openDirect(final String path) {
        final byte[] bytes = path.getBytes(StandardCharsets.UTF_8);
        final Memory name = allocate(bytes.length + 1);
        name.buffer(0, bytes.length + 1).put(bytes).put((byte) 0);
        try {
            return (int) OPEN.invokeExact(name.address, O_DIRECT | O_CLOEXEC);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static void close(final int fd) {
        try {
            final int ignored = (int) CLOSE.invokeExact(fd);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** A native-order view of {@code length} bytes at a raw address, such as an io_uring ring. */
    static ByteBuffer view(final long address, final long length) {
        try {
            final Object segment = REINTERPRET.invoke((Object) OF_ADDRESS.invokeExact(address), length);
            return ((ByteBuffer) AS_BYTE_BUFFER.invokeExact(segment)).order(ByteOrder.nativeOrder());
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private static final Cleaner CLEANER = Cleaner.create();
    private static final int PROT_READ_WRITE = 0x3;
    private static final int MAP_PRIVATE_ANONYMOUS = 0x02 | 0x20; // same on x86_64 and aarch64

    /**
     * Zeroed anonymous memory from {@code mmap}, outside the JVM's direct memory limit, and committed only as it is
     * touched. It is unmapped once the returned {@link Memory} is unreachable.
     *
     * @throws IOException if the mapping fails
     */
    static Memory mapAnonymous(final long size) throws IOException {
        final long address = mmap(size, PROT_READ_WRITE, MAP_PRIVATE_ANONYMOUS, -1, 0);
        if (address == -1L) {
            throw new IOException("mmap of " + size + " anonymous bytes failed");
        }
        try {
            final Memory memory = new Memory(REINTERPRET.invoke((Object) OF_ADDRESS.invokeExact(address), size), address);
            CLEANER.register(memory, () -> unmap(address, size));
            return memory;
        } catch (Throwable t) {
            unmap(address, size);
            throw new IOException("Could not map " + size + " anonymous bytes", t);
        }
    }

    private static void unmap(final long address, final long size) {
        try {
            final int ignored = (int) MUNMAP.invokeExact(address, size);
        } catch (Throwable t) {
            log.warn("munmap of {} bytes failed", size, t);
        }
    }

    /** Zeroed, page-aligned off-heap memory, freed by the GC once unreachable. Counts against direct memory. */
    static Memory allocate(final long size) {
        try {
            final Object segment = (Object) ALLOCATE.invokeExact((Object) OF_AUTO.invokeExact(), size, (long) PAGE);
            return new Memory(segment, (long) SEGMENT_ADDRESS.invokeExact(segment));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Off-heap memory; holding it keeps the memory alive. */
    static final class Memory {
        private final Object segment;
        final long address;

        private Memory(final Object segment, final long address) {
            this.segment = segment;
            this.address = address;
        }

        /** A little-endian view of {@code [offset, offset + length)}; {@code length} must fit an int. */
        ByteBuffer buffer(final long offset, final int length) {
            try {
                return ((ByteBuffer) AS_BYTE_BUFFER.invokeExact((Object) AS_SLICE.invokeExact(segment, offset, (long) length))).order(
                    ByteOrder.LITTLE_ENDIAN
                );
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }
    }

    private static RuntimeException rethrow(final Throwable t) {
        if (t instanceof RuntimeException e) {
            return e;
        }
        if (t instanceof Error e) {
            throw e;
        }
        return new IllegalStateException(t);
    }
}
