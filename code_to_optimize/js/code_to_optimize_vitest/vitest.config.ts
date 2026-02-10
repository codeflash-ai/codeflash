import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['tests/**/*.test.ts'],
    reporters: ['default', 'junit'],
    outputFile: {
      junit: '.codeflash/vitest-results.xml',
    },
  },
});
