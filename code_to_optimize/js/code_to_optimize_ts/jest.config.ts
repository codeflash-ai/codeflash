import type { Config } from 'jest';

const config: Config = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testMatch: [
    '**/tests/**/*.test.ts',
    '**/tests/**/*.spec.ts'
  ],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
  collectCoverageFrom: [
    '**/*.ts',
    '!**/node_modules/**',
    '!**/dist/**',
    '!jest.config.ts'
  ],
  reporters: [
    'default',
    [
      'jest-junit',
      {
        outputDirectory: '.codeflash',
        outputName: 'jest-results.xml',
        includeConsoleOutput: true
      }
    ]
  ],
  transform: {
    '^.+\\.tsx?$': ['ts-jest', {
      useESM: false
    }]
  }
};

export default config;
