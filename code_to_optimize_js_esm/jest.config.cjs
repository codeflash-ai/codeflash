// Jest config for ES Module project (using .cjs since package is type: module)
module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/tests/**/*.test.js'],
  reporters: ['default', ['jest-junit', { outputDirectory: '.codeflash' }]],
  verbose: true,
  transform: {},
  // Tell Jest to also look for modules in the project's node_modules when
  // resolving modules from symlinked packages (like codeflash)
  moduleDirectories: ['node_modules', '<rootDir>/node_modules'],
};
