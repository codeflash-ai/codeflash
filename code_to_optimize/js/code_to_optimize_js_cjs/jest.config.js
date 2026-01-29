module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/tests/**/*.test.js'],
  reporters: ['default', ['jest-junit', { outputDirectory: '.codeflash' }]],
  verbose: true,
};
