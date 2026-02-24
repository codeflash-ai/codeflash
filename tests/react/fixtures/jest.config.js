module.exports = {
	resetMocks: true,
	roots: ['<rootDir>'],
	testEnvironment: 'jsdom',
	transform: {
		'\\.[jt]sx?$': ['esbuild-jest', { sourcemap: true }],
	},
};
