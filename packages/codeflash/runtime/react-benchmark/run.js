const { requireFromRoot } = require("../utils");
const { Benchmark } = require("./Benchmark")

const React = requireFromRoot("react")
const { render, act, waitFor } = requireFromRoot("@testing-library/react")

module.exports = async function runBenchmark({ component, props, samples = 50, type = 'mount' }) {
	const ref = React.createRef();

	let results;
	let resolvePromise;
	const completionPromise = new Promise((resolve) => {
		resolvePromise = resolve;
	});

	const handleComplete = (res) => {
		results = res;
		resolvePromise(res);
	};

	render(
		React.createElement(Benchmark, {
			component,
			onComplete: handleComplete,
			ref,
			samples,
			componentProps: props,
			type,
		})
	);

	act(() => {
		ref.current?.start();
	});

	await completionPromise;
	return results;
}
