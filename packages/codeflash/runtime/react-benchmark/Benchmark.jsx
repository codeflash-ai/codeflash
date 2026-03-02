const Timing = require("./timing")
const { getMean, getMedian, getStdDev } = require("./math")
const {requireFromRoot} = require("../utils")
const React = requireFromRoot("react")

const sortNumbers = (a, b) => a - b;

// eslint-disable-next-line @typescript-eslint/ban-types
function BenchmarkInner(
	{
		component: Component,
		componentProps,
		includeLayout = false,
		onComplete,
		samples: numSamples,
		timeout = 10000,
		type = 'mount',
	},
	ref
) {
	const [{ running, cycle, samples, startTime }, dispatch] = React.useReducer(reducer, initialState);

	React.useImperativeHandle(ref, () => ({
		start: () => {
			dispatch({ type: 'START', payload: Timing.now() });
		},
	}));

	const shouldRender = getShouldRender(type, cycle);
	const shouldRecord = getShouldRecord(type, cycle);
	const isDone = getIsDone(type, cycle, numSamples);

	const handleComplete = React.useCallback(
		(startTime, endTime, samples) => {
			const runTime = endTime - startTime;
			const sortedElapsedTimes = samples.map(({ elapsed }) => elapsed).sort(sortNumbers);
			const mean = getMean(sortedElapsedTimes);
			const stdDev = getStdDev(sortedElapsedTimes);

			const result = {
				startTime,
				endTime,
				runTime,
				sampleCount: samples.length,
				samples,
				max: sortedElapsedTimes[sortedElapsedTimes.length - 1],
				min: sortedElapsedTimes[0],
				median: getMedian(sortedElapsedTimes),
				mean,
				stdDev,
				p70: mean + stdDev,
				p95: mean + stdDev * 2,
				p99: mean + stdDev * 3,
				layout: undefined,
			};

			if (includeLayout) {
				const sortedLayoutTimes = samples.map(({ layout }) => layout).sort(sortNumbers);
				const mean = getMean(sortedLayoutTimes);
				const stdDev = getStdDev(sortedLayoutTimes);
				result.layout = {
					max: sortedLayoutTimes[sortedLayoutTimes.length - 1],
					min: sortedLayoutTimes[0],
					median: getMedian(sortedLayoutTimes),
					mean,
					stdDev,
					p70: mean + stdDev,
					p95: mean + stdDev * 2,
					p99: mean + stdDev * 3,
				};
			}

			onComplete(result);

			dispatch({ type: 'RESET' });
		},
		[includeLayout, onComplete]
	);

	// useMemo causes this to actually run _before_ the component mounts
	// as opposed to useEffect, which will run after
	React.useMemo(() => {
		if (running && shouldRecord) {
			dispatch({ type: 'START_SAMPLE', payload: Timing.now() });
		}
	}, [cycle, running, shouldRecord]);

	React.useEffect(() => {
		if (!running) {
			return;
		}

		const now = Timing.now();

		if (shouldRecord && samples.length && samples[samples.length - 1].end < 0) {
			if (includeLayout && type !== 'unmount' && document.body) {
				document.body.offsetWidth;
			}
			const layoutEnd = Timing.now();

			dispatch({ type: 'END_SAMPLE', payload: now });
			dispatch({ type: 'END_LAYOUT', payload: layoutEnd - now });
			return;
		}

		const timedOut = now - startTime > timeout;
		if (!isDone && !timedOut) {
			setTimeout(() => {
				dispatch({ type: 'TICK' });
			}, 1);
			return;
		} else if (isDone || timedOut) {
			handleComplete(startTime, now, samples);
		}
	}, [includeLayout, running, isDone, samples, shouldRecord, shouldRender, startTime, timeout]);

	return running && shouldRender ? (
		// @ts-ignore forcing a testid for cycling
		<Component {...componentProps} data-testid={cycle} />
	) : null;
}

// eslint-disable-next-line @typescript-eslint/ban-types
export const Benchmark = React.forwardRef(BenchmarkInner);

function reducer(state, action) {
	switch (action.type) {
		case 'START':
			return {
				...state,
				startTime: action.payload,
				running: true,
			};

		case 'START_SAMPLE': {
			const samples = [...state.samples];
			samples.push({ start: action.payload, end: -Infinity, elapsed: -Infinity, layout: -Infinity });
			return {
				...state,
				samples,
			};
		}

		case 'END_SAMPLE': {
			const samples = [...state.samples];
			const index = samples.length - 1;
			samples[index].end = action.payload;
			samples[index].elapsed = action.payload - samples[index].start;
			return {
				...state,
				samples,
			};
		}

		case 'END_LAYOUT': {
			const samples = [...state.samples];
			const index = samples.length - 1;
			samples[index].layout = action.payload;
			return {
				...state,
				samples,
			};
		}

		case 'TICK':
			return {
				...state,
				cycle: state.cycle + 1,
			};

		case 'RESET':
			return initialState;

		default:
			return state;
	}
}

function getShouldRender(type, cycle) {
	switch (type) {
		// Render every odd iteration (first, third, etc)
		// Mounts and unmounts the component
		case 'mount':
		case 'unmount':
			return !((cycle + 1) % 2);
		// Render every iteration (updates previously rendered module)
		case 'update':
			return true;
		default:
			return false;
	}
}

function getShouldRecord(type, cycle) {
	switch (type) {
		// Record every odd iteration (when mounted: first, third, etc)
		case 'mount':
			return !((cycle + 1) % 2);
		// Record every iteration
		case 'update':
			return cycle !== 0;
		// Record every even iteration (when unmounted)
		case 'unmount':
			return !(cycle % 2);
		default:
			return false;
	}
}

function getIsDone(type, cycle, numSamples) {
	switch (type) {
		case 'mount':
		case 'unmount':
			return cycle >= numSamples * 2 - 1;
		case 'update':
			return cycle >= numSamples;
		default:
			return true;
	}
}
