# TODO

## Reference Graph

- [ ] Weight reference graph edges by existing unit test coverage
  - Large/complex functions pull in the most references but we're bad at generating good unit tests for them
  - If those functions already have existing unit tests, rank them higher
  - For the top 10-20 ranked functions, use a higher effort parameter for optimization

## Test Processing Cache

- [ ] Cache expensive test processing results using file hashes (like the reference graph does)
  - Some big tests take a long time to process — hash the test files and only redo the computation when the hash changes
