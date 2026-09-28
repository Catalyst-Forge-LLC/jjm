# Application boundary example

`consume-judgment.mjs` is a small, original Node-compatible example showing why an admitted judgment is not action authorization. It expects the full contract to have been validated already. It does not implement a server, contact a model, or perform a receipt/authentication check itself.

Its `receiptVerified` and `authorized` inputs must come from independent application checks. Passing booleans in a test is not equivalent to implementing those checks in production. A state revision change invalidates eligibility. The helper never executes an action.

Run the synthetic checks with:

```bash
node reference/consume-judgment.test.mjs
```

The tests were run on Node 22.16.0 in the preparation environment. This is not a claim that a JJM service or TypeScript SDK has been implemented. The production SDK remains an implementation task in the specification.
