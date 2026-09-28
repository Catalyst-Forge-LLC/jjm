# Synthetic examples

Every fixture is synthetic. No number comes from a model, no artifact payload exists, and no calibration study or benchmark was performed. Fields representing validated calibration illustrate a data shape, not evidence that any actual model is calibrated.

`index.json` names each request/response pair and states whether it must pass or fail the pack validator. Positive means valid under the proposed contract, not a correct real-world model answer. Negative fixtures deliberately violate the contract.

Start with `01-routing`, `03-missing-evidence`, and `04-not-observable` to see accepted inference, preflight abstention, and an evaluated but unobservable outcome.

Digest strings identify synthetic fixture labels. They are not computed fingerprints of live model assets, rendered prompts, or the full request. A production implementation must calculate the real RFC 8785 content digests and verify actual artifact payloads.
