# Interview walkthrough

“FeatureDock is a small ML data pipeline. It records when a sensor event happened and when it arrived, so a historical training row only uses information that was available at prediction time. I check a real PySpark backfill against the serving SQL, and test retries, bad messages, and recovery from a crash.”

## Explain the choices

- **Why two timestamps?** Event time alone can leak late-arriving data into historical features. Receipt time preserves what the system knew at that moment.
- **Why not discard late data?** It can be useful for later snapshots and investigation. Filtering by both clocks is enough for this retained-history implementation.
- **Why two IDs?** A transport retry and an underlying observation are different objects. Delivery-level and event-level deduplication solve different cases.
- **What is atomic?** Accepted event changes and inbox progress in one SQLite transaction. This does not make a separate broker, database and downstream service one distributed transaction.
- **Why a separate Spark implementation?** Batch training and serving often drift apart. An independent aggregation path plus hand-calculated tests provides evidence that the contract is consistent.
- **Why such a simple model?** The central contribution is the data path. Logistic regression makes the consumer small, reproducible, and inspectable.
- **Does the model predict equipment failures?** No. It predicts a threshold-based future condition in a synthetic simulator.
- **What does the benchmark prove?** Correctness and numerical agreement at the measured fixture size. It does not prove cluster-scale throughput or production reliability.
- **Why can the demo replay receipt timestamps but the API cannot accept them?** Historical import is a trusted operation. Allowing arbitrary API clients to backdate receipt time would undermine the leakage guarantee.

## Demonstrate it

1. Choose an asset and prediction hour. Show the two feature columns and event lineage.
2. Send a duplicate. Show the event count stays fixed.
3. Submit the invalid reading. Show its quarantine reason.
4. Submit a late reading. Show it is recorded but excluded from the historical prediction.
5. Open `spark_job.py`, the release report, and `test_crash_before_commit_leaves_checkpoint_and_events_unchanged`.

Read `store.py`, `features.py`, `contracts.py`, `spark_job.py` and `model.py` before discussing the project. Describe it as portfolio work, not commercial deployment. A useful follow-up is to connect a licensed real dataset and define a target whose labels and availability times can be verified.
