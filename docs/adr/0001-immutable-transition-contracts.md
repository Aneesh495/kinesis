# ADR-0001: Immutable Transition Contracts and Schema Versioning

## Status
Accepted

## Context
In reinforcement learning and world-model research, subtle silent data corruption often occurs when transition schemas evolve, fields mutate in-place during preprocessing, or data structures lack runtime validation. Furthermore, without deterministic serialization contracts, replay buffers cannot guarantee reproducibility across different worker processes and storage backends.

## Decision
We enforce frozen, immutable dataclasses for all core runtime contracts:
1. `Transition`: Contains action, observation, next_observation, reward, discount, step_type, and timing metadata.
2. `Episode`: Bounded sequence of transitions with cryptographic episode identifiers, seed tracking, total return, and termination reason.
3. `ObservationSpec` and `ActionSpec`: Strict dimensional and bounds enforcement with clipping helpers.
4. `ModelVersion`: Strongly-typed semantic versions ensuring compatibility between world models and planning policies.

All contracts must support round-trip JSON serialization and validation. Mutating an existing transition raises a `FrozenInstanceError`.

## Consequences
### Positive
- Guarantee of thread-safe concurrent collection across workers.
- Impossible for downstream policy or world model code to mutate recorded history in-place.
- Deterministic SHA-256 hashing of trajectory files and replay buffer segments.

### Negative
- Minor memory overhead when copying modified transitions, mitigated by shallow references for immutable NumPy arrays.
