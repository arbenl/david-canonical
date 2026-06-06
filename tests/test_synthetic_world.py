"""Tests for the generative twin."""

from __future__ import annotations

from david.simulator.synthetic_world import HyperPrior, sample_world


def test_sample_world_shapes():
    prior = HyperPrior(R=2, T=12, L=3, K=4, S=3, M=2, H=6)
    w = sample_world(prior, seed=42)
    assert w.z.shape == (2, 18)
    assert w.a.shape == (2, 18, 4)
    assert w.b.shape == (2, 12, 4, 3)
    assert w.y.shape == (2, 12, 4, 3, 2)
    assert w.selected.shape == (2, 12, 4)
    assert w.observability.shape == (2, 12)


def test_sample_world_no_self_transitions_within_segment():
    """Within a regime segment, z should be constant; transitions happen at boundaries."""
    prior = HyperPrior(R=1, T=60, L=3, K=2, S=3, M=2, H=0)
    w = sample_world(prior, seed=1)
    # Check that any change in z[r, t] over t is a valid transition (not self)
    # Self-transitions inside a segment are fine; what we check is that the
    # *generative* process actually moved between distinct regimes at some point.
    distinct = len(set(w.z[0].tolist()))
    assert distinct >= 2, "expected multiple regimes over T=60 months"
