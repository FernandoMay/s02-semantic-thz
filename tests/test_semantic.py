"""
Tests for Semantic-Aware THz Communication
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semantic_thz import (
    THzChannel, SemanticEncoder, SemanticResourceAllocator, SimulationRunner
)


class TestTHzChannel:
    def test_channel_creation(self):
        channel = THzChannel()
        assert channel.frequency == 0.3e12

    def test_molecular_absorption(self):
        channel = THzChannel()
        absorption = channel.molecular_absorption(distance=100)
        assert 0 < absorption <= 1

    def test_snr_calculation(self):
        channel = THzChannel()
        snr = channel.snr(tx_power=0.1, distance=100)
        assert snr > 0


class TestSemanticEncoder:
    def test_encoder_creation(self):
        encoder = SemanticEncoder(input_dim=128, latent_dim=16)
        assert encoder.input_dim == 128

    def test_encode_decode(self):
        encoder = SemanticEncoder(input_dim=128, latent_dim=16)
        x = np.random.randn(128)
        z = encoder.encode(x)
        x_rec = encoder.decode(z)
        assert x_rec.shape == (128,)

    def test_fidelity(self):
        encoder = SemanticEncoder(input_dim=128, latent_dim=16)
        original = np.random.randn(128)
        fidelity = encoder.compute_semantic_fidelity(original, original)
        assert fidelity == 1.0


class TestSimulationRunner:
    def test_runner_creation(self):
        runner = SimulationRunner()
        assert runner.channel is not None

    def test_experiment_run(self):
        runner = SimulationRunner()
        results = runner.run_experiment()
        assert 10 in results
        assert "snr_db" in results[10]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
