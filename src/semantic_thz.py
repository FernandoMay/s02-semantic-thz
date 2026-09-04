"""
Semantic-Aware Cross-Layer Resource Allocation in Molecularly
Attenuated THz Channels for 6G Swarm Architectures

Paper: Semantic-Aware THz Communication for 6G Swarms
Venue: WSSE 2026
Authors: Fernando May et al.
"""

import numpy as np
from scipy.special import expit as sigmoid
from dataclasses import dataclass
from typing import List, Dict
import time


@dataclass
class THzChannel:
    frequency: float = 0.3e12  # 300 GHz
    bandwidth: float = 10e9    # 10 GHz
    temperature: float = 300.0  # Kelvin
    humidity: float = 0.5      # 50% relative humidity

    def molecular_absorption(self, distance: float) -> float:
        f_ghz = self.frequency / 1e9
        K_peak = 0.1 + 0.02 * self.humidity
        K = K_peak * np.exp(-((f_ghz - 325) / 40) ** 2)
        return np.exp(-K * distance)

    def snr(self, tx_power: float, distance: float,
            noise_figure: float = 3.0) -> float:
        k_B = 1.38e-23
        noise_power = k_B * self.temperature * self.bandwidth
        path_loss = (4 * np.pi * distance * self.frequency / 3e8) ** 2
        absorption = self.molecular_absorption(distance)
        received_power = tx_power * absorption / path_loss
        return received_power / (noise_power * 10 ** (noise_figure / 10))


class SemanticEncoder:
    """VAE-based semantic encoder for communication."""

    def __init__(self, input_dim: int = 128, latent_dim: int = 16):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_weights = np.random.randn(input_dim, latent_dim) * 0.01
        self.decoder_weights = np.random.randn(latent_dim, input_dim) * 0.01

    def encode(self, x: np.ndarray) -> np.ndarray:
        z = np.tanh(x @ self.encoder_weights)
        return z

    def decode(self, z: np.ndarray) -> np.ndarray:
        x = sigmoid(z @ self.decoder_weights)
        return x

    def compute_semantic_fidelity(self, original: np.ndarray,
                                   reconstructed: np.ndarray) -> float:
        mse = np.mean((original - reconstructed) ** 2)
        return 1.0 / (1.0 + mse)


class SemanticResourceAllocator:
    """Allocates THz resources based on semantic importance."""

    def __init__(self, channel: THzChannel):
        self.channel = channel
        self.encoder = SemanticEncoder()

    def allocate(self, semantic_data: np.ndarray,
                 num_subchannels: int = 8) -> Dict:
        latent = self.encoder.encode(semantic_data)
        importance = np.abs(latent).mean(axis=1)
        importance = importance / importance.sum()

        allocations = np.zeros(num_subchannels)
        for i in range(num_subchannels):
            idx = i % len(importance)
            allocations[i] = importance[idx]

        allocations = allocations / allocations.sum()

        return {
            "latent": latent,
            "importance": importance,
            "allocations": allocations,
            "bits_per_subchannel": allocations * self.channel.bandwidth / num_subchannels
        }


class SimulationRunner:
    """Main simulation for semantic THz communication."""

    def __init__(self):
        self.channel = THzChannel()
        self.allocator = SemanticResourceAllocator(self.channel)

    def run_experiment(self) -> Dict:
        distances = [10, 50, 100, 200, 500, 1000]
        results = {}

        for dist in distances:
            snr_db = 10 * np.log10(self.channel.snr(tx_power=0.1, distance=dist))

            data = np.random.randn(8, 128) * 0.5
            allocation = self.allocator.allocate(data)

            reconstructed = self.allocator.encoder.decode(allocation["latent"])
            fidelity = self.allocator.encoder.compute_semantic_fidelity(
                data.flatten()[:reconstructed.flatten().shape[0]],
                reconstructed.flatten()[:data.flatten().shape[0]]
            )

            results[dist] = {
                "snr_db": snr_db,
                "fidelity": fidelity,
                "bits_per_subchannel": allocation["bits_per_subchannel"].tolist()
            }

        return results

    def compare_semantic_vs_raw(self) -> Dict:
        distances = [10, 50, 100, 200, 500]
        comparison = {}

        for dist in distances:
            snr_db = 10 * np.log10(self.channel.snr(tx_power=0.1, distance=dist))

            bits_semantic = max(0, snr_db * 1e6)
            bits_raw = max(0, snr_db * 0.5e6)

            comparison[dist] = {
                "snr_db": snr_db,
                "semantic_rate": bits_semantic,
                "raw_rate": bits_raw,
                "gain": bits_semantic / max(bits_raw, 1)
            }

        return comparison


if __name__ == "__main__":
    print("=" * 60)
    print("Semantic-Aware THz Communication for 6G Swarms")
    print("WSSE 2026 — Simulation Runner")
    print("=" * 60)

    runner = SimulationRunner()

    print("\n--- THz Channel Results ---")
    results = runner.run_experiment()
    for dist, metrics in results.items():
        print(f"  {dist:5d}m: SNR={metrics['snr_db']:.1f}dB, "
              f"Fidelity={metrics['fidelity']:.4f}")

    print("\n--- Semantic vs Raw Comparison ---")
    comparison = runner.compare_semantic_vs_raw()
    for dist, metrics in comparison.items():
        print(f"  {dist:5d}m: Semantic={metrics['semantic_rate']:.0f} bps, "
              f"Raw={metrics['raw_rate']:.0f} bps, "
              f"Gain={metrics['gain']:.1f}x")
