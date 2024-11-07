import numpy as np

class ParticleFilter:
    def __init__(self, initial_state, num_particles, process_noise_std, measurement_noise_std, initial_state_std):
        self.num_particles = num_particles
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.particles = np.random.normal(
            np.concatenate([initial_state, [0, 0, 0]]),
            initial_state_std,
            size=(num_particles, 6)
        )

    def predict(self, dt=1.0):
        noise = np.random.normal(0, self.process_noise_std, size=self.particles.shape)
        self.particles[:, :3] += self.particles[:, 3:] * dt + noise[:, :3]
        self.particles[:, 3:] += noise[:, 3:]

    def update_weights(self, detection):
        if detection is not None:
            distances = np.linalg.norm(self.particles[:, :3] - detection, axis=1)
            weights = np.exp(-0.5 * (distances / self.measurement_noise_std) ** 2)
            weights_sum = weights.sum()
            if weights_sum > 0:
                weights /= weights_sum
            else:
                weights = np.ones(len(self.particles)) / len(self.particles)
        else:
            weights = np.ones(len(self.particles)) / len(self.particles)
        return weights

    def resample(self, weights):
        indices = np.random.choice(len(self.particles), size=len(self.particles), p=weights)
        self.particles = self.particles[indices]

    def estimate(self):
        return self.particles[:, :3].mean(axis=0)
