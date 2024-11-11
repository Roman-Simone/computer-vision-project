import numpy as np

class ParticleFilter:
    """
    A Particle Filter for tracking an object in a 3D space with velocity components.
    
    Parameters:
        initial_state (array-like): initial state of the object in the form [x, y, z].
        num_particles (int): number of particles in the filter.
        process_noise_std (float): standard deviation of process noise affecting the state.
        measurement_noise_std (float): standard deviation of measurement noise affecting detections.
        initial_state_std (array-like): standard deviation of initial particle spread in [x, y, z, vx, vy, vz].
    """

    def __init__(self, initial_state, num_particles, process_noise_std, measurement_noise_std, initial_state_std):
        self.num_particles = num_particles
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std

        # Initialize particles around the initial state, including zero velocity [vx, vy, vz]
        self.particles = np.random.normal(
            np.concatenate([initial_state, [0, 0, 0]]),
            initial_state_std,
            size=(num_particles, 6)
        )

    def predict(self, dt=1.0):
        """
        Predicts the new state of each particle based on its current velocity, 
        applying process noise to account for uncertainty.
        
        Parameters:
            dt (float): time step for the prediction update, default is 1.0.
        """
        # Add process noise to particles
        noise = np.random.normal(0, self.process_noise_std, size=self.particles.shape)

        # Update position based on velocity and process noise
        self.particles[:, :3] += self.particles[:, 3:] * dt + noise[:, :3]
        self.particles[:, 3:] += noise[:, 3:]  # Update velocity with noise

    def update_weights(self, detection):
        """
        Updates the weights for each particle based on the likelihood of a given detection.
        
        Parameters:
            detection (array-like or None): observed position [x, y, z] of the object or None if no detection.
            
        Returns:
            np.ndarray: updated weights for each particle.
        """
        if detection is not None:
            # Calculate distances from each particle to the detection
            distances = np.linalg.norm(self.particles[:, :3] - detection, axis=1)

            # weights --> Gaussian likelihood of each particle’s distance to detection
            weights = np.exp(-0.5 * (distances / self.measurement_noise_std) ** 2)
            weights_sum = weights.sum()

            if weights_sum > 0:
                weights /= weights_sum
            else:
                weights = np.ones(len(self.particles)) / len(self.particles)
        else:
            # Uniform weights if no detection is provided
            weights = np.ones(len(self.particles)) / len(self.particles)

        return weights

    def resample(self, weights):
        """
        Resamples particles based on their weights to focus on particles closer to the detection.
        
        Parameters:
            weights (np.ndarray): probability weights for each particle.
        """

        indices = np.random.choice(len(self.particles), size=len(self.particles), p=weights)
        self.particles = self.particles[indices]

    def estimate(self):
        """
        Estimates the current position of the object by taking the mean position of all particles.
        
        Returns:
            np.ndarray: estimated position [x, y, z] of the object.
        """
        return self.particles[:, :3].mean(axis=0)
