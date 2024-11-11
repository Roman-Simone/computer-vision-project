import numpy as np
import cv2
import torch
size = 800

DISTANCE_THRESHOLD = 200

class ParticleFilter:
    """
    A Particle Filter class to track the position of an object using particle-based state estimation.
    
    Parameters:
        tracker_id (int): unique identifier for the tracker.
        color (tuple): color used for drawing particles and trajectory on the frame.
        frame_size (tuple): size of the frame in (width, height).
        num_particles (int): number of particles used for the filter. Default is 1000.
        process_noise (float): standard deviation for process noise. Default is 5.0.
        measurement_noise (float): standard deviation for measurement noise. Default is 2.0.
    """

    def __init__(self, tracker_id, color, frame_size, num_particles=1000, process_noise=5.0, measurement_noise=2.0):
        # Initialize tracker properties
        self.tracker_id = tracker_id
        self.color = color
        self.frame_width, self.frame_height = frame_size
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # Initialize particles randomly within the frame dimensions
        self.particles = np.random.rand(self.num_particles, 2) * [self.frame_width, self.frame_height]
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.ball_positions = []
        self.last_position = None
        self.active = True
        self.frames_without_detection = 0

    def predict(self):
        """
        Predicts the new state of particles based on previous velocity and process noise.
        Particles are updated with the estimated velocity, then perturbed by random noise.
        """
        # Estimate velocity if there are at least two past positions
        if len(self.ball_positions) >= 2:
            velocity = np.array(self.ball_positions[-1]) - np.array(self.ball_positions[-2])
            self.particles += velocity
        
        # Apply process noise to particles
        self.particles += np.random.randn(self.num_particles, 2) * self.process_noise
        
        # Clip particles within frame bounds
        self.particles = np.clip(self.particles, [0, 0], [self.frame_width - 1, self.frame_height - 1])

    def update(self, measurement):
        """
        Updates particle weights based on the measured position and resamples particles.
        
        Parameters:
            measurement (array-like or torch.Tensor): measured position of the object (x, y).
        """
        if isinstance(measurement, torch.Tensor):
            measurement = measurement.cpu().numpy()
        
        if measurement.shape != (2,):
            print(f"Invalid measurement shape: {measurement.shape}")
            return

        # Handle case where detection is missing (e.g., measurement is [-1, -1])
        if np.array_equal(measurement, np.array([-1, -1])):
            self.frames_without_detection += 1
            if self.frames_without_detection > 10:
                self.active = False  # Deactivate tracker if no detection for too long
            return

        self.frames_without_detection = 0  

        # Check if measurement is too far from the last position
        if self.last_position is not None:
            distance = np.linalg.norm(measurement - np.array(self.last_position))
            if distance > DISTANCE_THRESHOLD:
                self.reset_tracker(measurement)
                return

        self.last_position = measurement
        self.ball_positions.append(measurement)

        # distances between particles and the measurement
        distances = np.linalg.norm(self.particles - measurement, axis=1)

        # update weights based on distances
        self.weights = np.exp(-distances**2 / (2 * self.measurement_noise**2))
        self.weights += 1e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)  # Normalize weights

        # Resample particles according to weights
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]

        # Add noise to a subset of particles around the measurement
        num_measurement_particles = self.num_particles // 4
        measurement_particles = np.random.normal(measurement, self.measurement_noise / 2, size=(num_measurement_particles, 2))
        self.particles[-num_measurement_particles:] = measurement_particles

        # Reset weights to uniform distribution after resampling
        self.weights = np.ones(self.num_particles) / self.num_particles

    def reset_tracker(self, measurement):
        """
        Resets the tracker when a measurement exceeds the distance threshold, re-centering particles.
        
        Parameters:
            measurement (array-like): measurement (x, y) to re-center the tracker.
        """
        print(f"Resetting tracker {self.tracker_id}")
        self.ball_positions = [measurement]
        self.last_position = measurement
        spread = 20  
        self.particles = np.random.normal(measurement, spread, size=(self.num_particles, 2))
        self.weights = np.ones(self.num_particles) / self.num_particles  

    def estimate(self):
        """
        Estimates the current position of the tracked object.
        
        Returns:
            np.ndarray: estimated position (x, y) based on particle weights.
        """
        # Return last known position if available; otherwise, calculate weighted average
        if len(self.ball_positions) > 0:
            return np.array(self.ball_positions[-1]).astype(int)
        return np.average(self.particles, weights=self.weights, axis=0).astype(int)

    def draw_particles(self, frame):
        """
        Draws a subset of particles on the given frame.
        
        Parameters:
            frame (np.ndarray): The image frame to draw particles on.
        """
        subset_size = min(100, self.num_particles)
        particle_indices = np.random.choice(self.num_particles, subset_size, replace=False)
        for particle in self.particles[particle_indices]:
            pos = tuple(np.clip(particle, [0, 0], [self.frame_width - 1, self.frame_height - 1]).astype(int))
            cv2.circle(frame, pos, 1, self.color, -1)

    def draw_estimated_position(self, frame):
        """
        Draws the estimated position of the object on the given frame.
        
        Parameters:
            frame (np.ndarray): The image frame to draw the estimated position on.
        """
        if len(self.ball_positions) > 0:
            pos = tuple(np.clip(self.estimate(), [0, 0], [self.frame_width - 1, self.frame_height - 1]).astype(int))
            cv2.circle(frame, pos, 5, self.color, -1)

    def draw_trajectory(self, frame):
        """
        Draws the trajectory of the object based on its historical positions.
        
        Parameters:
            frame (np.ndarray): The image frame to draw the trajectory on.
        """
        if len(self.ball_positions) > 1:
            for i in range(1, len(self.ball_positions)):
                pt1 = tuple(np.clip(self.ball_positions[i-1], [0, 0], [self.frame_width - 1, self.frame_height - 1]).astype(int))
                pt2 = tuple(np.clip(self.ball_positions[i], [0, 0], [self.frame_width - 1, self.frame_height - 1]).astype(int))
                cv2.line(frame, pt1, pt2, self.color, 5)
