import numpy as np
import cv2
size = 800

DISTANCE_THRESHOLD = 200

class ParticleFilterBallTracker:
    def __init__(self, tracker_id, color, num_particles=1000, process_noise=5.0, measurement_noise=2.0):
        self.tracker_id = tracker_id
        self.color = color
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.particles = np.random.rand(self.num_particles, 2) * size
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.ball_positions = []
        self.last_position = None
        self.active = True
        self.frames_without_detection = 0
        
    def predict(self):
        if len(self.ball_positions) >= 2:
            # Calculate velocity from last two positions
            velocity = np.array(self.ball_positions[-1]) - np.array(self.ball_positions[-2])
            self.particles += velocity
        
        # Add noise to particles
        self.particles += np.random.randn(self.num_particles, 2) * self.process_noise
        self.particles = np.clip(self.particles, 0, size-1)

    def update(self, measurement):
        if measurement == (-1, -1):
            self.frames_without_detection += 1
            if self.frames_without_detection > 10:  # Deactivate tracker if no detections for too long
                self.active = False
            return

        self.frames_without_detection = 0
        
        if self.last_position is not None:
            distance = np.linalg.norm(np.array(measurement) - np.array(self.last_position))
            if distance > DISTANCE_THRESHOLD:
                self.reset_tracker(measurement)
                return
                
        self.last_position = measurement
        self.ball_positions.append(measurement)
        
        # Calculate weights based on distance to measurement
        distances = np.linalg.norm(self.particles - np.array(measurement), axis=1)
        self.weights = np.exp(-distances**2 / (2 * self.measurement_noise**2))
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)
        
        # Resample particles
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        
        # Add some particles near the measurement
        num_measurement_particles = self.num_particles // 4  # 25% of particles
        measurement_particles = np.random.normal(measurement, self.measurement_noise/2, 
                                               size=(num_measurement_particles, 2))
        self.particles[-num_measurement_particles:] = measurement_particles
        
        # Reset weights
        self.weights = np.ones(self.num_particles) / self.num_particles

    def reset_tracker(self, measurement):
        print(f"Resetting tracker {self.tracker_id}")
        self.ball_positions = [measurement]
        self.last_position = measurement
        spread = 20  # Initial spread around the measurement
        self.particles = np.random.normal(measurement, spread, size=(self.num_particles, 2))
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def estimate(self):
        if len(self.ball_positions) > 0:
            return np.array(self.ball_positions[-1]).astype(int)
        return np.average(self.particles, weights=self.weights, axis=0).astype(int)

    def draw_particles(self, frame):
        # Draw only a subset of particles for better visualization
        subset_size = min(100, self.num_particles)
        particle_indices = np.random.choice(self.num_particles, subset_size, replace=False)
        for particle in self.particles[particle_indices]:
            pos = tuple(np.clip(particle, 0, size-1).astype(int))
            cv2.circle(frame, pos, 1, self.color, -1)

    def draw_estimated_position(self, frame):
        if len(self.ball_positions) > 0:
            pos = tuple(np.clip(self.estimate(), 0, size-1).astype(int))
            cv2.circle(frame, pos, 5, self.color, -1)

    def draw_trajectory(self, frame):
        if len(self.ball_positions) > 1:
            for i in range(1, len(self.ball_positions)):
                pt1 = tuple(np.clip(self.ball_positions[i-1], 0, size-1).astype(int))
                pt2 = tuple(np.clip(self.ball_positions[i], 0, size-1).astype(int))
                cv2.line(frame, pt1, pt2, self.color, 2)
