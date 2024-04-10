# Copyright (c) 2024 Chikezie Wood. All rights reserved.

import numpy as np
from scipy.stats import norm

# Step 1: Read the data from likelihood.txt
def read_likelihood(file_path):
    likelihoods = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            likelihood_data = line.strip().split()
            if i == 0:
                object_class = 'bird'
            else:
                object_class = 'airplane'
            likelihoods[object_class] = {'velocity': {}}
            for j, value in enumerate(likelihood_data):
                # Calculate velocity value based on index (0-399) with 0.5 interval
                velocity = j * 0.5
                likelihoods[object_class]['velocity'][velocity] = {'mean': float(value)}
    return likelihoods

# Step 2: Read the data from dataset.txt
def read_dataset(file_path):
    tracks = {'bird': [], 'airplane': []}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            # Determine object class based on row index
            object_class = 'bird' if i < 10 else 'airplane'
            # Split the line and convert non-NaN values to floats
            velocities = [float(x) if x != 'NaN\n' else np.nan for x in line.strip().split()]
            tracks[object_class].append(velocities)
    return tracks

def calculate_acceleration(tracks):
    """
    Calculate acceleration values for each track in the dataset.

    Args:
    - tracks (dict): Dictionary containing velocity tracks for birds and airplanes.

    Returns:
    - dict: Dictionary containing acceleration tracks for birds and airplanes.
    """
    accelerations = {'bird': [], 'airplane': []}
    for object_class, velocity_tracks in tracks.items():
        for track in velocity_tracks:
            acceleration_track = []
            # Filter out NaN values
            valid_values = [value for value in track if not np.isnan(value)]
            for i in range(1, len(valid_values)):
                # Calculate acceleration as the difference between adjacent velocity values
                velocity_diff = valid_values[i] - valid_values[i-1]
                acceleration_track.append(velocity_diff)
            accelerations[object_class].append(acceleration_track)
    return accelerations

def calculate_average_acceleration(accelerations):
    """
    Calculate average acceleration values for birds and airplanes.

    Args:
    - accelerations (dict): Dictionary containing acceleration tracks for birds and airplanes.

    Returns:
    - tuple: Average acceleration for birds and airplanes.
    """
    bird_accelerations = []
    airplane_accelerations = []

    for object_class, acceleration_tracks in accelerations.items():
        for track in acceleration_tracks:
            if object_class == 'bird':
                bird_accelerations.extend(track)
            else:
                airplane_accelerations.extend(track)

    bird_mean_acceleration = np.mean(bird_accelerations)
    airplane_mean_acceleration = np.mean(airplane_accelerations)

    return bird_mean_acceleration, airplane_mean_acceleration

def calculate_acceleration_std(accelerations):
    """
    Calculate standard deviation of acceleration values for birds and airplanes.

    Args:
    - accelerations (dict): Dictionary containing acceleration tracks for birds and airplanes.

    Returns:
    - tuple: Standard deviation of acceleration for birds and airplanes.
    """
    bird_accelerations = []
    airplane_accelerations = []

    for object_class, acceleration_tracks in accelerations.items():
        for track in acceleration_tracks:
            # Calculate differences only if there are at least two valid values
            if len(track) > 1:
                if object_class == 'bird':
                    bird_accelerations.extend(track)
                else:
                    airplane_accelerations.extend(track)

    bird_acceleration_std = np.std(bird_accelerations)
    airplane_acceleration_std = np.std(airplane_accelerations)

    return bird_acceleration_std, airplane_acceleration_std



# calculate the probability of a given acceleration value being a bird or an airplane
def calculate_acceleration_likelihood(acceleration, bird_acceleration_std, airplane_acceleration_std):
    bird_likelihood = norm.pdf(acceleration, loc=0, scale=bird_acceleration_std)
    airplane_likelihood = norm.pdf(acceleration, loc=0, scale=airplane_acceleration_std)

    # Compare the likelihoods and pick the larger one
    if bird_likelihood > airplane_likelihood:
        return 'b'  # Bird
    elif airplane_likelihood > bird_likelihood:
        return 'a'  # Airplane
    else:
        return 'x'  # Inconclusive

class ObjectClassifier:
    def __init__(self, likelihoods, bird_avg_accel, airplane_avg_accel, bird_acceleration_std, airplane_acceleration_std, file_path):
        self.likelihoods = likelihoods
        self.bird_avg_accel = bird_avg_accel
        self.airplane_avg_accel = airplane_avg_accel
        self.bird_acceleration_std = bird_acceleration_std
        self.airplane_acceleration_std = airplane_acceleration_std
        self.file_path = file_path
        self.velocities = self._read_velocities()
        self.accelerations = self._calculate_accelerations()

    def _read_velocities(self):
        velocities = []
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                velocities.append([float(x) if x != 'NaN\n' else np.nan for x in line.strip().split()])
        return velocities
    
    def _calculate_accelerations(self):
        accelerations = []
        for velocity_track in self.velocities:
            acceleration_track = []
            for i in range(1, len(velocity_track)):
                # Calculate acceleration as the difference between adjacent velocity values
                velocity_diff = velocity_track[i] - velocity_track[i-1]
                acceleration_track.append(velocity_diff)
            accelerations.append(acceleration_track)
        return accelerations

    def _calculate_average_velocity(self):
        average_velocities = []
        for track in self.velocities:
            valid_velocities = [v for v in track if not np.isnan(v)]
            average_velocities.append(np.mean(valid_velocities))
        return average_velocities

    def _find_extremes(self, dataset):
        extremes = []
        for data in dataset:
            valid_data = [d for d in data if not np.isnan(d)]
            if valid_data:  # Check if the dataset is not empty
                lowest = min(valid_data)
                highest = max(valid_data)
                extremes.append((lowest, highest))
            else:
                extremes.append((None, None))
        return extremes
    
    def _calculate_likelihood(self, value, mean, std):
        # Calculate likelihood using normal distribution
        if not np.isnan(mean) and not np.isnan(std) and not np.isnan(value):
            likelihood = np.exp(-((value - mean) ** 2) / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))
            return round (likelihood, 8)
        else:
            return None

    def classify_objects(self):
        average_velocities = self._calculate_average_velocity()
        velocity_outliers = self._find_extremes(self.velocities)
        acceleration_outliers = self._find_extremes(self.accelerations)

        classifications = []
        objects_raw_data = []
        for i in range(len(self.velocities)):
            average_velocity = round(average_velocities[i]*2)/2
            min_velocity = round(velocity_outliers[i][0]*2)/2
            max_velocity = round(velocity_outliers[i][1]*2)/2
            min_acceleration = round(acceleration_outliers[i][0]*2)/2
            max_acceleration = round(acceleration_outliers[i][1]*2)/2

            # Step 1: Find corresponding likelihood values for average velocity
            bird_likelihood_avg_vel = self.likelihoods['bird']['velocity'].get(average_velocity, None).get('mean', None)
            airplane_likelihood_avg_vel = self.likelihoods['airplane']['velocity'].get(average_velocity, None).get('mean', None)

            # Step 2: Find corresponding likelihood values for velocity outliers
            bird_likelihood_min_vel = self.likelihoods['bird']['velocity'].get(min_velocity, None).get('mean', None)
            bird_likelihood_max_vel = self.likelihoods['bird']['velocity'].get(max_velocity, None).get('mean', None)
            airplane_likelihood_min_vel = self.likelihoods['airplane']['velocity'].get(min_velocity, None).get('mean', None)
            airplane_likelihood_max_vel = self.likelihoods['airplane']['velocity'].get(max_velocity, None).get('mean', None)

            # Step 3: Compare acceleration min and max to std values
            bird_min_acceleration_likelihood = self._calculate_likelihood(min_acceleration, 
                                                                          self.bird_avg_accel, 
                                                                          self.bird_acceleration_std)
            bird_max_acceleration_likelihood = self._calculate_likelihood(max_acceleration, 
                                                                          self.bird_avg_accel, 
                                                                          self.bird_acceleration_std)
            airplane_min_acceleration_likelihood = self._calculate_likelihood(min_acceleration, 
                                                                              self.airplane_avg_accel, 
                                                                              self.airplane_acceleration_std)
            airplane_max_acceleration_likelihood = self._calculate_likelihood(max_acceleration, 
                                                                              self.airplane_avg_accel, 
                                                                              self.airplane_acceleration_std)

            # Step 4: Store the classification information
            object_data = {
                'average_velocity': average_velocity,
                'min_velocity': min_velocity,
                'max_velocity': max_velocity,
                'min_acceleration': min_acceleration,
                'max_acceleration': max_acceleration,
                'bird_likelihood_avg_vel': bird_likelihood_avg_vel,
                'airplane_likelihood_avg_vel': airplane_likelihood_avg_vel,
                'bird_likelihood_min_vel': bird_likelihood_min_vel,
                'bird_likelihood_max_vel': bird_likelihood_max_vel,
                'airplane_likelihood_min_vel': airplane_likelihood_min_vel, 
                'airplane_likelihood_max_vel': airplane_likelihood_max_vel,
                'bird_min_acceleration_likelihood': bird_min_acceleration_likelihood,
                'bird_max_acceleration_likelihood': bird_max_acceleration_likelihood,
                'airplane_min_acceleration_likelihood': airplane_min_acceleration_likelihood,
                'airplane_max_acceleration_likelihood': airplane_max_acceleration_likelihood,
            }
            objects_raw_data.append(object_data)

            # Comparisons for classification
            count_a = 0
            count_b = 0
            # average velocity comparison
            if bird_likelihood_avg_vel > airplane_likelihood_avg_vel:
                count_b += 1    
            else: count_a += 1
            # min velocity comparison
            if bird_likelihood_min_vel > airplane_likelihood_min_vel:
                count_b += 1    
            else: count_a += 1
            # max velocity comparison
            if bird_likelihood_max_vel > airplane_likelihood_max_vel:
                count_b += 1    
            else: count_a += 1
            # min acceleration likelihood comparison
            if bird_min_acceleration_likelihood > airplane_min_acceleration_likelihood:
                count_b += 1    
            else: count_a += 1
            # max acceleration likelihood comparison
            if bird_max_acceleration_likelihood > airplane_max_acceleration_likelihood:
                count_b += 1    
            else: count_a += 1

            if count_a > count_b:
                classifications.append("a")
            else: classifications.append("b")

        
        return classifications

        
if __name__ == "__main__":
    likelihoods = read_likelihood('/Users/chikezie/Documents/Tufts/CS 131/P5/likelihood.txt')
    tracks = read_dataset('/Users/chikezie/Documents/Tufts/CS 131/P5/dataset.txt')

    accelerations = calculate_acceleration(tracks)
    bird_avg_accel, airplane_avg_accel = calculate_average_acceleration(accelerations)
    bird_acceleration_std, airplane_acceleration_std = calculate_acceleration_std(accelerations)

    classifier = ObjectClassifier(likelihoods, 
                                  bird_avg_accel, 
                                  airplane_avg_accel, 
                                  bird_acceleration_std, 
                                  airplane_acceleration_std, 
                                  '/Users/chikezie/Documents/Tufts/CS 131/P5/testing.txt')
    classifications = classifier.classify_objects()
    print("Final classifications for each track:")
    print(classifications)

