import numpy as np
import pygame
import cv2
from dataclasses import dataclass
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt

# Constants for pixel to meter conversion
PIXELS_PER_METER = 100
SCREEN_TO_WORLD_RATIO = 1 / PIXELS_PER_METER

def pixels_to_meters(pixels: float) -> float:
    return pixels * SCREEN_TO_WORLD_RATIO

def meters_to_pixels(meters: float) -> float:
    return meters / SCREEN_TO_WORLD_RATIO

@dataclass
class PhysicsParams:
    mass: float
    gravity: float
    drag: float


class ODESolver:
    @staticmethod
    def rk4_step(state: np.ndarray, dt: float, params: PhysicsParams) -> np.ndarray:
        k1 = ODESolver.derivatives(state, params)
        k2 = ODESolver.derivatives(state + 0.5 * dt * k1, params)
        k3 = ODESolver.derivatives(state + 0.5 * dt * k2, params)
        k4 = ODESolver.derivatives(state + dt * k3, params)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    @staticmethod
    def euler_step(state: np.ndarray, dt: float, params: PhysicsParams) -> np.ndarray:
        derivatives = ODESolver.derivatives(state, params)
        return state + dt * derivatives

    @staticmethod
    def adams_bashforth_step(states: list, dt: float, params: PhysicsParams) -> np.ndarray:
        if len(states) < 3:
            # Use RK4 for the first steps
            return ODESolver.rk4_step(states[-1], dt, params)

        # Adams-Bashforth 3-step method
        f_n = ODESolver.derivatives(states[-1], params)
        f_n1 = ODESolver.derivatives(states[-2], params)
        f_n2 = ODESolver.derivatives(states[-3], params)

        return states[-1] + (dt / 12) * (23 * f_n - 16 * f_n1 + 5 * f_n2)

    @staticmethod
    def trapezoidal_step(state: np.ndarray, dt: float, params: PhysicsParams) -> np.ndarray:
        # Predict with Euler (explicit)
        f_n = ODESolver.derivatives(state, params)
        predicted = state + dt * f_n

        # Correct with trapezoidal (implicit)
        f_np1 = ODESolver.derivatives(predicted, params)
        return state + 0.5 * dt * (f_n + f_np1)

    @staticmethod
    def derivatives(state: np.ndarray, params: PhysicsParams) -> np.ndarray:
        x, y, vx, vy = state
        ax = params.drag * vx / params.mass
        ay = params.gravity + params.drag * vy / params.mass
        return np.array([vx, vy, ax, ay])


class TrajectorySimulator:
    def __init__(self, params: PhysicsParams):
        self.params = params
        self.methods = {
            'RK4': ODESolver.rk4_step,
            'Euler': ODESolver.euler_step,
            'Trapezoidal': ODESolver.trapezoidal_step
        }

    def calculate_trajectory_cost(self, simulated_x: np.ndarray, simulated_y: np.ndarray,
                                  actual_x: np.ndarray, actual_y: np.ndarray) -> float:
        n_points = min(len(simulated_x), len(actual_x), 50)

        # Calculate ranges for normalization
        x_range = np.max(actual_x[:n_points]) - np.min(actual_x[:n_points])
        y_range = np.max(actual_y[:n_points]) - np.min(actual_y[:n_points])

        # Normalize errors by range
        mse_x = np.mean(((simulated_x[:n_points] - actual_x[:n_points]) / x_range) ** 2)
        mse_y = np.mean(((simulated_y[:n_points] - actual_y[:n_points]) / y_range) ** 2)

        total_cost = (mse_x + mse_y) / 2
        return total_cost

    def simulate_with_method(self, initial_state: np.ndarray, method: str,
                             dt: float = 0.01, num_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate trajectory using specified numerical method
        """
        if method not in self.methods and method != 'Adams-Bashforth':
            raise ValueError(f"Unknown method: {method}")

        if num_points is None:
            max_time = 5.0
            times = np.arange(0, max_time, dt)
        else:
            times = np.arange(num_points) * dt

        positions_x = []
        positions_y = []
        state = initial_state.copy()

        # For Adams-Bashforth
        states = [state.copy()]

        for _ in times:
            positions_x.append(state[0])
            positions_y.append(state[1])

            if len(positions_x) == num_points:
                break

            if method == 'Adams-Bashforth':
                state = ODESolver.adams_bashforth_step(states, dt, self.params)
                states.append(state.copy())
                if len(states) > 3:
                    states.pop(0)
            else:
                state = self.methods[method](state, dt, self.params)

            if state[1] < 0 or state[1] > 10:
                if num_points is not None:
                    while len(positions_x) < num_points:
                        positions_x.append(positions_x[-1])
                        positions_y.append(positions_y[-1])
                break

        return np.array(positions_x), np.array(positions_y)

    def plot_method_comparison(self, initial_state: np.ndarray, actual_x: np.ndarray,
                               actual_y: np.ndarray, num_points: int = 20):
        """
        Plot comparison of different numerical methods against actual trajectory
        """
        plt.figure(figsize=(15, 10))

        # Plot actual trajectory
        plt.plot(actual_x[:num_points], actual_y[:num_points], 'ko-',
                 label='Actual', linewidth=2, markersize=8)

        # Colors for different methods
        colors = {'RK4': 'b', 'Euler': 'r', 'Adams-Bashforth': 'g', 'Trapezoidal': 'm'}

        # Simulate and plot each method
        costs = {}
        for method in ['RK4', 'Euler', 'Adams-Bashforth', 'Trapezoidal']:
            sim_x, sim_y = self.simulate_with_method(
                initial_state, method, dt=0.01, num_points=num_points
            )

            # Calculate MSE
            cost = self.calculate_trajectory_cost(sim_x, sim_y, actual_x, actual_y)
            costs[method] = cost

            plt.plot(sim_x, sim_y, f'{colors[method]}o--',
                     label=f'{method} (MSE: {cost:.6f})', linewidth=2, markersize=6)

        plt.title('Numerical Methods Comparison\nFirst 20 Frames', fontsize=14, pad=20)
        plt.xlabel('X Position (meters)', fontsize=12)
        plt.ylabel('Y Position (meters)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        # Add physics parameters as text
        params_text = f'Physics Parameters:\n' \
                      f'Mass: {self.params.mass:.3f} kg\n' \
                      f'Gravity: {self.params.gravity:.3f} m/s²\n' \
                      f'Drag: {self.params.drag:.6f}'
        plt.text(0.02, 0.98, params_text, transform=plt.gca().transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        return costs

    def simulate(self, initial_state: np.ndarray, dt: float = 0.01,
                 num_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Original simulate method using RK4 for the actual simulation"""
        return self.simulate_with_method(initial_state, 'RK4', dt, num_points)


class ParameterEstimator:
    def __init__(self, positions_x: np.ndarray, positions_y: np.ndarray, times: np.ndarray):
        """
        positions_x: array of x coordinates in meters
        positions_y: array of y coordinates in meters
        times: array of timestamps
        """
        self.positions_x = positions_x
        self.positions_y = positions_y
        self.times = times
        self.num_points = len(positions_x)

    def estimate_parameters_and_velocities(self) -> Tuple[PhysicsParams, np.ndarray, np.ndarray]:
        """
        Unified estimation of physics parameters and velocities using Newton's shooting method.
        Returns:
            - PhysicsParams object containing mass, gravity, and drag coefficients
            - Arrays of x and y velocities for all time points
        """

        def shooting_function(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Returns the residual error and Jacobian for the combined parameter vector
            params: [mass, gravity, drag, vx_0, vy_0, vx_1, vy_1, ...]
            """
            # Extract parameters
            mass, gravity, drag = params[:3]
            velocities = params[3:].reshape(-1, 2)
            physics_params = PhysicsParams(mass, gravity, drag)
            simulator = TrajectorySimulator(physics_params)

            # Calculate residuals
            position_residuals = []
            physics_residuals = []

            # Position and velocity residuals
            for i in range(len(self.times) - 1):
                dt = self.times[i + 1] - self.times[i]
                initial_state = np.array([
                    self.positions_x[i],
                    self.positions_y[i],
                    velocities[i, 0],
                    velocities[i, 1]
                ])

                # Simulate trajectory for this segment
                sim_x, sim_y = simulator.simulate(initial_state, dt, num_points=2)

                # Position residuals
                pos_residuals = np.array([
                    sim_x[-1] - self.positions_x[i + 1],
                    sim_y[-1] - self.positions_y[i + 1]
                ])
                position_residuals.extend(pos_residuals)

                # Physics constraint residuals
                if i < len(self.times) - 2:
                    # Predicted velocity using physics equations
                    speed = np.sqrt(velocities[i, 0] ** 2 + velocities[i, 1] ** 2)
                    v_next_predicted = velocities[i] + dt * np.array([
                        -drag / mass * velocities[i, 0] * speed,
                        -gravity - drag / mass * velocities[i, 1] * speed
                    ])
                    physics_residuals.extend(velocities[i + 1] - v_next_predicted)

            # Combine residuals with weights
            position_weight = 1.0
            physics_weight = 0.1

            residuals = np.concatenate([
                position_weight * np.array(position_residuals),
                physics_weight * np.array(physics_residuals)
            ])

            # Calculate Jacobian numerically
            J = np.zeros((len(residuals), len(params)))
            eps = 1e-6

            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += eps

                # Recalculate residuals with perturbed parameter
                mass_p, gravity_p, drag_p = params_plus[:3]
                velocities_p = params_plus[3:].reshape(-1, 2)
                physics_params_p = PhysicsParams(mass_p, gravity_p, drag_p)
                simulator_p = TrajectorySimulator(physics_params_p)

                pos_residuals_p = []
                phys_residuals_p = []

                for j in range(len(self.times) - 1):
                    dt = self.times[j + 1] - self.times[j]
                    state_p = np.array([
                        self.positions_x[j],
                        self.positions_y[j],
                        velocities_p[j, 0],
                        velocities_p[j, 1]
                    ])

                    sim_x_p, sim_y_p = simulator_p.simulate(state_p, dt, num_points=2)
                    pos_residuals_p.extend([
                        sim_x_p[-1] - self.positions_x[j + 1],
                        sim_y_p[-1] - self.positions_y[j + 1]
                    ])

                    if j < len(self.times) - 2:
                        speed_p = np.sqrt(velocities_p[j, 0] ** 2 + velocities_p[j, 1] ** 2)
                        v_next_p = velocities_p[j] + dt * np.array([
                            -drag_p / mass_p * velocities_p[j, 0] * speed_p,
                            -gravity_p - drag_p / mass_p * velocities_p[j, 1] * speed_p
                        ])
                        phys_residuals_p.extend(velocities_p[j + 1] - v_next_p)

                residuals_p = np.concatenate([
                    position_weight * np.array(pos_residuals_p),
                    physics_weight * np.array(phys_residuals_p)
                ])

                J[:, i] = (residuals_p - residuals) / eps

            return residuals, J

        # Initial guess
        initial_velocities = np.zeros((self.num_points, 2))
        for i in range(self.num_points - 1):
            dt = self.times[i + 1] - self.times[i]
            initial_velocities[i] = [
                (self.positions_x[i + 1] - self.positions_x[i]) / dt,
                (self.positions_y[i + 1] - self.positions_y[i]) / dt
            ]
        initial_velocities[-1] = initial_velocities[-2]

        params = np.concatenate([
            [1.0, 9.81, 0.1],  # Initial physics parameters
            initial_velocities.flatten()  # Initial velocities
        ])

        # Newton's method iteration
        max_iterations = 10
        tolerance = 1e-6
        lambda_reg = 1e-6  # Regularization parameter

        for iteration in range(max_iterations):
            residuals, J = shooting_function(params)

            if np.linalg.norm(residuals) < tolerance:
                break

            # Solve Newton's step with regularization
            delta = np.linalg.solve(
                J.T @ J + lambda_reg * np.eye(len(params)),
                -J.T @ residuals
            )

            # Update parameters with damping if needed
            step_size = 1.0
            while step_size > 1e-10:
                new_params = params + step_size * delta

                # Apply bounds
                new_params[0] = np.clip(new_params[0], 0.1, 10.0)  # mass
                new_params[1] = np.clip(new_params[1], 8.0, 12.0)  # gravity
                new_params[2] = np.clip(new_params[2], 0.01, 1.0)  # drag

                # Check if update reduces residual
                new_residuals, _ = shooting_function(new_params)
                if np.linalg.norm(new_residuals) < np.linalg.norm(residuals):
                    params = new_params
                    break

                step_size *= 0.5

        # Extract final parameters
        mass, gravity, drag = params[:3]
        velocities = params[3:].reshape(-1, 2)

        return PhysicsParams(mass, gravity, drag), velocities[:, 0], velocities[:, 1]

class BallInterceptor:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()

        # Add animation parameters
        self.ball_radius = 10
        self.interceptor_radius = 8
        self.animation_speed = 0.5  # Reduced animation speed
        self.current_time = 0

        # Add targeting parameters
        self.target_radius = 10  # Radius of target area in pixels
        self.max_speed = 17.0  # Maximum speed in m/s

        self.frame_20_time = None  # Will store time at frame 20
        self.frame_20_x = None  # Will store x position at frame 20

    def process_video(self, video_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract ball positions and timestamps from video"""

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        positions_x = []
        positions_y = []
        times = []
        frame_count = 0

        while frame_count < 20:  # Process first 20 frames
            ret, frame = cap.read()
            if not ret:
                break

            pos = self._detect_ball(frame)
            if pos is not None:
                x, y = pos
                # Convert to meters immediately
                positions_x.append(pixels_to_meters(x))
                positions_y.append(pixels_to_meters(y))
                times.append(frame_count / fps)

                # Store position at frame 20
                if frame_count == 19:
                    self.frame_20_time = times[-1]
                    self.frame_20_x = positions_x[-1]

            frame_count += 1

        cap.release()

        if not positions_x:
            raise ValueError("No ball detected in video")

        return np.array(positions_x), np.array(positions_y), np.array(times)

    def _detect_ball(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detect ball position using Canny edge detection and DBSCAN clustering.

        Args:
            frame: Input frame in BGR format

        Returns:
            Tuple of (x, y) coordinates if ball is detected, None otherwise
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply light Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 30, 150)

        # Find non-zero points (edge pixels)
        points = np.column_stack(np.where(edges > 0))

        if len(points) == 0:
            return None



        # Normalize points for DBSCAN
        points = points.astype(np.float32)

        # Apply DBSCAN clustering
        from sklearn.cluster import DBSCAN

        # Parameters for DBSCAN:
        # eps: maximum distance between points in a cluster (adjust based on ball size)
        # min_samples: minimum points to form a cluster
        clustering = DBSCAN(eps=20, min_samples=1).fit(points)

        if len(set(clustering.labels_)) <= 0:  # No clusters found or only noise
            return None

        # Find the largest cluster (excluding noise which is labeled as -1)
        labels = clustering.labels_
        unique_labels = set(labels)
        max_cluster_size = 0
        max_cluster_points = None

        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            cluster_points = points[labels == label]
            if len(cluster_points) > max_cluster_size:
                max_cluster_size = len(cluster_points)
                max_cluster_points = cluster_points

        if max_cluster_points is None:
            return None

        # Calculate centroid of largest cluster
        center_y, center_x = np.mean(max_cluster_points, axis=0)

        return (center_x, center_y)

    def calculate_intercept_velocity(self,
                                     start_pos: np.ndarray,
                                     target_pos: np.ndarray,
                                     target_time: float,
                                     params: PhysicsParams) -> np.ndarray:
        """
        Calculate initial velocity needed to intercept target using Newton's shooting method.

        Args:
            start_pos: Starting position [x, y] in meters
            target_pos: Target position [x, y] in meters
            target_time: Time at which to intercept target in seconds
            params: Physics parameters

        Returns:
            Initial velocity vector [vx, vy] in m/s
        """

        def shooting_function(v0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Function that returns the final position error and its Jacobian
            given an initial velocity.
            """
            # Simulate trajectory with current initial velocity
            initial_state = np.array([start_pos[0], start_pos[1], v0[0], v0[1]])
            simulator = TrajectorySimulator(params)

            pos_x, pos_y = simulator.simulate(
                initial_state,
                dt=0.01,
                num_points=int(target_time * 100) + 1
            )

            # Get final position
            final_pos = np.array([pos_x[-1], pos_y[-1]])
            error = final_pos - target_pos

            # Calculate Jacobian numerically
            eps = 1e-6
            J = np.zeros((2, 2))

            for i in range(2):
                v0_plus = v0.copy()
                v0_plus[i] += eps




                initial_state_plus = np.array([start_pos[0], start_pos[1], v0_plus[0], v0_plus[1]])
                pos_x_plus, pos_y_plus = simulator.simulate(
                    initial_state_plus,
                    dt=0.01,
                    num_points=int(target_time * 100) + 1
                )
                final_pos_plus = np.array([pos_x_plus[-1], pos_y_plus[-1]])

                # Finite difference approximation of Jacobian
                J[:, i] = (final_pos_plus - final_pos) / eps

            return error, J

        # Initial guess: direct path with constant velocity
        displacement = target_pos - start_pos
        v0 = displacement / target_time

        # Apply initial speed constraint if needed
        speed = np.linalg.norm(v0)
        if speed > self.max_speed:
            v0 *= (self.max_speed / speed)

        # Newton's shooting method iteration
        max_iterations = 10
        tolerance = 0.01  # 1cm accuracy

        for _ in range(max_iterations):
            error, J = shooting_function(v0)

            if np.linalg.norm(error) < tolerance:
                break

            # Solve Newton's step with regularization to handle potential singular matrices
            lambda_reg = 1e-6
            delta_v = np.linalg.solve(
                J.T @ J + lambda_reg * np.eye(2),
                -J.T @ error
            )

            # Update velocity
            v0 += delta_v

            # Apply speed constraint
            speed = np.linalg.norm(v0)
            if speed > self.max_speed:
                v0 *= (self.max_speed / speed)

        return v0



    def setup_phase(self, ball_x: np.ndarray, ball_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Setup phase to let user choose launch position and target point"""
        running = True
        launch_pos = None
        target_point = None

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None, None, None

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if launch_pos is None:
                        # First click: set launch position
                        launch_pos = np.array([
                            pixels_to_meters(event.pos[0]),
                            pixels_to_meters(event.pos[1])
                        ])
                    elif target_point is None:
                        # Second click: set target point and find closest point on trajectory
                        mouse_pos = np.array([
                            pixels_to_meters(event.pos[0]),
                            pixels_to_meters(event.pos[1])
                        ])

                        # Find closest point on ball trajectory
                        distances = np.sqrt(
                            (ball_x - mouse_pos[0]) ** 2 +
                            (ball_y - mouse_pos[1]) ** 2
                        )
                        target_idx = np.argmin(distances)
                        target_point = np.array([ball_x[target_idx], ball_y[target_idx]])
                        target_time = target_idx * 0.01  # Based on simulation dt
                        running = False

            # Drawing
            self.screen.fill((0, 0, 0))

            # Draw ball trajectory
            ball_pixels = [
                (int(meters_to_pixels(x)), int(meters_to_pixels(y)))
                for x, y in zip(ball_x, ball_y)
            ]
            pygame.draw.lines(self.screen, (255, 0, 0), False, ball_pixels, 1)

            # Draw launch position if set
            if launch_pos is not None:
                launch_pixel_pos = (
                    int(meters_to_pixels(launch_pos[0])),
                    int(meters_to_pixels(launch_pos[1]))
                )
                pygame.draw.circle(self.screen, (0, 255, 0), launch_pixel_pos, self.interceptor_radius)

                # Show instruction text
                font = pygame.font.Font(None, 36)
                if target_point is None:
                    text = font.render("Click on ball trajectory to select target point", True, (255, 255, 255))
                    self.screen.blit(text, (10, 10))

            else:
                # Show instruction text
                font = pygame.font.Font(None, 36)
                text = font.render("Click to set launch position", True, (255, 255, 255))
                self.screen.blit(text, (10, 10))

            pygame.display.flip()
            self.clock.tick(60)

        return launch_pos, target_point, target_time

    def run_simulation(self, video_path: str):
        """Main simulation loop with setup phase"""
        # Process video and estimate parameters
        positions_x, positions_y, times = self.process_video(video_path)
        estimator = ParameterEstimator(positions_x, positions_y, times)
        params, vx, vy = estimator.estimate_parameters_and_velocities()

        print(f"Estimated parameters:")
        print(f"Mass: {params.mass:.3f} kg")
        print(f"Gravity: {params.gravity:.3f} m/s²")
        print(f"Drag coefficient: {params.drag:.6f}")
        print(f"Initial velocity: ({vx[0]:.2f}, {vy[0]:.2f}) m/s")

        # Set up simulation with estimated parameters and velocities
        simulator = TrajectorySimulator(params)
        initial_state = np.array([
            positions_x[0],
            positions_y[0],
            vx[0],
            vy[0]
        ])

        # [Rest of the simulation code remains the same]
        # Simulate ball trajectory
        ball_x, ball_y = simulator.simulate(initial_state)
        cost = simulator.calculate_trajectory_cost(ball_x, ball_y, positions_x, positions_y)
        print(f"\nTrajectory Cost (MSE) for first 20 frames: {cost:.6f} m²")

        # Show method comparison
        costs = simulator.plot_method_comparison(initial_state, positions_x, positions_y)
        print("\nMSE Costs by Method:")
        for method, cost in costs.items():
            print(f"{method}: {cost:.6f} m²")

        # Run setup phase
        launch_pos, target_point, target_time = self.setup_phase(ball_x, ball_y)
        if launch_pos is None:  # User closed window
            return

        # Calculate interceptor trajectory
        v0 = self.calculate_intercept_velocity(
            launch_pos, target_point, target_time, params
        )

        interceptor_state = np.array([
            launch_pos[0], launch_pos[1], v0[0], v0[1]
        ])
        interceptor_x, interceptor_y = simulator.simulate(interceptor_state)

        # Main simulation loop
        running = True
        self.current_time = 0
        ball_index = 0
        interceptor_index = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update animation
            self.current_time += self.clock.get_time() / 1000.0 * self.animation_speed

            # Update positions
            ball_index = min(int(self.current_time * 100), len(ball_x) - 1)
            interceptor_index = min(int(self.current_time * 100), len(interceptor_x) - 1)

            # Drawing
            self.screen.fill((0, 0, 0))

            # Draw trajectories
            ball_pixels = [(int(meters_to_pixels(x)), int(meters_to_pixels(y)))
                           for x, y in zip(ball_x, ball_y)]
            interceptor_pixels = [(int(meters_to_pixels(x)), int(meters_to_pixels(y)))
                                  for x, y in zip(interceptor_x, interceptor_y)]

            pygame.draw.lines(self.screen, (255, 0, 0), False, ball_pixels, 1)
            pygame.draw.lines(self.screen, (0, 255, 0), False, interceptor_pixels, 1)

            # Draw target area
            target_pixel_pos = (
                int(meters_to_pixels(target_point[0])),
                int(meters_to_pixels(target_point[1]))
            )
            pygame.draw.circle(self.screen, (255, 255, 0),
                               target_pixel_pos, self.target_radius, 1)

            # Draw current positions
            current_ball_pos = (
                int(meters_to_pixels(ball_x[ball_index])),
                int(meters_to_pixels(ball_y[ball_index]))
            )
            current_interceptor_pos = (
                int(meters_to_pixels(interceptor_x[interceptor_index])),
                int(meters_to_pixels(interceptor_y[interceptor_index]))
            )

            pygame.draw.circle(self.screen, (255, 0, 0), current_ball_pos, self.ball_radius)
            pygame.draw.circle(self.screen, (0, 255, 0), current_interceptor_pos, self.interceptor_radius)

            # Check for collision
            distance = np.hypot(
                ball_x[ball_index] - interceptor_x[interceptor_index],
                ball_y[ball_index] - interceptor_y[interceptor_index]
            )
            if distance < pixels_to_meters(self.ball_radius + self.interceptor_radius):
                collision_pos = current_ball_pos
                pygame.draw.circle(self.screen, (255, 255, 0), collision_pos, self.ball_radius * 2)

            # Reset animation if it reaches the end
            if ball_index >= len(ball_x) - 1:
                self.current_time = 0
                ball_index = 0
                interceptor_index = 0

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()



if __name__ == "__main__":
    game = BallInterceptor(1200, 800)
    # game.run_simulation("../CP2/project_video.mp4") #works great  100 pixels per meter
    # game.run_simulation("cut (1).mp4")  # works  200 ppr
    # game.run_simulation("../CP2/cut2.mp4")    #doesnt works, shound not   work  300 ppr
    # game.run_simulation("slow_throw_and_fall.mp4")  # 50 ppr
    game.run_simulation("fast_throw_and_fall.mp4") # doesnt work should not work




