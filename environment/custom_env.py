"""
Dermatology Clinic Triage Environment
Custom Gymnasium environment for reinforcement learning triage optimization.

Author: [Your Name]
Date: 2025-11-21
"""

import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional


class ClinicEnv(gym.Env):
    """
    Custom Gymnasium environment for dermatology clinic triage.

    The agent manages patient triage and resource allocation in a busy clinic.
    Patients arrive with varying severity levels, and the agent must:
    - Correctly triage patients to appropriate care levels
    - Manage room resources
    - Minimize wait times
    - Maximize correct diagnoses

    Observation Space (15 dimensions):
        [0] age_norm: Normalized patient age (0.0-1.0)
        [1] duration_norm: Symptom duration normalized (0.0-1.0)
        [2] fever_flag: Binary fever indicator (0.0 or 1.0)
        [3] infection_flag: Binary infection indicator (0.0 or 1.0)
        [4-11] symptom_embed: 8-dimensional symptom embedding (0.0-1.0 each)
        [12] room_avail: Room availability flag (0.0 or 1.0)
        [13] queue_len_norm: Normalized queue length (0.0-1.0)
        [14] time_of_day_norm: Normalized episode progress (0.0-1.0)

    Action Space (8 discrete actions):
        0: send_doctor - Send patient to dermatologist
        1: send_nurse - Send patient to nurse practitioner
        2: remote_advice - Provide telemedicine consultation
        3: escalate_priority - Mark as urgent (doctor + priority)
        4: defer_patient - Postpone to end of queue
        5: idle - No action (wait)
        6: open_room - Open additional exam room
        7: close_room - Close an exam room

    Reward Structure:
        Correct triage rewards:
            - Mild → remote_advice: +1.0
            - Moderate → nurse: +1.0
            - Severe → doctor: +2.0
            - Critical → escalate (fast): +3.0
            - Critical → escalate (slow): +2.0

        Penalties:
            - Incorrect triage: -1.5
            - Wait time: -0.01 × queue_size per step
            - Resource cost: -0.05 × num_open_rooms per step

    Episode Termination:
        - After max_steps timesteps (default 500)
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 6
    }

    # Severity level definitions
    SEVERITY_MILD = 0
    SEVERITY_MODERATE = 1
    SEVERITY_SEVERE = 2
    SEVERITY_CRITICAL = 3

    # Action definitions
    ACTION_SEND_DOCTOR = 0
    ACTION_SEND_NURSE = 1
    ACTION_REMOTE_ADVICE = 2
    ACTION_ESCALATE = 3
    ACTION_DEFER = 4
    ACTION_IDLE = 5
    ACTION_OPEN_ROOM = 6
    ACTION_CLOSE_ROOM = 7

    ACTION_NAMES = [
        "Send to Doctor",
        "Send to Nurse",
        "Remote Advice",
        "Escalate Priority",
        "Defer Patient",
        "Idle",
        "Open Room",
        "Close Room"
    ]

    def __init__(
        self,
        seed: Optional[int] = None,
        max_steps: int = 500,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the ClinicEnv.

        Args:
            seed: Random seed for reproducibility
            max_steps: Maximum steps per episode
            render_mode: Rendering mode ("human", "rgb_array", or "ansi")
        """
        super().__init__()

        self.max_steps = max_steps
        self.render_mode = render_mode

        # Observation space: 15 dimensions (FIXED from 14)
        obs_low = np.array([0.0] * 15, dtype=np.float32)
        obs_high = np.array([1.0] * 15, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Action space: 8 discrete actions
        self.action_space = spaces.Discrete(8)

        # Internal state variables
        self.step_count = 0
        self.num_open_rooms = 1
        self.queue = []  # List of patient dictionaries
        self.current_patient = None
        self.total_wait = 0.0
        self.last_render = None

        # Episode statistics for evaluation
        self.episode_stats = {
            "correct_triages": 0,
            "incorrect_triages": 0,
            "total_patients": 0,
            "total_wait_time": 0.0,
            "total_reward": 0.0
        }

        # Set random seed
        if seed is not None:
            self.seed(seed)

        # Initialize episode
        self.reset()

    def seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility."""
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def _sample_patient(self) -> Dict[str, Any]:
        """
        Generate a synthetic patient with clinically-inspired features.

        Severity levels are correlated with symptom features:
        - Mild (40%): Minor rash, low symptom scores
        - Moderate (35%): Moderate symptoms, may need physical exam
        - Severe (20%): Suspicious lesions, infection signs
        - Critical (5%): Urgent cases requiring immediate attention

        Returns:
            Dictionary containing patient attributes
        """
        # Sample severity (hidden ground truth)
        severity = np.random.choice(
            [self.SEVERITY_MILD, self.SEVERITY_MODERATE,
             self.SEVERITY_SEVERE, self.SEVERITY_CRITICAL],
            p=[0.4, 0.35, 0.2, 0.05]
        )

        # Generate correlated features
        age_norm = np.clip(np.random.normal(0.5, 0.15), 0.0, 1.0)
        duration_norm = np.clip(np.random.exponential(0.5), 0.0, 1.0)

        # Fever and infection more likely with higher severity
        fever_prob = 0.05 + 0.2 * severity
        infection_prob = 0.05 + 0.25 * severity
        fever_flag = 1.0 if np.random.rand() < fever_prob else 0.0
        infection_flag = 1.0 if np.random.rand() < infection_prob else 0.0

        # Symptom embedding: correlated with severity
        base_severity = 0.2 + 0.25 * severity
        symptom_embed = np.clip(
            np.random.normal(loc=base_severity, scale=0.08, size=(8,)),
            0.0, 1.0
        )

        patient = {
            "severity": int(severity),
            "age_norm": float(age_norm),
            "duration_norm": float(duration_norm),
            "fever_flag": float(fever_flag),
            "infection_flag": float(infection_flag),
            "symptom_embed": symptom_embed,
            "wait_time": 0.0
        }

        return patient

    def _form_observation(self, patient: Dict[str, Any]) -> np.ndarray:
        """
        Convert patient data to observation vector.

        Args:
            patient: Patient dictionary

        Returns:
            15-dimensional observation vector
        """
        vec = [
            patient["age_norm"],
            patient["duration_norm"],
            patient["fever_flag"],
            patient["infection_flag"],
        ]
        vec += list(patient["symptom_embed"])  # 8 dimensions
        vec += [
            1.0 if self.num_open_rooms > 0 else 0.0,  # room_avail
            np.clip(len(self.queue) / 10.0, 0.0, 1.0),  # queue_len_norm
            np.clip(self.step_count / self.max_steps, 0.0, 1.0)  # time_of_day_norm
        ]
        return np.array(vec, dtype=np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Optional random seed
            options: Optional reset options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if seed is not None:
            self.seed(seed)

        # Reset state
        self.step_count = 0
        self.num_open_rooms = 1
        self.queue = [self._sample_patient() for _ in range(3)]  # Warm start
        self.current_patient = None
        self.total_wait = 0.0

        # Reset statistics
        self.episode_stats = {
            "correct_triages": 0,
            "incorrect_triages": 0,
            "total_patients": 0,
            "total_wait_time": 0.0,
            "total_reward": 0.0
        }

        # Spawn initial patient
        self._maybe_spawn_next()

        obs = self._form_observation(self.current_patient)
        info = self._get_info()

        return obs, info

    def _maybe_spawn_next(self):
        """Spawn next patient from queue or create new one."""
        if self.current_patient is None and len(self.queue) > 0:
            self.current_patient = self.queue.pop(0)
        elif self.current_patient is None:
            # Create new patient if queue is empty
            self.current_patient = self._sample_patient()

    def _get_correct_action(self, severity: int) -> int:
        """
        Determine the optimal action for a given severity level.

        Args:
            severity: Patient severity level (0-3)

        Returns:
            Optimal action index
        """
        if severity == self.SEVERITY_MILD:
            return self.ACTION_REMOTE_ADVICE
        elif severity == self.SEVERITY_MODERATE:
            return self.ACTION_SEND_NURSE
        elif severity == self.SEVERITY_SEVERE:
            return self.ACTION_SEND_DOCTOR
        else:  # CRITICAL
            return self.ACTION_ESCALATE

    def _get_info(self) -> Dict[str, Any]:
        """Get current environment info."""
        if self.current_patient is None:
            return {"queue_length": len(self.queue)}

        return {
            "current_severity": int(self.current_patient["severity"]),
            "correct_action": int(self._get_correct_action(
                self.current_patient["severity"]
            )),
            "num_open_rooms": int(self.num_open_rooms),
            "queue_length": len(self.queue),
            "episode_stats": self.episode_stats.copy()
        }

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: Action to take (0-7)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self.step_count += 1
        patient = self.current_patient
        reward = 0.0

        # Determine correct action
        correct_action = self._get_correct_action(patient["severity"])

        # Reward for correct triage
        if action == correct_action:
            if patient["severity"] == self.SEVERITY_MILD:
                reward += 1.0
            elif patient["severity"] == self.SEVERITY_MODERATE:
                reward += 1.0
            elif patient["severity"] == self.SEVERITY_SEVERE:
                reward += 2.0
            else:  # CRITICAL
                # Bonus for fast escalation
                if patient["wait_time"] < 5.0:
                    reward += 3.0
                else:
                    reward += 2.0

            self.episode_stats["correct_triages"] += 1
        else:
            # Penalty for incorrect triage
            reward -= 1.5
            self.episode_stats["incorrect_triages"] += 1

        # Handle action side effects
        if action == self.ACTION_OPEN_ROOM:
            self.num_open_rooms += 1
        elif action == self.ACTION_CLOSE_ROOM and self.num_open_rooms > 0:
            self.num_open_rooms -= 1
        elif action == self.ACTION_DEFER:
            # Defer patient to end of queue
            patient["wait_time"] += 1.0
            self.queue.append(patient)
            self.current_patient = None
        else:
            # All other actions "treat" the patient
            self.current_patient = None

        # Update queue wait times
        wait_increment = 0.01 * len(self.queue)
        for p in self.queue:
            p["wait_time"] += 1.0
        self.total_wait += wait_increment

        # Wait penalty
        reward -= 0.01 * wait_increment

        # Resource cost penalty
        reward -= 0.05 * self.num_open_rooms

        # Spawn next patient
        self._maybe_spawn_next()

        # Update statistics
        self.episode_stats["total_patients"] += 1
        self.episode_stats["total_wait_time"] += wait_increment
        self.episode_stats["total_reward"] += reward

        # Get new observation
        obs = self._form_observation(self.current_patient)

        # Check termination
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """
        Render the environment.

        Returns RGB array or prints to console based on render_mode.
        """
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        elif self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            # For human mode, we could use pygame (implemented in rendering.py)
            return self._render_rgb_array()

    def _render_rgb_array(self) -> np.ndarray:
        """
        Render as RGB array (NumPy-based, fast for video generation).

        Returns:
            240x360 RGB numpy array
        """
        H, W = 240, 360
        canvas = np.ones((H, W, 3), dtype=np.uint8) * 255  # White background

        if self.current_patient is None:
            return canvas

        # --- Severity indicator bar ---
        sev = self.current_patient["severity"]
        sev_norm = sev / 3.0

        # Color gradient: green (mild) -> yellow -> red (critical)
        color = np.array([
            int(255 * sev_norm),           # R increases with severity
            int(180 * (1 - sev_norm)),     # G decreases with severity
            60                              # B constant
        ], dtype=np.uint8)

        canvas[20:200, 20:60] = color

        # --- Queue length indicator (blue bars) ---
        q_len = len(self.queue)
        q_h = min(q_len * 15, 150)
        canvas[20:20+q_h, 80:100] = [80, 80, 255]

        # --- Open rooms indicator (green bars) ---
        r_h = min(self.num_open_rooms * 20, 150)
        canvas[20:20+r_h, 120:140] = [50, 200, 50]

        # --- Progress bar ---
        progress = min(self.step_count / self.max_steps, 1.0)
        prog_w = int(progress * 320)
        canvas[220:230, 20:20+prog_w] = [100, 100, 100]

        self.last_render = canvas
        return canvas

    def _render_ansi(self) -> str:
        """Render as ANSI text."""
        if self.current_patient is None:
            return "No current patient"

        sev_names = ["MILD", "MODERATE", "SEVERE", "CRITICAL"]
        severity = self.current_patient["severity"]

        output = f"\n{'='*50}\n"
        output += f"Step: {self.step_count}/{self.max_steps}\n"
        output += f"Current Patient: {sev_names[severity]}\n"
        output += f"Queue Length: {len(self.queue)}\n"
        output += f"Open Rooms: {self.num_open_rooms}\n"
        output += f"Triage Accuracy: {self._get_triage_accuracy():.1f}%\n"
        output += f"{'='*50}\n"

        return output

    def _get_triage_accuracy(self) -> float:
        """Calculate current triage accuracy percentage."""
        total = self.episode_stats["correct_triages"] + self.episode_stats["incorrect_triages"]
        if total == 0:
            return 0.0
        return 100.0 * self.episode_stats["correct_triages"] / total

    def close(self):
        """Clean up resources."""
        pass


# Helper function for creating vectorized environments
def make_env(seed: int = 0, max_steps: int = 500):
    """
    Factory function for creating ClinicEnv instances.

    Args:
        seed: Random seed
        max_steps: Maximum steps per episode

    Returns:
        Callable that creates a ClinicEnv
    """
    def _init():
        env = ClinicEnv(seed=seed, max_steps=max_steps)
        return env
    return _init
