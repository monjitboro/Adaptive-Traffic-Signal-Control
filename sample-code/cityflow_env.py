import os
import json
import numpy as np
import cityflow

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


class CityFlowSingleIntersectionEnv(gym.Env):
    metadata = {"render_modes": []}
    SUPPORTED_REWARD_MODES = ("queue", "queue_delay", "queue_switch", "hybrid")

    def __init__(
        self,
        config_path,
        scenario_name,
        num_step=3600,
        decision_interval=5,
        reward_mode="queue",
        reward_delay_coef=0.5,
        reward_switch_coef=0.1,
    ):
        super().__init__()

        self.config_path = config_path
        self.scenario_name = scenario_name
        self.num_step = num_step
        self.decision_interval = decision_interval
        self.reward_mode = reward_mode
        self.reward_delay_coef = float(reward_delay_coef)
        self.reward_switch_coef = float(reward_switch_coef)

        if self.reward_mode not in self.SUPPORTED_REWARD_MODES:
            raise ValueError(
                f"Unsupported reward_mode={reward_mode!r}. "
                f"Choose from {self.SUPPORTED_REWARD_MODES}."
            )

        self.current_step = 0
        self.eng = None

        self.scenario_dir = os.path.join("data", scenario_name)
        roadnet_path = os.path.join(self.scenario_dir, "roadnet.json")

        with open(roadnet_path, "r") as f:
            self.roadnet = json.load(f)

        self.intersection_id = self._get_real_intersection_id()
        self.incoming_lanes = self._get_incoming_lanes(self.intersection_id)
        self.valid_phases = self._get_valid_phases(self.intersection_id)

        self.current_phase_pos = 0

        obs_dim = len(self.incoming_lanes) + len(self.valid_phases)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1000.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.valid_phases))

    def _get_real_intersection_id(self):
        real_intersections = [
            i for i in self.roadnet["intersections"]
            if not i.get("virtual", False)
        ]
        if len(real_intersections) == 0:
            raise ValueError("No real intersection found.")
        return real_intersections[0]["id"]

    def _get_valid_phases(self, intersection_id):
        inter = next(
            i for i in self.roadnet["intersections"]
            if i["id"] == intersection_id
        )
        phases = inter.get("trafficLight", {}).get("lightphases", [])

        # Keep only phases that actually allow some movement
        valid = []
        for idx, ph in enumerate(phases):
            if len(ph.get("availableRoadLinks", [])) > 0:
                valid.append(idx)

        if len(valid) == 0:
            # Fallback: use all phases
            valid = list(range(len(phases)))

        return valid

    def _get_incoming_lanes(self, intersection_id):
        incoming_roads = [
            r for r in self.roadnet["roads"]
            if r["endIntersection"] == intersection_id
        ]

        lane_ids = []
        for road in incoming_roads:
            lane_count = len(road.get("lanes", []))
            for lane_idx in range(lane_count):
                lane_ids.append(f"{road['id']}_{lane_idx}")
        return lane_ids

    def _make_engine(self):
        self.eng = cityflow.Engine(self.config_path, thread_num=1)

    def _collect_metrics(self):
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        lane_vehicle = self.eng.get_lane_vehicle_count()

        total_queue = float(sum(lane_waiting.get(lane_id, 0) for lane_id in self.incoming_lanes))
        total_vehicles = float(sum(lane_vehicle.get(lane_id, 0) for lane_id in self.incoming_lanes))
        incoming_lane_count = max(1, len(self.incoming_lanes))

        return {
            "total_queue": total_queue,
            "avg_queue": total_queue / incoming_lane_count,
            # CityFlow does not expose a direct per-step delay term here,
            # so we use the waiting/occupancy ratio as a delay proxy.
            "delay_proxy": total_queue / max(1.0, total_vehicles),
            "total_vehicles": total_vehicles,
            "avg_travel_time": float(self.eng.get_average_travel_time()),
        }

    def _get_obs(self):
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()

        queue_feats = []
        for lane_id in self.incoming_lanes:
            queue_feats.append(float(lane_waiting.get(lane_id, 0)))

        phase_one_hot = np.zeros(len(self.valid_phases), dtype=np.float32)
        phase_one_hot[self.current_phase_pos] = 1.0

        obs = np.concatenate([
            np.array(queue_feats, dtype=np.float32),
            phase_one_hot
        ]).astype(np.float32)

        return obs

    def _get_reward(self, metrics, phase_changed):
        queue_term = float(metrics["avg_queue"])
        delay_term = float(metrics["delay_proxy"])
        switch_term = float(phase_changed)

        if self.reward_mode == "queue":
            total_penalty = queue_term
        elif self.reward_mode == "queue_delay":
            total_penalty = queue_term + self.reward_delay_coef * delay_term
        elif self.reward_mode == "queue_switch":
            total_penalty = queue_term + self.reward_switch_coef * switch_term
        else:
            total_penalty = (
                queue_term
                + self.reward_delay_coef * delay_term
                + self.reward_switch_coef * switch_term
            )

        reward_terms = {
            "queue_term": queue_term,
            "delay_term": delay_term,
            "switch_term": switch_term,
        }
        return -float(total_penalty), reward_terms

    def _get_info(self, metrics, reward_terms, phase_changed):
        return {
            "queue": float(metrics["total_queue"]),
            "vehicles": float(metrics["total_vehicles"]),
            "avg_travel_time": float(metrics["avg_travel_time"]),
            "reward_mode": self.reward_mode,
            "delay_proxy": float(metrics["delay_proxy"]),
            "phase_changed": float(phase_changed),
            "reward_queue_term": float(reward_terms["queue_term"]),
            "reward_delay_term": float(reward_terms["delay_term"]),
            "reward_switch_term": float(reward_terms["switch_term"]),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.current_phase_pos = 0

        self._make_engine()

        # Set initial phase
        init_phase = self.valid_phases[self.current_phase_pos]
        self.eng.set_tl_phase(self.intersection_id, init_phase)

        obs = self._get_obs()
        metrics = self._collect_metrics()
        reward_terms = {
            "queue_term": float(metrics["avg_queue"]),
            "delay_term": float(metrics["delay_proxy"]),
            "switch_term": 0.0,
        }
        info = self._get_info(metrics, reward_terms, phase_changed=0.0)
        return obs, info

    def step(self, action):
        action = int(action)
        phase_changed = float(action != self.current_phase_pos)
        self.current_phase_pos = action
        phase_id = self.valid_phases[self.current_phase_pos]

        self.eng.set_tl_phase(self.intersection_id, phase_id)

        for _ in range(self.decision_interval):
            self.eng.next_step()
            self.current_step += 1

        obs = self._get_obs()
        metrics = self._collect_metrics()
        reward, reward_terms = self._get_reward(metrics, phase_changed)
        info = self._get_info(metrics, reward_terms, phase_changed)

        terminated = self.current_step >= self.num_step
        truncated = False

        return obs, reward, terminated, truncated, info
