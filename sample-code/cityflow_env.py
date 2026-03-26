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

    def __init__(self, config_path, scenario_name, num_step=3600, decision_interval=5):
        super().__init__()

        self.config_path = config_path
        self.scenario_name = scenario_name
        self.num_step = num_step
        self.decision_interval = decision_interval

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

    def _get_reward(self):
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        total_queue = sum(lane_waiting.get(lane_id, 0) for lane_id in self.incoming_lanes)

        # Start simple: minimize queue length
        reward = -float(total_queue) / max(1, len(self.incoming_lanes))
        return reward

    def _get_info(self):
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        lane_vehicle = self.eng.get_lane_vehicle_count()

        total_queue = sum(lane_waiting.get(lane_id, 0) for lane_id in self.incoming_lanes)
        total_vehicles = sum(lane_vehicle.get(lane_id, 0) for lane_id in self.incoming_lanes)

        return {
            "queue": float(total_queue),
            "vehicles": float(total_vehicles),
            "avg_travel_time": float(self.eng.get_average_travel_time())
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
        info = self._get_info()
        return obs, info

    def step(self, action):
        self.current_phase_pos = int(action)
        phase_id = self.valid_phases[self.current_phase_pos]

        self.eng.set_tl_phase(self.intersection_id, phase_id)

        for _ in range(self.decision_interval):
            self.eng.next_step()
            self.current_step += 1

        obs = self._get_obs()
        reward = self._get_reward()
        info = self._get_info()

        terminated = self.current_step >= self.num_step
        truncated = False

        return obs, reward, terminated, truncated, info