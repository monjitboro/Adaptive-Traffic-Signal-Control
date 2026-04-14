import cityflow
import numpy as np
import json
import os

class TrafficEnvironment:
    def __init__(self, config_file, num_steps=3600):
        self.config_file = config_file
        self.num_steps   = num_steps
        self.eng         = cityflow.Engine(config_file, thread_num=1)

        self.current_step   = 0
        self.current_phase  = 0
        self.phase_duration = 0
        self.prev_waiting   = 0
        self.prev_vehicles  = 0

        # Increased min green time
        self.min_green_time = 10

        self.lane_ids = list(self.eng.get_lane_waiting_vehicle_count().keys())

        roadnet_path = os.path.join(os.path.dirname(config_file), "roadnet.json")
        with open(roadnet_path) as f:
            roadnet = json.load(f)
        self.intersection_id = [
            i["id"] for i in roadnet["intersections"]
            if not i.get("virtual", False)
        ][0]

        self.state_size  = len(self.lane_ids) + 2
        self.action_size = 2

        # Stronger reward weights
        self.w_queue      = 0.6   # was 0.4
        self.w_delay      = 0.2   # was 0.3
        self.w_throughput = 0.1   # was 0.2
        self.w_pressure   = 0.05
        self.w_switch     = 0.05

        print("Environment initialized")
        print("  Intersection  : {}".format(self.intersection_id))
        print("  State size    : {}".format(self.state_size))
        print("  Action size   : {}".format(self.action_size))
        print("  Lanes         : {}".format(len(self.lane_ids)))

    def reset(self):
        self.eng.reset()
        self.current_step   = 0
        self.current_phase  = 0
        self.phase_duration = 0
        self.prev_waiting   = 0
        self.prev_vehicles  = 0
        return self._get_state()

    def step(self, action):
        switched = False

        if action == 1 and self.phase_duration >= self.min_green_time:
            self._switch_phase()
            switched = True

        waiting_before  = self._get_total_waiting()
        vehicles_before = self._get_total_vehicles()

        self.eng.next_step()
        self.current_step   += 1
        self.phase_duration += 1

        waiting_after  = self._get_total_waiting()
        vehicles_after = self._get_total_vehicles()

        reward     = self._calculate_reward(
            waiting_before, waiting_after,
            vehicles_before, vehicles_after,
            switched
        )
        next_state = self._get_state()
        done       = self.current_step >= self.num_steps

        self.prev_waiting  = waiting_after
        self.prev_vehicles = vehicles_after

        return next_state, reward, done

    def _get_state(self):
        waiting = self.eng.get_lane_waiting_vehicle_count()
        queue_state = np.array(
            [waiting[lane] / 20.0 for lane in self.lane_ids],
            dtype=np.float32
        )
        phase_state    = np.array([self.current_phase / 4.0],  dtype=np.float32)
        duration_state = np.array([min(self.phase_duration / 30.0, 1.0)], dtype=np.float32)
        return np.concatenate([queue_state, phase_state, duration_state])

    def _calculate_reward(self, waiting_before, waiting_after, vehicles_before, vehicles_after, switched):
        # Component 1 - Queue
        queue_reward = (waiting_before - waiting_after) / 20.0

        # Component 2 - Delay
        vehicle_speeds = self.eng.get_vehicle_speed()
        delayed = sum(1 for s in vehicle_speeds.values() if s < 0.1)
        delay_reward = -delayed / 20.0

        # Component 3 - Throughput
        throughput = max(0, vehicles_before - vehicles_after)
        throughput_reward = throughput / 10.0

        # Component 4 - Pressure
        waiting_counts = list(self.eng.get_lane_waiting_vehicle_count().values())
        if len(waiting_counts) >= 2:
            pressure = -(max(waiting_counts) - min(waiting_counts)) / 20.0
        else:
            pressure = 0

        # Component 5 - Switch penalty
        switch_reward = -0.1 if switched else 0.0

        total_reward = (
            self.w_queue      * queue_reward      +
            self.w_delay      * delay_reward      +
            self.w_throughput * throughput_reward +
            self.w_pressure   * pressure          +
            self.w_switch     * switch_reward
        )
        return total_reward
    
    def _switch_phase(self):
        self.current_phase = (self.current_phase + 1) % 4
        self.eng.set_tl_phase(self.intersection_id, self.current_phase)
        self.phase_duration = 0

    def _get_total_waiting(self):
        return sum(self.eng.get_lane_waiting_vehicle_count().values())

    def _get_total_vehicles(self):
        return sum(self.eng.get_lane_vehicle_count().values())

    def get_metrics(self):
        waiting  = self._get_total_waiting()
        vehicles = self._get_total_vehicles()
        speeds   = self.eng.get_vehicle_speed()
        delayed  = sum(1 for s in speeds.values() if s < 0.1)
        congestion = (delayed / vehicles * 100) if vehicles > 0 else 0
        return {
            "step"           : self.current_step,
            "waiting"        : waiting,
            "total_vehicles" : vehicles,
            "delayed"        : delayed,
            "congestion_rate": round(congestion, 2)
        }
