import os
import traci

SUMO_CFG = os.path.join(os.environ["SUMO_HOME"], "tools/game/cross.sumocfg")

def main():
    traci.start(["sumo", "-c", SUMO_CFG, "--no-step-log", "true", "--waiting-time-memory", "1000"])

    tls = traci.trafficlight.getIDList()[0]
    logic = traci.trafficlight.getAllProgramLogics(tls)[0]
    n_phases = len(logic.phases)

    switch_every = 30
    phase = 0
    steps = 0
    total_wait = 0.0

    while traci.simulation.getMinExpectedNumber() > 0:
        if steps % switch_every == 0:
            phase = (phase + 1) % n_phases
            traci.trafficlight.setPhase(tls, phase)

        traci.simulationStep()

        veh_ids = traci.vehicle.getIDList()
        total_wait += sum(traci.vehicle.getWaitingTime(v) for v in veh_ids)
        steps += 1

    traci.close()
    print("DONE steps:", steps, "avg_wait_per_step:", total_wait / max(steps, 1))

if __name__ == "__main__":
    main()
