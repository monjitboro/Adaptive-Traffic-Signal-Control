import traci

SUMO_CFG = "nets/grid.sumocfg"
USE_GUI = False

def sumo_cmd():
    binary = "sumo-gui" if USE_GUI else "sumo"
    return [binary, "-c", SUMO_CFG, "--no-step-log", "true", "--waiting-time-memory", "1000"]

def main():
    traci.start(sumo_cmd())

    tls_ids = traci.trafficlight.getIDList()
    print("TLS IDs:", tls_ids)
    assert tls_ids, "No traffic lights found. Did you run netconvert --tls.guess?"

    tls = tls_ids[0]
    logic = traci.trafficlight.getAllProgramLogics(tls)[0]
    n_phases = len(logic.phases)
    print("Controlling TLS:", tls, "phases:", n_phases)

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

        if steps % 300 == 0:
            print("step", steps, "vehicles_in_sim", len(veh_ids))

    traci.close()
    print("DONE steps:", steps, "avg_wait_per_step:", total_wait / max(steps, 1))

if __name__ == "__main__":
    main()
