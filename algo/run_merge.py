import os
import traci

sumocfg_file = os.path.join("..", "sumo_merge_example", "merge.sumocfg")
sumo_binary = "sumo-gui"

# 这里改了，增加 --remote-port 0
sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--remote-port", "0"]

traci.start(sumo_cmd)

step = 0
while step < 1000:
    traci.simulationStep()
    step += 1

    veh_ids = traci.vehicle.getIDList()
    for vid in veh_ids:
        speed = traci.vehicle.getSpeed(vid)
        print(f"Step {step}: Vehicle {vid} speed={speed:.2f} m/s")

traci.close()
print("Simulation finished.")
