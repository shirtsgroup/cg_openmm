#!/usr/bin/bash

hostname
date

printf -v beg '%(%s)T\n' -1

# Run replica exchange:
python run_replica_exchange.py > cg_openmm.out
printf -v sim_end '%(%s)T\n' -1

# Process replica exchange:
python process_replica_exchange.py > process_replica.out
printf -v proc_end '%(%s)T\n' -1

# Calculate heat capacity:
python run_ensemble_validation.py > ensemble_check.out
printf -v val_end '%(%s)T\n' -1

# Print timing stats:
echo simulation time: sim_elapsed=$((sim_end-beg))
echo processing time: proc_elapsed=$((proc_end-sim_end))
echo physical validation time: val_elapsed=$((val_end-proc_end))
echo total time: total_elapsed=$((val_end-beg))