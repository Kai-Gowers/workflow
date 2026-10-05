#!/bin/bash
#SBATCH -A m5370
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH -t 6:00:00
#SBATCH -J slidescan_bare_pbe

# Same layout as the production relaxation bat (2 nodes, 42 ranks x 6 threads, KPAR=7 in INCAR).
module load vasp/6.6.1-cpu
export OMP_NUM_THREADS=6
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
ulimit -s unlimited

cd "$SLURM_SUBMIT_DIR"
for d in slide/MoS2_*/*/; do
    [ -f "$d/OUTCAR" ] && grep -q "General timing" "$d/OUTCAR" && continue
    echo "=== $d $(date)"
    (cd "$d" && srun -n 42 -c 12 --cpu-bind=cores vasp_std > vasp.out 2>&1)
    grep "F=" "$d/OSZICAR" | tail -1
done
