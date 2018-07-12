Lax OpenMP is a simple project created to experiment with OMP features, particularly offloading to GPUs using OMP 4.5.

To compile: 
make OFFLOAD=[yes | no]

To run:
./OMP_LAX Npoints N_time_steps WRITE_FLAG

where: WRITE_FLAG=[0 | 1]

suggested use:
make OFFLOAD=yes

./OMP_LAX 100000 40000 0

