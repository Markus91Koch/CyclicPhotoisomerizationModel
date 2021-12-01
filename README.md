# CyclicPhotoisomerizationModel

The Cyclic Photoisomerization Model (CPM) is a simulation model for fully atomisitc MD simulations that implements the effects of UV--Vis light on azobenzene-containing systems. The CPM is inspired by a simulation approach by Bedrov et al. [1], which has been first implemented and then further extended/improved.

![cyclic_illustration_final2_italic4_PAPER_CPM-E-dif-colors](https://user-images.githubusercontent.com/47243285/144288668-0824764b-a2fd-4a77-9ccf-73960b3d2a3f.png)

## Requirements

The CPM as provided here has been developed for the open-source simulation software LAMMPS [2,3]. An installation of LAMMPS is required to run the provided simulation scripts. 

### Overview of all requirements:

#### Necessary: 

- LAMMPS
- Python 3.6+ 
- bash

#### Optional:

- catdcd (required for the additionally provided bash script *concatenate.sh*)
- VMD (possible visualization of the MD system)

## References

- [1] Bedrov, D., Hooper, J. B., Glaser, M. A., and Clark, N. A. "Photoinduced and Thermal Relaxation in Surface-Grafted Azobenzene-Based Monolayers: A Molecular Dynamics Simulation Study" *Langmuir* 2016, 32, 16, 4004–4015, https://doi.org/10.1021/acs.langmuir.6b00120
- [2] Thompson, A.P., Aktulga, H.M., Berger, R., Bolintineanu, D.S., Brown, M.W., Crozier, P.S., in ’t Veld, P.J., Kohlmeyer, A.,
Moore, S.G., Nguyen, T.D., Shan, R., Stevens, M., Tranchida, J., Trott, C., Plimpton, S.J. "LAMMPS - A flexible simulation tool
for particle-based materials modeling at the atomic, meso, and continuum scales" *Comput. Phys. Commun.* 2021, 108171 https://doi.org/10.1016/j.cpc.2021.108171
- [3] https://www.lammps.org/download.html
