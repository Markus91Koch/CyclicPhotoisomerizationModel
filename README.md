# CyclicPhotoisomerizationModel

The Cyclic Photoisomerization Model (CPM) is a **simulation model for fully atomistic MD simulations** that incorporates the **effects of UV–Vis light on azobenzene-containing systems**. The CPM is inspired by a simulation approach by Bedrov et al. [1], which has been first implemented and then further extended/improved.
Furthermore, we use a model by Heinz et al. [2] to realize the photoisomerization reactions (*trans* to *cis*, *cis* to *trans*) of individual azobenzene groups in the system.
 
The main focus of the CPM approach is to reproduce the **collective photoisomerization kinetics** of azobenzene groups under light. In particular, the approach assumes a constant light intensity and resonant excitation of the *trans* and the *cis* isomers of azobenzene. The CPM can simulate the effects of different (fixed) light intensities and of different (fixed) wavelengths of the light.

A detailed description of the CPM is provided in Ref. [3]. 
For the general course of a CPM simulation refer to the following illustration:

![cyclic_illustration_final2_italic4_PAPER_CPM-E-dif-colors](https://user-images.githubusercontent.com/47243285/144288668-0824764b-a2fd-4a77-9ccf-73960b3d2a3f.png)

## Requirements

The CPM as provided here has been developed for the open-source simulation software LAMMPS [4,5]. An installation of LAMMPS is required to run the provided simulation scripts. 

### Overview of all requirements:

#### Necessary: 

- LAMMPS
- Python 3.6+
- Numpy
- bash

#### Optional:

- Python packages: SciPy, Pandas, Matplotlib, os, glob (just for analyzing results)
- catdcd (required for the additionally provided bash script *concatenate.sh*)
- VMD (possible visualization and analysisof the MD trajectory)

## References

- [1] Bedrov, D., Hooper, J. B., Glaser, M. A., and Clark, N. A. "Photoinduced and Thermal Relaxation in Surface-Grafted Azobenzene-Based Monolayers: A Molecular Dynamics Simulation Study" *Langmuir* 2016, 32, 16, 4004–4015, https://doi.org/10.1021/acs.langmuir.6b00120
- [2] Heinz, H., Vaia, R. A., Koerner, H., and Farmer, B. L. Photoisomerization of Azobenzene Grafted to Layered Silicates: Simulation and
Experimental Challenges. *Chem. Mater.* 2008, 20, 6444–6456, https://doi.org/10.1021/cm801287d
- [3] Koch, M., Saphiannikova, M., and Guskova, O. "Cyclic Photoisomerization of Azobenzene in Atomistic Simulations: Modeling the Effect of Light on Columnar Aggregates of Azo Stars" *Molecules* 2021, 26, 7674, https://doi.org/10.3390/molecules26247674 
- [4] Thompson, A. P., Aktulga, H. M., Berger, R., Bolintineanu, D. S., Brown, W. M., Crozier, P. S., in ’t Veld, P. J., Kohlmeyer, A.,
Moore, S. G., Nguyen, T. D., Shan, R., Stevens, M., Tranchida, J., Trott, C., Plimpton, S. J. "LAMMPS - A flexible simulation tool
for particle-based materials modeling at the atomic, meso, and continuum scales" *Comput. Phys. Commun.* 2022, 271, 108171, https://doi.org/10.1016/j.cpc.2021.108171
- [5] https://www.lammps.org/download.html
