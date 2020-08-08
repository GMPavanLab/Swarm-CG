# Swarm-CG

Swarm-CG is designed for automatically optimizing the bonded terms of a coarse-grained (CG) molecular model, in explicit or implicit solvent, with respect to a reference all-atom (AA) trajectory and starting from a preliminary CG model (topology and non-bonded parameters). The package is designed for usage with Gromacs and contains 3 modules for:

1. Evaluating the bonded parametrization of a CG model
2. Optimizing bonded terms of a CG model
3. Monitoring an optimization procedure

![Swarm-CG](/images/TOC_Swarm-CG_paper.png)

### Publication

> Empereur-mot, C.; Pesce, L.; Bochicchio, D.; Perego, C.; Pavan, G.M. (2020) Swarm-CG: Automatic Parametrization of Bonded Terms in Coarse-Grained Models of Simple to Complex Molecules via Fuzzy Self-Tuning Particle Swarm Optimization. [ChemRxiv. Preprint](https://doi.org/10.26434/chemrxiv.12613427)

### Installation & Usage

Swarm-CG was tested using Python 3.6.8 and Gromacs 2018.6.

	pip3 install swarm-cg    # creates the 3 entrypoints/aliases below
	
	scg_evaluate -h          # see point 1
	scg_optimize -h          # see point 2
	scg_monitor -h           # see point 3

To better handle sampling in symmetrical molecules you can form groups of bonds/angles/dihedrals that Swarm-CG will consider identical, using line returns and/or comments in the topology (ITP) file. AA-mapped distributions will be averaged within groups to create the references used for evaluation (see point 1) or as target of the optimization procedure (see point 2). For optimization, identical parameters will be used for the bonds/angles/dihedrals within each group.

Here is an ITP file extract from the demonstration data of [PAMAM G1](https://github.com/GMPavanLab/Swarm-CG/tree/master/PAMAM_G1_DATA/cg_model.itp):

	[ bonds ]
	;   i     j   funct   length   force.c.   
	; bond group 1
	    1     2       1        0         0           ; B1
	; bond group 2
	    1     3       1        0         0           ; B2
	    1     9       1        0         0           ; B2
	; bond group 3
	    3     4       1        0         0           ; B3
	    9    10       1        0         0           ; B3

### 1. Evaluate bonded parametrization of a CG model

The module `scg_evaluate` enables quick evaluation of the fit of bond, angle and dihedral distributions between a CG model trajectory and a reference AA model trajectory of an identical molecule, by producing a single comprehensive figure.

	scg_evaluate -aa_tpr G1_DATA/aa_topol.tpr -aa_traj G1_DATA/aa_traj.xtc -cg_map G1_DATA/cg_map.ndx -cg_itp G1_DATA/cg_model.itp -cg_tpr G1_OPTI_mode1_200ns_valid/longer_run.tpr -cg_traj G1_OPTI_mode1_200ns_valid/longer_run.xtc

This is particularly useful to assess the need to run an optimization procedure (assuming one already has a CG model). It is also suited to the assessment of geometrical changes triggered by a modification of CG beads types (defining non-bonded parameters) or after manually editing bonded parameters while working on a model. This command also provides publication-quality figures to support the parametrization of your models (also in vectorized formats). Radius of gyration (Rg) and solvent accessible surface area (SASA) are also calculated. 

### 2. Optimize bonded terms of a CG model

The module `scg_optimize` allows to automatically optimize the bonded parameters of a CG model according to a reference AA trajectory. To this end, several simulations will be run to explore and evaluate the relevance of different sets of bonded parameters, using 3 optimization cycles.

For example, using demonstration data of [PAMAM G1](https://github.com/GMPavanLab/Swarm-CG/tree/master/PAMAM_G1_DATA):

	scg_optimize -in_dir G1_DATA/ -gmx gmx_2018.6_p

Which will use all default filenames of the software and is *exactly identical* to this command:

	scg_optimize -aa_tpr G1_DATA/aa_topol.tpr -aa_traj G1_DATA/aa_traj.xtc -cg_map G1_DATA/cg_map.ndx -cg_itp G1_DATA/cg_model.itp -cg_gro G1_DATA/start_conf.gro -cg_top G1_DATA/system.top -cg_mdp_mini G1_DATA/mini.mdp -cg_mdp_equi G1_DATA/equi.mdp -cg_mdp_md G1_DATA/md.mdp -gmx gmx_2018.6_p

We recommend to first prepare files in a directory to be fed to Swarm-CG using argument `-in_dir`.

The input is composed of:

1. An AA reference trajectory (TPR + XTC/TRR)
2. The AA to CG mapping (NDX)
3. A preliminary CG model (ITP, equilibrium values and force constants can be initialized arbitrarily to e.g. 0)
4. A CG configuration used as starting point of each iterative optimization run (GRO file, from a mapped AA frame and solvated if necessary)
5. Other simulation files (TOP and MDP, notably with your barostat and thermostat choices)

At all times during execution, the best parametrized model is accessible in the optimization output folder at `out_dir/optimized_CG_model/cg_model.itp`. The bonded parameters obtained via the Boltzmann inversion implemented in Swarm-CG with groups averaging (see paper sections 2.1 and 6.1) are also available at `out_dir/boltzmann_inv_CG_model/cg_model.itp`.

The AA trajectory is mapped on-the-fly (if atoms are mapped to multiple CG beads, atom masses are split accordingly). The AA trajectory must contain box information for PBC handling, otherwise it is assumed the molecule is "unwrapped" already. Only the MDP file provided via argument `-cg_mdp_md` will be modified to adjust simulation time (nsteps), taking into account the timestep you provided. To minimize the execution time of `scg_optimize`, equilibration should stay short (e.g. 50-500 fs) and so should the optimization cycles 1 and 2 (e.g. 10-20 ns). To maximize the precision of `scg_optimize`, optimization cycle 3 must always use longer simulation times (e.g. 25-100 ns). Execution times should vary between 4h to 24h according to parameters and hardware used.

For information about execution modes 1 and 2, please see paper sections 2.4 and 4 and command help (-h).

### 3. Monitor an ongoing CG model optimization

Optimization procedures can be monitored at any point during execution. The module `scg_monitor` produces a visual summary (see paper Fig. 3) of the progress of an optimization procedure started with module `scg_optimize`. The plot will be produced in the directory provided via argument `-opti_dir`.

	scg_monitor -opti_dir MODEL_OPTI__STARTED_03-07-2020_10h_12m_15s -gmx gmx_2018.6_p

See the help (-h) for a complete description of `scg_monitor` output. In particular, note that Rg and SASA might be rough estimates in this display, as they are calculated from short simulations used for optimization. These values must probably be validated using longer simulation times. Using `scg_evaluate` can be helpful to this end.

### Extended usage (untested)

In principle, Swarm-CG workflow is general and can be applied also for tuning bonded terms in coarser CG models (by mapping more than 3-5 atoms to each CG bead and providing adequate non-bonded parameters). To this end, it is possible to use an AA trajectory as reference for optimization, but also instead a high resolution CG trajectory (fine grain) for tuning the coarser CG model (see paper section 4 for a more detailed discussion about crossing CG scales).

Another possible use case would be the tuning of elastic networks in CG models of proteins, although this still requires a well sampled AA or fine CG reference trajectory.

Please feel free to open an [Issue](https://github.com/GMPavanLab/SwarmCG/issues) or email us if you are interested into extended usages and need help.

### Credits

Swarm-CG makes extensive use of [FST-PSO](https://doi.org/10.1016/j.swevo.2017.09.001) and [MDAnalysis](https://doi.org/10.1002/jcc.21787). We thank [Marco S. Nobile](http://msnobile.it/personal/) for his valuable insights.





