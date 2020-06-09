# SwarmCG

Required libraries:

`$ pip install numpy scipy matplotlib MDAnalysis pyemd networkx fst-pso # TODO: remove networkx`

### Optimize bonded parameters of a CG model, according to a reference AA trajectory

For optimizing bonded parameters of a model according to a reference AA trajectory, using the example data of PAMAM G1:

`$ ./optimize_model.py -input_dir G1_CG_SIM/ -gmx gmx_2018.6_p`

Which will use all default filenames of the software and is EXACTLY IDENTICAL to this command:

`$ ./optimize_model.py -aa_tpr G1_CG_SIM/aa_topol.tpr -aa_traj G1_CG_SIM/aa_traj.xtc -cg_map G1_CG_SIM/cg_map.ndx -cg_itp G1_CG_SIM/cg_model.itp -cg_gro G1_CG_SIM/start_conf.gro -cg_top G1_CG_SIM/system.top -cg_mdp_mini G1_CG_SIM/mini.mdp -cg_mdp_pre_md G1_CG_SIM/pre-md.mdp -cg_mdp_md G1_CG_SIM/md.mdp -gmx gmx_2018.6_p`

### Evaluate bonded parametrization of a CG model, with respect to a reference AA trajectory

