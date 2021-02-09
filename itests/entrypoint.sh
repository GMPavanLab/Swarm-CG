python3 swarmcg/evaluate_model.py -aa_tpr tests/data/aa_topol.tpr -aa_traj tests/data/aa_traj.xtc -cg_map tests/data/cg_map.ndx -cg_itp tests/data/cg_model.itp -cg_tpr tests/data/cg_topol.tpr -cg_traj tests/data/cg_traj.xtc
python3 swarmcg/optimize_model.py -gmx gmx -in_dir tests/data/ -aa_tpr tests/data/aa_topol.tpr -aa_traj tests/data/aa_traj.xtc -cg_map tests/data/cg_map.ndx -cg_itp tests/data/cg_model.itp -cg_top tests/data/system.top -cg_gro tests/data/start_conf.gro -out_dir MODEL_OPTI -sim_type TEST
python3 swarmcg/analyze_optimization.py -opti_dir MODEL_OPTI
