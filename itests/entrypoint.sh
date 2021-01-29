python3 swarmcg/evaluate_model.py -aa_tpr G1_DATA/aa_topol.tpr -aa_traj G1_DATA/aa_traj.xtc -cg_map G1_DATA/cg_map.ndx -cg_itp G1_DATA/cg_model.itp -cg_tpr G1_DATA/cg_topol.tpr -cg_traj G1_DATA/cg_traj.xtc
#python3 swarmcg/optimize_model.py -gmx gmx -in_dir G1_DATA -aa_tpr G1_DATA/aa_topol.tpr -aa_traj G1_DATA/aa_traj.xtc -cg_map G1_DATA/cg_map.ndx -cg_itp G1_DATA/cg_model.itp -cg_top G1_DATA/system.top -cg_gro G1_DATA/start_conf.gro -out_dir MODEL_OPTI -sim_type TEST
#python3 swarmcg/analyze_optimization.py -opti_dir MODEL_OPTI
