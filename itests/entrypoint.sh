cd ./Swarm-CG
#pytest tests
python3 swarmcg/evaluate_model.py -aa_tpr G1_DATA/aa_topol.tpr -aa_traj G1_DATA/aa_traj.xtc -cg_map G1_DATA/cg_map.ndx -cg_itp G1_DATA/cg_model.itp -cg_tpr G1_DATA/cg_topol.tpr -cg_traj G1_DATA/cg_traj.xtc