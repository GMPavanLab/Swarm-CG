import swarmcg
from .. import config

sep = '----------------------------------------------------------------------'
sep_close = '+---------------------------------------------------------------------------------------------+'
header_warning = '\n-- ! WARNING ! --\n'
header_error = '\n-- ! ERROR ! --\n'
header_gmx_error = sep + '\n  GMX ERROR MSG\n' + sep + '\n\n'

# String 'S m a r t  .  C G' Ivrit style Fitted/Full
def header_package(module_line):
	return f"""
            
        
             ███████╗██╗    ██╗ █████╗ ██████╗ ███╗   ███╗       ██████╗ ██████╗ 
             ██╔════╝██║    ██║██╔══██╗██╔══██╗████╗ ████║      ██╔════╝██╔════╝ 
             ███████╗██║ █╗ ██║███████║██████╔╝██╔████╔██║█████╗██║     ██║  ███╗
             ╚════██║██║███╗██║██╔══██║██╔══██╗██║╚██╔╝██║╚════╝██║     ██║   ██║
             ███████║╚███╔███╔╝██║  ██║██║  ██║██║ ╚═╝ ██║      ╚██████╗╚██████╔╝
             ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝       ╚═════╝ ╚═════╝   v {swarmcg.__version__ }
            {module_line}
{sep_close}
|                 Swarm-CG is distributed under the terms of the MIT License.                 |
|                                                                                             |
|                    Feedback, questions and bug reports are welcome at:                      |
|                        {config.github_url}/issues                          |
|                                                                                             |
|                 If you found Swarm-CG useful in your research, please cite:                 |
|            Swarm-CG: Automatic parametrization of bonded terms in MARTINI-based             |
|            CG models of simple to complex molecules via FST-PSO, 2020 ACS Omega,             |
|    Empereur-mot C., Pesce L., Doni G., Bochicchio D., Capelli R., Perego C., Pavan G.M.     |
|                                                                                             |
|                               Swarm-CG relies on FST-PSO:                                   |
|          Fuzzy Self-Tuning PSO: A settings-free algorithm for global optimization,          |
|  2018 Swarm Evo Comp, Nobile M.S., Cazzaniga P., Besozzi D., Colombo R., Mauri G., Pasia G. |
{sep_close}
"""

ANALYSE_DESCR = """
This module produces a visual summary (big plot) of an optimization procedure started with
module 'scg_optimize' to refine the bonded terms of a coarse-grained (CG) molecular model.
It works whether the optimization is ongoing or finished. The plot will be produced in the
directory provided via argument -opti_dir.

Top row displays bonded terms score (global and breakdown) together with radius of gyration
(Rg) and solvent accessible surface area (SASA) estimations. We call these estimations because
they are calculated on short simulations used during optimization (time depends on parameters
used for optimization), therefore one should always run a long simulation at the end of the
optimizaton process, from which one can calculate the real Rg and SASA values for your model.

Other rows display bond, angle and dihedral parameters tested together with their independant
score (distance from the AA distributions using EMD/Wasserstein). This allows to diagnose
issues, notably related to the topology defined in the ITP file, for example if the score
cannot go down for a specific group of bonds, angles or dihedrals. The optimization procedure
is in principle robust, as demonstrated in the paper, however problems can arise from the CG
representation used (e.g. if topology is too restrictive or incorrectly defined) and non-bonded
parameters (e.g. strong intra-molecular attractions that would not allow the molecule to adopt
extended conformations).
"""

EVALUATE_DESCR = """
This module enables quick evaluation of the fit of bond, angle and dihedral distributions between
a CG model trajectory and a reference AA model trajectory of an identical molecule, in a single
comprehensive figure. The figure's rows display bond, angle and dihedral distributions for groups
present in your system according to the ITP file.

Arguments allows to specify scaling of the AA bonds used as reference to optimize the CG model.
An image displaying all AA reference distributions will be created at the very beginning of the
optimization process. You can check it to make sure scaling is conform to your expectations.

The CG model preliminary ITP file follows the standard ITP format, with one subtlety. The file
can include groups of bonds, angles and dihedrals that will be considered identical. Their
distributions will be averaged within groups. This is important to obtain reliable results for
symmetrical molecules. Groups can be formed using empty line(s) or comment(s), like this:

  [ angles ]

  ; i     j     k    funct   angle  force.c.
  ; grp 1
    5     6    10        1     150       40  
    9     8    11        1     150       40  
  ; grp 2
    1     6    10        2     120        0      ; NOTE: either comment(s) or empty line(s)
    4     8    11        2     120        0      ;       separate groups of bonds/ang/dihe.

The AA trajectory is mapped on-the-fly using file from argument -cg_map, which uses gromacs NDX
file format. Periodic boundary conditions are handled internally if the input trajectories
contain box dimensions.
"""

OPTIMISE_DESCR = """
This module automatically optimizes the bonded parameters of a CG model to best match the bonds,
angles and dihedrals distributions of a reference AA model. Different sets of bonded parameters
are explored via swarm optimization (FST-PSO) and iterative CG simulations. Bonded parameters are
evaluated for the matching they produce between AA and CG distributions via a scoring function
relying on the Earth Movers' Distance (EMD/Wasserstein). The process is designed to execute in
4-24h on a standard desktop machine, according to hardware, molecule size and simulations setup. 

This module has 2 optimization modes:

(1) TUNE BOTH BONDS LENGTHS, ANGLES/DIHEDRALS VALUES AND THEIR FORCE CONSTANTS. First uses
    Boltzmann Inversion to estimate bonds lengths, angles/dihedrals values and their force
    constants, then runs optimization to best fit the reference AA-mapped distributions.

(2) TUNE ONLY FORCE CONSTANTS FOR BONDS/ANGLES/DIHEDRALS, WITH CONSTANT EQUILIBRIUM VALUES.
    Equilibrium values of angles/dihedrals provided in the preliminary CG ITP model are
    conserved while optimization will aim at best fitting reference AA-mapped distributions.

Independently of parameters, the expected input is:

(1) Atomistic trajectory of the molecule   (gromacs binary TPR + trajectory files XTC TRR)
(2) Mapping file, atoms to CG beads        (gromacs NDX format)
(3) CG model ITP file to be optimized      (group identical bonds/angles/dihedrals, see below)
(4) CG simulation files                    (initial configuration GRO + system TOP + MDP files)

You can prepare a directory using default input filenames, then provide only argument -in_dir.
If -in_dir is provided, all filenames provided as arguments will also be searched for within
this directory. Demonstration data are available at ''' + config.github_url + '''.

Arguments allows to specify scaling of the AA bonds used as reference to optimize the CG model.
An image displaying all AA reference distributions will be created at the very beginning of the
optimization process. You can check it to make sure scaling is conform to your expectations.

The CG model preliminary ITP file follows the standard ITP format, with one subtlety. The file
can include groups of bonds, angles and dihedrals that will be considered identical. Their
distributions will be averaged within groups. This is important to obtain reliable results for
symmetrical molecules. Groups can be formed using empty line(s) or comment(s), like this:

[ angles ]

; i     j     k    funct   angle  force.c.
; grp 1
    5     6    10        1     150       40      ; NOTE 1: force constants can be set to 0
    9     8    11        1     150       40      ;         in the prelim. model to optimize
; grp 2
    1     6    10        2     120        0      ; NOTE 2: either comment(s) or empty line(s)
    4     8    11        2     120        0      ;         separate groups of bonds/ang/dihe.

The AA trajectory is mapped on-the-fly using file from argument -cg_map, which uses gromacs NDX
file format. Periodic boundary conditions are handled internally if the input AA trajectory
contains box dimensions
"""