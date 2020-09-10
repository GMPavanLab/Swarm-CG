import swarmcg
from .. import config

sep = '----------------------------------------------------------------------'
sep_close = '+---------------------------------------------------------------------------------------------+'
header_warning = '\n-- ! WARNING ! --\n'
header_error = '\n-- ! ERROR ! --\n'
header_gmx_error = sep + '\n  GMX ERROR MSG\n' + sep + '\n\n'

# String 'S m a r t  .  C G' Ivrit style Fitted/Full
def header_package(module_line):
	return '''\
            
        
             ███████╗██╗    ██╗ █████╗ ██████╗ ███╗   ███╗       ██████╗ ██████╗ 
             ██╔════╝██║    ██║██╔══██╗██╔══██╗████╗ ████║      ██╔════╝██╔════╝ 
             ███████╗██║ █╗ ██║███████║██████╔╝██╔████╔██║█████╗██║     ██║  ███╗
             ╚════██║██║███╗██║██╔══██║██╔══██╗██║╚██╔╝██║╚════╝██║     ██║   ██║
             ███████║╚███╔███╔╝██║  ██║██║  ██║██║ ╚═╝ ██║      ╚██████╗╚██████╔╝
             ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝       ╚═════╝ ╚═════╝   v ''' + swarmcg.__version__ +'''
            ''' + module_line +'''
''' + sep_close + '''
|               Swarm-CG is distributed under the terms of the MIT License.                   |
|                                                                                             |
|                    Feedback, questions and bug reports are welcome at:                      |
|                        ''' + config.github_url +'''/issues                          |
|                                                                                             |
|                 If you found Swarm-CG useful in your research, please cite:                 |
|              Swarm-CG: Automatic parametrization of bonded terms in CG models               |
|                        of simple to complex molecules via FST-PSO                           |
|        Empereur-mot C., Pesce L., Bochicchio D., Perego C., Pavan G.M. ChemRxiv 2020        |
|                                                                                             |
|                               Swarm-CG relies on FST-PSO:                                   |
|          Fuzzy Self-Tuning PSO: A settings-free algorithm for global optimization           |
|  Nobile M.S., Cazzaniga P., Besozzi D., Colombo R., Mauri G., Pasia G. Swarm Evo Comp 2018  |
''' + sep_close + '\n'