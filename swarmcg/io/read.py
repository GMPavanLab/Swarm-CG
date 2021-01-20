import re
import warnings

import MDAnalysis as mda

from swarmcg import config
from swarmcg.shared import exceptions


def read_aa_traj(ns):
    """Read atomistic trajectory"""
    print('Reading All Atom (AA) trajectory')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                category=ImportWarning)  # ignore warning: "bootstrap.py:219: ImportWarning: can't resolve package from __spec__ or __package__, falling back on __name__ and __path__"
        ns.aa_universe = mda.Universe(ns.aa_tpr_filename, ns.aa_traj_filename,
                                      in_memory=True, refresh_offsets=True,
                                      guess_bonds=False)  # setting guess_bonds=False disables angles, dihedrals and improper_dihedrals guessing, which is activated by default in some MDA versions
    print('  Found', len(ns.aa_universe.trajectory), 'frames')


def read_itp(filename):
    """Read itp file and return string for processing"""
    with open(filename, 'r') as fp:
        try:
            itp_lines = fp.read().split('\n')
            itp_lines = [itp_line.split(';')[0].strip() for itp_line in itp_lines]
        except UnicodeDecodeError:
            msg = "Cannot read CG ITP, it seems you provided a binary file."
            raise exceptions.MissformattedFile(msg)
    return itp_lines


def verify_handled_functions(geom, func, line_nb):
    """Check if functions present in CG ITP file can be used by this program, if not we throw
    an error authorized functions are defined in config.py (we switch them on in config.py
    once we have tested them)"""
    try:
        func = int(func)
    except (ValueError, IndexError):
        msg = (
            f"Unexpected error while reading CG ITP file at line {line_nb}, please check this file."
        )
        raise exceptions.MissformattedFile(msg)

    if func not in config.handled_functions[geom]:
        functions_str = ", ".join(map(str, config.handled_functions[geom]))
        if functions_str == '':
            functions_str = 'None'
        msg = (
            f"Error while reading {geom} function in CG ITP file at line {line_nb}.\n"
            f"This potential function is not implemented in Swarm-CG at the moment.\n"
            f"Please use one of these {geom} potential functions: {functions_str}.\n\n"
            f"If you feel this is an important missing feature, please feel free to\n"
            f"open an issue on github at {config.github_url}/issues."
        )
        raise exceptions.MissformattedFile(msg)

    return func


# TODO: the 3 next functions below (section_switch, vs_error_control, read_cg_itp_file) could be isolated in a sort of
#       "topology reader" class, and next we would include the formats from other MD engines

def section_switch(section_read, section_active):
    """Sections switch for reading ITP sections"""
    for section_current in section_read:
        section_read[section_current] = False
    if section_active is not None:
        section_read[section_active] = True


def vs_error_control(cg_itp, bead_id, vs_type, func, line_nb, vs_def_beads_ids=None):
    """Check itpd fields; vs_type in [2, 3, 4, n], then they each have specific functions
    to define their positions"""
    if bead_id >= len(cg_itp['atoms']):
        msg = (
            f"A virtual site is defined for ID {bead_id + 1}, while this ID exceeds the number of atoms"
            f" defined in the CG ITP file."
        )
        raise exceptions.MissformattedFile(msg)

    if vs_def_beads_ids is not None:
        for bid in vs_def_beads_ids:
            if bid >= len(cg_itp['atoms']):
                msg = (
                    f"The definition of virtual site ID {bead_id + 1} makes use of ID {bid + 1}, while this ID exceeds"
                    f" the number of atoms defined in the CG ITP file."
                )
                raise exceptions.MissformattedFile(msg)

    if not cg_itp['atoms'][bead_id]['bead_type'].startswith('v'):
        msg = (
            f"CG bead number {bead_id + 1} is referenced to as a virtual site, but its bead type"
            f" does NOT start with letter 'v'."
        )
        raise exceptions.MissformattedFile(msg)

    vs_type_str = f'virtual_sites{vs_type}'
    func = verify_handled_functions(vs_type_str, func, line_nb)

    return func


def validate_cg_itp(cg_itp, **kwargs):
    # verify we have as many real CG beads (i.e. NOT virtual sites) in the ITP than in the mapping file
    if "all_beads" in kwargs:
        if len(kwargs["all_beads"]) != len(cg_itp["real_beads_ids"]):
            msg = (
                "The CG beads mapping (NDX) file does NOT include as many CG beads as the ITP file.\n"
                "Please check the NDX and ITP files you provided."
            )
            raise exceptions.MissformattedFile(msg)
    else:
        return True


def read_cg_itp_file(ns):
    """Read coarse-grain ITP"""
    print('Reading Coarse-Grained (CG) ITP file')
    cg_itp = {'moleculetype': {'molname': '', 'nrexcl': 0}, 'atoms': [], 'constraint': [],
              'bond': [], 'angle': [], 'dihedral': [], 'virtual_sites2': {},
              'virtual_sites3': {}, 'virtual_sites4': {}, 'virtual_sitesn': {}, 'exclusion': []}
    real_beads_ids, vs_beads_ids = [], []
    nb_constraints, nb_bonds, nb_angles, nb_dihedrals = -1, -1, -1, -1

    itp_lines = read_itp(ns.cg_itp_filename)

    section_read = {
        'moleculetype': False,
        'atom': False,
        'constraint': False,
        'bond': False,
        'angle': False,
        'dihedral': False,
        'vs_2': False,
        'vs_3': False,
        'vs_4': False,
        'vs_n': False,
        'exclusion': False
    }

    def msg_force_boundaries(line, min_fct, max_fct, str_arg):
        msg = (
            f'You activated the option to take into account the parameters provided via\n'
            f'the ITP file, but the input force constant provided at line {line} is outside\n'
            f'of the range of authorized values: [{min_fct}, {max_fct}].\n'
            f'Please either modify the argument {str_arg} or modify the force\n'
            f'constant in your ITP file.'
        )
        return msg

    for i in range(len(itp_lines)):
        itp_line = itp_lines[i]
        if itp_line != '':

            if bool(re.search(r'\[.*moleculetype.*\]', itp_line)):
                section_switch(section_read, 'moleculetype')
            elif bool(re.search(r'\[.*atoms.*\]', itp_line)):
                section_switch(section_read, 'atom')
            elif bool(re.search(r'\[.*constraint.*\]', itp_line)):
                section_switch(section_read, 'constraint')
            elif bool(re.search(r'\[.*bond.*\]', itp_line)):
                section_switch(section_read, 'bond')
            elif bool(re.search(r'\[.*angle.*\]', itp_line)):
                section_switch(section_read, 'angle')
            elif bool(re.search(r'\[.*dihedral.*\]', itp_line)):
                section_switch(section_read, 'dihedral')
            elif bool(re.search(r'\[.*virtual_sites2.*\]', itp_line)):
                section_switch(section_read, 'vs_2')
            elif bool(re.search(r'\[.*virtual_sites3.*\]', itp_line)):
                section_switch(section_read, 'vs_3')
            elif bool(re.search(r'\[.*virtual_sites4.*\]', itp_line)):
                section_switch(section_read, 'vs_4')
            elif bool(re.search(r'\[.*virtual_sitesn.*\]', itp_line)):
                section_switch(section_read, 'vs_n')
            elif bool(re.search(r'\[.*exclusion.*\]', itp_line)):
                section_switch(section_read, 'exclusion')
            elif bool(re.search(r'\[.*\]', itp_line)):  # all other sections
                section_switch(section_read, None)

            else:
                sp_itp_line = itp_line.split()

                if section_read['moleculetype']:

                    cg_itp['moleculetype']['molname'], cg_itp['moleculetype']['nrexcl'] = \
                        sp_itp_line[0], int(sp_itp_line[1])

                elif section_read['atom']:

                    # TODO: test what happens if there are VS in the middle of real CG beads in the [ atoms ] section
                    #       because most probably this won't be OK -- Not sure who does this though, but seems possible

                    if len(sp_itp_line) == 7:
                        bead_id, bead_type, resnr, residue, atom, cgnr, charge = sp_itp_line[:7]
                        mass = None

                    # In case the masses are ABSENT in the ITP file (probably the most normal case with
                    # MARTINI usage), then we will read the CG masses from the TPR file to avoid having
                    # to look into TOP and potentially multiple ITP files:
                    #
                    #  - from evaluate_model.py this means a TPR has been provided already, if the user is not using
                    #    the script for exclusive AA distributions inspection (in which case we don't need the masses
                    #    at all because we will use mapped/splitted weights of the atoms into CG beads exclusively anyway)
                    #  - from optimize_model.py all the ITP included in the TOP file are read to find
                    #    appropriate masses

                    elif len(sp_itp_line) == 8:
                        bead_id, bead_type, resnr, residue, atom, cgnr, charge, mass = sp_itp_line[
                                                                                       :8]
                        mass = float(mass)
                    else:
                        msg = (
                            "The atom description from the input itp file: \n\n {} \n\n"
                            "does not contain the correct number of fields. Please insert "
                            "the following information: \n\n  bead_id, bead_type, resnr, "
                            "residue, atom, cgnr, charge, [mass] \n\n".format(itp_line)
                        )
                        raise exceptions.MissformattedFile(msg)

                    # discriminate between real beads and virtual sites
                    if bead_type.startswith('v'):
                        vs_beads_ids.append(int(bead_id) - 1)
                    else:
                        real_beads_ids.append(int(bead_id) - 1)

                    # assignment of the variables value
                    cg_itp['atoms'].append(
                        {'bead_id': int(bead_id) - 1, 'bead_type': bead_type, 'resnr': int(resnr),
                         'residue': residue, 'atom': atom, 'cgnr': int(cgnr),
                         'charge': float(charge), 'mass': mass, 'vs_type': None})
                    # here there is still MASS and VS_TYPE that are subject to later modification

                    if not len(cg_itp['atoms']) == int(bead_id):
                        msg = (
                            f"Swarm-CG handles .itp files with atoms indexed consecutively starting from 1.\n"
                            f"The bead numbered {bead_id + 1} does not follow this formatting."
                        )
                        raise exceptions.MissformattedFile(msg)

                elif section_read['constraint']:

                    # beginning of a new group
                    if itp_lines[i - 1] == '' or itp_lines[i - 1].startswith(';') or bool(
                            re.search(r'\[.*constraint.*\]', itp_lines[i - 1])):
                        nb_constraints += 1
                        if itp_lines[i - 1].startswith('; constraint type'):
                            geom_type = itp_lines[i - 1].split()[
                                3]  # if the current CG ITP was generated with our package

                        else:
                            geom_type = str(len(cg_itp['constraint']) + 1)
                        cg_itp['constraint'].append(
                            {'geom_type': geom_type, 'beads': [], 'func': [], 'value': [],
                             'value_user': []})  # initialize storage for this new group

                    try:
                        cg_itp['constraint'][nb_constraints]['beads'].append(
                            [int(bead_id) - 1 for bead_id in sp_itp_line[
                                                             0:2]])  # retrieve indexing from 0 for CG beads IDS for MDAnalysis
                    except ValueError:
                        msg = (
                            "Incorrect reading of the CG ITP file within [constraints] section.\n"
                            "Please check this file."
                        )
                        raise exceptions.MissformattedFile(msg)

                    func = verify_handled_functions('constraint', sp_itp_line[2], i + 1)
                    cg_itp['constraint'][nb_constraints]['func'].append(func)
                    cg_itp['constraint'][nb_constraints]['value'].append(
                        float(sp_itp_line[3]))

                elif section_read['bond']:

                    # beginning of a new group
                    if itp_lines[i - 1] == '' or itp_lines[i - 1].startswith(';') or bool(
                            re.search(r'\[.*bond.*\]', itp_lines[i - 1])):
                        nb_bonds += 1
                        if itp_lines[i - 1].startswith('; bond type'):
                            geom_type = itp_lines[i - 1].split()[
                                3]  # if the current CG ITP was generated with our package
                        else:
                            geom_type = str(len(cg_itp['bond']) + 1)
                        cg_itp['bond'].append(
                            {'geom_type': geom_type, 'beads': [], 'func': [], 'value': [],
                             'value_user': [], 'fct': [],
                             'fct_user': []})  # initialize storage for this new group

                    try:
                        cg_itp['bond'][nb_bonds]['beads'].append(
                            [int(bead_id) - 1 for bead_id in sp_itp_line[
                                                             0:2]])  # retrieve indexing from 0 for CG beads IDS for MDAnalysis
                    except ValueError:
                        msg = (
                            "Incorrect reading of the CG ITP file within [bonds] section.\n"
                            "Please check this file."
                        )
                        raise exceptions.MissformattedFile(msg)

                    func = verify_handled_functions('bond', sp_itp_line[2], i + 1)
                    cg_itp['bond'][nb_bonds]['func'].append(func)
                    cg_itp['bond'][nb_bonds]['value'].append(float(sp_itp_line[3]))
                    cg_itp['bond'][nb_bonds]['value_user'].append(float(sp_itp_line[3]))
                    cg_itp['bond'][nb_bonds]['fct'].append(float(sp_itp_line[4]))
                    cg_itp['bond'][nb_bonds]['fct_user'].append(float(sp_itp_line[4]))

                    if ns.user_input and not 0 <= float(
                            sp_itp_line[4]) <= ns.default_max_fct_bonds_opti:
                        raise exceptions.MissformattedFile(
                            msg_force_boundaries(i + 1, 0, ns.default_max_fct_bonds_opti,
                                                 '-max_fct_bonds_f1'))

                elif section_read['angle']:

                    # beginning of a new group
                    if itp_lines[i - 1] == '' or itp_lines[i - 1].startswith(';') or bool(
                            re.search(r'\[.*angle.*\]', itp_lines[i - 1])):
                        nb_angles += 1
                        if itp_lines[i - 1].startswith('; angle type'):
                            geom_type = itp_lines[i - 1].split()[
                                3]  # if the current CG ITP was generated with our package
                        else:
                            geom_type = str(len(cg_itp['angle']) + 1)
                        cg_itp['angle'].append(
                            {'geom_type': geom_type, 'beads': [], 'func': [], 'value': [],
                             'value_user': [], 'fct': [],
                             'fct_user': []})  # initialize storage for this new group

                    try:
                        cg_itp['angle'][nb_angles]['beads'].append(
                            [int(bead_id) - 1 for bead_id in sp_itp_line[
                                                             0:3]])  # retrieve indexing from 0 for CG beads IDS for MDAnalysis
                    except ValueError:
                        msg = (
                            "Incorrect reading of the CG ITP file within [angles] section.\n"
                            "Please check this file."
                        )
                        raise exceptions.MissformattedFile(msg)

                    func = verify_handled_functions('angle', sp_itp_line[3], i + 1)
                    cg_itp['angle'][nb_angles]['func'].append(func)
                    cg_itp['angle'][nb_angles]['value'].append(float(sp_itp_line[4]))
                    cg_itp['angle'][nb_angles]['value_user'].append(float(sp_itp_line[4]))
                    cg_itp['angle'][nb_angles]['fct'].append(float(sp_itp_line[5]))
                    cg_itp['angle'][nb_angles]['fct_user'].append(float(sp_itp_line[5]))

                    if ns.user_input:
                        if func == 1 and not 0 <= float(
                                sp_itp_line[5]) <= ns.default_max_fct_angles_opti_f1:
                            raise exceptions.MissformattedFile(
                                msg_force_boundaries(i + 1, 0, ns.default_max_fct_angles_opti_f1,
                                                     '-max_fct_angles_f1'))
                        elif func == 2 and not 0 <= float(
                                sp_itp_line[5]) <= ns.default_max_fct_angles_opti_f2:
                            raise exceptions.MissformattedFile(
                                msg_force_boundaries(i + 1, 0, ns.default_max_fct_angles_opti_f2,
                                                     '-max_fct_angles_f2'))

                elif section_read['dihedral']:

                    # beginning of a new group
                    if itp_lines[i - 1] == '' or itp_lines[i - 1].startswith(';') or bool(
                            re.search(r'\[.*dihedral.*\]', itp_lines[i - 1])):
                        nb_dihedrals += 1
                        if itp_lines[i - 1].startswith('; dihedral type'):
                            geom_type = itp_lines[i - 1].split()[
                                3]  # if the current CG ITP was generated with our package
                        else:
                            geom_type = str(len(cg_itp['dihedral']) + 1)
                        cg_itp['dihedral'].append(
                            {'geom_type': geom_type, 'beads': [], 'func': [], 'value': [],
                             'value_user': [], 'fct': [], 'fct_user': [],
                             'mult': []})  # initialize storage for this new group

                    try:
                        cg_itp['dihedral'][nb_dihedrals]['beads'].append(
                            [int(bead_id) - 1 for bead_id in sp_itp_line[
                                                             0:4]])  # retrieve indexing from 0 for CG beads IDS for MDAnalysis
                    except ValueError:
                        msg = (
                            "Incorrect reading of the CG ITP file within [dihedrals] section.\n"
                            "Please check this file."
                        )
                        raise exceptions.MissformattedFile(msg)

                    func = verify_handled_functions('dihedral', sp_itp_line[4], i + 1)
                    cg_itp['dihedral'][nb_dihedrals]['func'].append(func)
                    cg_itp['dihedral'][nb_dihedrals]['value'].append(float(
                        sp_itp_line[5]))  # issue happens here for functions that are not handled
                    cg_itp['dihedral'][nb_dihedrals]['value_user'].append(
                        float(sp_itp_line[5]))
                    cg_itp['dihedral'][nb_dihedrals]['fct'].append(float(sp_itp_line[6]))
                    cg_itp['dihedral'][nb_dihedrals]['fct_user'].append(float(sp_itp_line[6]))

                    if ns.user_input:
                        if func in config.dihedral_func_with_mult and not -ns.default_abs_range_fct_dihedrals_opti_func_with_mult <= float(
                                sp_itp_line[
                                    6]) <= ns.default_abs_range_fct_dihedrals_opti_func_with_mult:
                            raise exceptions.MissformattedFile(msg_force_boundaries(i + 1,
                                                                                    -ns.default_abs_range_fct_dihedrals_opti_func_with_mult,
                                                                                    ns.default_abs_range_fct_dihedrals_opti_func_with_mult,
                                                                                    '-max_fct_dihedrals_f149'))
                        elif func == 2 and not -ns.default_abs_range_fct_dihedrals_opti_func_without_mult <= float(
                                sp_itp_line[
                                    6]) <= ns.default_abs_range_fct_dihedrals_opti_func_without_mult:
                            raise exceptions.MissformattedFile(msg_force_boundaries(i + 1,
                                                                                    -ns.default_abs_range_fct_dihedrals_opti_func_without_mult,
                                                                                    ns.default_abs_range_fct_dihedrals_opti_func_without_mult,
                                                                                    '-max_fct_dihedrals_f2'))

                    # handle multiplicity if function assumes multiplicity
                    if func in config.dihedral_func_with_mult:
                        try:
                            cg_itp['dihedral'][nb_dihedrals]['mult'].append(
                                int(sp_itp_line[7]))
                        except (IndexError, ValueError):  # incorrect read of multiplicity
                            msg = f"Incorrect read of multiplicity in dihedral with potential function {func} at ITP line {i + 1}."
                            raise exceptions.MissformattedFile(msg)
                    else:  # no multiplicity parameter is expected
                        cg_itp['dihedral'][nb_dihedrals]['mult'].append(None)

                elif section_read['vs_2']:

                    vs_type = 2
                    bead_id = int(sp_itp_line[0]) - 1
                    vs_def_beads_ids = [int(bid) - 1 for bid in sp_itp_line[1:3]]
                    func = sp_itp_line[
                        3]  # will be casted to int in the verification below (for factorizing checks)
                    func = vs_error_control(cg_itp, bead_id, vs_type, func, i + 1,
                                            vs_def_beads_ids)  # i is the line number
                    vs_params = float(sp_itp_line[4])
                    cg_itp['atoms'][bead_id]['vs_type'] = vs_type
                    cg_itp['virtual_sites2'][bead_id] = {'bead_id': bead_id, 'func': func,
                                                            'vs_def_beads_ids': vs_def_beads_ids,
                                                            'vs_params': vs_params}

                elif section_read['vs_3']:

                    vs_type = 3
                    bead_id = int(sp_itp_line[0]) - 1
                    vs_def_beads_ids = [int(bid) - 1 for bid in sp_itp_line[1:4]]
                    func = sp_itp_line[
                        4]  # will be casted to int in the verification below (for factorizing checks)
                    func = vs_error_control(cg_itp, bead_id, vs_type, func, i + 1,
                                            vs_def_beads_ids)  # i is the line number
                    if func in [1, 2, 3]:
                        vs_params = [float(param) for param in sp_itp_line[5:7]]
                    elif func == 4:
                        vs_params = [float(param) for param in sp_itp_line[5:8]]
                    cg_itp['atoms'][bead_id]['vs_type'] = vs_type
                    cg_itp['virtual_sites3'][bead_id] = {'bead_id': bead_id, 'func': func,
                                                            'vs_def_beads_ids': vs_def_beads_ids,
                                                            'vs_params': vs_params}

                elif section_read['vs_4']:

                    vs_type = 4
                    bead_id = int(sp_itp_line[0]) - 1
                    vs_def_beads_ids = [int(bid) - 1 for bid in sp_itp_line[1:5]]
                    func = sp_itp_line[
                        5]  # will be casted to int in the verification below (for factorizing checks)
                    func = vs_error_control(cg_itp, bead_id, vs_type, func, i + 1,
                                            vs_def_beads_ids)  # i is the line number
                    vs_params = [float(param) for param in sp_itp_line[6:9]]
                    cg_itp['atoms'][bead_id]['vs_type'] = vs_type
                    cg_itp['virtual_sites4'][bead_id] = {'bead_id': bead_id, 'func': func,
                                                            'vs_def_beads_ids': vs_def_beads_ids,
                                                            'vs_params': vs_params}

                elif section_read['vs_n']:

                    vs_type = 'n'
                    bead_id = int(sp_itp_line[0]) - 1
                    func = sp_itp_line[
                        1]  # will be casted to int in verification below (for factorizing checks)
                    # here we do the check in 2 steps, because the reading of beads_ids depends on the function
                    func = vs_error_control(cg_itp, bead_id, vs_type, func, i + 1,
                                            vs_def_beads_ids=None)  # i is the line number
                    if func == 3:
                        vs_def_beads_ids = [int(sp_itp_line[2:][i]) - 1 for i in
                                            range(0, len(sp_itp_line[2:]), 2)]
                        vs_params = [float(sp_itp_line[2:][i]) for i in
                                     range(1, len(sp_itp_line[2:]), 2)]
                    else:
                        vs_def_beads_ids = [int(bid) - 1 for bid in sp_itp_line[2:]]
                        vs_params = None
                    func = vs_error_control(cg_itp, bead_id, vs_type, func, i + 1,
                                            vs_def_beads_ids)  # i is the line number
                    cg_itp['atoms'][bead_id]['vs_type'] = vs_type
                    cg_itp['virtual_sitesn'][bead_id] = {'bead_id': bead_id, 'func': func,
                                                            'vs_def_beads_ids': vs_def_beads_ids,
                                                            'vs_params': vs_params}

                elif section_read['exclusion']:

                    cg_itp['exclusion'].append([int(bead_id) - 1 for bead_id in sp_itp_line])

    # error handling, verify that funct, value and fct are all identical within the group, as they should be, and reduce arrays to single elements
    # TODO: make these messages more clear and CORRECT for the dihedral function handling -- also explain this is the current Opti.CG implementation, function 9 might come in next version
    # TODO: check what kind of error or processing is done when a correct line is duplicated within a group ?? probably it goes on in a bad way

    def msg(geom, grp_geom):
        str_msg = (
            f"In the provided CG ITP file {geom}s have been grouped, but {geom}s group "
            f"{str(grp_geom + 1)} holds lines that have different parameters.\nParameters should be "
            f"identical within a group, only CG beads IDs should differ.\n"
            f"Please correct the CG ITP file and separate groups using a blank or commented line."
        )
        return str_msg

    for geom in ['constraint']:  # constraints only
        for grp_geom in range(len(cg_itp[geom])):
            for var in ['func', 'value', 'value_user']:
                var_set = set(cg_itp[geom][grp_geom][var])
                if len(var_set) == 1:
                    cg_itp[geom][grp_geom][var] = var_set.pop()
                else:
                    raise exceptions.MissformattedFile(msg(geom, grp_geom))

    for geom in ['bond', 'angle']:  # bonds and angles only
        for grp_geom in range(len(cg_itp[geom])):
            for var in ['func', 'value', 'value_user', 'fct', 'fct_user']:
                var_set = set(cg_itp[geom][grp_geom][var])
                if len(var_set) == 1:
                    cg_itp[geom][grp_geom][var] = var_set.pop()
                else:
                    raise exceptions.MissformattedFile(msg(geom, grp_geom))

    for geom in ['dihedral']:  # dihedrals only
        for grp_geom in range(len(cg_itp[geom])):
            for var in ['func', 'value', 'value_user', 'fct', 'fct_user']:
                var_set = set(cg_itp[geom][grp_geom][var])
                if len(var_set) == 1:
                    cg_itp[geom][grp_geom][var] = var_set.pop()
                else:
                    raise exceptions.MissformattedFile(msg(geom, grp_geom))

            for var in ['mult']:
                var_set = set(cg_itp[geom][grp_geom][var])
                if len(var_set) == 1:
                    cg_itp[geom][grp_geom][var] = var_set.pop()
                else:
                    raise exceptions.MissformattedFile(msg(geom, grp_geom))

    nb_constraints += 1
    nb_bonds += 1
    nb_angles += 1
    nb_dihedrals += 1
    print(f'  Found {len(real_beads_ids)} beads')
    print(f'  Found {len(vs_beads_ids)} virtual sites')
    print(f'  Found {nb_constraints} constraints groups')
    print(f'  Found {nb_bonds} bonds groups')
    print(f'  Found {nb_angles} angles groups')
    print(f'  Found {nb_dihedrals} dihedrals groups')
    cg_itp["real_beads_ids"] = real_beads_ids
    cg_itp["vs_beads_ids"] = vs_beads_ids
    cg_itp["nb_bonds"] = nb_bonds
    cg_itp["nb_angles"] = nb_angles
    cg_itp["nb_dihedrals"] = nb_dihedrals
    cg_itp["nb_constraints"] = nb_constraints
    return cg_itp
