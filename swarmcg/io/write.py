def write_cg_itp_file(itp_obj, out_path_itp,
                      print_sections=['constraint', 'bond', 'angle', 'dihedral', 'exclusion']):
    """Print coarse-grain ITP. Here we have a switch for print_sections because we might want
    to optimize constraints/bonds/angles/dihedrals separately, so we can leave some out with
    the switch and they will be optimized later"""
    with open(out_path_itp, 'w') as fp:

        fp.write('[ moleculetype ]\n')
        fp.write('; molname        nrexcl\n')
        fp.write('{0:<4} {1:>13}\n'.format(itp_obj['moleculetype']['molname'],
                                           itp_obj['moleculetype']['nrexcl']))

        fp.write('\n\n[ atoms ]\n')
        fp.write('; id type resnr residue   atom  cgnr    charge     mass\n\n')

        for i in range(len(itp_obj['atoms'])):
            # if the ITP did NOT contain masses, they are set at 0 in this field during ITP reading
            if itp_obj['atoms'][i]['mass'] is not None:
                fp.write(
                    '{0:<4} {1:>4}    {6:>2}  {2:>6} {3:>6}  {4:<4} {5:9.5f}     {7:<5.2f}\n'.format(
                        itp_obj['atoms'][i]['bead_id'] + 1, itp_obj['atoms'][i]['bead_type'],
                        itp_obj['atoms'][i]['residue'], itp_obj['atoms'][i]['atom'], i + 1,
                        itp_obj['atoms'][i]['charge'],
                        itp_obj['atoms'][i]['resnr'], itp_obj['atoms'][i]['mass']))
            else:
                fp.write('{0:<4} {1:>4}    {6:>2}  {2:>6} {3:>6}  {4:<4} {5:9.5f}\n'.format(
                    itp_obj['atoms'][i]['bead_id'] + 1, itp_obj['atoms'][i]['bead_type'],
                    itp_obj['atoms'][i]['residue'], itp_obj['atoms'][i]['atom'], i + 1,
                    itp_obj['atoms'][i]['charge'],
                    itp_obj['atoms'][i]['resnr']))

        if 'constraint' in print_sections and 'constraint' in itp_obj and len(
                itp_obj['constraint']) > 0:
            fp.write('\n\n[ constraints ]\n')
            fp.write(';   i     j   funct   length\n')

            for j in range(len(itp_obj['constraint'])):

                constraint_type = itp_obj['constraint'][j]['geom_type']
                fp.write('\n; constraint type ' + constraint_type + '\n')
                grp_val = itp_obj['constraint'][j]['value']

                for i in range(len(itp_obj['constraint'][j]['beads'])):
                    fp.write('{beads[0]:>5} {beads[1]:>5} {0:>7} {1:8.3f}      ; {2}\n'.format(
                        itp_obj['constraint'][j]['func'], grp_val, constraint_type,
                        beads=[bead_id + 1 for bead_id in itp_obj['constraint'][j]['beads'][i]]))

        if 'bond' in print_sections and 'bond' in itp_obj and len(itp_obj['bond']) > 0:
            fp.write('\n\n[ bonds ]\n')
            fp.write(';   i     j   funct   length   force.c.\n')

            for j in range(len(itp_obj['bond'])):

                bond_type = itp_obj['bond'][j]['geom_type']
                fp.write('\n; bond type ' + bond_type + '\n')
                grp_val, grp_fct = itp_obj['bond'][j]['value'], itp_obj['bond'][j]['fct']

                for i in range(len(itp_obj['bond'][j]['beads'])):
                    fp.write(
                        '{beads[0]:>5} {beads[1]:>5} {0:>7} {1:8.3f}  {2:7.2f}           ; {3}\n'.format(
                            itp_obj['bond'][j]['func'], grp_val, grp_fct, bond_type,
                            beads=[bead_id + 1 for bead_id in itp_obj['bond'][j]['beads'][i]]))

        if 'angle' in print_sections and 'angle' in itp_obj and len(itp_obj['angle']) > 0:
            fp.write('\n\n[ angles ]\n')
            fp.write(';   i     j     k   funct     angle   force.c.\n')

            for j in range(len(itp_obj['angle'])):

                angle_type = itp_obj['angle'][j]['geom_type']
                fp.write('\n; angle type ' + angle_type + '\n')
                grp_val, grp_fct = itp_obj['angle'][j]['value'], itp_obj['angle'][j]['fct']

                for i in range(len(itp_obj['angle'][j]['beads'])):
                    fp.write(
                        '{beads[0]:>5} {beads[1]:>5} {beads[2]:>5} {0:>7} {1:9.2f}   {2:7.2f}           ; {3}\n'.format(
                            itp_obj['angle'][j]['func'], grp_val, grp_fct, angle_type,
                            beads=[bead_id + 1 for bead_id in itp_obj['angle'][j]['beads'][i]]))

        if 'dihedral' in print_sections and 'dihedral' in itp_obj and len(itp_obj['dihedral']) > 0:
            fp.write('\n\n[ dihedrals ]\n')
            fp.write(';   i     j     k     l   funct     dihedral   force.c.   mult.\n')

            for j in range(len(itp_obj['dihedral'])):

                dihedral_type = itp_obj['dihedral'][j]['geom_type']
                fp.write('\n; dihedral type ' + dihedral_type + '\n')
                grp_val, grp_fct = itp_obj['dihedral'][j]['value'], itp_obj['dihedral'][j]['fct']

                for i in range(len(itp_obj['dihedral'][j]['beads'])):

                    # handle writing of multiplicity
                    multiplicity = itp_obj['dihedral'][j]['mult']
                    if multiplicity == None:
                        multiplicity = ''

                    fp.write(
                        '{beads[0]:>5} {beads[1]:>5} {beads[2]:>5} {beads[3]:>5} {0:>7}    {1:9.2f} {2:7.2f}       {4}     ; {3}\n'.format(
                            itp_obj['dihedral'][j]['func'], grp_val, grp_fct, dihedral_type,
                            multiplicity,
                            beads=[bead_id + 1 for bead_id in itp_obj['dihedral'][j]['beads'][i]]))

        # here starts 4 almost identical blocks, that differ only by vs_2, vs_3, vs_4, vs_n
        # but we could still need to write several of these sections (careful if factorizing this)
        if len(itp_obj['virtual_sites2']) > 0:
            fp.write('\n\n[ virtual_sites2 ]\n')
            fp.write(';  vs     i     j  func   param\n')
            for bead_id in itp_obj['virtual_sites2']:
                fp.write('{0:>5} {beads[0]:>5} {beads[1]:>5} {1:>5}   {2}\n'.format(
                    str(itp_obj['virtual_sites2'][bead_id]['bead_id'] + 1),
                    str(itp_obj['virtual_sites2'][bead_id]['func']),
                    itp_obj['virtual_sites2'][bead_id]['vs_params'],
                    beads=[bid + 1 for bid in
                           itp_obj['virtual_sites2'][bead_id]['vs_def_beads_ids']])
                )

        if len(itp_obj['virtual_sites3']) > 0:
            fp.write('\n\n[ virtual_sites3 ]\n')
            fp.write(';  vs     i     j     k  func   params\n')
            for bead_id in itp_obj['virtual_sites3']:
                fp.write('{0:>5} {beads[0]:>5} {beads[1]:>5} {beads[2]:>5} {1:>5}   {2}\n'.format(
                    str(itp_obj['virtual_sites3'][bead_id]['bead_id'] + 1),
                    str(itp_obj['virtual_sites3'][bead_id]['func']),
                    '  '.join(map(str, itp_obj['virtual_sites3'][bead_id]['vs_params'])),
                    beads=[bid + 1 for bid in
                           itp_obj['virtual_sites3'][bead_id]['vs_def_beads_ids']])
                )

        if len(itp_obj['virtual_sites4']) > 0:
            fp.write('\n\n[ virtual_sites4 ]\n')
            fp.write(';  vs     i     j     k     l  func   params\n')
            for bead_id in itp_obj['virtual_sites4']:
                fp.write(
                    '{0:>5} {beads[0]:>5} {beads[1]:>5} {beads[2]:>5} {beads[3]:>5} {1:>5}   {2}\n'.format(
                        str(itp_obj['virtual_sites4'][bead_id]['bead_id'] + 1),
                        str(itp_obj['virtual_sites4'][bead_id]['func']),
                        '  '.join(map(str, itp_obj['virtual_sites4'][bead_id]['vs_params'])),
                        beads=[bid + 1 for bid in
                               itp_obj['virtual_sites4'][bead_id]['vs_def_beads_ids']])
                )

        if len(itp_obj['virtual_sitesn']) > 0:
            fp.write('\n\n[ virtual_sitesn ]\n')
            fp.write(';  vs  func     def\n')
            for bead_id in itp_obj['virtual_sitesn']:
                params = []
                if itp_obj['virtual_sitesn'][bead_id]['func'] == 3:
                    for i in range(len(itp_obj['virtual_sitesn'][bead_id]['vs_def_beads_ids'])):
                        params.append('{} {}'.format(
                            itp_obj['virtual_sitesn'][bead_id]['vs_def_beads_ids'][i] + 1,
                            itp_obj['virtual_sitesn'][bead_id]['vs_params'][i]))
                    params = '  '.join(params)
                else:
                    params = ' '.join(['{:>4}'.format(bid + 1) for bid in
                                       itp_obj['virtual_sitesn'][bead_id]['vs_def_beads_ids']])
                fp.write('{:>5} {:>5}   {}\n'.format(
                    itp_obj['virtual_sitesn'][bead_id]['bead_id'] + 1,
                    itp_obj['virtual_sitesn'][bead_id]['func'],
                    params)
                )

        if 'exclusion' in print_sections and 'exclusion' in itp_obj and len(
                itp_obj['exclusion']) > 0:
            fp.write('\n\n[ exclusions ]\n')
            fp.write(';   i     j\n\n')

            for j in range(len(itp_obj['exclusion'])):
                fp.write(('{:>4} ' * len(itp_obj['exclusion'][j]) + '\n').format(
                    *[bead_id + 1 for bead_id in itp_obj['exclusion'][j]]))

        fp.write('\n\n')
