import os
import sys

des = sys.argv[1]
with open('toaig.abc', 'w') as f:
    f.write('read_blif {}.blif \n'.format(des))
    f.write('st;ps;zero \n')
    f.write('write_aiger -s {}.aig'.format(des))

os.system('abc -f toaig.abc')
os.system('rm {}.blif'.format(des))
os.system('mv {}.aig ../aigers/'.format(des))
