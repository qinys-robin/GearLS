import os

def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root,file))
    return file_paths

def gen_read_script(design, mod_map):
    dir = '../RTLlib/'+design
    all_files = get_all_file_paths(dir)
    scr_file = 'read_and_hier.ys'
    with open(scr_file, 'w') as f:
        for rtl in all_files:
            f.write('read_verilog -nolatches {}\n'.format(rtl))
        f.write('hierarchy -check -top {}\n'.format(mod_map[design]))
        f.write('proc -noopt; fsm; memory \n')
        f.write('techmap \n')
        f.write('stat')

def gen_write_script(design, mod_map):
    scr_file = 'write_aig_sog.ys'
    f = open(scr_file, 'w')
    '''SOG mapping'''
    f.write('flatten \n')
    f.write('abc -fast -g AND,OR,MUX,XOR \n')
    f.write('write_verilog -noattr ../netlist/{0}/{0}_noopt_sog.v \n'.format(design))
    '''AIG mapping'''
    f.write('async2sync \n')
    f.write('dfflegalize -cell $_DFF_P_ 0 \n')
    f.write('aigmap \n')
    f.write('write_blif {}.blif \n'.format(design))
    '''Standard cell mapping'''
    f.write('dfflibmap -liberty ../asap7/asap7sc7p5t_SEQ_RVT_SS_nldm_220123.lib \n')
    f.write('abc -liberty ../asap7/asap7sc7p5t_RVT_SS_nldm.lib -constr IO_constr.txt \n')
    f.write('write_verilog -noattr ../netlist/{0}/{0}_noopt.v'.format(design))
    f.close()

module_map = {'ac97_ctrl':'ac97_top', 'aes_core':'aes_cipher_top', 'ethernet':'eth_top',
              'pci':'pci_bridge32','spi':'spi_top','tv80':'tv80s', 'vga_lcd':'vga_enh_top',
              'wb_conmax':'wb_conmax_top'}
designs = [d for d in module_map.keys()]
des = 'ethernet'
if not os.path.exists('../netlist/'+des):
    os.mkdir('../netlist/'+des)
gen_read_script(des, module_map)
gen_write_script(des, module_map)
#os.system('yosys read_and_hier.ys')
