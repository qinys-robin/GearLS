'''
Author: Qin Yusen
email: qinys2001@163.com
LastEditTime: 2026-04-15 15:36:19
Description: 
'''
import os
import aiger
from flex_aig import *
import pickle
import multiprocessing

designs = ['ac97_ctrl','aes_core','aor3000','des','double_fpu','ethernet',
           'gfx_cuvz','gfx_transform','jpeg','md5','sha512','pci','spi','tv80']

def batch_convert(designs):
    for des in designs:
        d_aig = aiger.load(f'../aigers/{des}.aig')
        df_aig = FlexAIG()
        df_aig.copy_aiger(d_aig)
        splist = []
        for node in df_aig.outport_map.values():
            if node.Type == 'Latch':
                splist.append(node.name)
        for node in df_aig.latch_map.values():
            if node.Type == 'Latch':
                splist.append(node.name)
        print(len(splist))
        mapback = df_aig.toCombinational()
        df_aig.toAIG().write(f'{des}_comb.aag')
        os.system(f'aigtoaig {des}_comb.aag {des}.aig')
        os.system(f'rm {des}_comb.aag')
        os.system(f'mv {des}.aig ../aigers/comb/')
        with open(f'../aigers/comb/{des}_mapback.pkl', 'wb') as f:
            pickle.dump(mapback, f)

tasks = [['ac97_ctrl','aes_core','des','double_fpu','ethernet'],
         ['gfx_cuvz','gfx_transform','sha512','spi'],['md5','pci','tv80'],
         ['aor3000']]
if __name__ == '__main__':
   with multiprocessing.Pool(processes=4) as pool:
       pool.map(batch_convert, tasks)
