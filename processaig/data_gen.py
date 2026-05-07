'''
Author: Qin Yusen, qinys2001@163.com
Date: 2025-03-10 14:49:37
LastEditTime: 2026-05-07 16:21:04
FilePath: /processaig/data_gen.py
Description: 
'''
import time
import os
import aiger
from flex_aig import *
import json

design_list = ['ac97_ctrl', 'aes_core', 'des','double_fpu','ethernet','gfx_cuvz',
               'gfx_transform','ifft64','jpeg','aor3000','pci','sha512','spi', 'tv80']
training_set = ['ac97_ctrl', 'aes_core', 'des','double_fpu','ethernet',
               'gfx_transform','jpeg','aor3000','sha512']

skip_eps = {'pci': ['$auto$async2sync.cc:228:execute$23180'],
            'tv80': ['i_tv80_core.TmpAddr']}

#with open('../place_timing/critical_endpoints.json', 'r') as f:
#    selected = json.load(f)

def extractCones(design, eps):
    if len(eps) > 600:
        eps = eps[:600]
    d_aig = aiger.load(f'../aigers/{design}.aig')
    df_aig = FlexAIG()
    df_aig.copy_aiger(d_aig)
    cone_num = 0
    w_th = 0.75
    num_t = 20
    if design == 'ac97_ctrl':
        w_th = 0.5

    if not os.path.exists('../aigers/subcones/'+design):
        os.makedirs('../aigers/subcones/'+design, exist_ok=True)
    for ep in eps:
        #print(ep)
        slfag = False
        if design in skip_eps:
            for skips in skip_eps[design]:
                if ep.startswith(skips):
                    print(f'skip {ep}')
                    slfag = True
        if slfag:
            continue
        _, splited_cone, cut_weights = df_aig.extractCone(ep, cal_cut_weight=True)
        splited_cone.toAIG().write(f'{design}_cone.aag')
        ep_re = ep.replace('$', '\\$')
        os.system(f'aigtoaig {design}_cone.aag {ep_re}.aig')
        os.system(f'rm {design}_cone.aag')
        os.system(f'mv {ep_re}.aig ../aigers/subcones/{design}/{ep_re}.aig')

        with open(f'../aigers/subcones/{design}/{ep}_cutweights.json', 'w') as f:
            json.dump(cut_weights, f, indent=4)
        cone_num += 1

def selectCrticalEpsFromLabel(des):
    with open(f'../netlist_beta/{des}/noopt_label.json', 'r') as f:
        label_dict = json.load(f)
    sorted_eps = sorted(label_dict.keys(), key=lambda x: label_dict[x], reverse=True)
    max_delay = label_dict[sorted_eps[0]]
    selected_eps = []
    uni_delay = set()
    delay_th = 0.75
    if des == 'ac97_ctrl':
        delay_th = 0.6

    for ep in sorted_eps:
        if label_dict[ep] < max_delay * delay_th:
            break
        delay_rounded = round(label_dict[ep], 2)
        if delay_rounded not in uni_delay:
            selected_eps.append(ep)
            uni_delay.add(delay_rounded)
        if len(selected_eps) >= 400:
            break

    print(f'{des}: {len(selected_eps)} critical endpoints selected.')
    return selected_eps
        
for des in design_list:
    c_eps = selectCrticalEpsFromLabel(des)
    extractCones(des, c_eps)
