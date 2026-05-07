'''
Author: Qin Yusen
email: qinys2001@163.com
LastEditTime: 2026-05-07 15:49:00
Description: 
'''
from trainingFuncs_new import train_process, continue_train_process
from stateGCNandSeqEncode import RLNetwork
import torch
import os
import json

dataset_dir = '../aigers/subcones'
training_set = ['sha512', 'aes_core', 'des','double_fpu','ethernet',
               'aor3000', 'gfx_transform','jpeg', 'ac97_ctrl']
test_set = ['pci', 'spi', 'tv80', 'ifft64']
category = {
    'ac97_ctrl': 'control',
    'aes_core': 'crypto',
    'des': 'crypto',
    'double_fpu': 'mcu',
    'ethernet': 'control',
    'gfx_transform': 'mcu',
    'jpeg': 'video',
    'aor3000': 'mcu',
    'sha512': 'crypto',
    'gfx_cuvz': 'mcu',
    'ifft64': 'crypto',
    'pci': 'control',
    'spi': 'control',
    'tv80': 'mcu'
}
category = {
    'ac97_ctrl': 'crypto',
    'aes_core': 'crypto',
    'des': 'control',
    'double_fpu': 'mcu',
    'ethernet': 'control',
    'gfx_transform': 'crypto',
    'jpeg': 'video',
    'aor3000': 'mcu',
    'sha512': 'mcu',
    'gfx_cuvz': 'mcu',
    'ifft64': 'crypto',
    'pci': 'control',
    'spi': 'control',
    'tv80': 'mcu'
}
if __name__ == '__main__':
    crypt, control, mcu, video = [], [], [], []
    '''
    crypt_drv, control_drv, mcu_drv, video_drv = [], [], [], []
    crypt_dly, control_dly, mcu_dly, video_dly = [], [], [], []
    crypt_cels, control_cels, mcu_cels, video_cels = [], [], [], []
    crypt_regs, control_regs, mcu_regs, video_regs = [], [], [], []
    
    with open('circuit_info/cell_num.json', 'r') as f:
        cell_num_dict = json.load(f)
    '''
    crypt_cut, control_cut, mcu_cut, video_cut = [], [], [], []

    for design in training_set:
        des_cone_dir = os.path.join(dataset_dir, design)
        # cells_num = cell_num_dict[design]
        cone_files = []
        cut_weight_files = []
        # cone_dlys, cone_drvs = [], []
        with open('circuit_info/{}_drivers.json'.format(design), 'r') as f:
            drivers = json.load(f)
        with open('circuit_info/{}_pred.json'.format(design), 'r') as f:
            delays = json.load(f)
        for file in os.listdir(des_cone_dir):
            if file.endswith('.aig'):
                cone_files.append(os.path.join(des_cone_dir, file))
                ep = file[:-4]
                # cone_dlys.append(delays[ep])
                # cone_drvs.append(drivers[ep])
            if file.endswith('.json'):
                cut_weight_files.append(os.path.join(des_cone_dir, file))
        
        if category[design] == 'crypto':
            crypt.extend(cone_files)
            crypt_cut.extend(cut_weight_files)
            # crypt_dly.extend(cone_dlys)
            # crypt_drv.extend(cone_drvs)
            # crypt_cels.extend([cells_num[0]] * len(cone_files))
            # crypt_regs.extend([cells_num[1]] * len(cone_files))
        elif category[design] == 'control':
            control.extend(cone_files)
            control_cut.extend(cut_weight_files)
            # control_dly.extend(cone_dlys)
            # control_drv.extend(cone_drvs)
            # control_cels.extend([cells_num[0]] * len(cone_files))
            # control_regs.extend([cells_num[1]] * len(cone_files))
        elif category[design] == 'mcu':
            mcu.extend(cone_files)
            mcu_cut.extend(cut_weight_files)
            # mcu_dly.extend(cone_dlys)
            # mcu_drv.extend(cone_drvs)
            # mcu_cels.extend([cells_num[0]] * len(cone_files))
            # mcu_regs.extend([cells_num[1]] * len(cone_files))
        elif category[design] == 'video':
            video.extend(cone_files)
            video_cut.extend(cut_weight_files)
            # video_dly.extend(cone_dlys)
            # video_drv.extend(cone_drvs)
            # video_cels.extend([cells_num[0]] * len(cone_files))
            # video_regs.extend([cells_num[1]] * len(cone_files))
    
    cone_lists = []
    # delay_lists, driver_lists, cells_lists = [], [], []
    # regs_lists = []
    weight_lists = []
    def split_list(input_list):
        n = len(input_list) // 4
        return input_list[:n], input_list[n:2*n], input_list[2*n:3*n], input_list[3*n:]
    def type_split(out_lists, crypt, control, mcu, video):
        for type_list in [crypt, control, mcu, video]:
            if type_list is None:
                continue
            for splited in split_list(type_list):
                out_lists.append(splited)
    #type_split(cone_lists, crypt, control, mcu, video)
    # type_split(delay_lists, crypt_dly, control_dly, mcu_dly,video_dly)
    # type_split(driver_lists, crypt_drv, control_drv, mcu_drv, video_drv)
    # type_split(cells_lists, crypt_cels, control_cels, mcu_cels, video_cels)
    # type_split(regs_lists, crypt_regs, control_regs, mcu_regs, video_regs)
    #type_split(weight_lists, crypt_cut, control_cut, mcu_cut, video_cut)
    type_split(cone_lists, crypt, control, mcu, None)
    type_split(weight_lists, crypt_cut, control_cut, mcu_cut, None)
    
    for design in test_set:
        des_cone_dir = os.path.join(dataset_dir, design)
        # cells_num = cell_num_dict[design]
        cone_files = []
        # cone_dlys, cone_drvs = [], []
        cut_weight_files = []
        # with open('circuit_info/{}_drivers.json'.format(design), 'r') as f:
        #     drivers = json.load(f)
        # with open('circuit_info/{}_pred.json'.format(design), 'r') as f:
        #     delays = json.load(f)
        for file in os.listdir(des_cone_dir):
            if file.endswith('.aig'):
                cone_files.append(os.path.join(des_cone_dir, file))
                ep = file[:-4]
                # cone_dlys.append(delays[ep])
                # cone_drvs.append(drivers[ep])
            if file.endswith('.json'):
                cut_weight_files.append(os.path.join(des_cone_dir, file))
        cone_lists.append(cone_files)
        # delay_lists.append(cone_dlys)
        # driver_lists.append(cone_drvs)
        # cells_lists.append([cells_num[0]] * len(cone_files))
        # regs_lists.append([cells_num[1]] * len(cone_files))
        weight_lists.append(cut_weight_files)
    
    compute = torch.device('cuda:0')
    #rl_nn = RLNetwork()
    test_model = 'checkpoints/20251016/model_train_point_873600_steps.zip'
    #result = train_process(cone_lists, weight_lists, compute, rwd='shaped')
    result = continue_train_process(test_model, cone_lists, weight_lists, compute, more_timesteps=360000)
