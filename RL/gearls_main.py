'''
Author: Qin Yusen
email: qinys2001@163.com
LastEditTime: 2026-05-07 16:27:35
Description: 
'''
import sys, os
import time
import shutil
import multiprocessing as mp
from multiprocessing import Pool
import torch
import numpy as np
import aiger
import json, math, pickle
import argparse
from tqdm import tqdm
import pandas as pd
from collections import Counter
from RLEnv import RLEnvironment, ABC_PATH, YOSYS_PATH, STA_PATH
from stable_baselines3 import PPO
parent_dir = os.path.abspath(os.path.dirname(__file__)) + '/..'
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from processaig.flex_aig import FlexAIG

def find_target_endpoint(design_in_file, len_eps = 10, abc_path=ABC_PATH, sta_path=STA_PATH):
    design = design_in_file.split('/')[-1].split('.')[0]
    rmaig = False
    if design_in_file not in os.listdir('.'):
        os.system(f'cp {design_in_file} .')
        rmaig = True
    
    new_din = design + '.aig'
    abc_cmd = f'{abc_path} -q "read {new_din}; read_lib ../asap7/asap7sc7p5t_RVT_SS_nldm.lib; map; topo; \
        buffer; write_verilog {design}.v"'
    os.system(abc_cmd)
    f = open('find_port.tcl', 'w')
    f.write(f'read_verilog {design}.v \n')
    f.write('read_liberty ../asap7/asap7sc7p5t_RVT_SS_nldm.lib \n')
    f.write(f'link_design {design} \n')
    f.write(f'report_checks -unconstrained -unique_paths_to_endpoint -group_count {len_eps} > report.txt \n')
    f.write('exit')
    f.close()
    os.system(f'{sta_path} -no_splash find_port.tcl')
    os.remove('find_port.tcl')
    os.remove(f'{design}.v')

    ports = []
    with open('report.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            if l.startswith('Endpoint'):
                port = l.split()[1]
                ports.append(port)
    os.remove('report.txt')

    if rmaig:
        os.remove(new_din)
    return ports

def optimize_cone_worker(args):
    """
    Worker function for parallel processing of a single logic cone.
    Extracts, optimizes, merges, evaluates, and returns results.
    args: (design, original_aig_path, endpoint, model_path, device, clk_prd, task_id,
           abc_path, yosys_path, sta_path, mbert_path)
    """
    design, original_aig_path, endpoint, model_path, device, clk_prd, task_id, abc_path, yosys_path, sta_path, mbert_path = args
    
    # Create a unique temporary directory for each process
    temp_dir = f"./temp_{design}_{task_id}"
    os.makedirs(temp_dir, exist_ok=True)
    tot_neg_time = 0.0
    
    try:
        # 1. Load the model and environment in the subprocess
        model = PPO.load(model_path, device=device)
        env_kwargs = dict(device=device, rwd='native', flow_len=5,
                          abc_path=abc_path, yosys_path=yosys_path, sta_path=sta_path)
        if mbert_path:
            env_kwargs['bert_dir'] = mbert_path
        RLEnv = RLEnvironment([original_aig_path], **env_kwargs)

        # 2. Extract the logic cone
        # print(f"[Task {task_id}] Extracting cone for endpoint: {endpoint}")
        RLEnv.reset()
        obs, neg_time = RLEnv.extract_largest_cone_features(target_node=endpoint, dir=temp_dir)
        tot_neg_time += neg_time

        # 3. Optimize using the RL agent
        # obs, _ = RLEnv.reset(cone_aig_path)
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, rwd, done, _, info = RLEnv.step(action.item(), cal_reward=False)
            obs, neg_time = RLEnv.extract_largest_cone_features(target_node=endpoint, dir=temp_dir)
            tot_neg_time += neg_time
        opt_flow = RLEnv.readGenedFlow()

        # 5. Merge the optimized cone back into a copy of the original circuit
        version_output = os.path.join(temp_dir, f"{design}_v.aig")
        RLEnv._abc.write(version_output)

        # 6. Evaluate timing (scl_mapping + report_timing)
        to_sequential(version_output, design, out_dir=temp_dir)
        scl_mapping(os.path.join(temp_dir, design+'_opt.aig'), design, out_dir=temp_dir,
                abc_path=abc_path, yosys_path=yosys_path)
        wns, tns = report_timing(design, f"{temp_dir}/{design}.v", target_clock=clk_prd,
                     task_dir=temp_dir, sta_path=sta_path)

        return wns, tns, opt_flow, temp_dir, tot_neg_time

    except Exception as e:
        raise RuntimeError(f"Error in worker {task_id} for endpoint {endpoint}: {e}")
    finally:
        # Keep merged_aig_path for later selection; clean other temporary files
        # Final cleanup is performed in the main process
        pass

def circuit_opt(design, aig_file, target_period, model_path, device, current_tns = -math.inf,
                abc_path=ABC_PATH, yosys_path=YOSYS_PATH, sta_path=STA_PATH, mbert_path=None):
    """
    Single round optimization for a circuit-level aiger.
    """
    critical_ports = find_target_endpoint(aig_file, len_eps=10, abc_path=abc_path, sta_path=sta_path)
    if not critical_ports:
        print(f"No critical ports found for {design}. Skipping optimization.")
        return aig_file, 0.0

    best_aig_this_round = aig_file
    tasks = [(design, best_aig_this_round, port, model_path, device, target_period, idx,
              abc_path, yosys_path, sta_path, mbert_path) for idx, port in enumerate(critical_ports)]
        
    results = []
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=len(tasks)) as pool:
        results = list(pool.imap(optimize_cone_worker, tasks))
    # for task in tasks:
    #     result = optimize_cone_worker(task)
    #     results.append(result)

    results = sorted(results, key=lambda x: x[1], reverse=True)
    _, tns, opt_flow, best_dir, neg_time = results[0]
    neg_time = max(results, key=lambda x: x[4])[4]
    if tns > current_tns:
        shutil.copy(f'{best_dir}/{design}_v.aig', f"{design}_opt.aig")
        shutil.copy(f'{best_dir}/{design}.v', f"../output/{design}.v")
        print(f"\nBest TNS improved to: {tns}, Optimize flow: {opt_flow}\n")
    for _, _, _, dir_path,_ in results:
        shutil.rmtree(dir_path)

    return tns, neg_time

def circuit_process(design, ori_aig_in, model_path, device, target_period, iter_round = 2,
                    abc_path=ABC_PATH, yosys_path=YOSYS_PATH, sta_path=STA_PATH, mbert_path=None):
    aig_input = ori_aig_in
    current_tns = -math.inf
    tot_neg_time = 0.0
    for i in range(iter_round):
        print(f"Processing {design}, iteration {i+1}/{iter_round}")
        current_tns, neg_time = circuit_opt(design, aig_input, target_period, model_path, device, current_tns,
                                            abc_path=abc_path, yosys_path=yosys_path, sta_path=sta_path,
                                            mbert_path=mbert_path)
        aig_input = design+'_opt.aig'
        tot_neg_time += neg_time

    return tot_neg_time
    
def batch_process(designs, periods, device, model_path, iter_round = 2,
                  abc_path=ABC_PATH, yosys_path=YOSYS_PATH, sta_path=STA_PATH, mbert_path=None):
    runtime_dict = {}
    for i, design in enumerate(designs):
        time_start = time.time()
        ori_aig_in = '../aigers/comb/'+design+'.aig'
        neg_time = circuit_process(design, ori_aig_in, model_path, device, periods[i], iter_round=iter_round,
                                   abc_path=abc_path, yosys_path=yosys_path, sta_path=sta_path, mbert_path=mbert_path)
        time_end = time.time()
        print(f"Time for {design}: {time_end - time_start} seconds")
        print(f"Clean Optimize time cost: {time_end - time_start - neg_time:.4f} sec.")
        runtime_dict[design] = {'total_time_sec': time_end - time_start,
                                'clean_opt_time_sec': time_end - time_start - neg_time}


def to_sequential(aig_file, design, out_dir=None):
    aig_obj = aiger.load(aig_file)
    f_aig = FlexAIG()
    f_aig.copy_aiger(aig_obj)
    with open(f'../aigers/comb/{design}_mapback.pkl', 'rb') as f:
        reg_map_dict = pickle.load(f)
    loop_pi = []
    loop_po = []
    for pi in f_aig.inputs:
        if pi in reg_map_dict:
            loop_pi.append(pi)
            po = pi[:-2] + '_D'
            loop_po.append(po)
    f_aig.loopBack(loop_pi, loop_po)
    if out_dir is None:
        f_aig.toAIG().write(design+'_opt.aag')
        os.system(f'aigtoaig {design}_opt.aag {design}_opt.aig')
        os.remove(design+'_opt.aag')
    else:
        f_aig.toAIG().write(out_dir+'/'+design+'_opt.aag')
        os.system(f'aigtoaig {out_dir}/{design}_opt.aag {out_dir}/{design}_opt.aig')
        os.remove(out_dir+'/'+design+'_opt.aag')

def scl_mapping(in_file, design, lib = '../asap7/asap7sc7p5t_RVT_SS_nldm.lib', seq_lib = '../asap7/asap7sc7p5t_SEQ_RVT_SS_nldm_220123.lib', out_dir='.', abc_path=ABC_PATH, yosys_path=YOSYS_PATH):
    verilog_out = design + '_temp.v'
    if out_dir is not None:
        verilog_out = out_dir + '/' + verilog_out
    scl_map_cmd = f'{abc_path} -q "read {in_file}; read_lib {lib}; read_constr ../aigers/IO_constr.txt; map; topo; buffer; write_verilog {verilog_out}"'
    os.system(scl_map_cmd)

    ys_script = 'dffmap.ys'
    if out_dir is not None:
        ys_script = out_dir + '/' + ys_script
    with open(ys_script, 'w') as f:
        f.write('read_verilog '+verilog_out+' \n')
        f.write('proc -noopt; techmap; rename -top '+design+' \n')
        f.write('dfflibmap -liberty '+seq_lib+' \n')
        f.write('abc -fast -constr ../aigers/IO_constr.txt -liberty '+lib+' \n')
        f.write(f'write_verilog -noattr {out_dir}/'+design+'.v \n')
    os.system(f'{yosys_path} -q {ys_script}')
    os.remove(verilog_out)
    os.remove(ys_script)

def report_timing(design, file_in, target_clock, task_dir=None, sta_path=STA_PATH):
    sta_script = f'{task_dir}/wns_tns.tcl' if task_dir is not None else 'wns_tns.tcl'
    report_out = f'{task_dir}/wns_tns.rpt' if task_dir is not None else 'wns_tns.rpt'
    with open(sta_script , 'w') as f:
        f.write('read_verilog '+file_in+' \n')
        f.write('read_liberty ../asap7/asap7sc7p5t_RVT_SS_nldm.lib \n')
        f.write('read_liberty ../asap7/asap7sc7p5t_SEQ_RVT_SS_nldm_220123.lib \n')
        f.write(f'link_design {design} \n')
        f.write(f'create_clock -name clk -period {target_clock} {{clock}} \n')
        f.write('report_wns > '+report_out+' \n')
        f.write('report_tns >> '+report_out+' \n')
        f.write('exit')
    os.system(' '.join([sta_path, '-no_splash', sta_script]))

    with open(report_out, 'r') as f:
        lines = f.readlines()
    wns = float(lines[0].split()[-1])
    tns = float(lines[1].split()[-1])
    os.remove(sta_script)
    os.remove(report_out)
    return wns, tns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GearLS optimization in batch or single-design mode')
    parser.add_argument('--design', type=str, help='Design name to process (e.g., sha512)')
    parser.add_argument('--period', type=float, help='Target clock period for the design')
    parser.add_argument('--abc', type=str, default=ABC_PATH, help='Path to ABC binary')
    parser.add_argument('--yosys', type=str, default=YOSYS_PATH, help='Path to Yosys binary')
    parser.add_argument('--sta', type=str, default=STA_PATH, help='Path to OpenSTA binary')
    parser.add_argument('--mbert', type=str, default=None, help='Path to MobileBERT model directory')
    parser.add_argument('--device', type=str, default='cuda', help='Compute device, e.g., cpu, cuda, cuda:0')
    parser.add_argument('--model', type=str, default='checkpoints/trained_model.zip', help='Trained RL model path')
    parser.add_argument('--iter_round', type=int, default=2, help='Iterations per circuit')

    # No extra args -> run batch_process (default behavior)
    if len(sys.argv) == 1:
        designs = ['sha512', 'aes_core', 'des','double_fpu','ethernet',
                   'gfx_transform','pci', 'spi', 'tv80', 
                   'ifft64', 'aor3000','jpeg', 'gfx_cuvz']
        clk_p = [2600, 800, 600, 1200, 950, 1450, 900, 850, 1300, 1150, 2600, 1350, 1800]
        device = torch.device('cuda')
        batch_process(designs, clk_p, device, 'checkpoints/trained_model.zip', iter_round=2,
                      abc_path=ABC_PATH, yosys_path=YOSYS_PATH, sta_path=STA_PATH, mbert_path=None)
        sys.exit(0)

    if '-h' in sys.argv or '--help' in sys.argv:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if args.design is None:
        parser.error('--design is required when passing command-line arguments')
    if args.period is None:
        parser.error('--period is required when --design is provided')

    device = torch.device(args.device)
    ori_aig_in = f'../aigers/comb/{args.design}.aig'
    print(f"Running single-design optimization for {args.design} with period {args.period}")
    time_st = time.time()
    neg_time = circuit_process(args.design, ori_aig_in, args.model, device, args.period,
                               iter_round=args.iter_round,
                               abc_path=args.abc, yosys_path=args.yosys, sta_path=args.sta, mbert_path=args.mbert)
    time_end = time.time()
    print(f"Time for {args.design}: {time_end - time_st} seconds")
    print(f"Clean Optimize time cost: {time_end - time_st - neg_time:.4f} sec.")

