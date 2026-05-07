'''
Author: Qin Yusen
email: qinys2001@163.com
LastEditTime: 2026-05-07 15:03:48
Description: 
'''
import sys, os
parent_dir = os.path.abspath(os.path.dirname(__file__)) + '/..'
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from stateGCNandSeqEncode import RLNetwork
import abc_py as abcpy
import torch_geometric
import torch_geometric.data
import gymnasium as gym
from gymnasium import spaces
from collections import OrderedDict
from transformers import AutoTokenizer, MobileBertModel
import torch
import numpy as np
import re, shutil, time
import math, json
from processaig.flex_aig import FlexAIG
import aiger

# Modify these
BERT_MODEL = '/home/qinyusen23/mylib/mobileBERT'
ABC_PATH = 'abc'
YOSYS_PATH = '/home/qinyusen23/mylib/yosys/build/yosys'
STA_PATH = '/home/qinyusen23/mylib/OpenSTA/build/sta'

synthesisOpToPosDic = \
{
     0: "refactor",
     1: "refactor -z",
     2: "rewrite" ,
     3: "rewrite -z" ,
     4: "resub" ,
     5: "resub -z",
     6: "balance" ,
     7: "refactor -z -l",
     8: "rewrite -z -l",
     9: "resub -z -l"
}

simplified_actions = {
     "refactor": "rf",
     "refactor-z": "rfz",
     "rewrite": "rw" ,
     "rewrite-z": "rwz" ,
     "resub": "rs" ,
     "resub-z": "rsz",
     "balance": "b" ,
     "refactor-z-l": "rfz -l",
     "rewrite-z-l": "rwz -l",
     "resub-z-l": "rsz -l"
}

def normalize(tensor):
    abs_val = torch.abs(tensor)
    min_val = abs_val.min()
    max_val = abs_val.max()
    return (5*tensor) / (max_val)

def signed_log(tensor):
    tensor = tensor.clone()
    tensor[tensor == 0] = 1e-10
    return torch.sign(tensor) * torch.log(torch.abs(tensor))

class RLEnvironment(gym.Env):
    def __init__(self, input_aiger_list, device, abc_path=ABC_PATH, yosys_path=YOSYS_PATH, \
                 sta_path=STA_PATH, bert_dir = BERT_MODEL, rwd='native', flow_len=5, cut_weight:list=None):
        super().__init__()
        self._abc = abcpy.AbcInterface()
        self._abc.start()
        self.ori_aigs = input_aiger_list
        self.abc_path = abc_path
        self.yosys_path = yosys_path
        self.sta_path = sta_path
        if rwd == 'native':
            self.reward_type = 'native'
        elif rwd == 'shaped':
            assert cut_weight is not None and len(input_aiger_list) == len(cut_weight)
            self.reward_type = 'shaped'
            self.cut_weight_files = cut_weight
        else:
            raise ValueError("Invalid reward type. Choose 'shaped' or 'native'.")
        print('input_aiger_list:', input_aiger_list[0])
        self.seq_lenth = flow_len
        self.select_aig = -1
        self.libfile = '../asap7/asap7sc7p5t_RVT_SS_nldm.lib'
        self.step_num = 0
        self.gened_flow = []
        # Load tokenizer and MobileBERT model only if a directory/path is provided.
        self.tokenizer = None
        self.bert_model = None
        self.bert_dir = bert_dir
        if bert_dir is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)
                self.bert_model = MobileBertModel.from_pretrained(bert_dir).to(device)
            except Exception as e:
                print(f"Warning: failed to load MobileBERT from {bert_dir}: {e}")
                self.tokenizer = None
                self.bert_model = None
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Dict(OrderedDict({
            'global_feat': spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
        #     'graphData': spaces.Dict({
        #         'x':spaces.Box(low=-np.inf, high=np.inf, shape=(20000,7), dtype=np.float32),
        #         'edge_index':spaces.Box(low=0, high=19999, shape=(2, 20000), dtype=np.int64)
        # }),
            # Not real space description, just a placeholder
            'graphData': spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
            'seq_embedding': spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=np.float32)
        }))
        self.compute_device = device

    def reset(self, seed=None, input_aiger_list=None, cut_weight:list=None):
        if input_aiger_list is not None:
            self.ori_aigs = input_aiger_list
            if self.reward_type == 'shaped':
                assert cut_weight is not None and len(input_aiger_list) == len(cut_weight)
                self.cut_weight_files = cut_weight
            self.select_aig = -1
        self.switchAig()
        global_feat, graphData, seq = self.getStateFeat()
        seq_embedding = self.convertSeqToEmbedding(seq)
        obs = self.wrapState(global_feat, graphData, seq_embedding)
        super().reset(seed=seed)
        info = self.currentInfo()
        return obs,info
    
    def wrapState(self, global_feat, graphData, seq_embedding):
        state = {}
        state['global_feat'] = global_feat.detach().cpu()
        state['graphData'] = graphData.detach().to('cpu')
        state['seq_embedding'] = seq_embedding.detach().cpu()
        return state
    
    def currentInfo(self):
        current_ep = self.ori_aigs[self.select_aig]
        current_step = self.step_num
        current_action = '; '.join(self.gened_flow)
        info = {
            'endpoint': current_ep,
            'steps': current_step,
            'actions': current_action
        }
        return info
    
    def takeAction(self, act_id):
        act_command = synthesisOpToPosDic[act_id]
        if act_id == 0:
            self._abc.refactor(l=False, z=False) #rf
        elif act_id == 1:
            self._abc.refactor(l=False, z=True) #rf -z
        elif act_id == 2:
            self._abc.rewrite(l=False, z=False) #rw -z
        elif act_id == 3:
            self._abc.rewrite(l=False, z=True) #rw -z
        elif act_id == 4:
            self._abc.resub(k=8,n=1,l=False, z=False) #rs
        elif act_id == 5:
            self._abc.resub(k=8,n=1,l=False, z=True) #rs -z
        elif act_id == 6:
            self._abc.balance(l=False) #balance
        elif act_id == 7:
            self._abc.refactor(l=True, z=True)
        elif act_id == 8:
            self._abc.rewrite(l=True, z=True)
        elif act_id == 9:
            self._abc.resub(k=8,n=1,l=True, z=True)
        else:
            print("Invalid action ID")
            return False
        self.step_num += 1
        self.gened_flow.append(act_command.replace(' ',''))

    def isDone(self):
        if self.step_num >= self.seq_lenth:
            return True
        else:
            return False
        
    def switchAig(self):
        self.select_aig += 1
        if self.select_aig >= len(self.ori_aigs):
            self.select_aig = 0
        self._abc.end()
        self._abc.start()
        self._abc.read(self.ori_aigs[self.select_aig])
        self.step_num = 0
        self.gened_flow = []
        
    def getStateFeat(self):
        '''
        Extract all state features: Global handcraft feat, GCN feat, opt sequence

        Global: total AND num, max logic level, num of I/O, total signal num, inverted signal num, average fanout, max fanout
        
        GCN: node type, fanout
        '''
        graphstat = self._abc.aigStats()
        tot_nodes = self._abc.numNodes()
        #print('tot_nodes:', tot_nodes)
        tot_ands = graphstat.numAnd
        max_level = graphstat.lev
        num_in = math.log(graphstat.numIn, 10)
        num_out = math.log(graphstat.numOut, 10)
        tot_sig = 0
        inv_sig = 0
        tot_fanout, max_fanout = 0, 0
        node_feat = torch.zeros((tot_nodes, 7), dtype=torch.float32)
        edge_index = [[],[]]
        for nodeId in range(tot_nodes):
            node = self._abc.aigNode(nodeId)
            ntype = node.nodeType()
            encoded_type = torch.nn.functional.one_hot(torch.tensor(ntype), num_classes=6)
            node_fanout = torch.tensor(node.numFanouts()).unsqueeze(0)
            node_feat[nodeId] = torch.cat((encoded_type.float(), node_fanout.float()), dim=0).type(torch.float32)
            for i in range(node.numFanouts()):
                if node.fanout(i) >= tot_nodes:
                    continue
                edge_index[0].append(nodeId)
                edge_index[1].append(node.fanout(i))
            if ntype < 3:
                continue
            tot_fanout += node.numFanouts()
            max_fanout = max(max_fanout, node.numFanouts())
            tot_sig += 2
            if ntype == 4:
                inv_sig += 1
            if ntype == 5:
                inv_sig += 2
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        graphData = torch_geometric.data.Data(edge_index=edge_index.contiguous(), x=node_feat)
        avg_fanout = float(tot_fanout) / float(tot_ands)
        tot_ands = float(tot_ands) / float(max_level)
        inv_sig = 10 * float(inv_sig) / float(tot_sig)
        tot_sig = math.log(tot_sig, 10)
        global_feat = torch.tensor([tot_ands, max_level, num_in, num_out, tot_sig, inv_sig, avg_fanout, max_fanout], dtype=torch.float32)
        global_feat = normalize(global_feat)
        #global_feat = global_feat.view(1, -1)
        return global_feat, graphData, '; '.join(self.gened_flow)
    
    def convertSeqToEmbedding(self, seq):
        encoded = self.tokenizer.encode_plus(
                text=seq,  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = 32,  # maximum length of a sentence
                padding='max_length',  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )
        encoded.to(self.compute_device)
        #self.bert_model.to(self.compute_device)
        self.bert_model.eval()
        with torch.no_grad():
            seq_embedding = self.bert_model(**encoded)
            seq_embedding = seq_embedding.pooler_output
        seq_embedding = seq_embedding.squeeze()
        seq_embedding = signed_log(seq_embedding)
        return normalize(seq_embedding)

    
    def step(self, action, cal_reward=True):
        reward = 0.0
        self.takeAction(action)
        global_feat, graphData, seq = self.getStateFeat()
        seq_embedding = self.convertSeqToEmbedding(seq)
        obs = self.wrapState(global_feat, graphData, seq_embedding)
        info = self.currentInfo()
        if self.isDone() and cal_reward:
            if self.reward_type == 'native':
                reward, _,_ = self.getNativeReward()
            elif self.reward_type == 'shaped':
                reward, _,_ = self.getShapedReward()
            else:
                raise ValueError("Invalid reward type or missing parameters for prediction.")
        return obs, reward, self.isDone(), False, info
    
    def readGenedFlow(self):
        return '; '.join([simplified_actions[op] for op in self.gened_flow])
    
    def getShapedReward(self):
        file_basename = os.path.splitext(os.path.basename(self.ori_aigs[self.select_aig]))[0]
        file_basename = file_basename.replace('$', '_')
        file_basename = file_basename.replace(':', '_')
        file_basename = file_basename.replace('.', '_')
        file_basename = file_basename.replace('[', '_')
        file_basename = file_basename.replace(']', '_')
        out_file = file_basename+'_'.join(self.gened_flow)+'.aig'
        self._abc.write(out_file)

        temp_ori_file = file_basename+'.aig'
        shutil.copy(self.ori_aigs[self.select_aig], temp_ori_file)

        ori_delay_dict = self.mapAIGandTiming(temp_ori_file, rm_aig=True, verbose_timing=True) 
        new_delay_dict = self.mapAIGandTiming(out_file, rm_aig=True, verbose_timing=True)
        ori_score, new_score = self.weightedDelayShape(ori_delay_dict, new_delay_dict)
        return 10*(ori_score - new_score) / ori_score, ori_score, new_score
    
    def getNativeReward(self):
        file_basename = os.path.splitext(os.path.basename(self.ori_aigs[self.select_aig]))[0]
        file_basename = file_basename.replace('$', '_')
        file_basename = file_basename.replace(':', '_')
        out_file = file_basename+'_'.join(self.gened_flow)+'.aig'
        self._abc.write(out_file)
        ori_delay = self.mapAIGandTiming(self.ori_aigs[self.select_aig]) 
        new_delay = self.mapAIGandTiming(out_file, rm_aig=True)
        return 10*(ori_delay - new_delay) / ori_delay, ori_delay, new_delay
    
    def weightedDelayShape(self, ori_delay_dict, new_delay_dict):
        '''Return: Ori_score, New_score'''
        ori_weighted_sum, new_weighted_sum = 0.0, 0.0
        count = 0
        with open(self.cut_weight_files[self.select_aig], 'r') as f:
            using_weight = json.load(f)
        for ep, delay in ori_delay_dict.items():
            if not ep.startswith('Temp_port'):
                ori_weighted_sum += delay
                new_weighted_sum += new_delay_dict[ep]
                count += 1
                if len(using_weight) == 0:
                    return ori_weighted_sum / count, new_weighted_sum / count
            if ep in using_weight and ep in new_delay_dict:
                weight = using_weight[ep]
                ori_weighted_sum += delay / weight
                new_weighted_sum += new_delay_dict[ep] / weight
                count += 1
        return ori_weighted_sum / count, new_weighted_sum / count

    def mapAIGandTiming(self, in_file, rm_aig=False, resyn=False, verbose_timing=False):
        in_file_reg = in_file.replace('$', '\\$')
        file_basename = os.path.splitext(os.path.basename(in_file))[0]
        file_basename = file_basename.replace('$', '_')
        file_basename = file_basename.replace(':', '_')
        log_file = file_basename+'.log'
        if verbose_timing:
            eval_cmd = f'{self.abc_path} -c "read {in_file_reg}; read_lib {self.libfile}; read_constr cone_constr.txt; map; topo; buffer; write_verilog {file_basename}.v"'
        elif not resyn:
            eval_cmd = f'{self.abc_path} -c "read {in_file_reg}; read_lib {self.libfile}; read_constr cone_constr.txt; map; topo; buffer; stime" > '+log_file
        else:
            eval_cmd = f'{self.abc_path} -c "read {in_file_reg}; read_lib {self.libfile}; read_constr cone_constr.txt; resyn2; map; topo; buffer; stime" > '+log_file
        os.system(eval_cmd)

        if verbose_timing:
            sta_scr = open(file_basename+'.tcl', 'w')
            sta_scr.write(f'read_verilog {file_basename}.v\n')
            sta_scr.write(f'read_lib {self.libfile}\n')
            sta_scr.write(f'link_design {file_basename}\n')
            sta_scr.write(f'report_checks -unconstrained -unique_paths_to_endpoint -group_count 200 > {log_file}\n')
            sta_scr.write('exit\n')
            sta_scr.close()
            os.system(f'{self.sta_path} -no_splash {file_basename}.tcl')
            os.remove(file_basename+'.tcl')
            os.remove(file_basename+'.v')
            delay_dict = {}
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    if l.startswith('Endpoint'):
                        parts = l.split()
                        endpoint = parts[1]
                    if 'data arrival time' in l:
                        parts = l.split()
                        delay = float(parts[0])
                        delay_dict[endpoint] = delay
            os.remove(log_file)
            if rm_aig:
                os.remove(in_file)
            return delay_dict

        with open(log_file, 'r') as f:
            lines = f.readlines()
        delay = re.findall(r'[0-9]+\.[0-9]+', lines[-1])
        delay = float(delay[-2])
        os.remove(log_file)
        if rm_aig:
            os.remove(in_file)
        return delay
    
    def close(self):
        self._abc.end()
    
    def extract_largest_cone_features(self, target_node=None, dir = '.'):
        """
        Extracts the largest logic cone from a given AIG file and returns its features.
        """
        # 1. Load the AIG file
        rw_time = 0.0
        t_s = time.time()
        self._abc.write(dir+'/temp_full.aig')
        rw_time += time.time() - t_s
        aig_file_path = dir+'/temp_full.aig'
        try:
            t_s = time.time()
            aig_obj = aiger.load(aig_file_path)
            f_aig = FlexAIG()
            f_aig.copy_aiger(aig_obj)
            rw_time += time.time() - t_s
        except Exception as e:
            print(f"Error loading AIG file {aig_file_path}: {e}")
            return None, None, None

        # 2. Identify all logic cones and find the largest one
        largest_cone_size = -1
        largest_cone_endpoint = None

        if target_node is not None:
            _, cone, _ = f_aig.extractCone(target_node)
            temp_aig_file = dir+"/temp_largest_cone.aag"
            t_s = time.time()
            cone.toAIG().write(temp_aig_file)
            os.system(f'aigtoaig {temp_aig_file} {dir}/temp_largest_cone.aig')
            rw_time += time.time() - t_s
            os.remove(temp_aig_file)
            temp_aig_file = dir+"/temp_largest_cone.aig"
            self._abc.read(temp_aig_file)
            global_feat, graphData, opt_flow = self.getStateFeat()

            t_s = time.time()
            self._abc.read(dir+'/temp_full.aig')
            os.remove(temp_aig_file)
            os.remove(dir+'/temp_full.aig')
            rw_time += time.time() - t_s
            seq_embedding = self.convertSeqToEmbedding(opt_flow)
            return self.wrapState(global_feat, graphData, seq_embedding), rw_time
        
        # Assuming outputs are endpoints for cones
        endpoints = list(aig_obj.outputs)
        if not endpoints:
            print("No outputs found in the AIG file.")
            return None, None, None

        for ep in endpoints:
            _, cone = f_aig.extractCone(ep)
            if cone.andsNum() > largest_cone_size:
                largest_cone_size = cone.andsNum()
                largest_cone_endpoint = ep

        if largest_cone_endpoint is None:
            print("Could not determine the largest cone.")
            return None, None, None
            
        print(f"Largest cone found at endpoint: {largest_cone_endpoint} with {largest_cone_size} ANDs.")

        # 3. Extract the largest cone and save it to a temporary file
        _, largest_cone = f_aig.extractCone(largest_cone_endpoint)
        temp_aig_file = "temp_largest_cone.aag"
        largest_cone.toAIG().write(temp_aig_file)
        os.system(f'aigtoaig {temp_aig_file} temp_largest_cone.aig')
        os.remove(temp_aig_file)
        temp_aig_file = "temp_largest_cone.aig"

        # 4. Use abcpy to read the cone and get features
        self._abc.read(temp_aig_file)
        
        # 5. Get features using the existing method
        global_feat, graphData, opt_flow = self.getStateFeat()
        self._abc.read('temp_full.aig')

        # 6. Clean up temporary file
        os.remove(temp_aig_file)
        os.remove('temp_full.aig')
        seq_embedding = self.convertSeqToEmbedding(opt_flow)

        return self.wrapState(global_feat, graphData, seq_embedding), rw_time
    
if __name__ == '__main__':
    import json
    compute = torch.device('cuda:1')
    env = RLEnvironment(['../aigers/subcones/ac97_ctrl/u6.rp[2].aig'], compute, rwd="shaped", cut_weight=['../aigers/subcones/ac97_ctrl/u6.rp[2]_cutweights.json'])
    env.reset()
    net = RLNetwork()
    net.to(compute)
    #state = env.reset()
    obs, reward, _, done, info = env.step(2, cal_reward=False)
    obs, reward, _, done, info = env.step(7, cal_reward=False)
    obs, reward, _, done, info = env.step(6, cal_reward=False)
    print('test step')
    (logits, value), _ = net(obs)
    act = torch.argmax(logits).item()
    print(act, value)
    print(done, obs)
    reward, _,_ = env.getShapedReward()
    print(reward)
