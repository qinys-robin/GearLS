'''
Author: Qin Yusen, qinys2001@163.com
Date: 2025-03-03 10:22:35
LastEditTime: 2025-10-27 20:04:14
FilePath: /processaig/flex_aig.py
Description: Dynamic storage form of AIG
'''
import aiger
import aiger.common
import numpy as np
from aiger.aig import Node, AndGate, Inverter, LatchIn, Input, ConstFalse
import copy, time
import attr
import pymetis
import networkx as nx

@attr.frozen(auto_detect=True)
class Output(Node):
    input: Node
    name: str

    @property
    def children(self):
        return (self.input, )

    def __eq__(self, other) -> bool:
        return isinstance(other, Output) and (self.name == other.name)

    def __hash__(self) -> int:
        return hash(("input", self.name))
    
@attr.frozen(auto_detect=True)
class DummyNode(Node):
    name: str=None

    @property
    def children(self):
        return ()

    def __eq__(self, other) -> bool:
        return isinstance(other, DummyNode)

    def __hash__(self) -> int:
        return hash(("DummyNode"))

'''
Replace whole Node object with pointer(id)
'''
class FlexNode:
    def __init__(self, from_node: Node, id:int, Children : list[int], node_type:str=None):
        self.Id = id
        self.children = Children
        self.name = None
        self.fanout = []
        if node_type is not None:
            self.Type = node_type
            if node_type == 'Latch' or node_type == 'Input':
                self.name = from_node.name
        elif isinstance(from_node, AndGate):
            self.Type = 'And'
        elif isinstance(from_node, Inverter):
            self.Type = 'Inv'
        elif isinstance(from_node, LatchIn):
            self.Type = 'Latch'
            self.name = from_node.name
        elif isinstance(from_node, Input):
            self.Type = 'Input'
            self.name = from_node.name
        elif isinstance(from_node, ConstFalse):
            self.Type = 'Const'
        elif isinstance(from_node, Output):
            self.Type = 'Output'
            self.name = from_node.name

    def changeType(self, new_type : str):
        self.Type = new_type

    def changeName(self, new_name : str):
        self.name = new_name

    def addFanout(self, fanout_id : int):
        self.fanout.append(fanout_id)

    def clearFanout(self):
        self.fanout = []

'''
In order to speed-up, using dynamic link storage
'''
class FlexAIG:

    '''Construct via variables'''
    def __init__(self, _id_map:dict[int, FlexNode]=None, 
                 _out_map:dict[str, FlexNode]=None, 
                 _inputs:set[str]=None, 
                 _lat_map:dict[str, FlexNode]=None, 
                 _lat_init:dict[str, bool] = None, _comm:tuple[str]=()):
        self.id_node_map = _id_map if _id_map is not None else dict()
        self.outport_map = _out_map if _out_map is not None else dict()
        self.inputs = _inputs if _inputs is not None else set()
        self.latch_map = _lat_map if _lat_map is not None else dict()
        self.latch_init = _lat_init if _lat_init is not None else dict()
        self.comments = _comm
        self.max_id = 0
        self.latin_name_id_map = dict()
        self.to_latout_id = dict[int, list]()

        # Reorganize fanout info for cone aiger initialization
        if _id_map is not None:
            for nid in self.id_node_map.keys():
                node = self.id_node_map[nid]
                for child in node.children:
                    self.id_node_map[child].addFanout(nid)

        for lat, n in self.latch_map.items():
            if n.Id not in self.to_latout_id:
                self.to_latout_id[n.Id] = [lat]
            else:
                self.to_latout_id[n.Id].append(lat)
        for port, n in self.outport_map.items():
            if n.Id not in self.to_latout_id:
                self.to_latout_id[n.Id] = [port]
            else:
                self.to_latout_id[n.Id].append(port)

    def nodesNum(self):
        return len(self.id_node_map)

    def andsNum(self):
        and_count = 0
        for node in self.id_node_map.values():
            if node.Type == 'And':
                and_count += 1
        return and_count
    
    def copy_aiger(self, source:aiger.AIG):
        '''Copy from pyaiger object'''
        id_count = 0
        origin_node_to_id = {}
        self.comments = source.comments
        self.latch_init = dict(source.latch2init)
        self.inputs = set(source.inputs)
        # Arrange id for each node
        for ori_node in aiger.common.dfs(source):
            if id(ori_node) not in origin_node_to_id:
                origin_node_to_id[id(ori_node)] = id_count
                id_count += 1
            child_list = []
            for child in ori_node.children:
                if id(child) not in origin_node_to_id:
                    origin_node_to_id[id(child)] = id_count
                    id_count += 1
                child_list.append(origin_node_to_id[id(child)])
            self.id_node_map[origin_node_to_id[id(ori_node)]] = FlexNode(
                ori_node, origin_node_to_id[id(ori_node)], child_list
            )
            if isinstance(ori_node, LatchIn) or isinstance(ori_node, Input):
                self.latin_name_id_map[ori_node.name] = \
                            origin_node_to_id[id(ori_node)]
        
        for nid in range(id_count):
            node = self.id_node_map[nid]
            for child in node.children:
                self.id_node_map[child].addFanout(nid)
        
        # Debugger
        # for inp in source.inputs:
        #     if inp not in self.latin_name_id_map:
        #         print(f'Warning: Missing PI {inp} in node map!')
        
        '''Copy output cone. CAUTION for multi-fanout cone!!!'''
        for outport, outcone in source.node_map.iteritems():
            cone_id = origin_node_to_id[id(outcone)]
            self.outport_map[outport] = self.id_node_map[cone_id]
            self.insertConeToLatout(cone_id, outport)
        '''Copy latch cone. CAUTION for multi-fanout cone!!!'''
        for latch, latcone in source.latch_map.iteritems():
            if latch not in self.latin_name_id_map:
                self.latin_name_id_map[latch] = id_count
                pseudo_node = LatchIn(name=latch)
                self.id_node_map[id_count] = FlexNode(pseudo_node, id_count, [])
                id_count += 1
            cone_id = origin_node_to_id[id(latcone)]
            self.latch_map[latch] = self.id_node_map[cone_id]
            self.insertConeToLatout(cone_id, latch)
        self.max_id = id_count

    def arrangeIdForOutputs(self):
        '''For Partition use. Outputs don't need to have Node objects'''
        id_count = self.max_id
        for out, driver in self.outport_map.items():
            pseudo_node = Output(input=driver.Id, name=out)
            self.id_node_map[id_count] = FlexNode(pseudo_node, id_count, [driver.Id])
            id_count += 1
        self.max_id = id_count

    def calAdjacencyMatrix(self):
        '''For Partition use'''
        # self.arrangeIdForOutputs()
        node_num = len(self.id_node_map)
        adj_matrix = [ [] for _ in range(node_num)]
        for nid, node in self.id_node_map.items():
            if node.children == []:
                continue
            for child in node.children:
                assert nid < node_num and child < node_num
                adj_matrix[child].append(nid)
        return adj_matrix

    def toCombinational(self):
        '''
        For reproduce other's work:
        Convert all FFs to IO, each split into D-port and Q-port
        Save name mapping dict
        '''
        map_back_dict = dict()
        for id,node in self.id_node_map.items():
            if node.Type != 'Latch':
                continue
            # if node.name == 'pci_target_unit.fifos.pcir_fifo_storage.do_reg_b[34]':
            #     print('debug')
            node.changeType('Input')
            lat_name = node.name
            self.outport_map[lat_name+'_D'] = self.latch_map[lat_name]
            self.latch_map.pop(lat_name)
            self.latch_init.pop(lat_name)
            self.latin_name_id_map.pop(lat_name)
            node.changeName(lat_name+'_Q')
            self.latin_name_id_map[lat_name+'_Q'] = id
            self.inputs.add(lat_name+'_Q')
            if id in self.to_latout_id:
                for latout in self.to_latout_id[id]:
                    if latout in self.outport_map:
                        self.outport_map[latout] = node
                    elif latout+'_D' in self.outport_map:
                        self.outport_map[latout+'_D'] = node
                    else:
                        self.latch_map[latout] = node
            map_back_dict[lat_name+'_Q'] = lat_name
            map_back_dict[lat_name+'_D'] = lat_name
        '''Patch: process latch with no fanout, which couldn't be traversed during construction'''
        remain_latch_dict = copy.copy(self.latch_map)
        for latname, cone in remain_latch_dict.items():
            if latname+'_D' not in map_back_dict:
                self.outport_map[latname+'_D'] = cone
                self.latch_map.pop(latname)
                self.latch_init.pop(latname)
                map_back_dict[latname+'_D'] = latname
            else:
                raise ValueError('Missing latch!')
        return map_back_dict

    def insertConeToLatout(self, cone_id:int, latout:str):
        if cone_id not in self.to_latout_id:
            self.to_latout_id[cone_id] = [latout]
        else:
            self.to_latout_id[cone_id].append(latout)

    def removeLatoutOfCone(self, cone_id:int, latout:str):
        self.to_latout_id[cone_id].remove(latout)
        if len(self.to_latout_id[cone_id]) == 0:
            self.to_latout_id.pop(cone_id)
    
    def restoreNode(self, root_node:FlexNode, cache_map:dict) -> Node:
        if root_node.Id in cache_map:
            return cache_map[root_node.Id]
        if root_node.Type == 'Latch':
            restore =  LatchIn(name=root_node.name)
        elif root_node.Type == 'Input':
            restore =  Input(name=root_node.name)
        elif root_node.Type == 'Const':
            restore =  ConstFalse()
        elif root_node.Type == 'Inv':
            child = self.id_node_map[root_node.children[0]]
            restore = Inverter(input=self.restoreNode(child, cache_map))
        elif root_node.Type == 'And':
            lchild = self.id_node_map[root_node.children[0]]
            rchild = self.id_node_map[root_node.children[1]]
            restore = AndGate(left=self.restoreNode(lchild, cache_map), 
                           right=self.restoreNode(rchild, cache_map))
        else:
            raise ValueError('Wrong node type!')
        cache_map[root_node.Id] = restore
        return restore
        
    def toAIG(self) -> aiger.AIG:
        s_map = dict()
        return aiger.AIG(inputs=self.inputs, 
                node_map={out : self.restoreNode(cone_root,s_map) for out,cone_root \
                           in self.outport_map.items()},
                latch_map={lat : self.restoreNode(cone_root,s_map) for lat,cone_root \
                           in self.latch_map.items()},
                latch2init=self.latch_init,
                comments=self.comments)
    
    def countFanout(self, ignore_input = True) -> dict:
        node_fanout = dict()
        for nid, node in self.id_node_map.items():
            if node.children == []:
                continue
            for child in node.children:
                fanin_type = self.id_node_map[child].Type
                if fanin_type == 'Const' or (fanin_type == 'Input' and ignore_input):
                    continue
                if child not in node_fanout:
                    node_fanout[child] = 1
                else:
                    node_fanout[child] += 1
        # Considering nodes connected to Latch and PO
        for node in self.outport_map.values():
            if node.Type == 'Const' or (node.Type == 'Input' and ignore_input):
                continue
            if node.Id not in node_fanout:
                node_fanout[node.Id] = 1
            else:
                node_fanout[node.Id] += 1
        for node in self.latch_map.values():
            if node.Type == 'Const' or (node.Type == 'Input' and ignore_input):
                continue
            if node.Id not in node_fanout:
                node_fanout[node.Id] = 1
            else:
                node_fanout[node.Id] += 1
        return node_fanout
    
    def findNearestEndpoint(self, find_root: int, exclude_ep: str, stop_level = -1):
        '''return first reachable latch or output AND searched levels
           
           If early stop, return negative values'''
        visited = set()
        queue = [(find_root, 0)]
        while queue:
            current, level = queue.pop(0)
            if stop_level >=0 and level > stop_level:
                return -1, -1
            if current in visited:
                continue
            visited.add(current)
            node = self.id_node_map[current]
            if current in self.to_latout_id:
                dest_ep = self.to_latout_id[current]
                if exclude_ep not in dest_ep:
                    return dest_ep[0], level
            for fo in node.fanout:
                if fo not in visited:
                    queue.append((fo, level+1))
        return -1, -1
    
    def extractCone(self, target:str, cal_cut_weight = False, thres_weight = 0.75, truncate_num = 20):
        '''
        Cone could directly represent an AIG

        Convert latches into PIs/PO after DFS-like process

        For nodes with multi-fanout, add temporary output

        Cut weight indicates path-level connected from the cut
        '''
        cone_node_id_map = {}
        cone_input = set()
        cut_map = dict()
        cut_weight = dict()
        cut_path_levels = dict()
        if target in self.latch_map:
            root_node = self.latch_map[target]
        elif target in self.outport_map:
            root_node = self.outport_map[target]
        else:
            raise ValueError('Target not in Latch or Output!')
        self.updateCone(root_node, cone_node_id_map, cone_input)
        root_node_new = cone_node_id_map[root_node.Id]
        cone_aiger = FlexAIG(_id_map = cone_node_id_map, 
                             _out_map = {target+'_temp':root_node_new}, 
                             _inputs = cone_input)
        cone_aiger.latin_name_id_map = {inp : self.latin_name_id_map[inp]
                                         for inp in cone_input}
        ori_fanout_dict = self.countFanout()
        cone_fanout_dict = cone_aiger.countFanout(ignore_input=False)
        cone_level_map = countLevelsOfNode(cone_aiger)
        max_level_cone = max(cone_level_map.values())

        temp_port_count = 0
        not_pi_cut_num = 0
        for nid, node in cone_node_id_map.items():
            # For Latches with multi-fanout, PI -> PO
            if node.name in self.inputs or node.Type == 'Const':
                continue
            if ori_fanout_dict[nid] > cone_fanout_dict[nid]:
                # Connections outside the cone, add an PO
                cone_aiger.outport_map['Temp_port'+str(temp_port_count)] = node
                cut_map[nid] = 'Temp_port'+str(temp_port_count)
                temp_port_count += 1
                if node.Type != 'Input' and node.Type != 'Const':
                    not_pi_cut_num += 1

                if cal_cut_weight:
                    cut_level = cone_level_map[nid]
                    if cut_level == 0:
                        continue
                    _, path_level = self.findNearestEndpoint(nid, target, stop_level=int(0.5*cut_level))
                    if path_level < 0:
                        continue
                    total_outpath_level = cut_level + path_level
                    inpath_weight = cut_level / total_outpath_level
                    out_path_weight = total_outpath_level / max_level_cone
                    if out_path_weight < thres_weight:
                        continue
                    cut_path_levels['Temp_port'+str(temp_port_count)] = (total_outpath_level, inpath_weight)
        
        if cal_cut_weight:
            selected_cuts = sorted(cut_path_levels.keys(), key=lambda x: cut_path_levels[x][0], reverse=True)
            for i, cut in enumerate(selected_cuts):
                if i >= truncate_num:
                    break
                cut_weight[cut] = cut_path_levels[cut][1]
        # print(f'Cone extracted for {target}, {not_pi_cut_num} non-PI cuts, total {temp_port_count} cuts.')
        return cut_map, cone_aiger, cut_weight
    
    def selfRemaining(self, extracted_cone, target_latch, cut_map):
        '''
        For self-loop FF, simply discard this FF

        (There would be an PO connected to the FF in cone, 
        make sure re-connection)

        Otherwise convert to PI
        '''
        self_loop = True
        cone_PI = extracted_cone.inputs
        '''
        After modifying, inputs containing:
        Original PIs, cut nodes, extracted latch
        '''
        if target_latch not in cone_PI:
            self_loop = False
            self.inputs.add(target_latch)
        for temp_port in cut_map.values():
            self.inputs.add(temp_port)

        for eid in extracted_cone.id_node_map.keys():
            if self.id_node_map[eid].Type == 'Input' or \
            self.id_node_map[eid].Type == 'Const':
                continue
            if eid in cut_map:
                self.id_node_map[eid].changeType('Input')
                self.id_node_map[eid].changeName(cut_map[eid])
                self.id_node_map[eid].children = []
                # Make sure port and latch map follows object change
                if eid in self.to_latout_id:
                    self.modifyLatoutCone(eid)
                self.latin_name_id_map[cut_map[eid]] = eid
            else:
                if eid != self.latin_name_id_map[target_latch]:
                    self.id_node_map.pop(eid)
        
        target_in_id = self.latch_map[target_latch].Id
        self.removeLatoutOfCone(target_in_id, target_latch)
        if not self_loop:
            self.latch_map.pop(target_latch)
            self.latch_init.pop(target_latch)
        if self_loop:
            self.id_node_map.pop(self.latin_name_id_map[target_latch])
            self.latin_name_id_map.pop(target_latch)
        for cov_lat in cone_PI:
            if cov_lat in self.inputs:
                continue
            if cov_lat != target_latch:
                self.outport_map[cov_lat+'_temp'] = self.latch_map[cov_lat]
                target_in_id = self.latch_map[cov_lat].Id
                #if cov_lat not in self.outport_map:
                self.removeLatoutOfCone(target_in_id, cov_lat)
                self.insertConeToLatout(target_in_id, cov_lat+'_temp')
                self.latin_name_id_map.pop(cov_lat)
            self.latch_map.pop(cov_lat)
            self.latch_init.pop(cov_lat)

    def modifyLatoutCone(self, cone_id):
        ''' Update FlexNode Object in outport and latch map
        (NOT changing driver node, UPDATE driver info)'''
        connected = self.to_latout_id[cone_id]
        for lat_out in connected:
            if lat_out in self.latch_map:
                self.latch_map[lat_out] = self.id_node_map[cone_id]
            elif lat_out in self.outport_map:
                self.outport_map[lat_out] = self.id_node_map[cone_id]

    def split(self, target_latch):
        cut_map, split_cone, _ = self.extractCone(target_latch)
        self.selfRemaining(split_cone, target_latch, cut_map)
        return split_cone

    def updateCone(self, search_node:FlexNode, cone_map:dict[int, FlexNode], 
                   cone_input:set[str]):
        if search_node.Id in cone_map:
            return
        cone_map[search_node.Id] = copy.deepcopy(search_node)
        cone_map[search_node.Id].clearFanout()
        if search_node.Type == 'Input':
            cone_input.add(search_node.name)
        elif search_node.Type == 'Latch':
            cone_map[search_node.Id].changeType('Input')
            cone_input.add(search_node.name)
        elif search_node.Type == 'And':
            lchild = self.id_node_map[search_node.children[0]]
            rchild = self.id_node_map[search_node.children[1]]
            self.updateCone(lchild, cone_map, cone_input)
            self.updateCone(rchild, cone_map, cone_input)
        elif search_node.Type == 'Inv':
            child = self.id_node_map[search_node.children[0]]
            self.updateCone(child, cone_map, cone_input)

    def cascadeFrom(self, other):
        '''The cascading ports should share the same name'''
        id_count = self.max_id
        map_old_new = dict()

        def arrange_and_map_node(inid, innode):
            nonlocal id_count
            if inid not in map_old_new:
                id_count = max(id_count, self.arrangeIdForNewNode(
                    innode, id_count, map_old_new))
            return map_old_new[inid]

        # Cascading cut-inputs from other's outport
        for port, innode in other.outport_map.items():
            inid = innode.Id
            if port not in self.latin_name_id_map:
                self.outport_map[port] = arrange_and_map_node(inid, innode)
                self.insertConeToLatout(map_old_new[inid], port)
            # Only count first fanin. Multi fanin would be handled later
            elif inid not in map_old_new:
                map_old_new[inid] = self.latin_name_id_map.pop(port)
                self.inputs.remove(port)

        # Process all nodes in other's id_node_map
        arr_flag = False
        rearr_map = dict()
        for inid, innode in other.id_node_map.items():
            new_id = arrange_and_map_node(inid, innode)
            modified_child = [arrange_and_map_node(child, other.id_node_map[child])
                                        for child in innode.children]
            self.id_node_map[new_id] = copy.deepcopy(innode)
            self.id_node_map[new_id].Id = new_id
            self.id_node_map[new_id].children = modified_child

            '''
            Notice: After resyn, multi temp-port may share one fanin
            '''
            if inid in other.to_latout_id and len(other.to_latout_id[inid]) > 1:
                for port in other.to_latout_id[inid]:
                    if port not in self.latin_name_id_map:
                        continue
                    equal_id = self.latin_name_id_map[port]
                    rearr_map[equal_id] = new_id
                    arr_flag = True
                    self.latin_name_id_map.pop(port)
                    self.inputs.remove(port)
            '''
            Update FlexNode Object in outport and latch map
            Note that these map store Object, need to change though id not change
            '''
            if new_id in self.to_latout_id:
                self.modifyLatoutCone(new_id)
        
        '''Make sure Node --> Id mapping is unique'''
        if arr_flag:
            self.removeClearedNode(rearr_map)
            
        # Add inputs and latches from other to self
        for inp in other.inputs:
            if inp not in self.latin_name_id_map:
                self.inputs.add(inp)
                self.latin_name_id_map[inp] = map_old_new[other.latin_name_id_map[inp]]

        for lat, cone in other.latch_map.items():
            assert lat not in self.latch_map
            self.latch_map[lat] = self.id_node_map[map_old_new[cone.Id]]
            self.latch_init[lat] = other.latch_init[lat]
            self.latin_name_id_map[lat] = map_old_new[other.latin_name_id_map[lat]]
            self.insertConeToLatout(map_old_new[cone.Id], lat)

        self.max_id = id_count
    
    def arrangeIdForNewNode(self, to_insert:FlexNode, id_count, arr_map) -> int:
        '''Only re-arrange id! Mapping would be done later'''
        if to_insert.name in self.latin_name_id_map:
            arr_map[to_insert.Id] = self.latin_name_id_map[to_insert.name]
            return self.latin_name_id_map[to_insert.name]
        arr_map[to_insert.Id] = id_count
        return id_count + 1
    
    def removeClearedNode(self, rearr_map):
        for clear_id in rearr_map.keys():
            reserve_id = rearr_map[clear_id]
            self.id_node_map.pop(clear_id)
            if clear_id in self.to_latout_id:
                for latout in self.to_latout_id[clear_id]:
                    self.insertConeToLatout(reserve_id, latout)
                    if latout in self.latch_map:
                        self.latch_map[latout] = self.id_node_map[reserve_id]
                    if latout in self.outport_map:
                        self.outport_map[latout] = self.id_node_map[reserve_id]
                self.to_latout_id.pop(clear_id)
        
        # Rearrange connection
        for nid, node in self.id_node_map.items():
            if node.children == []:
                continue
            for i in range(len(node.children)):
                if node.children[i] in rearr_map:
                    self.id_node_map[nid].children[i] = \
                          rearr_map[node.children[i]]
    
    def loopBack(self, inputs:list[str], outputs:list[str]):
        assert len(inputs) == len(outputs)
        for i in range(len(inputs)):
            assert inputs[i] in self.inputs
            assert outputs[i] in self.outport_map
            if inputs[i] not in self.latin_name_id_map:
                self.inputs.remove(inputs[i])
                self.outport_map.pop(outputs[i])
                continue
            nid = self.latin_name_id_map[inputs[i]]
            self.id_node_map[nid].changeType('Latch')
            self.latch_map[inputs[i]] = self.outport_map[outputs[i]]
            cone_id = self.outport_map[outputs[i]].Id
            self.removeLatoutOfCone(cone_id, outputs[i])
            self.insertConeToLatout(cone_id, inputs[i])
            self.latch_init[inputs[i]] = False
            self.inputs.remove(inputs[i])
            self.outport_map.pop(outputs[i])

    def eliminateCuts(self):
        '''For merging partitions, connect existing cut edges (I/O pair with same name)'''
        for nid, node in self.id_node_map.items():
            # Connect cut edge
            new_children = []
            for child in node.children:
                if self.id_node_map[child].Type == 'Input' and \
                   self.id_node_map[child].name.startswith('Cut_port_'):
                    cut_name = self.id_node_map[child].name
                    if cut_name not in self.latin_name_id_map:
                        raise ValueError('Missing cut port in other part!')
                    new_id = self.outport_map[cut_name].Id
                    new_children.append(new_id)
                else:
                    new_children.append(child)
            self.id_node_map[nid].children = new_children

            # Update to_latout_id map
            if nid in self.to_latout_id:
                self.modifyLatoutCone(nid)

        for nid, node in self.id_node_map.items():
            if node.Type == 'Input' and node.name.startswith('Cut_port_'):
                new_id = self.outport_map[node.name].Id
                self.outport_map.pop(node.name)
                self.inputs.remove(node.name)
                if nid not in self.to_latout_id:
                    continue
                for outport in self.to_latout_id[nid]:
                    self.outport_map[outport] = self.id_node_map[new_id]
                    self.insertConeToLatout(new_id, outport)
                self.to_latout_id.pop(nid)

def countLevelsOfNode(in_aig:FlexAIG):
    level_map = dict()
    def dfs_level(nid):
        if nid in level_map:
            return level_map[nid]
        node = in_aig.id_node_map[nid]
        if node.Type == 'Input' or node.Type == 'Const' or node.Type == 'Latch':
            level_map[nid] = 0
            return 0
        max_level = 0
        for child in node.children:
            child_level = dfs_level(child)
            if child_level > max_level:
                max_level = child_level
        level_map[nid] = max_level + 1
        return level_map[nid]
    
    for nid in in_aig.id_node_map.keys():
        dfs_level(nid)
    return level_map

def mergeFlexAIG(sub_cone:FlexAIG, remain:FlexAIG, target_latch):
    # Restore target latch first if self-loop
    self_loop = False
    if target_latch in sub_cone.inputs:
        self_loop = True
        sub_cone.loopBack([target_latch], [target_latch+'_temp'])
    remain.cascadeFrom(sub_cone)
    cone_latch_in = [l for l in sub_cone.inputs if l+'_temp' in remain.outport_map]
    remain_latch_out = [l+'_temp' for l in cone_latch_in]
    # Restore target latch with no self-loop
    if not self_loop:
        cone_latch_in.append(target_latch)
        remain_latch_out.append(target_latch+'_temp')
    remain.loopBack(cone_latch_in, remain_latch_out)
    # for key,val in remain.latch_map.items():
    #     name = val.name
    #     if name is not None and name.startswith('Temp_port'):
    #         print(f"Warning: {key} connects to a temporary port, please check! Enpoint: {target_latch}")
    #         raise ValueError()
    # for key,val in remain.outport_map.items():
    #     name = val.name
    #     if name is not None and name.startswith('Temp_port'):
    #         print(f"Warning: {key} connects to a temporary port, please check! Enpoint: {target_latch}")
    #         raise ValueError()

'''
def partitionAIG(source:FlexAIG, nodes_per_part:int = 5000):
    ''' '''Currently only partitioning combinational circuit''' '''
    part_num = int(np.ceil(len(source.id_node_map) / nodes_per_part))
    if part_num < 2:
        raise ValueError('Too few nodes to partition!')
    adj_matrix = source.calAdjacencyMatrix()
    (edgecuts, parts) = pymetis.part_graph(part_num, adjacency=adj_matrix)
    part_map = dict()
    for i in range(len(parts)):
        if parts[i] not in part_map:
            part_map[parts[i]] = [i]
        else:
            part_map[parts[i]].append(i)
    part_list = []
    part_connections = nx.DiGraph()
    part_cut_list = {}                 # (from_part, to_part) -> [(cut_node_id, cut_num_postfix)]
    subgraph_id_count = source.max_id
    node_cut_count_map = dict()

    for i in range(part_num):
        if i not in part_map:
            raise ValueError('Empty partition!')
        id_set = set(part_map[i])
        sub_id_map = dict()
        sub_out_map = dict()
        sub_inputs = set()
        sub_lat_map = dict()
        sub_lat_init = dict()
        cut_input_map = {}

        for nid in id_set:
            node = source.id_node_map[nid]
            if node.Type == 'Input':
                sub_inputs.add(node.name)
            child_list = []
            for child in node.children:
                if child in id_set:
                    child_list.append(child)
                else:
                    # Cut node, convert to PI
                    # Record part connection for merge, cut node would be PO in other part (processed later)
                    # Same part, Same cut node -> Same PI; Different part, Same cut node -> Different PI
                    if child not in cut_input_map:
                        if child not in node_cut_count_map:
                            node_cut_count_map[child] = 0
                        else:
                            node_cut_count_map[child] += 1
                        sub_inputs.add('Cut_port_'+str(child)+'_'+str(node_cut_count_map[child]))
                        temp_input = FlexNode(Input(name='Cut_port_'+str(child)+'_'+str(node_cut_count_map[child])), subgraph_id_count, [])
                        sub_id_map[subgraph_id_count] = temp_input
                        cut_input_map[child] = subgraph_id_count
                        subgraph_id_count += 1
                        if (parts[child], i) not in part_cut_list:
                            part_cut_list[(parts[child], i)] = [(child, node_cut_count_map[child])]
                        else:
                            part_cut_list[(parts[child], i)].append((child, node_cut_count_map[child]))

                    child_list.append(cut_input_map[child])
                    part_connections.add_edge(parts[child], i)
            new_node = copy.deepcopy(node)
            new_node.children = child_list
            sub_id_map[nid] = new_node
            
        for out, cone in source.outport_map.items():
            if cone.Id in id_set:
                sub_out_map[out] = sub_id_map[cone.Id]
        part_list.append(FlexAIG(_id_map=sub_id_map, _out_map=sub_out_map,
                                 _inputs=sub_inputs, _lat_map=sub_lat_map,
                                 _lat_init=sub_lat_init, _comm=source.comments))
        
    merge_start = -1
    for i in range(part_num):
        if part_connections.out_degree(i) == 0:
            merge_start = i
            continue
        for j in part_connections.successors(i):
            if (i,j) not in part_cut_list:
                raise ValueError('Missing cut record!')
            cut_nodes = part_cut_list[(i,j)]
            for cut_node in cut_nodes:
                part_list[i].outport_map['Cut_port_'+str(cut_node[0])+'_'+str(cut_node[1])] = part_list[i].id_node_map[cut_node[0]]
                part_list[i].insertConeToLatout(cut_node[0], 'Cut_port_'+str(cut_node[0])+'_'+str(cut_node[1]))
    
    return part_list, merge_start, part_connections
'''
