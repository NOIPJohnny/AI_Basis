from copy import deepcopy
from queue import Queue

from interface.state import StateBase
from utils.show_path import show_reversed_path

# 广度优先搜索全部解
class BreadthFirstSearch:
    
    def __init__(self, state:StateBase):
        assert isinstance(state, StateBase)
        self.initial_state = deepcopy(state)
    
    # tree_search决定是否判断重复状态，require_path决定是否记录路径
    def search(self, tree_search:bool=True, require_path:bool=True) -> None:
        # 添加队列大小峰值记录
        queue_peak_size = 0
        
        # 宽搜状态队列
        states_queue = Queue()
        
        # 记录状态的前一个状态，用于记录路径
        last_state_of = dict()
        
        # 防止重复状态，判断哪些状态已经访问过
        explored_states = set()
        
        states_queue.put(self.initial_state)
        explored_states.add(self.initial_state)
        
        while not states_queue.empty():
            # 更新队列峰值
            queue_peak_size = max(queue_peak_size, states_queue.qsize())
            
            state = states_queue.get()
            
            if state.success():
                '''if require_path:
                    show_reversed_path(last_state_of, state)
                else:
                    state.show()'''
                continue
            
            if state.fail():
                continue
            
            # 考虑所有可能动作，扩展全部子状态
            for action in state.action_space():
                
                new_state = state.next(action)
                
                if tree_search:
                    # TODO：扩展待访问的结点集合——1行
                    states_queue.put(new_state)
                    if require_path:
                        # TODO：在 last_state_of 中记录路径——1行
                        last_state_of[new_state] = state
                
                elif new_state not in explored_states:
                    # TODO：扩展待访问的结点集合——2行
                    states_queue.put(new_state)
                    explored_states.add(new_state)
                    if require_path:
                        # TODO：在 last_state_of 中记录路径——1行
                        last_state_of[new_state] = state
        
        # 输出队列峰值
        print(f"Queue peak size: {queue_peak_size}")
