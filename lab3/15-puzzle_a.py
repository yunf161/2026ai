import heapq
target_pos={1:(0,0),2:(0,1),3:(0,2),4:(0,3),
            5:(1,0),6:(1,1),7:(1,2),8:(1,3),
            9:(2,0),10:(2,1),11:(2,2),12:(2,3),
            13:(3,0),14:(3,1),15:(3,2),0:(3,3)}
target=tuple(range(1,16))+tuple([0])
ZERO_NEIGHBORS = {
0: (1, 4), 1: (0, 2, 5), 2: (1, 3, 6), 3: (2, 7),
4: (0, 5, 8), 5: (1, 4, 6, 9), 6: (2, 5, 7, 10), 7: (3, 6, 11),
8: (4, 9, 12), 9: (5, 8, 10, 13), 10: (6, 9, 11, 14), 11: (7, 10, 15),
12: (8, 13), 13: (9, 12, 14), 14: (10, 13, 15), 15: (11, 14)
}
#全局变量ZERO_NEIGHBORS和target_pos，分别表示0的邻居位置和目标状态中每个数字的位置



def h1(state): #启发式函数，计算当前状态与目标状态之间的曼哈顿距离和线性冲突
    manhattan = 0
    conflict = 0

    for i in range(16):
        x = i // 4
        y = i % 4
        value = state[i]
        if value == 0:
            continue
        gx, gy = target_pos[value]
        manhattan += abs(x - gx) + abs(y - gy)

    for row in range(4):
        row_tiles = []
        for col in range(4):
            value = state[row * 4 + col]
            if value == 0:
                continue
            gx, gy = target_pos[value]
            if gx == row:
                row_tiles.append(gy)
        for i in range(len(row_tiles)):
            for j in range(i + 1, len(row_tiles)):
                if row_tiles[i] > row_tiles[j]:
                    conflict += 1

    for col in range(4):
        col_tiles = []
        for row in range(4):
            value = state[row * 4 + col]
            if value == 0:
                continue
            gx, gy = target_pos[value]
            if gy == col:
                col_tiles.append(gx)
        for i in range(len(col_tiles)):
            for j in range(i + 1, len(col_tiles)):
                if col_tiles[i] > col_tiles[j]:
                    conflict += 1

    return manhattan + 2 * conflict

def tuple_swap(state, i, j): #交换状态中的两个位置，并返回新的状态和移动的数字
    state_list = list(state)
    state_list[i], state_list[j] = state_list[j], state_list[i]
    return tuple(state_list), state_list[j]+state_list[i]

def A_star(start, h): #A*算法
    frontier = []
    dict_current_cost = {start: 0} #记录每个状态的当前成本，用于剪枝
    heapq.heappush(frontier, (h(start),h(start), start))
    visited = set()
    parent_map={start:(None,None)} #记录每个状态的父状态和移动的数字
   
    while frontier:
        current_ch,current_h,current= heapq.heappop(frontier)
        cost=current_ch-current_h
        if current == target:
            return current,parent_map
        if current in visited:
            continue
        visited.add(current)
        zero_index = current.index(0)
        for neighbor_index in ZERO_NEIGHBORS[zero_index]:
            temp,moved_value=tuple_swap(current, zero_index, neighbor_index)
            if temp not in visited:    
                temp_cost=cost+1   
                if  dict_current_cost.get(temp, None) is None or dict_current_cost[temp]>temp_cost:
                    temp_f=h(temp)+temp_cost
                    dict_current_cost[temp]=temp_cost
                    heapq.heappush(frontier, (temp_f,temp_f-temp_cost, temp))
                    parent_map[temp]=(current,moved_value)
                
    return None,None

def print_solution_A_star(end,parent_map): #打印A*算法的解
    print_list=[]
    while parent_map[end][0] is not None:  
        print_list.append(parent_map[end][1])
        end=parent_map[end][0]
    print_list.reverse()
    print(print_list)

import time

def main():
    f = open('grid1.txt','r')
    count=0
    while(count<10):
        start_time = time.time()
        print('-----------------')
        print("测试",count+1)
        line=f.readline()
        if not line:
            break
        line=line.strip().split(',')
        num=0
        line_tuple=()
        for i in line:
            line_tuple+= (int(i),)
            num+=1
        count+=1
        end,parent_map = A_star(line_tuple,h1)
        if end is not None:
            print_solution_A_star(end,parent_map)
        else:
            print("No solution found.")
        print("-----------------")
        end_time = time.time()
        print("耗时:",end_time-start_time)
        
    f.close()

if __name__=="__main__":
    main()
       

           

        

    
    



