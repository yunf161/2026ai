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

def tuple_swap(state, i, j): 
    state_list = list(state)
    state_list[i], state_list[j] = state_list[j], state_list[i]
    return tuple(state_list), state_list[j]+state_list[i]

def dfs(start,max_depth,path_visited,print_list,cost,h,h_cache,zero_index,zero_index_pre):
    path_visited.add(start)
    if start in h_cache: #如果当前状态已经计算过启发式值，直接使用
      h_value = h_cache[start] + cost
    else:
        h_value = h(start) + cost
        h_cache[start] = h_value - cost
    if start == target:
        return start
    if h_value>max_depth[0]:
        if len(max_depth)==1:
            max_depth.append(h_value)
        else:
            max_depth[1]=min(max_depth[1],h_value)
        path_visited.remove(start) #局部visited集合，回退时需要移除当前状态
        return None
    
    for neighbor_index in ZERO_NEIGHBORS[zero_index]:
            if zero_index_pre==neighbor_index: #避免回退到上一个状态
             continue
            temp,moved_value=tuple_swap(start, zero_index, neighbor_index)
            if temp not in path_visited:
             print_list.append(moved_value)
             temp_result=dfs(temp,max_depth,path_visited,print_list,cost+1,h,h_cache,neighbor_index,zero_index)
             if temp_result!=None:
                path_visited.remove(start)
                return temp_result
             print_list.pop()
    path_visited.remove(start)
    return None



def IDA_star(start, h):
    checker=None
    path_visited=set()
    max_depth=[]
    max_depth.append(h(start))
    print_list=[]
    h_cache = {}
    while True:
        checker=dfs(start,max_depth,path_visited,print_list,0,h,h_cache,start.index(0),-1)
        if checker!=None or len(max_depth)==1: #如果找到目标状态，或者没有下一次的max_depth，说明没有更优的路径，直接退出
            break
        temp=max_depth[1]
        max_depth.clear()
        max_depth.append(temp)
    if checker==None:
        print("ida失败")
    else:
        print("ida找到目标状态路径：")
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
        IDA_star(line_tuple,h1)
        end_time = time.time()
        print("耗时:",end_time-start_time)
        print("-----------------")
        
    f.close()

if __name__=="__main__":
    main()