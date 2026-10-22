import numpy as np
import random
class  GeneticAlgTSP:
    cross_rate=0.85
    mutation_rate=0.2
    def __init__(self,filename):
        self.best_node_static=[] #全局精英保留
        self.best_node_static_fitness=float('inf')
        self.fitness_cache=[]
        self.population_size=20
        self.best_nodes_num=2 #一定设为偶数，否则会出错，这样设计未来方便交叉操作
        self.population=[]
        self.cities=np.array([])
        temp_cities=[]
        with open(filename,'r') as f:
            lines=f.readlines()
            begin=False
            for line in lines:
                if not begin:
                    if line.startswith('NODE_COORD_SECTION'):
                        begin=True
                        continue
                else:
                    parts=line.split()
                    if len(parts)==3:
                        temp_cities.append((float(parts[1]),float(parts[2])))
        self.cities=np.array(temp_cities,dtype=float)
        diff = self.cities[:, np.newaxis, :] - self.cities[np.newaxis, :, :]
        self.dist=np.sqrt(np.sum(diff**2,axis=2))  #预计算城市之间的距离，避免重复计算
        for i in range(self.population_size//3):  # 生成30%的随机解
            node=random.sample(range(0,self.cities.shape[0]),self.cities.shape[0])
            while node  in self.population:
                node=random.sample(range(0,self.cities.shape[0]),self.cities.shape[0])
            self.population.append(node)
            self.fitness_cache.append(self.fitness(node))
        start={}
       
        sample_start_points=random.sample(range(0,self.cities.shape[0]),self.population_size-self.population_size//3)
        for i in range(self.population_size-self.population_size//3):  # 生成70%的解，随机选择起点，依次选择最近的未访问城市
            way=[]
            start_point=sample_start_points[i]
            way.append(start_point)
            visited={start_point:True}
            while len(way)<self.cities.shape[0]:
                min_distance=float('inf')
                next_point=None
                for j in range(self.cities.shape[0]): # 选择最近的未访问城市
                    if j not in visited:
                        if self.dist[way[-1]][j]<min_distance:
                            min_distance=self.dist[way[-1]][j]
                            next_point=j
                way.append(next_point)
                visited[next_point]=True
            if not self.check(way):
                raise Exception('Invalid node generated')
            self.population.append(way)
            self.fitness_cache.append(self.fitness(way))
           # print('generated :',way)
        print('Initial population generated')

    def determine_cross_edge(self, edge1_start, edge1_end, edge2_start, edge2_end):
        point1 = self.cities[edge1_start]  #判断是否相交，最终未使用
        point2 = self.cities[edge1_end]
        point3 = self.cities[edge2_start]
        point4 = self.cities[edge2_end]
        cross1 = ((point2[0] - point1[0]) * (point3[1] - point1[1])
                  - (point2[1] - point1[1]) * (point3[0] - point1[0]))
        cross2 = ((point2[0] - point1[0]) * (point4[1] - point1[1])
                  - (point2[1] - point1[1]) * (point4[0] - point1[0]))
        cross3 = ((point4[0] - point3[0]) * (point1[1] - point3[1])
                  - (point4[1] - point3[1]) * (point1[0] - point3[0]))
        cross4 = ((point4[0] - point3[0]) * (point2[1] - point3[1])
                  - (point4[1] - point3[1]) * (point2[0] - point3[0]))
        return cross1 * cross2 < 0 and cross3 * cross4 < 0
    
    def two_opt(self, node):  # 2-opt优化，穷举两条不相邻的边，反转两条边中间的节点
        improved = False
        best_node = node[:]
        

        for i in range(len(best_node) - 1):
            for j in range(i + 2, len(best_node)):
                if i == 0 and j == len(best_node) - 1:
                    continue

                edge1_start = best_node[i]   # 选择两条不相邻的边i，i+1和j，j+1
                edge1_end = best_node[i + 1]
                edge2_start = best_node[j]
                edge2_end = best_node[(j + 1) % len(best_node)]

              #  if not self.determine_cross_edge(edge1_start, edge1_end, edge2_start, edge2_end):
                    #continue

                candidate = best_node[:]
                candidate[i + 1:j + 1] = reversed(candidate[i + 1:j + 1])  #反转i+1到j的节点
                a = best_node[i]
                b = best_node[i + 1]
                c = best_node[j]
                d = best_node[(j + 1) % len(best_node)]
                if self.dist[a][c] + self.dist[b][d] < self.dist[a][b] + self.dist[c][d]:
                    improved = True
                    return improved, candidate   #如果改进了就返回改进后的路径，否则继续穷举其他边

        return improved, best_node

           
    def check(self,node):  # 检查路径是否有效
        node_copy=[i for i in node]
        node_copy.sort()
        for i in range(0,len(node_copy)):
            if node_copy[i]!=i:
                return False
        return len(node)==self.cities.shape[0]
    def fitness(self,node):  # 计算路径的适应度，即路径的总距离
        total_distance=0
        for i in range(0,len(node)):
            city1=node[i]
            if i==len(node)-1:
                city2=node[0]
            else:
                city2=node[i+1]
            distance=self.dist[city1][city2]
            total_distance+=distance
        return total_distance   

    def ox_child(self, parent1, parent2):   #ox交叉变异
        index1=random.randint(0,len(parent1)-2)
        index2=random.randint(index1+1,len(parent1)-1)
        child=[None]*len(parent1)
        child[index1:index2]=parent1[index1:index2]  # 保留父1的基因，从index1到index2

        fill_values=parent2[index2:]+parent2[:index2]
        fill_positions=list(range(index2,len(parent1)))+list(range(0,index2))  # 从index2到父1的末尾，再从0到index2，依次填充从父2中从index2开始的基因，
        # 直到父1的末尾，再从父2的开头到index2，依次填充父1的开头到index2
        fill_index=0

        for city in fill_values:
            if city in child:
                continue
            while child[fill_positions[fill_index]] is not None:
                fill_index+=1
            child[fill_positions[fill_index]]=city

        return child

    def inherit(self): #遗传，完全随机选择父代，概率是路径越短越大，生成新的子代
        new_population=[]
        
        weights=[1/self.fitness_cache[i] for i in range(len(self.population))]
        new_cache=[]
               # best_nodes=[]
      #  for i in range(self.population_size):
          #  if len(best_nodes)<self.best_nodes_num:
               # best_nodes.append((weights[i],i))
                #best_nodes.sort()
        #    else:
               # if weights[i]>best_nodes[0][0]:
                 #   best_nodes[0]=(weights[i],i)
               # best_nodes.sort()
        #for i in range(self.best_nodes_num):
        #    new_population.append(self.population[best_nodes[i][1]])   
       # for i in range(self.population_size-self.best_nodes_num):
        for i in range(self.population_size):
            new_node_idx=random.choices(range(len(self.population)),weights=weights,k=1)[0]
            if not self.check(self.population[new_node_idx]):
                raise Exception('Invalid node generated')
            new_population.append( [i for i in self.population[new_node_idx]] )
            new_cache.append(self.fitness_cache[new_node_idx])
        self.population=new_population
        self.fitness_cache=new_cache
    def crossover(self):  # 交叉操作
        new_population=[]
        new_cache=[]
        weights=[1/self.fitness_cache[i] for i in range(len(self.population))]
        best_nodes=[]
        for i in range(self.population_size):
            if len(best_nodes)<self.best_nodes_num:
                best_nodes.append((weights[i],i))
                best_nodes.sort()
            else:
                if weights[i]>best_nodes[0][0]:
                    best_nodes[0]=(weights[i],i)
                best_nodes.sort()
        for i in range(self.best_nodes_num):
            new_population.append( [i for i in self.population[best_nodes[i][1]]] )
            new_cache.append(self.fitness_cache[best_nodes[i][1]])
        current_best_idx=best_nodes[-1][1]
        current_best_fitness=self.fitness_cache[current_best_idx]
        if self.best_node_static_fitness>current_best_fitness:
            self.best_node_static_fitness=current_best_fitness
            self.best_node_static=[i for i in self.population[current_best_idx]]
        is_visited=[]
        for i in range((self.population_size-self.best_nodes_num)//2): #轮盘赌选择父代，概率是路径越短越大，生成新的子代
             num1=random.randint(0,len(self.population)-1)
             while num1 in is_visited:
                 num1=random.randint(0,len(self.population)-1)
             is_visited.append(num1)
             num2=random.randint(0,len(self.population)-1)
             while num2 in is_visited:
                 num2=random.randint(0,len(self.population)-1)
             is_visited.append(num2)
             new_node1=[i for i in self.population[num1] ]
             new_node2=[i for i in self.population[num2] ]
             if random.random()>self.cross_rate:
                 new_population.append(new_node1)
                 new_cache.append(self.fitness_cache[num1])
                 new_population.append(new_node2)
                 new_cache.append(self.fitness_cache[num2])
                 continue
             else:
                 child1=self.ox_child(new_node1,new_node2)  #交叉变异
                 child2=self.ox_child(new_node2,new_node1)
                 if not self.check(child1) or not self.check(child2):
                     raise Exception('Invalid node generated')
                 new_population.append(child1)
                 new_population.append(child2)
                 new_cache.append(self.fitness(child1))
                 new_cache.append(self.fitness(child2))
        self.population=new_population
        self.fitness_cache=new_cache
    def mutation(self):  # 变异操作，随机交换两个城市的位置，或者反转两个城市之间的路径
        weights=[1/self.fitness_cache[i] for i in range(len(self.population))]
        best_nodes=[]  #局部精英保留，将适应度最高的best_nodes_num个路径直接加入新种群
        for i in range(self.population_size):
            if len(best_nodes)<self.best_nodes_num:
                best_nodes.append((weights[i],i))
                best_nodes.sort()
            else:
                if weights[i]>best_nodes[0][0]:
                    best_nodes[0]=(weights[i],i)
                best_nodes.sort()
        current_best_idx=best_nodes[-1][1]
        current_best_fitness=self.fitness_cache[current_best_idx]
        if self.best_node_static_fitness>current_best_fitness:  #全局精英保留，将最短路径加入全局精英保留
            self.best_node_static_fitness=current_best_fitness
            self.best_node_static=[i for i in self.population[current_best_idx]]
        best_nodes_idx=set()
        for i in range(self.best_nodes_num):
            best_nodes_idx.add(best_nodes[i][1])
        for i in range(0,len(self.population)):
            if i in best_nodes_idx:
                continue
            if random.random()>self.mutation_rate:
                continue
            else:
                index1=random.randint(0,len(self.population[i]) -1)
                index2=random.randint(0,len(self.population[i]) -1)
                while index1==index2:
                    index2=random.randint(0,len(self.population[i]) -1)
                self.population[i][index1],self.population[i][index2]=self.population[i][index2],self.population[i][index1]
                index3=random.randint(0,len(self.population[i]) -2)
                index4=random.randint(index3+1,len(self.population[i]) -1)
                self.population[i]=self.population[i][:index3]+self.population[i][index3:index4][::-1]+self.population[i][index4:]
                self.fitness_cache[i]=self.fitness(self.population[i])
        
    def run(self,gen_num):
        
        while gen_num>0:
            import time
            timer=time.time()   
            self.inherit()
            self.crossover()
            self.mutation()
            if gen_num%10000==0:  #每10代进行一次two-opt优化，对适应度最高的best_nodes_num个路径进行优化
                weights=[1/self.fitness_cache[i] for i in range(len(self.population))]
                best_nodes=[]
                for i in range(self.population_size):
                    if len(best_nodes)<self.best_nodes_num:
                        best_nodes.append((weights[i],i))
                        best_nodes.sort()
                    else:
                        if weights[i]>best_nodes[0][0]:
                            best_nodes[0]=(weights[i],i)
                            best_nodes.sort()
                for i in range(0,self.best_nodes_num):
                        best_node=self.population[best_nodes[i][1]]
                        improved, best_node=self.two_opt(best_node)
                        while improved:
                            self.population[best_nodes[i][1]]=best_node    
                            self.fitness_cache[best_nodes[i][1]]=self.fitness(best_node)
                            improved, best_node=self.two_opt(best_node)
            gen_num-=1
            print('Generation ',150-gen_num,' completed in ',time.time() - timer)

        idx=0
        min_fitness=float('inf')
        for i in range(0,len(self.population)):
            fit=self.fitness_cache[i]
            if fit<min_fitness:
                min_fitness=fit
                idx=i
        if self.best_node_static_fitness<min_fitness:
            min_fitness=self.best_node_static_fitness 
            idx=-1
        print('最短路径长度：',min_fitness)
        print('最短路径：',self.best_node_static if idx==-1 else self.population[idx])
        
def main():
    import time
    timer=time.time()
    filename='mu1979.tsp'
    genetic_alg=GeneticAlgTSP(filename)
    genetic_alg.run(1500)
    print('Time taken:', time.time() - timer)
if __name__=='__main__':  
    main()            