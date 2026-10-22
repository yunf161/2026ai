import numpy as np
import random
class  GeneticAlgTSP:
    cross_rate=0.8
    mutation_rate=0.1
    def __init__(self,filename):
        self.population_size=20
        self.best_nodes_num=2
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
        for i in range(self.population_size//3):
            node=random.sample(range(0,self.cities.shape[0]),self.cities.shape[0])
            while node  in self.population:
                node=random.sample(range(0,self.cities.shape[0]),self.cities.shape[0])
            self.population.append(node)
        start={}
        diff = self.cities[:, np.newaxis, :] - self.cities[np.newaxis, :, :]
        self.dist=np.sqrt(np.sum(diff**2,axis=2))
        sample_start_points=random.sample(range(0,self.cities.shape[0]),self.population_size-self.population_size//3)
        for i in range(self.population_size-self.population_size//3):
            way=[]
            start_point=sample_start_points[i]
            way.append(start_point)
            visited={start_point:True}
            while len(way)<self.cities.shape[0]:
                min_distance=float('inf')
                next_point=None
                for j in range(self.cities.shape[0]):
                    if j not in visited:
                        if self.dist[way[-1]][j]<min_distance:
                            min_distance=self.dist[way[-1]][j]
                            next_point=j
                way.append(next_point)
                visited[next_point]=True
            if not self.check(way):
                raise Exception('Invalid node generated')
            self.population.append(way)
           # print('generated :',way)
        print('Initial population generated')

    def determine_cross_edge(self, edge1_start, edge1_end, edge2_start, edge2_end):
        point1 = self.cities[edge1_start]
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
    
    def two_opt(self, node):
        improved = False
        best_node = node[:]
        

        for i in range(len(best_node) - 1):
            for j in range(i + 2, len(best_node)):
                if i == 0 and j == len(best_node) - 1:
                    continue

                edge1_start = best_node[i]
                edge1_end = best_node[i + 1]
                edge2_start = best_node[j]
                edge2_end = best_node[(j + 1) % len(best_node)]

              #  if not self.determine_cross_edge(edge1_start, edge1_end, edge2_start, edge2_end):
                    #continue

                candidate = best_node[:]
                candidate[i + 1:j + 1] = reversed(candidate[i + 1:j + 1])
                a = best_node[i]
                b = best_node[i + 1]
                c = best_node[j]
                d = best_node[(j + 1) % len(best_node)]
                if self.dist[a][c] + self.dist[b][d] < self.dist[a][b] + self.dist[c][d]:
                    improved = True
                    return improved, candidate

        return improved, best_node

           
    def check(self,node):
        node_copy=[i for i in node]
        node_copy.sort()
        for i in range(0,len(node_copy)):
            if node_copy[i]!=i:
                return False
        return len(node)==self.cities.shape[0]
    def fitness(self,node):
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
    def inherit(self):
        new_population=[]
        weights=[1/self.fitness(node) for node in self.population]
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
            new_node=[i for i in random.choices(self.population,weights=weights,k=1)[0] ]
            if not self.check(new_node):
                raise Exception('Invalid node generated')
            new_population.append(new_node)
        self.population=new_population
    def crossover(self):
        new_population=[]
        weights=[1/self.fitness(node) for node in self.population]
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
            new_population.append(self.population[best_nodes[i][1]])   
        is_visited=[]
        for i in range((self.population_size-self.best_nodes_num)//2):
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
                 new_population.append(new_node2)
                 continue
             else:
                 index1=random.randint(0,len(new_node1)-2)
                 index2=random.randint(index1+1,len(new_node1)-1)
                 child=[]
                 for i in range(0,len(new_node2)):
                     if new_node2[i] not in new_node1[index1:index2]:
                         child.append(new_node2[i])
                 child=child[:index1]+new_node1[index1:index2]+child[index1:]
                 if not self.check(child):
                     raise Exception('Invalid node generated')
                 new_population.append(child)
                 new_population.append( new_node2 if self.fitness(new_node2)<self.fitness(new_node1) else new_node1)
        self.population=new_population
    def mutation(self):
        weights=[1/self.fitness(node) for node in self.population]
        best_nodes=[]
        for i in range(self.population_size):
            if len(best_nodes)<self.best_nodes_num:
                best_nodes.append((weights[i],i))
                best_nodes.sort()
            else:
                if weights[i]>best_nodes[0][0]:
                    best_nodes[0]=(weights[i],i)
                best_nodes.sort()
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
        
    def run(self,gen_num):
        while gen_num>0:
            self.inherit()
            self.crossover()
            self.mutation()
            if gen_num%10==0:
                weights=[1/self.fitness(node) for node in self.population]
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
                            improved, best_node=self.two_opt(best_node)
            gen_num-=1
        idx=0
        min_fitness=float('inf')
        for i in range(0,len(self.population)):
            fit=self.fitness(self.population[i])
            if fit<min_fitness:
                min_fitness=fit
                idx=i
        print('最短路径长度：',min_fitness)
        print('最短路径：',self.population[idx])
        
def main():
    filename='dj38.tsp'
    genetic_alg=GeneticAlgTSP(filename)
    genetic_alg.run(1500)

if __name__=='__main__':  
    main()            