import numpy as np
import random
class  GeneticAlgTSP:
    cross_rate=0.8
    mutation_rate=0.1
    def __init__(self,filename):
        self.population_size=200
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
        for i in range(self.population_size):
            node=random.sample(range(0,self.cities.shape[0]),self.cities.shape[0])
            while node  in self.population:
                node=random.sample(range(0,self.cities.shape[0]),self.cities.shape[0])
            self.population.append(node)
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
            city1=self.cities[node[i]]
            if i==len(node)-1:
                city2=self.cities[node[0]]
            else:
                city2=self.cities[node[i+1]]
            distance=np.sqrt((city1[0]-city2[0])**2+(city1[1]-city2[1])**2)
            total_distance+=distance
        return total_distance   
    def inherit(self):
        new_population=[]
        weights=[1/self.fitness(node) for node in self.population]
        for i in range(self.population_size):
            new_node=[i for i in random.choices(self.population,weights=weights,k=1)[0] ]
            if not self.check(new_node):
                raise Exception('Invalid node generated')
            new_population.append(new_node)
        self.population=new_population
    def crossover(self):
        new_population=[]
        is_visited=[]
        for i in range(self.population_size//2):
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
        for i in range(0,len(self.population)):
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
    genetic_alg.run(1000)

if __name__=='__main__':  
    main()            