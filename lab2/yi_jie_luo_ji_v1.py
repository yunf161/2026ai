from yibanheyi import mgu    
from functools import total_ordering
from yibanheyi import set_new_line,get_term,to_list


@total_ordering
class ziju:              #定义一个子句类，存放在子句集中
    line=tuple()
    nums=0
    xu_hao=0
    is_visited=False
    constant_count=0
    def __init__(self,line):        #构造函数，包含拷贝构造和普通构造两种情况
        if isinstance(line,ziju):
            self.line=line.line
            self.nums=line.nums
            self.is_visited=False
            self.xu_hao=line.xu_hao
            self.constant_count=line.constant_count
        else:
            self.line=line
            self.nums=len(line)
            self.is_visited=False
            for i in list(self.line):
                temp_list=to_list(i)
                for j in temp_list:
                    if(get_term(j)==1):
                            self.constant_count+=1
                    else:
                        continue
    def change_xu_hao(self,xu_hao):   #每个子句都有序号，是加入子句集的顺序，方便回溯归结步骤
        self.xu_hao=xu_hao
    def get_tuple_line(self):  #tuple类型的子句，方便比较和存储
        return self.line
    def get_xu_hao(self):    #获取子句的序号
        return self.xu_hao
    def get_is_visited(self):    #获取子句是否被访问过，在主归结逻辑中，需要一次选择未访问过的子句进行归结，访问过的子句不再选择
        return self.is_visited
    def set_is_visited(self,visited):    #设置子句是否被访问过
        self.is_visited=visited
        return self.is_visited
    def __lt__(self, other):    #定义子句的比较规则，先按原子数排序，再按常量数排序，这样会先选择原子数少的子句进行归结，原子数相同的子句，常量数多的优先选择，来提升效率
        if(self.nums==other.nums):
            if(self.constant_count==other.constant_count):
                return str(self.line) < str(other.line)
            return self.constant_count>other.constant_count  
        return self.nums<other.nums
    def __eq__(self, other):    #定义子句的相等比较规则，根据内容是否相同来判断，方便去重
        return self.line==other.line
    def __hash__(self): #定义子句的哈希函数，根据内容来计算哈希值，方便存储在集合中进行去重
        return hash(self.line)

def readProp(file): #读取文件，将文件中的子句读取到子句集中，返回子句集
    lines=set()       #输入格式，主函数只能处理一个子句集输入，子句集一行，子句间用空格隔开，子句中的每个文字（谓词）项按照分号分开，函数的参数按照逗号分开
    with open(file,'r') as f:
       line=list(f.readline().split(' '))
       for i in range(len(line)):
           a=list(line[i].split(';'))
           a.sort()
           a=tuple(a)
           ziju_obj=ziju(a)
           lines.add(ziju_obj)
    lines=list(lines)
    lines.sort()
    for i in range(len(lines)):    #排序后给每个子句设置序号，方便回溯归结步骤
        lines[i].change_xu_hao(i)
    return lines 

def quchong(line_new):     #去重逻辑，先删除完全重复的文字（P 和 P），再删除互补对 P 和 ~P
    # 第一步：先删除完全重复的文字（P 和 P）
    line_no_duplicate = list(dict.fromkeys(line_new))  # 保留顺序，去重
    # 第二步：删除互补对 P 和 ~P
    i = 0
    while i < len(line_no_duplicate):
        current = line_no_duplicate[i]
        if current.startswith('~'):
            complement = current[1:]
        else:
            complement = '~' + current
        if complement in line_no_duplicate:
            #print(f"去重:{current} 和 {complement}")
            line_no_duplicate.remove(current)
            line_no_duplicate.remove(complement)
            i = 0  # 删除后列表变化，重新开始
        else:
            i += 1
    return line_no_duplicate

def Gui_Jie_step(line1,line2,index1,index2):   #单步归结，在找到匹配的文字后，进行一次归结，如果有多个匹配的文字，在去重中处理
    a=list(line1.get_tuple_line())             
    b=list(line2.get_tuple_line())
    temp_str,temp_str1,dict_re,check=mgu(a[index1],b[index2]) #调用mgu函数，判断是否匹配
    if(check==False):
        return None,False,None
    a.remove(a[index1])
    b.remove(b[index2])
    line_new=a+b
    line_new.sort()
    set_new_line(line_new,dict_re)  #将新的子句中的文字进行替换，替换的映射关系在dict_re中，替换后可能会有重复的文字，需要进行去重
    line_new=quchong(line_new)   #去重逻辑，先删除完全重复的文字（P 和 P），再删除互补对 P 和 ~P
    line_new=tuple(line_new)
    return line_new,True,dict_re

def find_chongyan(line1,line2):   #查找匹配的文字，如果有多个匹配的文字，只返回第一个匹配的文字的索引，其他的在去重逻辑里面处理
    #先找到匹配的文字，再调用单步归结函数进行归结，返回新的子句和是否成功归结的标志，以及匹配的文字的索引，方便输出归结步骤
    a=list(line1.get_tuple_line())
    b=list(line2.get_tuple_line())
    find_negative=True
    for i in range(len(a)):
        if a[i].startswith('~'):
            find_negative=False
        for j in range(len(b)):
            if find_negative and a[i][:a[i].find('(')]==b[j][1:b[j].find('(')]:
                new_line,check,dict_re=Gui_Jie_step(line1,line2,i,j)
                if(check==True):
                    return new_line,True,i,j,dict_re
            if not find_negative and a[i][1:a[i].find('(')]==b[j][:b[j].find('(')]:
                new_line,check,dict_re=Gui_Jie_step(line1,line2,i,j)
                if(check==True):
                    return new_line,True,i,j,dict_re
    return None,False,-1,-1,None

def guijie_main(lines,output_list): #主归结逻辑，先选择一个未访问过的子句进行归结，找到与其匹配的子句进行归结，归结成功后将新的子句加入子句集中，并标记被访问过的子句，继续选择下一个未访问过的子句进行归结，直到找到空子句或者没有未访问过的子句为止
    visited_nums=0
    while visited_nums<len(lines) and visited_nums<50:
        for line1 in lines:   #选择一个未访问过的子句进行归结
            check=False   
            found=False
            cnt=0  #每个子句最多归结两次，这是出于硬件限制考虑，如果每个子句归结次数过多，可能会导致内存溢出或者运行时间过长，所以设置一个归结次数的限制，超过这个限制就不再进行归结，直接跳出循环，这样可以保证程序的稳定性和效率
            if line1.get_is_visited():       #但是这样可能会过度剪枝，导致无法找到解，所以这个归结次数的限制需要根据具体情况进行调整，或者在程序中添加一些启发式策略，本来想使用支持集策略，但是限制于输入格式无法实现
                continue
            for line2 in lines:  #被访问过的子句在这里仍然可以参与归结
                if line2 is line1:
                    continue
                new_line,check,index1,index2,dict_re=find_chongyan(line1,line2)
                if(check):
                    found=True                      
                    new_line_obj=ziju(new_line)
                    new_line_obj.change_xu_hao(len(lines))
                    cun_zai=False
                    for i in lines: #不能加入重复的子句
                        if i.get_tuple_line()==new_line:
                            cun_zai=True
                            break
                        else:
                            continue
                    if not cun_zai:
                        lines.append(new_line_obj)
                        cnt+=1
                        line1.set_is_visited(True)
                        visited_nums+=1        #输出归结步骤，如果有多个匹配的文字，在去重逻辑中处理，输出时只输出第一个匹配的文字的索引，其他的在去重中处理
                        output_list.append('%d R[%d%s,%d%s]{%s}:%s' % (
                         len(lines)-1,
                         line1.get_xu_hao(), chr(ord('a') + index1),
                         line2.get_xu_hao(), chr(ord('a') + index2),
                         str(dict_re),
                         str(new_line)
                           ))
                        if len(new_line)==0:
                            return True

            if cnt==2:
                 lines.sort() 
                 break
    
        if not found:
            return False
       
    return False 

def get_index(line):   #优化策略，删除无用步骤，这个函数用获取output中的子句编号
    num_str = line.split()[0]
    return int(num_str)

def find_parent(line):   #优化策略，删除无用步骤，这个函数用来找到每个子句的父句
    father=-1
    mother=-1
    if 'R[' in line:
        start=line.find('R[')+2
        end=line.find(']{')
        parent_part=line[start:end]
        father_str,mother_str=parent_part.split(',')
        father=int(father_str[:-1])
        mother=int(mother_str[:-1])
    return father,mother

def tag_change(line,dict_key):   #优化策略，删除无用步骤，这个函数用来将归结步骤中的索引替换为新的索引
    for key in dict_key:
        if key==dict_key[key]:
            continue
        while(line.find(key)!=-1):
            line=line.replace(key,dict_key[key])
    return line
    

def handle_output(output_list,cin_line):   #优化策略，删除无用步骤，这个是这部分优化的主函数
    last_line=output_list[-1]
    last_line_index=get_index(last_line)
    list_index_selected=[]
    list_index_selected.append(last_line_index)
    dict_key={}
    for i in range(len(output_list)-1,-1,-1): #解析成功步骤的R，逐层回溯父句
        line=output_list[i]
        line_index=get_index(line)
       
        if line_index in list_index_selected:
            father,mother=find_parent(line)
            if father!=-1 and father not in list_index_selected:
                list_index_selected.append(father)
            if mother!=-1 and mother not in list_index_selected:
                list_index_selected.append(mother)
    list_index_selected.sort()
    list_selected=[]
    count=0
    for i in output_list:   #建立索引映射关系
        line_index=get_index(i)
        if line_index in list_index_selected :
            list_selected.append(i)
            dict_key[str(line_index)]=str(count+cin_line)
            count+=1
    for i in range(len(list_selected)): #改变索引
        list_selected[i]=tag_change(list_selected[i],dict_key)
    return list_selected


def output(output_list):  #输出函数，将归结步骤输出到控制台
    for i in output_list:
        print(i)

def main():   #主函数
    file='yijie2.txt'
    lines=readProp(file)
    cin_line=len(lines)
    for i in lines:   
        print(i.get_xu_hao(),i.get_tuple_line())
    output_list=[]
    check=guijie_main(lines,output_list)
    if(check==True):
        output_list=handle_output(output_list,cin_line)
        output(output_list)
        print("归结成功")
    else:
        output(output_list)
        print("归结失败")



     
if __name__=='__main__':
      main()


