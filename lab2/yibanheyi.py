def find_diff(l1,l2):   #找到两个列表不同的第一个位置
    for i in range(len(l1)):
        if(l1[i]!=l2[i]):
            return i
    return len(l1)

def cacukuohao(str):   #辅助函数，计算括号数量，帮助to_list函数正确分割参数
    count=0
    for i in range(len(str)):
        if(str[i]=='('):
            count+=1
        elif(str[i]==')'):
            count-=1
    return count

def to_list(str):   #将字符串转换为列表，处理括号和逗号，帮助mgu函数处理参数
    begin1=str.find('(')+1
    end1=len(str)-1
    list1=str[begin1:end1].split(',')  #先按照逗号分割
    list2=[]
    count=0
    nw=''
    for i in range(len(list1)):  #遍历list1，每个元素可能包含括号，如果括号数量不为0，说明这个元素还没有分割完，需要继续拼接，否则直接添加到list2中
        temp=cacukuohao(list1[i])
        if count==0 and temp==0:
            list2.append(list1[i])
        elif count==0 and temp>0:
            nw+=list1[i]
            count+=temp
        elif count!=0:
            nw+=','
            nw+=list1[i]
            count+=temp
            if count==0:
                list2.append(nw)
                nw='' 
    return list2

def is_var(str):   #判断字符串是否为变量，变量列出
    list2={'x','y','z','w','v','u','t','s'}
    if str in list2:
            return True
    return False

def is_func(str):   #判断字符串是否为函数，函数包含括号
    return  '('in str and ')'in str

def get_term(str):  #判断字符串是常量、变量还是函数，常量返回1，变量返回2，函数返回3，不是函数，变量，就是常量
    if(is_var(str)):
        return 2
    elif(is_func(str)):
        return 3
    else :
        return 1
    
def turn_var_to_constant_in_func(str_fun,var,constant):  #将函数str_fun中的变量var替换为常量constant
    list1=to_list(str_fun)
    for i in range(len(list1)):
        if(is_var(list1[i])):
            if(list1[i]==var):
                list1[i]=constant
        elif(is_func(list1[i])):
            list1[i]= turn_var_to_constant_in_func(list1[i],var,constant) #处理函数嵌套的情况
    str_re=str_fun[0:str_fun.find('(')+1]
    for i in list1:
        str_re+=i+','
    str_re = str_re[:-1] + ')'
    return str_re

def var_constant(l1,l2,dict_re,var,constant):    #将l1中的var替换为constant
    dict_re[var]=constant
    for i in range(len(l1)):
        if(is_var(l1[i])):
            if(l1[i]==var):
                l1[i]=constant
        elif(is_func(l1[i])):
            l1[i]= turn_var_to_constant_in_func(l1[i],var,constant)
    return True

def var_in_func_check(str_fun,var):  #判断函数str_fun中是否有变量var
    list1=to_list(str_fun)
    for i in range(len(list1)):
        if(is_var(list1[i])):
            if(list1[i]==var):
                return True
        elif(is_func(list1[i])):
            if(var_in_func_check(list1[i],var)):
                return True
    return False

def turn_var_to_function_in_func(str_fun,var,function):  #将函数str_fun中的变量var替换为函数function
    list1=to_list(str_fun)
    for i in range(len(list1)):
        if(is_var(list1[i])):
            if(list1[i]==var):
                list1[i]=function
        elif(is_func(list1[i])):
            list1[i]= turn_var_to_function_in_func(list1[i],var,function) #处理函数嵌套的情况
    str_re=str_fun[0:str_fun.find('(')+1]
    for i in list1:
        str_re+=i+','
    str_re = str_re[:-1] + ')'
    return str_re

def var_function(l1,l2,dict_re,var,function):  #将l1中的var替换为函数function
        if(var_in_func_check(function,var)):
            return False
        else:
            dict_re[var]=function
            for i in range(len(l1)):
                if(is_var(l1[i])):
                    if(l1[i]==var):
                        l1[i]=function
                elif(is_func(l1[i])):
                    l1[i]= turn_var_to_function_in_func(l1[i],var,function)
        return True

def var_var(l1,l2,dict_re,var,var2):    #将l1中的var替换为var2
    dict_re[var]=var2
    for i in range(len(l1)):
        if(is_var(l1[i])):
            if(l1[i]==var):
                l1[i]=var2
        elif(is_func(l1[i])):
            l1[i]= turn_var_to_function_in_func(l1[i],var,var2)
    return True

def fun_fun(l1,l2,dict_re,function,function2):   #将函数function替换为函数function2
    if(function[:function.find('(')]!=function2[:function2.find('(')]):
        return False
    f3,f4,dict_re2,check= mgu(function,function2)
    if(check==False):
        return False
    for key in dict_re2:
        # 替换 l1 所有出现 key 的地方
        for i in range(len(l1)):
            if is_var(l1[i]) and l1[i] == key:
                l1[i] = dict_re2[key]
            elif is_func(l1[i]):
                l1[i] = turn_var_to_function_in_func(l1[i], key, dict_re2[key])
        # 替换 l2 所有出现 key 的地方
        for i in range(len(l2)):
            if is_var(l2[i]) and l2[i] == key:
                l2[i] = dict_re2[key]
            elif is_func(l2[i]):
                l2[i] = turn_var_to_function_in_func(l2[i], key, dict_re2[key])
    dict_re.update(dict_re2)
    return True

def mgu(str1,str2):  #实现mgu，mgu部分主函数，输入两个字符串，输出替换后的字符串和替换的映射关系，以及是否匹配成功的标志
    dict_re={}
    l1=to_list(str1)
    l2=to_list(str2)
    diff=find_diff(l1,l2)
    check=True
    while(diff<len(l1)):
        a=get_term(l1[diff])*10+get_term(l2[diff])   #常量1 变量2 函数3
        temp1=l1[diff]
        temp2=l2[diff]
        match a:
            case 11:
                check=False
                return str1,str2,dict_re,check
            case 12: 
                var_constant(l1,l2,dict_re,temp2,temp1) 
                var_constant(l2,l1,dict_re,temp2,temp1)  #def var_constant(l1,l2,dict_re,var,constant):  
            case 13:
                check=False
                return str1,str2,dict_re,check
            case 21:   #var_function(l1,l2,dict_re,var,function): 
                var_function(l2,l1,dict_re,temp1,temp2)
                var_function(l1,l2,dict_re,temp1,temp2)   
            case 22:#def var_var(l1,l2,dict_re,var,var2):    #将l1中的var替换为var2
                var_var(l1,l2,dict_re,temp1,temp2)
                var_var(l2,l1,dict_re,temp1,temp2)
            case 23:
                var_function(l1,l2,dict_re,temp1,temp2)
                var_function(l2,l1,dict_re,temp1,temp2)
            case 31:
                check=False
                return str1,str2,dict_re,check
            case 32:
                var_function(l2,l1,dict_re,temp2,temp1)
                var_function(l1,l2,dict_re,temp2,temp1)
            case 33:
                check= fun_fun(l1,l2,dict_re,temp1,temp2)
                if check==False:
                    return str1,str2,dict_re,check   
            case _:
                pass
        diff=find_diff(l1,l2)
    str1_result = str1[:str1.find('(')+1] + ','.join(l1) + ')'
    str2_result = str2[:str2.find('(')+1] + ','.join(l2) + ')'
    return str1_result,str2_result,dict_re,check

def set_new_line(line_new,dict_re):   #与mgu无关，将一个子句中其他谓词也进行变量替换，写在这里是为了减少import
    for i in range(len(line_new)):
        can_shu_list=to_list(line_new[i])
        for j in range(len(can_shu_list)):
            if(is_var(can_shu_list[j])):
                if(can_shu_list[j] in dict_re.keys()):
                    can_shu_list[j]=dict_re[can_shu_list[j]]
            elif(is_func(can_shu_list[j])):
                    for ss in dict_re.keys():
                        can_shu_list[j]=turn_var_to_function_in_func(can_shu_list[j],ss,dict_re[ss])
        prefix = line_new[i][:line_new[i].find('(') + 1]
        line_new[i]=prefix+','.join(can_shu_list)+')'



def main():   #测试mgu主函数，输入要求两个谓词间分号分隔，谓词内参数逗号分隔
    file='text.txt'
    with open(file,'r') as f:
        line=f.readline()
        str1=line.split(';')[0]
        str2=line.split(';')[1]
        print(str1)
        print(str2)
        str1_result,str2_result,dict_re,check=mgu(str1,str2)
        print(str1_result,str2_result,dict_re,check)
 
if __name__=='__main__':
    main()

   

    
    

