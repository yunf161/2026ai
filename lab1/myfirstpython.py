message1 ="hello world"  # python 的声明变量不需要类型，每一行也没有分号
message2 ="hello python"
print(message1,message2) # 可以同时打印多个变量，用逗号隔开
print(message1,message2,sep="___",end="!!!\n") #有sep参数指定多个输出变量之间的分隔符，默认空格
#有end参数指定输出的结尾，默认换行符
#message3=input("请输入文本：")
#接受stdin的输入input函数，参数是提示词字符串可选，返回值是字符串
# print(message3)
print(3/5) #除法，结果是浮点数
print(3//5)# 地板除，结果是整数
a=round(3/7,3) #四舍五入，保留k位小数   round（x，k） k默认0
print(a)
message4=" "+message1+" "+message2+' ' #z 字符串可以用加号拼接
print(message4)
print(message1*3) #字符串可以用乘号重复输出
print(message1.upper()) #大写输出
print(message4.strip()) #strip删除首尾空白字符
print(message4.lstrip())   #删除左侧空白字符，rstrip删除右侧空白字符
print(message4.split(' ')) #split方法以指定分隔符将字符串分割成列表，默认空格
print(message4.replace('hello','hi')) #replace 后替换前，前面必须完全匹配
print(message4.replace('helloi','hi')) #替换无效
print(int(3.5)) #强制类型转换
#python的代码块由缩进表示，通常是4个空格
print(0 and 20)
print(1 and 20)
#and运算符返回值是 如果第一个操作数假，是第一个操作数，否则是第二个操作数
print(0 or 20)
print(1 or 20)
#or运算符返回值是  如果第一个操作数真，返回第一个操作数，否则第二个操作数
print(not 0)
print(not 1)
#not 的返回值是布尔类型
#range(start,end,step(间隔)) 生成一个整数序列，默认start=0,step=1,不包含end包含start
sum=0
for i in range(100,0,-1):
    sum+=i
print(sum) #5050

    