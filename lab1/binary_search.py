def binary_search(arr,target):
     left=-1
     right=len(arr)
     while(left+1!=right):
          mid=(left+right)//2
          if(arr[mid]>target):
               right=mid
          else: 
               left=mid
     return left if left>=0 and arr[left]==target else -1

arr=[1,3,5,7,9]
print(binary_search(arr,4)) #1
print(binary_search(arr,5)) #2