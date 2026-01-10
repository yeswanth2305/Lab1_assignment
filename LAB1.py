import random

def count_pairs(list1):
    count=0
    for i in range(len(list1)):
        for j in range(i+1,len(list1)):
            if list1[i]+list1[j]==10:
                count+=1
    return count

def difference(values):
    if len(values)<3:
        return "range determination not possible"
    else:
        return max(values)-min(values)

def multiply_matrices(matrixa,matrixb):
    size=len(matrixa)
    result=[[0 for k in range(size)] for h in range(size)]
    for q in range(size):
        for w in range(size):
            for e in range(size):
                result[q][w]+=matrixa[q][e]*matrixb[e][w]
    return result

def matrix_power(matrixa,m):
    result=matrixa
    for t in range(1,m):
        result=multiply_matrices(result,matrixa)
    return result

def count_characters(s):
    max_char=""
    max_num = 0
    for char in s:
        if char.isalpha():
            count=s.count(char)
            if count>max_num:
                max_num = count
                max_char=char
    return max_num,max_char

def mean_median_mode(nums):
    total=0
    for number in nums:
        total=total+number
    mean=total/len(nums)

    nums.sort()
    middle=len(nums)//2
    median=nums[middle]

    numbers_count={}
    for number in nums:
        if number in numbers_count:
            numbers_count[number]=numbers_count[number]+1
        else:
            numbers_count[number]=1

    mode=nums[0]
    max_times=0
    for key in numbers_count:
        if numbers_count[key]>max_times:
            max_times=numbers_count[key]
            mode=key
    return mean,median,mode

list1=[2,7,4,1,3,6]
print("counting pairs of elements with sum equal to 10:",count_pairs(list1))

list2=[5,3,8,1,0,4]
print("range of list:",difference(list2))

matrixa=[
    [1,2],
    [3,4]
]

m=2
print("matrix a raised to power",m)
print(matrix_power(matrixa,m))

s= input("enter the string: ")
char,count=count_characters(s)
print(f"the maximum number of occurences of a character in the string is {count_characters(s)}")
print("the highest occurring character is:",char)
print("number of occurrence:",count)

random_numbers=[]
for u in range(25):
    random_numbers.append(random.randint(1,10))
mean,median,mode=mean_median_mode(random_numbers)

print("Random numbers:",random_numbers)
print("mean:",mean)
print("median",median)
print("mode",mode)