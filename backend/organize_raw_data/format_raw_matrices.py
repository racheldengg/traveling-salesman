import math
nums = []
f = open("format.txt", "r")
lines = f.readlines()
f.close()

for l in lines:
    nums.extend(l.split())
dest = open("format.txt", "w")


lengths = [len(s) for s in nums]
longest_index = lengths.index(max(lengths))
max_len = len(nums[longest_index])
empty = " " * max_len

k = len(nums)
size = math.floor(math.sqrt(k * 2 - 1))
print(size)
line_num = 0
row_num = 0

UPPER_TRIANGULAR = 0
LOWER_TRIANGULAR = 1
UPPER_DIAG = 2
LOWER_DIAG = 3

mode = 0

match mode:
    case 0 | 2:
        for n in nums:
            dest.write(n + empty[len(n):] + " ")
            line_num += 1

            if line_num == size:
                row_num += 1
                line_num = row_num
                dest.write("\n")
                dest.write(" " * (row_num * max_len) + " " * row_num)
    case 1 | 3:
        row_num = 1
        for n in nums:
            dest.write(n + empty[len(n):] + " ")
            line_num += 1

            if line_num == row_num:
                row_num += 1
                line_num = 0
                dest.write("\n")
    case 5:
        mat = []
        for i in range(size):
            mat.append([])
        
        row = 0
        line = 0
        for n in nums:
            mat[row][line] = n
            mat[line][row] = n
            line += 1
            if line == size:
                row += 1
                line = 0
        
        line_num = 0
        for n in mat:
            dest.write(n + empty[len(n):] + " ")
            if line_num == size:
                dest.write("\n")
                line_num = 0
    
    case 6:
        line_num = 0
        for n in nums:
            dest.write(n + empty[len(n):] + " ")
            line_num += 1
            if line_num == size:
                dest.write("\n")
                line_num = 0
