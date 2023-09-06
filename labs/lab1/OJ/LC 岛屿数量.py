#
# 解题思路：
# 1. grid 网格遍历问题 => 按照行列进行遍历
# 2. 





#%%
# 方法：按照连续的1/0进行遍历；右边+下边
# 问题1：有可能会有没有遍历到地
# 问题2：用dictionary判断有无遍历过，占过多内存并且耗时过长


d = {"Apple":1,"Pear":2,"Orange":3}
print(d.setdefault("Apple",0))

if d.setdefault("Pineapple",0) == 0:
    d["Pineapple"] = 1
else:
    print(...)

print(d)


class Solution:

    def numIslands(grid = [
        ["1","1","1","1","0"],
        ["1","1","0","1","0"],
        ["1","1","0","0","0"],
        ["0","0","0","0","0"]]) -> int:
        m,n = len(grid),len(grid[0])
        onedic = {}
        zerodic = {}
        oneque,zeroque = list(),list()
        count = 0
        
        if grid[0][0] == "1":
            onedic[(0,0)] = "Exist"
            oneque.append((0,0))
            count = count + 1
        else:
            zerodic[(0,0)] = "Exist"
            zeroque.append((0,0))
            
        while len(oneque)!=0 or len(zeroque)!=0:
            if len(oneque)!=0:
                cur = oneque.pop(0)
                curm,curn = cur[0],cur[1]
                
                if curn+1 < n:
                    if grid[curm][curn+1] == "1":
                        if grid[curm][curn+1] not in onedic.keys():
                            onedic[(curm,curn+1)] = "Exist"
                            oneque.append((curm,curn+1))
                    if grid[curm][curn+1] == "0":
                        if grid[curm][curn+1] not in zerodic.keys():
                            zerodic[(curm,curn+1)] = "Exist"
                            zeroque.append((curm,curn+1))

                elif curm + 1 < m:
                    if grid[curm+1][curn] == "1":
                        if grid[curm+1][curn] not in onedic.keys():
                            onedic[(curm+1,curn)] = "Exist"
                            oneque.append((curm+1,curn))
                    if grid[curm+1][curn] == "0":
                        if grid[curm+1][curn] not in zerodic.keys():
                            zerodic[(curm+1,curn)] = "Exist"
                            zeroque.append((curm+1,curn))

                if len(oneque) == 0:
                    count = count + 1

            else:
                cur = zeroque.pop(0)
                curm,curn = cur[0],cur[1]

                if curn+1 < n:
                    if grid[curm][curn+1] == "1":
                        if grid[curm][curn+1] not in onedic.keys():
                            onedic[(curm,curn+1)] = "Exist"
                            oneque.append((curm,curn+1))
                    if grid[curm][curn+1] == "0":
                        if grid[curm][curn+1] not in zerodic.keys():
                            zerodic[(curm,curn+1)] = "Exist"
                            zeroque.append((curm,curn+1))

                elif curm + 1 < m:
                    if grid[curm+1][curn] == "1":
                        if grid[curm+1][curn] not in onedic.keys():
                            onedic[(curm+1,curn)] = "Exist"
                            oneque.append((curm+1,curn))
                    if grid[curm+1][curn] == "0":
                        if grid[curm+1][curn] not in zerodic.keys():
                            zerodic[(curm+1,curn)] = "Exist"
                            zeroque.append((curm+1,curn))


        print(count)

Solution.numIslands()

#%%
# 广度优先搜索；上下左右
# 方法： 直接按照行列进行遍历；用(grid[r][c] = "2")表示已经遍历过的位置；
# 复杂度： O(mn) （访问）
'''
执行用时：120 ms, 在所有 Python3 提交中击败了57.93%的用户
内存消耗：23.7 MB, 在所有 Python3 提交中击败了95.00%的用户
'''


grid_ = [
        ["1","1","1","1","0"],
        ["1","1","0","1","0"],
        ["1","1","0","0","0"],
        ["0","0","0","0","0"]]

grid = [
    ["1","1","0","0","0"],
    ["1","1","0","0","0"],
    ["0","0","1","0","0"],
    ["0","0","0","1","1"]]

count = 0
m,n = len(grid),len(grid[0])

for r in range(m):
    for c in range(n):
        if grid[r][c] == "1":
            count += 1
            myque = [(r,c)]
            grid[r][c] = "2"
            while myque:
                curm,curn = myque.pop(0)
                # 上下左右都要判断；用for循环减少代码行数
                for x,y in [(curm-1,curn),(curm,curn-1),(curm+1,curn),(curm,curn+1)]:
                    if 0 <= x < m and 0 <= y < n and grid[x][y] == "1":
                        myque.append((x,y))
                        grid[x][y] = "2"

print(count)

#%%
# 深度优先DFS
# 复杂度： O(mn) 
'''
执行用时：108 ms, 在所有 Python3 提交中击败了82.39%的用户
内存消耗：23.7 MB, 在所有 Python3 提交中击败了87.99%的用户
'''

grid = [
    ["1","1","0","0","0"],
    ["1","1","0","0","0"],
    ["0","0","1","0","0"],
    ["0","0","0","1","1"]]

count = 0
m,n = len(grid),len(grid[0])

for r in range(m):
    for c in range(n):
        if grid[r][c] == "1":
            grid[r][c] = "2"
            count += 1
            stack = [(r,c)]
            while stack:
                curm,curn = stack.pop()
                for x,y in [(curm-1,curn),(curm,curn-1),(curm+1,curn),(curm,curn+1)]:
                    if 0<=x<m and 0<=y<n and grid[x][y]=="1":
                        grid[x][y]="2"
                        stack.append((x,y))


print(count)



