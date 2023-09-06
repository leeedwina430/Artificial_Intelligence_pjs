#%%
import heapq

f=open("input112.txt")

nodes = []
edges = []
while True:
#    str = input()
    str = f.readline().strip()
    
    if str == "END":
        break
    begin,end,leng = str.split()
    leng = eval(leng)
    nodes.append(begin);nodes.append(end)

    edges.append((begin,end,leng))

flag = 1
nodes = list(set(nodes))
if not 'Start' in nodes:
    flag = 0
m = len(nodes)
map = [['inf' for _ in range(m)] for i in range(m)]
mapos = {}
for i in range(len(nodes)):
    mapos[nodes[i]] = i

for edge in edges:
    map[mapos[edge[0]]][mapos[edge[1]]] = edge[2]


explored = []
explored_node = []
frontier = []
heapq.heappush(frontier,(0,"Start","Null"))

while frontier and flag == 1:

    curvalue,node,parent = heapq.heappop(frontier)

    explored.append((curvalue,node,parent))
    explored_node.append(node)

    # 回溯路径
    if node == "Goal":
        path = []
        #cost = 0
        curnode = node
        explored_ = [x for _,x,_ in explored]
        while parent in explored_:
            path.append(curnode)
            #cost += map[mapos[parent]][mapos[curnode]]
            curnode = parent
            parent = explored[explored_.index(curnode)][2]
        path.append(curnode)
        path.reverse()
        print('->'.join(path))
        print(curvalue)
        break

    for j in range(m):
        cost = map[mapos[node]][j]
        child = nodes[j]
        
        if cost != "inf" and child not in explored_node:
            frontier_ = [x for _,x,_ in frontier]

            if child in frontier_:
                minvalue = frontier[frontier_.index(child)][0]

                if cost+curvalue < minvalue:
                    frontier.pop(frontier_.index(child))
                    heapq.heappush(frontier,(cost+curvalue,child,node))
            else:
                heapq.heappush(frontier,(cost+curvalue,child,node))

else:
    print("Unreachable")
