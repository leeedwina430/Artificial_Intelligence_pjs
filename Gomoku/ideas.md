
---

### **更多优化的思考：**

\1. 考虑利用多线程，让算法实现并行计算，每个线程负责不同子树，可提高速率；

\2. 落子的影响范围有限，搜索过程中很多点的估值并没有改变，可以考虑把一些点的估值记录下来，以后只要遇到搜索到的节点，就可以直接得到结果，可提高速率；

\3. 由于算法的固定性，所以一担玩家一次获胜，按照相同的走法，必然会再次获胜。但除了必杀招或者必防招，一个局面很多时候没有绝对最好的走法，而是有一些都不错的走法。考虑把这些评分差不多走法汇集起来，然后随机选择它们中的一种走法，避免走法的固定。

\4. 可利用神经网络思想来提高速率，存储结点信息，增加学习功能，避免再次犯错。


1. 将一些棋局形式以hash值（或用列表）存储，判断局面时可以先查表，加快速度（后端）
2. 对于评分相近（grader<=threshold）的走法，随机抽取
3. 每落下一个新的子，回溯；不对全棋盘进行再判断


---

### 打包

```css
pyinstaller example.py Board.py NeuralNetwork.py pisqpipe.py --name pbrain-pyRL.exe --onefile

pyinstaller example.py Board.py NeuralNetwork.py pisqpipe.py Estimator.py --name pbrain-pyRL3.exe --onefile

pyinstaller example.py Board.py mapping.py pisqpipe.py --name pbrain-pyAB.exe --onefile

pyinstaller example.py Board.py Estimator.py pisqpipe.py --name pbrain-pyAB.exe --onefile

pyinstaller example.py Board.py mappings.py pisqpipe.py --name pbrain-pyAB3.exe --onefile
```

==注意：提交时一定要删清所有调试输出！！==

---

### Q-learning 训练踩坑记

```css
pyinstaller example.py Board.py NeuralNetwork.py pisqpipe.py --name pbrain-pyRL.exe --onefile
```

e_greedy没有返回action

还是在 (5,6) 落子过后，没找到点

经过实验，直接 $random.choice(self.actions)$ 可以正常运行。
发现是 $self.Q_value(self.actions)$ 没找到 


问题：Y_hat是个float，没有shape参数


将Q_value()函数中选取besta的判断式改为 <=


增加了平局的判断之后，仍然会有问题


最后残留的问题：会有除以0的情况（train的时候也会有权重为inf）



---

- [x] 对某点判断pattern：提取四条线对应8邻域字符串，转为int()打到patternTable里


- [ ] 每落一子，更新棋盘的nShape，四条线的patterns；

- [ ] nShape是个[ []\*16, []\*16 ] 的list，每次落子增加对应nShape点，悔子直接pop对应nShape点

- [ ] 在进行VCT的时候，用Set(nShape)进行操作



已经去除了很多种情况….原来的情况有很多是不可能发生的….👍



#### 1230继续努力：

1. 把RL改成能够融入现在的表示的形式
2. 重新开跑，试试效果
3. VCF挑挑bug+尝试改进


#### 1231：

1.  改好打包问题，尝试用了学习的参数效果如何 => 有用
2. 和之前的对战试试 
3. VCF / VCT改掉
4. 魔改之后再看看是否有可能搞改进
5. <font color=red>记得最后删掉logdebug</font>

---

### Alpha-Beta 踩坑记

```css
pyinstaller example.py Board.py mapping.py pisqpipe.py --name pbrain-pyAB.exe --onefile

pyinstaller example.py Board.py Estimator.py pisqpipe.py --name pbrain-pyAB1.exe --onefile
```

1. 先不实现保存每行pattern；
2. 先进行fullpattern的全搜索试试（或者采取branchfactor的方法）


#### AB0-初始版VCT

#### AB1-高级版VCT


1. VCF/VCT有两次落在了(0,0)点：Debug

可能的(4,4)点返回：Vcf / Vct 内部没有搜索到合法点

2. 有几率严重超时？？==（时间都在哪里？）==

慢（会超时）+有些点搜索得还是有点问题

3. 超时的可能原因：在进行VCF，VCT搜索的时候，最后一步还是遍历了所有可能，没有从开始点出发


这里的搜索可能超时

解决：加上对于初始点的判断；只搜索初始点附近的几个点判断条件


没有判断是否可以落子（isFree）
为什么会找到错误的点？——没有找到这个点


4. 关于用了set()后用pop()会有什么结果


另：只要保证在循环中，a.pop()完之后还能加回来原来的点即可


##### 关于用了numpy超内存的问题

1. 用import numpy as np 还是 from numpy import xxx 两者没有区别


##### yjh建议

1. VCT初始时候从step > 5时候开始（根据经验）
2. 打包加速：用那个神奇压缩软件
3. VCF和VCT的合并？
3. 估值函数的改变
3. 减少一定minimax层数
3. 尽可能加深minimax



