import random
from itertools import product, filterfalse
import numpy as np
from mappings import *
from collections import OrderedDict, Counter
from PriorSearch import PriorSearch


class Line(object): 
    """   The representation of a line   """
    def __init__(self,Type="row",Index=0,String=""):
        self.type = Type
        self.index = Index
        self.string = String
        self.patterns = [[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
        self.value = [0,0]      # [player1, player2]

    def update_value(self):
        self.value[0] = sum(map(lambda x: x[0]*x[1], zip(valueTableVector,self.patterns[0])))
        self.value[1] = sum(map(lambda x: x[0]*x[1], zip(valueTableVector,self.patterns[1])))

    def update_pattern(self):
        self.patterns = [[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
        for i in range(4,len(self.string)-4):
            player = int(self.string[i])
            if player < Empty:
                self.patterns[player][patternDict[self.string[i-4:i]+self.string[i+1:i+5]][player]] += 1

    def hasFive(self,player):
        return self.patterns[player][7] > 0


class Board(object):
    """
    The representation of the board.
    """
    # __slots__ = ['row', 'col', 'diagpri', 'diagsub', 'score']

    def __init__(self, network, turn=1, offend=1, explore_prob=0.2):
        self.network = network         # 神经网络
        self.height = Height + 8
        self.width = Width + 8
        self.moves = OrderedDict()     # {(x,y):[nShape0,nShape1]},  moves we've done
        self.legalMoves = Counter()     # {(x,y):# moves around}, points that we will consider
        self.row = [Line(Type='row',Index=x,String="0"*self.width) for x in range(self.height)]
        self.col = [Line(Type='col',Index=x,String="0"*self.height) for x in range(self.width)]
        self.diagpri = [Line(Type='diagpri',Index=x,String="0") for x in range(self.height+self.width-1)]
        self.diagsub = [Line(Type='diagsub',Index=x,String="0") for x in range(self.height+self.width-1)]
        self.initialBoard()

        self.step = 0
        self.my = Player1
        self.opp = Player2
        self.Cross = [[set() for _ in range(16)] for _ in range(2)]  # 双方下一步能成的棋型 [my=[...],opp=[...]]
        self.best = [4,4]
        self.solved = False # debug

        self.turn = turn
        self.explore_prob = explore_prob
        self.offend = offend
        self.feature = self.extract_feature()
        self.MAX_POINT = 5


    def initialBoard(self): 
        """ Initial the lines  """
        margin = [[Out]*(self.width)] * 4
        self.board = margin + [([Out]*4) + [Empty for _ in range(self.width-8)] + ([Out]*4) for _ in range(self.height-8)] + margin
        for row_idx in range(self.height):
            str_row = ''.join(map(str, self.board[row_idx]))
            self.row[row_idx].string = str_row

        for col_idx in range(self.width):
            col = [row[col_idx] for row in self.board]
            str_col = ''.join(map(str, col))
            self.col[col_idx].string = str_col

        for i in range(-self.width + 1, self.height):
            if i < 0:
                TL_x, TL_y = (0, -i)
                BR_x, BR_y = (self.width + i, self.width)
            else:
                TL_x, TL_y = (i, 0)
                BR_x, BR_y = (self.height, self.height - i)

            str_diagonal = ''.join(map(lambda pos: str(self.board[pos[0]][pos[1]]), zip(range(TL_x, BR_x), range(TL_y, BR_y))))
            self.diagpri[self.width - 1 + TL_x - TL_y].string = str_diagonal

        for i in range(self.width + self.height - 1):
            if i < self.height:
                BL_x, BL_y = (i, 0)
                TR_x, TR_y = (-1, i + 1)
            else:
                BL_x, BL_y = (self.height - 1, i - self.height + 1)
                TR_x, TR_y = (i - self.height - 2, self.width)

            str_diagonal = ''.join(map(lambda pos: str(self.board[pos[0]][pos[1]]), zip(range(BL_x, TR_x, -1), range(BL_y, TR_y))))
            self.diagsub[BL_x + BL_y].string = str_diagonal


    def setPoint(self,x,y,piece):
        """ update the lines  """
        self.row[x].string = self.row[x].string[:y] + str(piece) + self.row[x].string[y+1:]
        self.col[y].string = self.col[y].string[:x] + str(piece) + self.col[y].string[x+1:]
        mi = min(x,y)
        self.diagpri[self.width-1+x-y].string = self.diagpri[self.width-1+x-y].string[:mi] + str(piece) + self.diagpri[self.width-1+x-y].string[mi+1:]
        mi = min(y, self.height - x - 1)
        self.diagsub[x+y].string = self.diagsub[x+y].string[:mi] + str(piece) + self.diagsub[x+y].string[mi+1:]

        
    def getPointPattern(self,x,y):
        ''' return: patterns[row,col,diag,subdiag], nShape[my,opp]'''
        cur_patterns = [[0,0,0,0],[0,0,0,0]]
        # row
        string = self.row[x].string[y-4:y] + self.row[x].string[y+1:y+5]
        cur_patterns[0][0] = patternDict[string][0]
        cur_patterns[1][0] = patternDict[string][1]
        # col
        string = self.col[y].string[x-4:x] + self.col[y].string[x+1:x+5]
        cur_patterns[0][1] = patternDict[string][0]
        cur_patterns[1][1] = patternDict[string][1]
        # pri-diag
        mi = min(x,y)
        string = self.diagpri[self.width-1+x-y].string[mi-4:mi] + self.diagpri[self.width-1+x-y].string[mi+1:mi+5]
        cur_patterns[0][2] = patternDict[string][0]
        cur_patterns[1][2] = patternDict[string][1]
        # sub-diag
        mi = min(y, self.height - x - 1)
        string = self.diagsub[x+y].string[mi-4:mi] + self.diagsub[x+y].string[mi+1:mi+5]
        cur_patterns[0][3] = patternDict[string][0]
        cur_patterns[1][3] = patternDict[string][1]
        return cur_patterns, [crossTable[cur_patterns[0][0]][cur_patterns[0][1]][cur_patterns[0][2]][cur_patterns[0][3]],\
                              crossTable[cur_patterns[1][0]][cur_patterns[1][1]][cur_patterns[1][2]][cur_patterns[1][3]]]

    def updateNbCross(self,x,y,sign):   # sign = +1 / -1
        ''' update the cross patterns of neighbours of (x,y) '''
        # vertical, horizontal, Main diagonal, Sub-diagonal
        dx = (1, 0, 1, 1)
        dy = (0, 1, 1, -1)

        for direction in range(0,4):
            x_0 = x - 4*dx[direction]
            y_0 = y - 4*dy[direction]
            for j in range(0,9):
                if j == 4: continue
                xx = x_0 + j*dx[direction]
                yy = y_0 + j*dy[direction]
                if self.board[xx][yy] != Empty: continue
                _,full = self.getPointPattern(xx,yy)
                if sign == '+':
                    self.Cross[0][full[0]].add((xx,yy))
                    self.Cross[1][full[1]].add((xx,yy))
                if sign == '-':
                    self.Cross[0][full[0]].discard((xx,yy))
                    self.Cross[1][full[1]].discard((xx,yy))


    def move(self, x, y, scope=2):
        ''' put a chess at (x,y) '''
        _, prefull = self.getPointPattern(x,y)
        self.Cross[0][prefull[0]].discard((x,y))    
        self.Cross[1][prefull[1]].discard((x,y))
        self.moves[(x,y)] = prefull 
        self.updateNbCross(x,y,'-')    

        # put chess and update information
        self.board[x][y] = self.my
        self.setPoint(x,y,self.my)
        self.updateNbCross(x,y,'+')    
        self.step += 1
        self.setplayer(self.opp)
        
        # update legalMoves
        for xx, yy in product(range(x - scope, x + scope + 1), range(y - scope, y + scope + 1)):
            if self.isFree(xx,yy) and (xx != x or yy!= y):
                self.legalMoves[(xx,yy)] += 1


    def delmove(self,scope=2):
        ''' restore a chess; completed with stack '''
        premove,prefull = self.moves.popitem()
        x,y = premove
        self.Cross[0][prefull[0]].add((x,y))
        self.Cross[1][prefull[1]].add((x,y))
        self.updateNbCross(x,y,'-')

        # get off chess and update information
        self.board[x][y] = Empty
        self.setPoint(x,y,Empty)
        self.updateNbCross(x,y,'+')  
        self.setplayer(self.opp)
        self.step -= 1

        # update legalMoves
        for xx, yy in product(range(x - scope, x + scope + 1), range(y - scope, y + scope + 1)):
            if self.isFree(xx,yy) and (xx != x or yy!= y):
                self.legalMoves[(xx,yy)] -= 1


    def calculate_utility(self, x, y):
        self.move(x, y)
        ans = self.evaluation()
        self.delmove()
        return ans

    
    def extract_feature(self):
        ''' feature extracter '''
        lst = []
        # my
        for i in range(len(self.Cross[Player1])):
            num = len(self.Cross[Player1][i])      
            if num >= 1:   
                lst.append(int(self.turn == Player1))
                lst.append(int(self.turn == Player2))
            else:
                lst.append(0)
                lst.append(0)
            if i == 0:  continue
            for _ in range(4):      
                if num >= 1:
                    lst.append(1)
                    num -= 1
                else:  lst.append(0)
            lst.append(num / 2) 
        lst.append(int(self.turn == Player1))

        # opp
        for i in range(len(self.Cross[Player2])):
            num = len(self.Cross[Player2][i])
            if num >= 1:   
                lst.append(int(self.turn == Player1))
                lst.append(int(self.turn == Player2))
            else:
                lst.append(0)
                lst.append(0)
            if i == 0:  continue
            for _ in range(4): 
                if num >= 1:
                    lst.append(1)
                    num -= 1
                else:   lst.append(0)
            lst.append(num / 2)  
        lst.append(int(self.turn == Player2))

        # offend       
        lst.append(int(self.offend == Player1))
        lst.append(int(self.offend == Player2))
        f = np.array(lst).reshape(len(lst), 1)     
        return f     

    
    def evaluation(self, feature=None):
        '''get Q estimator'''
        if feature is None:  feature = self.extract_feature()
        params_values = self.network.params
        value, _ = self.network.full_forward_propagation(feature, params_values) # 当前feature作为输入，通过神经网络获得Q估计
        return value


    def Q_value(self, legal_actions=None):
        '''get Q real'''
        if legal_actions is None:  legal_actions = self.getLegalMoves()
        besta = None
        if self.turn == Player1:       
            bestq = float('-inf')
            for a in legal_actions:
                self.move(a[0], a[1])
                q = self.evaluation() + np.random.rand()/1000   
                self.delmove()
                if q >= bestq:
                    bestq = q
                    besta = a
        else:                       
            bestq = float('inf')
            for a in legal_actions:
                self.move(a[0], a[1])
                q = self.evaluation() - np.random.rand()/1000 
                self.delmove()
                if q <= bestq:     
                    bestq = q
                    besta = a

        return bestq, besta


    def e_greedy(self, e=None):
        '''epsilon-greedy to find best action'''
        actions = self.getLegalMoves()
        if e is None:  e = self.explore_prob

        if np.random.rand() > e:  value, action = self.Q_value(actions)
        else:  action = random.choice(actions)

        return action

    
    def is_win(self):
        ''' return the winner player1 or player2, none return -1'''
        for lines in [self.row, self.col, self.diagpri, self.diagsub]:
            for line in lines:
                line.update_pattern()
                if line.hasFive(Player1): return Player1
                if line.hasFive(Player2): return Player2

        return -1

    
    def isFull(self):
        ''' judge whether it's full or not '''
        return np.sum(np.array(self.board)[4:self.height-4, 4:self.width-4]) <= 200


    def isFree(self, x, y):
        ''' judge whether it's full or not '''
        return x >= 4 and y >= 4 and x < self.width-4 and y < self.height-4


    def setplayer(self, player):
        self.my = player
        self.opp = player ^ 1
    

    def getMoveValue(self, x, y):
        ''' get move estimate value; for pruning '''
        cur_patterns = self.getPointPattern(x,y)[0]
        score = [valueTable[cur_patterns[0][0]][cur_patterns[0][1]][cur_patterns[0][2]][cur_patterns[0][3]],\
                valueTable[cur_patterns[1][0]][cur_patterns[1][1]][cur_patterns[1][2]][cur_patterns[1][3]]]
        
        if score[self.my] >= 200 or score[self.opp] >= 200:
            if score[self.my] >= score[self.opp]:
                return 2 * score[self.my]
            else:
                return score[self.opp]
        return score[self.my] * 2 + score[self.opp]


    def getLegalMoves(self):  
        ''' get legal moves; we only consider points in our neighbour '''
        for set in [self.Cross[self.my][VCF6], self.Cross[self.opp][VCF6], self.Cross[self.my][VCF5]]:
            if len(set) > 0: return list(set)

        if len(self.Cross[self.opp][VCF5]) > 0:  
            moves_new = []
            for type in [VCF6,VCF5,VCF4,VCF3,VCF2,VCF1]:
                moves_new.extend(list(self.Cross[0][type]))
                moves_new.extend(list(self.Cross[1][type]))
            return moves_new

        moves = Counter()       
        for move in [move for move,_ in (self.legalMoves - Counter()).items()]:
            if self.board[move[0]][move[1]]==Empty: moves[move] = self.getMoveValue(move[0],move[1])
        moves = sorted(moves,key=lambda x: moves[x], reverse=True)

        return moves


####################################################################


    def rootSearch(self, ps):
        ''' searching from the root '''
        move_list = self.getLegalMoves()
        if len(move_list) == 0:    return (4, 4)   
        move_list = move_list[0:min(len(move_list), BRANCHFACTOR)]  

        # defend opponent's
        bestpoint = [move_list[0][0], move_list[0][1]]  
        Lose = True
        maxloseStep = 0
        deletels = []
        for move in move_list: 
            self.move(move[0], move[1]) 
            threatpoint = ps.VcfStart() 
            if threatpoint != -1:
                result = ps.Vct3(self.my, 0, 10, threatpoint) 
                if result > 0:  
                    deletels.append(move)
                    if Lose and (result > maxloseStep):
                        maxloseStep = result
                        bestpoint = move
                else:
                    if Lose:
                        Lose = False
                        bestpoint = move
            self.delmove() 

        # avoid opponent's 
        move_list = [move for move in move_list if (move not in deletels)]   
        n = len(move_list)
        if n == 0:   return  bestpoint
        if n == 1:   return  move_list[0]

        # search for best move
        return self.e_greedy(e=0)


    def Search(self):
        ''' We will do our Prior Search first then do use min max search '''
        # starting point
        if self.step == 0: return self.width // 2, self.height // 2

        # find if there exsits ps points
        ps = PriorSearch(self) 
        if ps.Vcf1() > 0: return self.best
        if ps.Vct1() > 0: return self.best

        # further finding 
        return self.rootSearch(ps)  