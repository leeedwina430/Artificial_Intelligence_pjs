# -*- coding: utf-8 -*-


from functools import reduce

# You should read those classes carefully before you begin to write your code
class BayesNet: 
    '''贝叶斯网络: 
    属性: nodes(list),variables(list)
    函数: variable_node(var) -> node (获取对应变量的node); 
          variable_values(var) -> [values] (获取对应变量的所有可能取值)
    '''
    def __init__(self, node_specs=[]):      # node_specs ???
        self.nodes = []     # 节点列表
        self.variables = []     # 变量列表，和节点对应
        for node_spec in node_specs:
            self.add(node_spec)

    def add(self, node_spec):      
        """Add a node to the net. Its parents must already be in the
        net, and its variable must not."""
        node = BayesNode(*node_spec)
        assert node.variable not in self.variables    # 
        self.nodes.append(node)
        self.variables.append(node.variable)
        for parent in node.parents:
            self.variable_node(parent).children.append(node)

    def variable_node(self, var):   # 获取代表这个var的节点
        for n in self.nodes:
            if n.variable == var:
                return n
        raise Exception("No such variable: %s" % var)   # 字符串的操作

    def variable_values(self, vars):        # 获取var的所有可能取值
        return [True, False]

    def __repr__(self):
        return 'BayesNet(%r)' % self.nodes


class BayesNode:
    '''贝叶斯网络节点: 
    属性: variable(str), parents(list[str])
          cpt(dic{(variable,):probability}), conditional probability table
          children(list[BayesNode])
    函数: p(value,evidence) -> probability of value | evidence 
         evidence 是个 {variable(str):value(Boolean)}
    '''

    """A conditional probability distribution for a boolean variable,
    P(X | parents). Part of a BayesNet."""

    def __init__(self, x, parents, cpt):

        if isinstance(parents, str):        # isinstance: 检查parents是不是一个str
            parents = parents.split()

        # We store the table always in the third form above.
        if isinstance(cpt, (float, int)):  # no parents, 0-tuple    # isinstance: 检查cpt是不是float或者int
            cpt = {(): cpt}
        elif isinstance(cpt, dict):
            # one parent, 1-tuple
            if cpt and isinstance(list(cpt.keys())[0], bool):
                cpt = dict(((v,), p) for v, p in list(cpt.items())) # key:value 搞成(key,):value

        assert isinstance(cpt, dict)
        for _, p in list(cpt.items()):  # 检查是不是所有概率满足概率条件
            assert 0 <= p <= 1

        self.variable = x
        self.parents = parents
        self.cpt = cpt
        self.children = []

    def p(self, value, event):  # event是个dic
        """A function that will return the conditional probability given the
        value of X and all evidence(a dist), You can Use it as BayesNode.p(True, e)
        which will return the probability P(X=True | parents)"""
        assert isinstance(value, bool)      # 检查是否是布尔值
        ptrue = self.cpt[event_values(event, self.parents)] # cpt[e的概率] ？？ 返回真值时候的概率
        return (ptrue if value else 1 - ptrue)

    def __repr__(self):     # print(bayesnode) <=> print(bayesnode.__repr__())
        return repr((self.variable, ' '.join(self.parents)))


class ProbDist:
    '''概率分布
    属性: prob(dic{value:probability}) 对应取值的概率
          varname(str) 该分布对应变量的名字
          values(list[value]) 变量所有可能取值
    函数: getitem(value) -> probability 对应取值的概率值
          setitem(value, probability) 设置对应取值的概率值
          normalize() 归一化self.prob
          show_approx -> print 打印对应概率
    '''
    def __init__(self, varname='?', freqs=None):
        """If freqs is given, it is a dictionary of value: frequency pairs,
        and the ProbDist then is normalized."""
        self.prob = {}
        self.varname = varname
        self.values = []
        if freqs:
            for (v, p) in list(freqs.items()):
                self[v] = p     # <=> self.__getitem__(val) => self.prob[val]
            self.normalize()

    def __getitem__(self, val):
        "Given a value, return P(value)."
        try:
            return self.prob[val]
        except KeyError:
            return 0

    def __setitem__(self, val, p):
        "Set P(val) = p."
        "You can use ProbDist(x_i) = p to assign X=x_i with probability p"
        if val not in self.values:
            self.values.append(val)
        self.prob[val] = p
    
    ########################################
    def getitem(self, val):
        "Given a value, return P(value)."
        try:
            return self.prob[val]
        except KeyError:
            return 0

    def setitem(self, val, p):
        "Set P(val) = p."
        "You can use ProbDist(x_i) = p to assign X=x_i with probability p"
        if val not in self.values:
            self.values.append(val)
        self.prob[val] = p

    ########################################

    def normalize(self):
        """Make sure the probabilities of all values sum to 1.
        Returns the normalized distribution.
        Raises a ZeroDivisionError if the sum of the values is 0."""
        total = sum(self.prob.values())
        if not isclose(total, 1.0):     # <=> total == 1 in 1e-9
            for val in self.prob:
                self.prob[val] = self.prob[val] * 1.0 / total
        return self

    def show_approx(self, numfmt='%.3g'):
        """Show the probabilities rounded and sorted by key, for the
        sake of portable doctests."""
        return ', '.join([('%s: ' + numfmt) % (v, p) for (v, p) in sorted(self.prob.items())])


class Factor:
    '''因子: 
    属性: variables(list[str]) 该因子的所有变量
          cpt(dic{(variable,):probability}), conditional probability table ? 
    函数: p(evidence) 返回对应的概率; evidence(dic) 包括了查询变量
          normalize() 归一化self.prob
    '''
    def __init__(self, variables, cpt):
        self.variables = variables
        self.cpt = cpt

    def __getitem__(self, val):
        "Given a value, return P(value)."
        try:
            return self.cpt[val]
        except KeyError:
            return 0

    def p(self, e):     # e=event, 返回
        return self.cpt[event_values(e, self.variables)]

    def normalize(self):
        total = sum(self.cpt.values())
        if not isclose(total, 1.0):
            for val in self.cpt:
                self.cpt[val] = self.cpt[val] * 1.0 / total
        return self     # return了自己

# You can skip the function below and that should be ok for your implementation
# Begining of functions that are optional to read
def joint_probability(x_group, d):
    prob_enumeration = enumerate_all(bn.variables, d, bn)
    return prob_enumeration

def conditional_probability(x_group, parent_vars, all_evidence, parent_evidence, children_TorF_tuple):
    if len(x_group) > 1:
        x_group_all = x_group + parent_vars
        joint_all, _ = joint_probability(x_group_all, all_evidence)
        condition, _ = joint_probability(parent_vars, parent_evidence)
        prob_enumeration = joint_all * 1.0 / condition
    else:
        a1 = enumeration_ask(x_group[0], parent_evidence, bn)
        #a2 = elimination_ask(x_group[0], parent_evidence, bn)
        prob_enumeration = a1[children_TorF_tuple]
    return prob_enumeration

def process_P_Query(query):
    for i in range(len(query)):
        if query[i] == '(':
            l = i
        if query[i] == ')':
            r = i
    content = query[l + 1:r]
    prob_enumeration = 0
    if '|' not in content:  # joint/ marginal distribution
        groups = content.split(', ')
        d = dict()
        x_group = []
        for group in groups:
            if '=' in group:
                x, y = group.split(' = ')
                TorF = processTF(y)
                d[x] = TorF
                x_group.append(x)
        prob_enumeration = joint_probability(x_group, d)

    else:  # conditional distribution
        children,  parents = content.split(' | ')
        parent_evidence = dict()
        all_evidence = dict()
        x_group = []
        parent_vars = []
        if ',' in parents:
            parent_groups = parents.split(', ')
            for i in range(len(parent_groups)):
                if '=' in parent_groups[i]:
                    x, y = parent_groups[i].split(' = ')
                    parent_evidence[x] = processTF(y)   # evidence x is a str
                    parent_vars.append(x)
        else:
            if '=' in parents:
                x, y = parents.split(' = ')
                parent_evidence[x] = processTF(y)
                parent_vars.append(x)

        if ',' in children:
            all_evidence = parent_evidence.copy()
            child_group = children.split(', ')
            children_TorF_tuple = []
            for i in range(len(child_group)):
                child = child_group[i]
                if '=' in child:
                    child, y = child.split(' = ')
                    x_group.append(child)
                    all_evidence[child] = processTF(y)
                    children_TorF_tuple.append(processTF(y))
        else:
            if '=' in children:
                child, y = children.split(' = ')
                children_TorF_tuple = processTF(y)
                x_group = [child]
        prob_enumeration = conditional_probability(x_group, parent_vars, all_evidence, parent_evidence, children_TorF_tuple)

    return prob_enumeration


def processTF(symbol):
    if symbol == "+":
        return True
    else:
        return False

def event_values(event, variables): # 返回 variables 对应的 概率value; event是个dic？
    if isinstance(event, tuple) and len(event) == len(variables):
        return event
    else:
        return tuple([event[var] for var in variables])

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    "Return true if numbers a and b are close to each other."
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def all_events(variables, e):
    if not variables:
        yield e
    else:
        X, rest = variables[0], variables[1:]
        for e1 in all_events(rest, e):
            for x in [True, False]:
                yield extend(e1, X, x)
# End of functions that are optional to read


## Begining of functions to be implemented
def enumeration_ask(X, e, bn):
    '''
    X: query variable
    e: (Dict) evidence variable the same format as X
    bn: (BayesNet) instance 
    #             Hint             #
    you can use Q(x_i) = p to assign X=x_i with probability p
    you can use extend(e,X,True/False) to extend the evidence with X=True/False
    '''
    Q = ProbDist(X)
    # YOUR_CODE_HERE (hint: only two-four lines)
    for xi in bn.variable_values(X):        # 要用模板的枚举
        Q.setitem(xi,enumerate_all(bn.variables, extend(e,X,xi),bn))
    return Q.normalize()

    # END_YOUR_CODE
#$
def enumerate_all(variables, e, bn):
    if not variables:
        return 1.0
    Y, rest = variables[0], variables[1:]
    Ynode = bn.variable_node(Y)
    # YOUR_CODE_HERE (hint: an if-statement)
    if Ynode.variable in e.keys():
        return Ynode.p(e[Ynode.variable],e) * enumerate_all(rest,e,bn)
    else:
        return sum(Ynode.p(y,e) * enumerate_all(rest, extend(e,Y,y), bn) for y in bn.variable_values(Y))
    # END_YOUR_CODE

## end of functions to be implemented


## Begining of functions that will be directly used in your implementation
def extend(s, var, val):    # 会返回一个新的s
    # extend s by seting var = val
    # s: evidence that will be extended
    # var: variable you want to extend
    # val: set var=val
    s2 = s.copy()
    s2[var] = val
    return s2

## End of functions that will be directly used in your implementation

if __name__=='__main__':
    bn = BayesNet()
    query = []
    finish_add_query = 0

    while True:
        try:
            line = input().strip()
            if line != '******' and finish_add_query == 0:
                query.append(line)
                continue

            if line == '******' and finish_add_query == 0:
                finish_add_query = 1

            if finish_add_query == 1:
                if '***' not in line:
                    if '|' not in line:  # no parent node
                        node = line
                        p = input().strip()
                        prob = float(p)
                        bn.add((node, '', prob))
                    else:  # parent node
                        cpt = dict()
                        node, parents = line.split(' | ')
                        line = input().strip()
                        every_parents = parents.split()

                        for i in range(int(2**len(every_parents))):
                            match = line.split()
                            if len(match) == 2:
                                match[1] = processTF(match[1])
                                cpt[match[1]] = float(match[0])
                            if len(match) == 3:
                                match[1] = processTF(match[1])
                                match[2] = processTF(match[2])
                                cpt[(match[1], match[2])] = float(match[0])
                            if len(match) == 4:
                                match[1] = processTF(match[1])
                                match[2] = processTF(match[2])
                                match[3] = processTF(match[3])
                                cpt[(match[1], match[2], match[3])] = float(match[0])
                            line = input().strip()
                        if not line:
                            bn.add((node, parents, cpt))
                        if line == '***' or line == '******':
                            bn.add((node, parents, cpt))
            if not line:
                break

        except EOFError:
            break


    for i in query:
        prob_enumeration = process_P_Query(i)
        print("probability by enumeration: {:.3f}".format(round(prob_enumeration, 3)))
        print("**********")