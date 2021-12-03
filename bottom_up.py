import itertools

INT = "int"
BOOL = "bool"
NODE = "node"
TYPES = [
    INT,
    BOOL,
    NODE
]

VARIABLE_NAME = "var"
INDEX_VARIABLE_NAME = "i"

DEPTH_LIMIT = 3
VARIABLES_LIMIT = 3
VARIABLES = {f"__{VARIABLE_NAME}_{i}__" for i in range(VARIABLES_LIMIT)}

INDEX_VARIABLES_LIMIT = 2
INDEX_VARIABLES = {
    f"__{INDEX_VARIABLE_NAME}_{i}__" for i in range(INDEX_VARIABLES_LIMIT)}

PARTITIONS_CACHE = {}


class SymmetryException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Expr(object):
    """
    Abstract Expression Object
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return "(Abstract Expr)"

    def __repr__(self) -> str:
        return str(self)

    def evaluate(self, ctx):
        raise RuntimeError("Cannot Evaluate Abstract Expr.")

    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)


class NewNode(Expr):
    """
    Represents New Node
    e.g. new Node();
    """

    def __init__(self, val, nxt) -> None:
        super().__init__()
        if type(nxt) == Variable:
            raise SymmetryException()
        self.val = val
        self.next = nxt

    def __str__(self) -> str:
        return f"(new Node(next={self.next}; val={self.val}))"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def typ():
        return ((INT, NODE), NODE)

    def evaluate(self, ctx):
        return self

    def can_eval(self):
        return True

    def __eq__(self, __o: object) -> bool:
        if type(__o) != NewNode:
            return False
        return self.val == __o.val and self.next == __o.next

    def __hash__(self) -> int:
        return hash(self.next) + hash(self.val)

    def within(self, variables):
        return self.next.within(variables)


class GetNext(Expr):
    """
    Represents node.next
    """

    def __init__(self, node) -> None:
        super().__init__()
        if node == Null():
            raise SymmetryException()
        if type(node) == NewNode:
            raise SymmetryException()
        if type(node) == GetNext:
            raise SymmetryException()
        self.node = node

    def __str__(self) -> str:
        return f"({self.node}.next)"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def typ():
        return ((NODE,), NODE)

    def evaluate(self, ctx):
        return self.node.next

    def can_eval(self):
        return False

    def within(self, variables):
        return self.node.within(variables)

    def __eq__(self, __o: object) -> bool:
        if type(__o) != GetNext:
            return False
        return self.node == __o.node

    def __hash__(self) -> int:
        return hash(self.node)


class GetVal(Expr):
    """
    Represents node.val
    """

    def __init__(self, node) -> None:
        super().__init__()
        if node == Null():
            raise SymmetryException()
        if type(node) == NewNode:
            raise SymmetryException()
        if type(node) == GetNext:
            raise SymmetryException()
        self.node = node

    def __str__(self) -> str:
        return f"({self.node}.val)"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def typ():
        return ((NODE,), INT)

    def evaluate(self, ctx):
        return self.node.val

    def can_eval(self):
        return False

    def within(self, variables):
        return self.node.within(variables)

    def __eq__(self, __o: object) -> bool:
        if type(__o) != GetVal:
            return False
        return self.node == __o.node

    def __hash__(self) -> int:
        return hash(self.node)


class Add(Expr):
    """
    Represents (+ l r)
    """

    def __init__(self, l: Expr, r: Expr) -> None:
        super().__init__()
        if l == Zero() or r == Zero():
            raise SymmetryException()
        if l == One() and r == NegativeOne():
            raise SymmetryException()
        if l == NegativeOne() and r == One():
            raise SymmetryException()
        if type(r) == Index or isinstance(l, Number):
            raise SymmetryException()
        self.l = l
        self.r = r

    def __str__(self) -> str:
        return f"(+ {self.l} {self.r})"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def typ():
        return ((INT, INT), INT)

    def evaluate(self, ctx):
        return self.l.evaluate(ctx) + self.r.evaluate(ctx)

    def can_eval(self):
        return self.l.can_eval() and self.r.can_eval()

    def within(self, variables):
        return self.l.within(variables) and self.r.within(variables)

    def __eq__(self, __o: object) -> bool:
        if type(__o) != Add:
            return False
        return self.l == __o.l and self.r == __o.r or self.r == __o.l and self.l == __o.r  # add commutes

    def __hash__(self) -> int:
        return hash(self.l) + hash(self.r)


class Mult(Expr):
    """
    Represents (* l r)
    """

    def __init__(self, l: Expr, r: Expr) -> None:
        super().__init__()
        if l == One() or r == One():
            raise SymmetryException()
        if l == Zero() or r == Zero():
            raise SymmetryException()
        if l == NegativeOne() or r == NegativeOne():
            raise SymmetryException()
        if type(r) == Index or isinstance(l, Number):
            raise SymmetryException()
        self.l = l
        self.r = r

    def __str__(self) -> str:
        return f"(* {self.l} {self.r})"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def typ():
        return ((INT, INT), INT)

    def evaluate(self, ctx):
        return self.l.evaluate(ctx) * self.r.evaluate(ctx)

    def can_eval(self):
        return self.l.can_eval() and self.r.can_eval()

    def within(self, variables):
        return self.l.within(variables) and self.r.within(variables)

    def __eq__(self, __o: object) -> bool:
        if type(__o) != Mult:
            return False
        # mult commutes
        return self.l == __o.l and self.r == __o.r or self.r == __o.l and self.l == __o.r

    def __hash__(self) -> int:
        return hash(self.l) + hash(self.r)


class Mod(Expr):
    """
    Represents (modulo l r)
    """

    def __init__(self, l: Expr, r: Expr) -> None:
        super().__init__()
        if l == r:
            raise SymmetryException()
        if l == Zero() or r == Zero():
            raise SymmetryException()
        if l == One() or r == One():
            raise SymmetryException()
        if l == NegativeOne() or r == NegativeOne():
            raise SymmetryException()
        if l == Two():
            raise SymmetryException()
        self.l = l
        self.r = r

    def __str__(self) -> str:
        return f"(modulo {self.l} {self.r})"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def typ():
        return ((INT, INT), INT)

    def evaluate(self, ctx):
        right = self.r.evaluate(ctx)
        if right == 0:
            return self.l.evaluate(ctx)
        return self.l.evaluate(ctx) % self.r.evaluate(ctx)

    def can_eval(self):
        return self.l.can_eval() and self.r.can_eval()

    def within(self, variables):
        return self.l.within(variables) and self.r.within(variables)

    def __eq__(self, __o: object) -> bool:
        if type(__o) != Mod:
            return False
        return self.l == __o.l and self.r == __o.r

    def __hash__(self) -> int:
        return hash(self.l) + hash(self.r)


class Equal(Expr):
    """
    Represents (?= l r)
    """

    def __init__(self, l: Expr, r: Expr) -> None:
        super().__init__()
        if l == r:
            raise SymmetryException()
        if type(r) == Index or isinstance(l, Number):
            raise SymmetryException()
        if type(l) == Index and type(r) == Index:
            raise SymmetryException()
        if isinstance(l, Number) and isinstance(r, Number):
            raise SymmetryException()
        self.l = l
        self.r = r

    def __str__(self) -> str:
        return f"(?= {self.l} {self.r})"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def typ():
        return ((INT, INT), BOOL)

    def evaluate(self, ctx):
        return self.l.evaluate(ctx) == self.r.evaluate(ctx)

    def can_eval(self):
        return self.l.can_eval() and self.r.can_eval()

    def within(self, variables):
        return self.l.within(variables) and self.r.within(variables)

    def __eq__(self, __o: object) -> bool:
        if type(__o) != Equal:
            return False
        # equal commutes
        return self.l == __o.l and self.r == __o.r or self.r == __o.l and self.l == __o.r

    def __hash__(self) -> int:
        return hash(self.l) + hash(self.r)


class NotEqual(Expr):
    """
    Represents (not (?= l r))
    """

    def __init__(self, l: Expr, r: Expr) -> None:
        super().__init__()
        if l == r:
            raise SymmetryException()
        if type(r) == Index or isinstance(l, Number):
            raise SymmetryException()
        if type(l) == Index and type(r) == Index:
            raise SymmetryException()
        if isinstance(l, Number) and isinstance(r, Number):
            raise SymmetryException()
        self.l = l
        self.r = r

    def __str__(self) -> str:
        return f"(not (?= {self.l} {self.r}))"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def typ():
        return ((INT, INT), BOOL)

    def evaluate(self, ctx):
        return self.l.evaluate(ctx) != self.r.evaluate(ctx)

    def can_eval(self):
        return self.l.can_eval() and self.r.can_eval()

    def within(self, variables):
        return self.l.within(variables) and self.r.within(variables)

    def __eq__(self, __o: object) -> bool:
        if type(__o) != NotEqual:
            return False
        return self.l == __o.l and self.r == __o.r

    def __hash__(self) -> int:
        return hash(self.l) + hash(self.r)


class NodeEqual(Expr):
    """
    Represents (?= l r)
    """

    def __init__(self, l: Expr, r: Expr) -> None:
        super().__init__()
        if l == r:
            raise SymmetryException()
        if type(r) == Variable or isinstance(l, Number):
            raise SymmetryException()
        if type(l) == Variable and type(r) == Variable:
            raise SymmetryException()
        self.l = l
        self.r = r

    def __str__(self) -> str:
        return f"(?= {self.l} {self.r})"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def typ():
        return ((NODE, NODE), BOOL)

    def evaluate(self, ctx):
        return self.l.evaluate(ctx) == self.r.evaluate(ctx)

    def can_eval(self):
        return self.l.can_eval() and self.r.can_eval()

    def within(self, variables):
        return self.l.within(variables) and self.r.within(variables)

    def __eq__(self, __o: object) -> bool:
        if type(__o) != NodeEqual:
            return False
        # equal commutes
        return self.l == __o.l and self.r == __o.r or self.r == __o.l and self.l == __o.r

    def __hash__(self) -> int:
        return hash(self.l) + hash(self.r)


class NodeNotEqual(Expr):
    """
    Represents (not (?= l r))
    """

    def __init__(self, l: Expr, r: Expr) -> None:
        super().__init__()
        if l == r:
            raise SymmetryException()
        if type(r) == Variable or isinstance(l, Number):
            raise SymmetryException()
        if type(l) == Variable and type(r) == Variable:
            raise SymmetryException()
        self.l = l
        self.r = r

    def __str__(self) -> str:
        return f"(not (?= {self.l} {self.r}))"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def typ():
        return ((NODE, NODE), BOOL)

    def evaluate(self, ctx):
        return self.l.evaluate(ctx) != self.r.evaluate(ctx)

    def can_eval(self):
        return self.l.can_eval() and self.r.can_eval()

    def within(self, variables):
        return self.l.within(variables) and self.r.within(variables)

    def __eq__(self, __o: object) -> bool:
        if type(__o) != NodeNotEqual:
            return False
        return self.l == __o.l and self.r == __o.r

    def __hash__(self) -> int:
        return hash(self.l) + hash(self.r)


class Number(Expr):
    pass


class Two(Number):
    """
    Represents 2
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return f"2"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def typ():
        return ((), INT)

    def evaluate(self, ctx):
        return 2

    def can_eval(self):
        return True

    def within(self, variables):
        return True

    def __eq__(self, __o: object) -> bool:
        return type(__o) == Two

    def __hash__(self) -> int:
        return hash(2)


class One(Number):
    """
    Represents 1
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return f"1"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def typ():
        return ((), INT)

    def evaluate(self, ctx):
        return 1

    def can_eval(self):
        return True

    def within(self, variables):
        return True

    def __eq__(self, __o: object) -> bool:
        return type(__o) == One

    def __hash__(self) -> int:
        return hash(1)


class Zero(Number):
    """
    Represents 0
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return f"0"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def typ():
        return ((), INT)

    def evaluate(self, ctx):
        return 0

    def can_eval(self):
        return True

    def within(self, variables):
        return True

    def __eq__(self, __o: object) -> bool:
        return type(__o) == Zero

    def __hash__(self) -> int:
        return hash(0)


class NegativeOne(Number):
    """
    Represents -1
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return f"-1"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def typ():
        return ((), INT)

    def evaluate(self, ctx):
        return -1

    def can_eval(self):
        return True

    def within(self, variables):
        return True

    def __eq__(self, __o: object) -> bool:
        return type(__o) == NegativeOne

    def __hash__(self) -> int:
        return hash(-1)


class Variable(Expr):
    """
    Represents the variable __var_n__ = (Some node); Variables can only Node type
    """

    def __init__(self, v) -> None:
        super().__init__()
        self.v = v

    def __str__(self) -> str:
        return f"{self.v}"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def typ():
        return ((), NODE)

    def evaluate(self, ctx):
        return ctx[self.v]

    def can_eval(self):
        return False

    def within(self, variables):
        return self.v in variables

    def __eq__(self, __o: object) -> bool:
        if type(__o) != Variable:
            return False
        return self.v == __o.v

    def __hash__(self) -> int:
        return hash(self.v)


class Index(Expr):
    """
    Represents the index variable i = 0; Index Variables can only be int type
    """

    def __init__(self, v) -> None:
        super().__init__()
        self.v = v

    def __str__(self) -> str:
        return f"{self.v}"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def typ():
        return ((), INT)

    def evaluate(self, ctx):
        return ctx[self.v]

    def can_eval(self):
        return False

    def within(self, variables):
        return self.v in variables

    def __eq__(self, __o: object) -> bool:
        if type(__o) != Index:
            return False
        return self.v == __o.v

    def __hash__(self) -> int:
        return hash(self.v)


class Null(Expr):
    """
    Represents null
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return f"null"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def typ():
        return ((), NODE)

    def evaluate(self, ctx):
        return None

    def can_eval(self):
        return True

    def within(self, variables):
        return True

    def __eq__(self, __o: object) -> bool:
        return type(__o) == Null

    def __hash__(self) -> int:
        return hash(None)


EXPR_CLASSES = [
    NewNode,
    GetNext,
    GetVal,
    Add,
    Mult,
    Mod,
    Equal,
    NotEqual,
    NodeEqual,
    NodeNotEqual,
    Two,
    One,
    Zero,
    NegativeOne,
    Variable,
    Index,
    Null,
]

TERMINAL_CLASSES = [eclass for eclass in EXPR_CLASSES if len(eclass.typ()[
    0]) == 0]

PRODUCTION_CLASSES = [
    eclass for eclass in EXPR_CLASSES if eclass not in TERMINAL_CLASSES]


def gen_bottom_up():
    size_type_to_values = {}
    current_size_to_values = {eclass.typ()[1]: [] for eclass in EXPR_CLASSES}
    for depth in range(1, DEPTH_LIMIT + 1):
        for typ in TYPES:
            size_type_to_values[(depth, typ)] = []
    # depth 1 case
    for eclass in TERMINAL_CLASSES:
        ret_typ = eclass.typ()[1]
        if eclass == Variable:
            for v in VARIABLES:
                size_type_to_values[(1, ret_typ)].append(eclass(v))
                current_size_to_values[ret_typ].append(eclass(v))
        elif eclass == Index:
            for v in INDEX_VARIABLES:
                size_type_to_values[(1, ret_typ)].append(eclass(v))
                current_size_to_values[ret_typ].append(eclass(v))
        else:
            size_type_to_values[(1, ret_typ)].append(eclass())
            current_size_to_values[ret_typ].append(eclass())
    values_cache = set()
    for depth in range(2, DEPTH_LIMIT + 1):
        grow(depth, PRODUCTION_CLASSES,
             size_type_to_values, current_size_to_values, values_cache)
    return size_type_to_values


def grow(new_size, operators, size_type_to_values, current_size_to_values, values_cache):
    # Code from Jonathan Tran HWK 1
    for op in operators:
        argument_types = op.typ()[0]
        num_args = len(argument_types)
        ret_type = op.typ()[1]
        first_partitions = integer_partitions(
            new_size - 1, num_args, PARTITIONS_CACHE)
        partitions = list(
            filter(lambda p: len(p) == num_args, first_partitions))
        new_expr_lst = []
        for p in partitions:
            lsts = []
            for i, size in enumerate(p):
                arg_typ = argument_types[i]
                lsts.append(size_type_to_values[(size, arg_typ)])
            args_lsts = itertools.product(*lsts)
            new_exprs = []
            for args in args_lsts:
                try:
                    new_expr = op(*args)
                except:
                    continue
                if new_expr.can_eval():
                    if new_expr.evaluate({}) in values_cache:
                        continue
                    else:
                        values_cache.add(new_expr.evaluate({}))
                new_exprs.append(new_expr)
            new_expr_lst += new_exprs
        size_type_to_values[(new_size, ret_type)] += new_expr_lst
        current_size_to_values[ret_type] += new_expr_lst
    return


def integer_partitions(target_value, number_of_arguments, partitions_cache):
    """
    CODE FROM PROFESSOR ELLIS HWK 1

    Returns all ways of summing up to `target_value` by adding `number_of_arguments` nonnegative integers
    You may find this useful when implementing `bottom_up_generator`:

    Imagine that you are trying to enumerate all expressions of size 10, and you are considering using an operator with 3 arguments.
    So the total size has to be 10, which includes +1 from this operator, as well as 3 other terms from the 3 arguments, which together have to sum to 10.
    Therefore: 10 = 1 + size_of_first_argument + \
        size_of_second_argument + size_of_third_argument
    Also, every argument has to be of size at least one, because you can't have a syntax tree of size 0
    Therefore: 10 = 1 + (1 + size_of_first_argument_minus_one) + (
        1 + size_of_second_argument_minus_one) + (1 + size_of_third_argument_minus_one)
    So, by algebra:
         10 - 1 - 3 = size_of_first_argument_minus_one + size_of_second_argument_minus_one + size_of_third_argument_minus_one
    where: size_of_first_argument_minus_one >= 0
           size_of_second_argument_minus_one >= 0
           size_of_third_argument_minus_one >= 0
    Therefore: the set of allowed assignments to {size_of_first_argument_minus_one,size_of_second_argument_minus_one,size_of_third_argument_minus_one} is just the integer partitions of (10 - 1 - 3).
    """
    if (target_value, number_of_arguments) in partitions_cache:
        return partitions_cache[(target_value, number_of_arguments)]
    if target_value <= 0:
        partitions_cache[(target_value, number_of_arguments)] = []
        return []

    if number_of_arguments == 1:
        partitions_cache[(target_value, number_of_arguments)] = [
            [target_value]]
        return [[target_value]]

    result = [[x1] + x2s
              for x1 in range(1, target_value + 1)  # - number of arguments
              for x2s in integer_partitions(target_value - x1, number_of_arguments - 1, partitions_cache)]
    partitions_cache[(target_value, number_of_arguments)] = result
    return result


def python_list_to_nodes(lst: list):
    if lst == []:
        return Null()
    return NewNode(lst[0], python_list_to_nodes(lst[1:]))


if __name__ == "__main__":
    bottom_up_exprs = gen_bottom_up()
    for key in bottom_up_exprs:
        print(key)
        print(bottom_up_exprs[key])
