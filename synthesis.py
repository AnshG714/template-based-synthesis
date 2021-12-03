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

DEPTH_LIMIT = 5
VARIABLES_LIMIT = 3
VARIABLES = {f"__{VARIABLE_NAME}_{i}__" for i in range(VARIABLES_LIMIT)}

PARTITIONS_CACHE = {}


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


class NewNode(Expr):
    """
    Represents New Node
    e.g. new Node();
    """

    def __init__(self, val, nxt) -> None:
        super().__init__()
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

    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)

    def __hash__(self) -> int:
        return hash(self.next) + hash(self.val)


class GetNext(Expr):
    """
    Represents node.next
    """

    def __init__(self, node) -> None:
        super().__init__()
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


class GetVal(Expr):
    """
    Represents node.val
    """

    def __init__(self, node) -> None:
        super().__init__()
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


class Add(Expr):
    """
    Represents (+ l r)
    """

    def __init__(self, l: Expr, r: Expr) -> None:
        super().__init__()
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


class Mult(Expr):
    """
    Represents (* l r)
    """

    def __init__(self, l: Expr, r: Expr) -> None:
        super().__init__()
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


class Mod(Expr):
    """
    Represents (modulo l r)
    """

    def __init__(self, l: Expr, r: Expr) -> None:
        super().__init__()
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


class Equal(Expr):
    """
    Represents (?= l r)
    """

    def __init__(self, l: Expr, r: Expr) -> None:
        super().__init__()
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


class NotEqual(Expr):
    """
    Represents (not (?= l r))
    """

    def __init__(self, l: Expr, r: Expr) -> None:
        super().__init__()
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


class NodeEqual(Expr):
    """
    Represents (?= l r)
    """

    def __init__(self, l: Expr, r: Expr) -> None:
        super().__init__()
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


class NodeNotEqual(Expr):
    """
    Represents (not (?= l r))
    """

    def __init__(self, l: Expr, r: Expr) -> None:
        super().__init__()
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


class One(Expr):
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


class NegativeOne(Expr):
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
    One,
    NegativeOne,
    Variable,
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
                new_expr = op(*args)
                if new_expr.can_eval():
                    if new_expr.evaluate({}) in values_cache:
                        continue
                    else:
                        values_cache.add(new_expr.evaluate({}))
                # Simple NULL Check: Not fully accurate
                if type(new_expr) in [GetNext, GetVal]:
                    if type(new_expr.node) == Null:
                        continue
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

    lst = python_list_to_nodes([1, 2, 3, 4, 5])
    print(lst)
