from bottom_up import SymmetryException, python_list_to_nodes, gen_bottom_up, Index, Variable, NodeNotEqual, Null, GetNext, BOOL, NODE, VARIABLES, INDEX_VARIABLES, INT

STMT_DEPTH_LIMIT = 10
HEAD_VARIABLE = "head"

NUM_HEAD_STMTS = 3
NUM_TAIL_STMTS = 3


class Stmt(object):
    """
    Abstract Statement Object
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return "(Abstract Stmt)"

    def __repr__(self) -> str:
        return str(self)

    def evaluate(self, ctx):
        raise RuntimeError("Cannot Evaluate Abstract Stmt.")


class AssignVal(Stmt):
    """
    Abstract Assignment Statement for Node
    """

    def __init__(self, v, expr) -> None:
        super().__init__()
        self.v = v
        self.expr = expr

    def __str__(self) -> str:
        return f"(set! {self.v}.val {self.expr})"

    def __repr__(self) -> str:
        return str(self)

    def evaluate(self, ctx):
        ctx[self.v].val = self.expr.evaluate(ctx)


class AssignNext(Stmt):
    """
    Abstract Assignment Statement for Node
    """

    def __init__(self, v, expr) -> None:
        super().__init__()
        self.v = v
        self.expr = expr

    def __str__(self) -> str:
        return f"(set! {self.v}.next {self.expr})"

    def __repr__(self) -> str:
        return str(self)

    def evaluate(self, ctx):
        ctx[self.v].next = self.expr.evaluate(ctx)


class Assign(Stmt):
    """
    Abstract Assignment Statement
    """

    def __init__(self, v, expr) -> None:
        super().__init__()
        if type(expr) == Variable:
            raise SymmetryException()
        if type(expr) == Index:
            raise SymmetryException()
        self.v = v
        self.expr = expr

    def __str__(self) -> str:
        return f"(set! {self.v} {self.expr})"

    def __repr__(self) -> str:
        return str(self)

    def evaluate(self, ctx):
        ctx[self.v] = self.expr.evaluate(ctx)


class If(Stmt):
    """
    Abstract If Statement
    """

    def __init__(self, b, assign) -> None:
        super().__init__()
        self.b = b
        self.assign = assign

    def __str__(self) -> str:
        return f"(when {self.b} {self.assign})"

    def __repr__(self) -> str:
        return str(self)

    def evaluate(self, ctx):
        if self.b.evaluate(ctx):
            self.assign.evaluate(ctx)


class While(Stmt):
    """
    Abstract While Statement
    """

    def __init__(self, b, body) -> None:
        super().__init__()
        self.b = b
        self.body = body

    def __str__(self) -> str:
        return f"(while {self.b} {self.body})"

    def __repr__(self) -> str:
        return str(self)

    def evaluate(self, ctx):
        while self.b.evaluate(ctx):
            for s in self.body:
                self.s.evaluate(ctx)


def gen_if(bottom_up_exprs, prev_defined_vars, all_vars, index_vars):
    bool_exprs = bottom_up_exprs[BOOL]
    if_stmts_lst = []
    for bexpr in bool_exprs:
        if not bexpr.within(prev_defined_vars):
            continue
        assignments_lst = gen_assign(
            bottom_up_exprs, prev_defined_vars, all_vars, index_vars)
        for a, new_defined_vars in assignments_lst:
            if_stmt = If(bexpr, a)
            if_stmts_lst.append((if_stmt, new_defined_vars))
    return if_stmts_lst


def gen_assign(bottom_up_exprs, prev_defined_vars, all_vars, index_vars):
    node_exprs = bottom_up_exprs[NODE]
    assignments_lst = []
    for lhs_v in all_vars:
        for rhs_node in node_exprs:
            if not rhs_node.within(prev_defined_vars):
                continue
            try:
                assignment = Assign(lhs_v, rhs_node)
            except:
                continue
            assignments_lst.append(
                (assignment, prev_defined_vars.union({lhs_v})))

    # for lhs_v in all_vars:
    #     for rhs_node in node_exprs:
    #         if not rhs_node.within(prev_defined_vars):
    #             continue
    #         assignment = AssignNext(lhs_v, rhs_node)
    #         assignments_lst.append(
    #             (assignment, prev_defined_vars.union({lhs_v})))

    int_exprs = bottom_up_exprs[INT]
    for lhs_v in index_vars:
        for rhs_int in int_exprs:
            if not rhs_int.within(prev_defined_vars):
                continue
            try:
                assignment = Assign(lhs_v, rhs_int)
            except:
                continue
            assignments_lst.append(
                (assignment, prev_defined_vars.union({lhs_v})))

    # for lhs_v in index_vars:
    #     for rhs_int in int_exprs:
    #         if not rhs_int.within(prev_defined_vars):
    #             continue
    #         assignment = AssignVal(lhs_v, rhs_int)
    #         assignments_lst.append(
    #             (assignment, prev_defined_vars.union({lhs_v})))

    return assignments_lst


def gen_int_assign(bottom_up_exprs, prev_defined_vars, all_vars, index_vars):
    int_exprs = bottom_up_exprs[INT]
    assignments_lst = []
    for lhs_v in index_vars:
        for rhs_int in int_exprs:
            if not rhs_int.within(prev_defined_vars):
                continue
            assignment = Assign(lhs_v, rhs_int)
            yield assignment
    raise RuntimeError("Ran Out of Int Assignment Statements")


def gen_node_assign(bottom_up_exprs, prev_defined_vars, all_vars, index_vars):
    pass


def gen_header(bottom_up_exprs, prev_defined_vars, all_vars, index_vars):
    pass


def gen_while_body(bottom_up_exprs, prev_defined_vars, all_vars, index_vars):
    pass


def gen_top_down(bottom_up_exprs, prev_defined_vars, all_vars, index_vars):
    """
    Yields programs

    We sh
    """
    pass


def check_input_outputs(inputs_outputs, bottom_up_exprs, prev_defined_vars, all_vars, index_vars):
    for program in gen_top_down(bottom_up_exprs, prev_defined_vars, all_vars, index_vars):
        correct = True
        for inpt, outpt in inputs_outputs:
            ctx = {HEAD_VARIABLE: inpt}
            # ASSUMES TERMINATION
            try:
                for line in program:
                    line.evaluate(ctx)
                result = ctx[HEAD_VARIABLE]
                if result != outpt:
                    correct = False
                    break
            except:
                correct = False
                break
        if correct:
            return program
    raise RuntimeError("No Synthesized Programs Satisfy Input Output Examples")


def convert_bottom_up(bottom_up_exprs):
    type_dict = {}
    for (_, typ) in bottom_up_exprs:
        type_dict[typ] = []
    for (size, typ) in bottom_up_exprs:
        type_dict[typ] += bottom_up_exprs[(size, typ)]
    return type_dict


def convert_inputs_outputs(inputs_outputs):
    result = []
    for inpt, outpt in inputs_outputs:
        result.append((python_list_to_nodes(inpt),
                       python_list_to_nodes(outpt)))
    return result


if __name__ == "__main__":
    bottom_up_exprs = convert_bottom_up(gen_bottom_up())
    # for k in bottom_up_exprs:
    #     for e in bottom_up_exprs[k]:
    #         print(e)
    all_vars = VARIABLES
    index_vars = INDEX_VARIABLES
    prev_defined_vars = {HEAD_VARIABLE, '__var_0__', '__var_2__', '__i_0__'}
    result = gen_if(bottom_up_exprs, prev_defined_vars, all_vars, index_vars)
    for r in result:
        print(r[0])
    print(len(result))
    assigns = gen_assign(
        bottom_up_exprs, prev_defined_vars, all_vars, index_vars)
    for a in assigns:
        print(a[0])
    print(len(assigns))
    inputs_outputs = [
        ([1, 2, 3, 4], [1, 2, 3, 4]),
        ([], []),
    ]
    inputs_outputs = convert_inputs_outputs(inputs_outputs)
    print(inputs_outputs)
    # base_program = [
    #     [],
    #     While(
    #         NodeNotEqual(Variable(HEAD_VARIABLE),  Null()),
    #         [Assign(
    #             HEAD_VARIABLE,
    #             GetNext(HEAD_VARIABLE)
    #         )]
    #     ),
    #     []
    # ]
    # gen_top_down()
