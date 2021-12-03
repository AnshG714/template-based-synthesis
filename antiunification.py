from _typeshed import NoneType


class AntiUnify(object):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return "Abstract AntiUnify Object"

    def __repr__(self) -> str:
        return "Abstract AntiUnify Object"

    def get_subtrees(self) -> list:
        raise RuntimeError(
            "Cannot Get Subtrees from AntiUnify Abstract Object")

    def set_subtrees(self, subtrees) -> NoneType:
        raise RuntimeError(
            "Cannot Assign Subtrees to AntiUnify Abstract Object")


class NotUnified(AntiUnify):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return "(Not Unified)"

    def __repr__(self) -> str:
        return str(self)

    def get_subtrees(self) -> list:
        return []

    def set_subtrees(self, subtrees) -> NoneType:
        raise RuntimeError("Cannot Assign Subtrees to Not Unified")


class While(AntiUnify):
    # While(b) do C
    pass


class IfThenElse(AntiUnify):
    # if b then C1 else C2
    pass


class Nop(AntiUnify):
    # PASS
    pass


class Assign(AntiUnify):
    # X = e
    pass


class FieldAssign(AntiUnify):
    # obj.field = e
    pass


class FieldRef(AntiUnify):
    # obj.e
    pass


def antiunify(term1, term2):
    assert isinstance(term1, AntiUnify)
    assert isinstance(term2, AntiUnify)

    if type(term1) == type(term2):
        subtrees1 = term1.get_subtrees()
        subtrees2 = term2.get_subtrees()
        if subtrees1 == [] and subtrees2 == []:
            return term1
        if len(subtrees1) != len(subtrees2):
            return NotUnified()

        unified_trees = []
        for subtree1, subtree2 in zip(subtrees1, subtrees2):
            unified_tree = antiunify(subtree1, subtree2)
            unified_trees.append(unified_tree)
        new_obj = term1.clone()
        new_obj.set_subtrees(unified_trees)
        return new_obj

    return NotUnified()


def main():
    pass


if __name__ == "__main__":
    main()
