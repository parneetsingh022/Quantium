from functools import lru_cache
from typing import Tuple, Union

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantium.units.registry import UnitsRegistry

# --- Plan node types ------------------------------------------------
# ("name", <str>)
# ("pow", <plan>, <int>)
# ("mul", <plan>, <plan>)
# ("div", <plan>, <plan>)
Plan = Tuple[str, Union[str, int, "Plan"], Union[int, "Plan", None]]

# ---------------- Parser that builds a PLAN (no registry lookups!) ----------------
class _UnitExprParser:
    """
    Grammar (no numbers except signed integers after **):
      expr   := term (('*' | '/') term)*
      term   := factor ['**' signed_int]?
      factor := NAME | '(' expr ')'
      NAME   := [A-Za-z_][A-Za-z0-9_]*
      signed_int := ['+'|'-']? [0-9]+
    """
    def __init__(self, text: str):
        self.s = text
        self.n = len(text)
        self.i = 0

    def parse(self) -> Plan:
        plan = self._parse_expr()
        self._skip_ws()
        if self.i != self.n:
            raise ValueError(f"Unexpected trailing input at {self.i}: {self.s[self.i:self.i+10]!r}")
        return plan

    # expr := term (('*' | '/') term)*
    def _parse_expr(self) -> Plan:
        left = self._parse_term()
        while True:
            self._skip_ws()
            if self._peek('*') and not self._peek('**'):
                self._eat('*')
                right = self._parse_term()
                left = ("mul", left, right)
            elif self._peek('/'):
                self._eat('/')
                right = self._parse_term()
                left = ("div", left, right)
            else:
                break
        return left

    # term := factor ['**' signed_int]?
    def _parse_term(self) -> Plan:
        base = self._parse_factor()
        self._skip_ws()
        if self._peek('**'):
            self._eat('**')
            exp = self._parse_signed_int()
            base = ("pow", base, exp)
        return base

    # factor := NAME | '(' expr ')'
    def _parse_factor(self) -> Plan:
        self._skip_ws()
        if self._peek('('):
            self._eat('(')
            val = self._parse_expr()
            self._skip_ws()
            self._eat(')')
            return val
        name = self._parse_name()
        if not name:
            ch = self.s[self.i:self.i+1]
            raise ValueError(f"Expected unit name or '(' at {self.i}, got {ch!r}")
        return ("name", name, None)

    # ---- token helpers ----
    def _parse_name(self):
        self._skip_ws()
        i0 = self.i
        if i0 < self.n and (self.s[i0].isalpha() or self.s[i0] == '_'):
            self.i += 1
            while self.i < self.n and (self.s[self.i].isalnum() or self.s[self.i] == '_'):
                self.i += 1
            return self.s[i0:self.i]
        return None

    def _parse_signed_int(self) -> int:
        self._skip_ws()
        i0 = self.i
        if self.i < self.n and self.s[self.i] in '+-':
            self.i += 1
        i1 = self.i
        while self.i < self.n and self.s[self.i].isdigit():
            self.i += 1
        if i1 == self.i:
            raise ValueError(f"Expected integer exponent at {self.i}")
        return int(self.s[i0:self.i])

    def _skip_ws(self):
        s, n, i = self.s, self.n, self.i
        while i < n and s[i].isspace():
            i += 1
        self.i = i

    def _peek(self, tok: str) -> bool:
        self._skip_ws()
        if tok == '**':
            return self.s[self.i:self.i+2] == '**'
        return self.i < self.n and self.s[self.i] == tok

    def _eat(self, tok: str):
        if not self._peek(tok):
            got = self.s[self.i:self.i+len(tok)]
            raise ValueError(f"Expected {tok!r} at {self.i}, got {got!r}")
        self.i += len(tok)

# ---------------- Evaluation of a plan against a given registry ----------------
def _eval_plan(plan: Plan, reg: "UnitsRegistry"):
    kind = plan[0]
    if kind == "name":
        name = plan[1]
        try:
            return reg.get(name)  # late binding to the provided registry
        except Exception as e:
            raise ValueError(f"Unknown unit '{name}': {e}") from None
    elif kind == "pow":
        base = _eval_plan(plan[1], reg)
        exp = plan[2]  # int
        return base ** exp
    elif kind == "mul":
        left = _eval_plan(plan[1], reg)
        right = _eval_plan(plan[2], reg)
        return left * right
    elif kind == "div":
        left = _eval_plan(plan[1], reg)
        right = _eval_plan(plan[2], reg)
        return left / right
    else:
        raise RuntimeError(f"Invalid plan node: {plan!r}")

# ---------------- Public API with caching-safe compilation ----------------
# Cache the *compiled plan* only. Safe across registries because there's no bound objects inside.
@lru_cache(maxsize=4096)
def _compile_unit_expr(expr: str) -> Plan:
    # cheap prefilter to reject disallowed characters early.
    # Note: '+' is allowed only as a sign in exponents; this prefilter doesn't distinguish positions
    # but keeps things simple—parser will give precise errors if needed.
    disallowed = set('~!@#$%^&|+=,:;?<>\'\"`\\[]{}')
    if any(c in disallowed for c in expr):
        raise ValueError("Only *, /, **, parentheses, unit names, and signed integer exponents are allowed.")
    return _UnitExprParser(expr).parse()

def extract_unit_expr(expr: str, reg: "UnitsRegistry"):
    """
    Fast custom parser for unit expressions like 'kg*m/(nF**2 * s**2)'.

    Caching-safety:
      * We cache a compiled syntax plan keyed by `expr` only (no registry state).
      * Evaluation binds names to units from the *provided* `reg` at call time.

    Allowed syntax:
      * Operators: '*', '/', '**' with integer (optionally signed) exponents.
      * Parentheses and unit names [A-Za-z_][A-Za-z0-9_]*
      * Anything else raises ValueError.

    Returns:
      A unit object from the given registry (whatever `reg.get()` returns for names).
    """
    plan = _compile_unit_expr(expr)
    return _eval_plan(plan, reg)
