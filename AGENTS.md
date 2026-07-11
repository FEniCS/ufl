# UFL

UFL (Unified Form Language) is a domain-specific language, embedded in Python, for declaring finite
element variational forms and the function spaces they live on. It is part of the FEniCS Project and is
the shared symbolic layer for FFCx, TSFC (used by Firedrake), and DOLFINx — none of which re-implement
form algebra or differentiation; they all lower whatever `compute_form_data` produces.

## Project Architecture

* **`Expr` (primal) and `BaseForm` (dual) are separate hierarchies.** `Expr` (`ufl/core/expr.py`):
  scalars, tensors, `Coefficient`, `Argument`. `BaseForm` (`ufl/form.py`): `Form`, `FormSum`, `Matrix`,
  `Cofunction`, `Coargument`, `Action`, `Adjoint`, `ZeroBaseForm`. Neither subclasses the other — code
  dispatching on `Expr` (`isinstance`, or a `singledispatchmethod` registered on `Expr`) silently misses
  every `BaseForm` node.
* **`compute_form_data` is the one pipeline every form compiler uses** (`ufl/algorithms/compute_form_data.py`):
  derivative expansion (`apply_derivatives`) → pullback/geometry lowering → default restrictions →
  integral scaling, converging on a small node vocabulary so FFCx/TSFC don't need to understand all of
  UFL.
* **Differentiation is a tree of `DAGTraverser` subclasses, each a `singledispatchmethod` dispatch
  table**, not one function: `GenericDerivativeRuleset` → `GateauxDerivativeRuleset` →
  `BaseFormOperatorDerivativeRuleset`, plus `DerivativeRuleDispatcher` at the top
  (`apply_derivatives.py`). Because `singledispatchmethod` does not inherit across subclasses
  (bpo-36457), every subclass re-declares its own `process` and falls back to `super().process(o)` — a
  parent's rule is reachable only through that explicit chain, never plain MRO.
* **`map_integrands` is the generic recursive form transform** almost everything else (differentiation,
  `replace`, splitting, action/adjoint construction) is built on. Its `Form` and `FormSum` branches are
  *not* symmetric — see Anti-Patterns before assuming a fix to one applies to the other.
* **`ZeroBaseForm` is the dual zero, and it carries shape**: which `Argument`s the now-vanished quantity
  depended on. A bare `0` or `Form([])` does not carry this — the loss is invisible until something
  downstream calls `.arguments()`.

## Core Working Rules

* **Fix the rule gap, not the test.** A crash tracing to a missing `process.register` case needs the
  rule added on the right class, not a try/except at the call site.
* **Extend an existing dispatch table** (`GenericDerivativeRuleset`, `map_integrands`, `extract_type`)
  rather than special-casing one algorithm.
* **Match the surrounding module's style**: Google-style `Args:`/`Returns:` docstrings, not numpydoc;
  keep diffs minimal and local.
* **The `Expr`/`BaseForm` split, and `Cofunction`/`Coargument`/`Interpolate`/`ZeroBaseForm`, are recent.**
  Trained knowledge describing "a form is a sum of integrals over `Expr`s" predates them and will be
  wrong about duals, `Action`, `Adjoint`. Read the actual class before assuming a method exists.
* **Document the present, not the past.** Don't explain what a removed workaround used to do; say what
  is true now and why.

## Coding Style And Conventions

* **Type hints + Google-style docstrings** on new code, matching `ufl/algorithms/`.
* **`# type: ignore` on narrowed `process.register` handlers.** A handler typed
  `(self, o: SomeType) -> BaseForm` fails `mypy` against the base `process(self, o: Expr) -> Expr` even
  when the runtime dispatch is correct — match the existing `Matrix`/`Interpolate`/`ExternalOperator`
  convention rather than widening the type hints.
* **`ruff` line length is 100**, not flake8's 79 default — `pyproject.toml`'s `[tool.ruff]` is
  authoritative.

## Testing Requirements

* Every PR needs a test demonstrating the fix or feature.
* **Test mathematical correctness, not just that it runs or looks structurally right.** Neither "no
  exception was raised" nor `.signature()`/`==` agreement between two expressions proves the result is
  correct — two independently-built expressions can match structurally while sharing the same wrong
  derivative or simplification rule, and a simplification can produce something smaller without being
  equivalent. Verify the actual mathematical claim: evaluate numerically (or `assemble` it in a
  downstream consumer) and compare against a hand-computed or finite-difference value, or use a Taylor
  test for anything claiming to be a derivative. Reserve structural/`.signature()` comparisons for
  checking that two *constructions* agree, never as a stand-in for checking that either one is right.
* Extend the existing file that already covers the feature (e.g. `test/test_duals.py` for
  `BaseForm`/`Action`/`Adjoint`/`ZeroBaseForm`). Don't create a new file for a single fix.
* Build elements/domains via `test/utils.py`'s `LagrangeElement`/`FiniteElement` helpers — no
  form-compiler stack needed to exercise the symbolic layer.

## Pull Request Expectations

* Only maintainers push to `origin` (`FEniCS/ufl`). Push to `fork`, then PR from `<fork-org>:<branch>`
  to `FEniCS/ufl:main`:
  ```bash
  git checkout -b <branch> main   # branch off an up-to-date origin/main
  git push -u fork <branch>
  gh pr create --repo FEniCS/ufl --base main --head <fork-org>:<branch> --title ... --body ...
  ```
  A permission error pushing to `origin` is expected, not a sign of misconfiguration.
* Run all three `lint.yml` stages before pushing — `ruff check .` passing alone is not enough:
  ```bash
  ruff check . && ruff format --check . && mypy -p ufl && mypy test/ && (cd test && mypy ../demo/)
  ```

## Development Toolchain

### Environment Setup

* UFL's `main`/`release` pair with FFCx/TSFC(Firedrake)/DOLFINx's `main`/`release`. When co-developing a
  fix across both, `pip install -e .` UFL into the consumer's venv and check the consumer's own UFL
  branch before assuming a failure is local to the package you're editing.
* `test/utils.py` provides lightweight element/domain constructors (`LagrangeElement`, `FiniteElement`,
  `MixedElement`) — reach for these first in a reproduction script.

### Testing

* `pytest test/` runs the full suite in seconds, no MPI/parallel infrastructure.
* Reproduce standalone first: a `FunctionSpace` on one or two elements plus the one call in question,
  under twenty lines — the symbolic layer has no external dependencies to configure.

### Debugging

* **Print a `BaseForm`/`Expr` before trusting `.arguments()` or `.empty()`.** A stray, un-expanded
  `CoefficientDerivative` is immediately visible in `repr()`, whereas `.arguments()` alone can look
  plausible while being wrong.
* **Compare `.signature()`, not `==`, for independently-built expressions.** Two structurally identical
  forms built via separate calls can carry differently-numbered dummy summation indices; `==` sees them
  as different, `.signature()` does not.
* **`AssertionError: Rule not set for <type>`** means no `process.register` handler for that type is
  reachable from the class actually dispatching (check `type(traverser)` at the failure point, not the
  traceback's outer frames) — either directly or via its `super().process(o)` fallback chain.
* **A `DAGTraverser` traceback is mostly cache-miss noise.** `DAGTraverser.__call__` memoizes by raising
  and catching `KeyError` on every call, so a real failure shows up as a long stack of "During handling
  of the above exception, another exception occurred" — skip to the *last* exception, and read
  `self = <ClassName ...>` at each frame to see exactly which dispatcher/ruleset was active, rather than
  guessing from the top-level error alone.
* **When a narrow code path disagrees with the full pipeline, run both on the same input and diff the
  output structurally, not just pass/fail.** `apply_derivatives`/`expand_derivatives` on the identical
  expression shows what the "correct" shape looks like (e.g. `Indexed(Grad(w), ...)` vs. an
  un-normalized `Grad(Indexed(w, ...))`) — this turns "why does this crash" into "which node differs and
  why."

## Pattern Matching For Planning And Debugging

* **Borrow a design from one layer down the stack before inventing one.** UFL sits below nothing and
  above several form compilers (FFCx, TSFC/GEM) that lower whatever `compute_form_data` produces. A
  simplification problem at the UFL layer has often already been solved, in a structurally analogous
  form, one layer further down — GEM already cancels Kronecker-delta-producing index contractions in
  its own optimization passes, and `cancel_jacobian_products.py`'s `JacobianCanceller`/
  `IdentityEliminator` pair is a deliberate port of that GEM pass, run earlier, before pullback lowering
  destroys the structure GEM would otherwise have to rediscover. Grep the downstream compiler for a pass
  that already names the pattern before designing a new algorithm; porting a design that has already
  shipped and been tested is strictly better than re-deriving one, and its tests hand you the edge cases
  for free. Treat this as good practice, not a shortcut to apologize for.
* **Inside UFL, match new rules to the nearest *structurally* similar existing one, not the nearest
  class in the hierarchy.** A dispatch rule for a new type should mirror an existing rule for a type
  playing the same abstract role (terminal-like dual object, zero-producing simplification,
  index-contraction rewrite), even in an unrelated class hierarchy — e.g. `Matrix`/`Cofunction`/
  `Coargument` are all "dual objects independent of a plain `Coefficient`." Copy the boilerplate too
  (`super().process(o)` fallback redeclaration, `# type: ignore` on a narrowed signature); it exists for
  concrete reasons (Python class-body name resolution, `mypy` callable variance), and skipping it
  reproduces failures someone already solved.
* **Classify which structural category a fix belongs to before copying it to a sibling code path.** Two
  branches that look parallel are not automatically the same problem — one may only ever *add*
  information, where its sibling *replaces* it (the `map_integrands` `Form`-vs-`FormSum` case in
  Anti-Patterns). State in one sentence which category applies before generalizing; a full local
  test-suite pass is not proof the generalization is safe, and it may take a downstream consumer's tests
  to reveal the mismatch.
* **A fix that trades one failing test for another is a context signal, not a reason to add more special
  cases.** Narrowing a dispatcher to "only handle X, leave everything else untouched" assumes node
  *type* is enough to decide — but the same node type can need different treatment depending on which
  recursive call reached it (inside the region being narrowed for, vs. outside it), and a type-keyed
  registry alone cannot express that. When a targeted fix (e.g. registering one more type) fixes the
  case you're chasing but breaks an unrelated, previously-passing test, look for a way to change *which
  traverser handles which subtree* instead of adding more type registrations. This is what fixing
  `CoefficientDerivativeRuleDispatcher` required: `Grad` needed full normalization when reached from
  *inside* a `CoefficientDerivative`'s own content, but no treatment at all when reached from *outside*
  one — the fix was recursing into that content with a separate, full `DerivativeRuleDispatcher`
  instance, not registering `Grad` on the narrowed dispatcher itself.
* **When a generalization regresses something only a downstream consumer's tests catch, isolate with a
  one-delta script before re-reading the whole diff.** Build the smallest example that drives exactly the
  two competing code paths (e.g. a node's pre-transform vs. post-transform arguments, or the same
  expression through both the narrow and the full pipeline) through the algorithm directly, and `repr()`
  the result at each step — this localizes which rule produced the wrong answer far faster than
  reasoning about the full pipeline in the abstract.

## Anti-Patterns

### Using A Bare `0` Or `Form([])` For A Form That Turned Out To Be Zero

WRONG — Once an algorithm establishes that a form is identically zero, returning a bare Python `0` (or
an empty, argument-less `Form([])`) throws away which function spaces the zero result lived on:

```python
# Anti-pattern: the caller cannot recover which Arguments this was supposed to have
def compute_something(form):
    result = ...  # turns out every integral cancelled
    if not result:
        return Form([])
    return result
```

RIGHT — Construct a `ZeroBaseForm` carrying the arguments the (now-vanished) result would have had.
Downstream code that calls `.arguments()`, assembles into a specific block, or feeds the result into
further form algebra then keeps working exactly as it would for a nonzero result:

```python
def compute_something(form):
    result = ...
    if not result:
        return ZeroBaseForm(form.arguments())
    return result
```

This is precisely the shape of the bug that produced `AssertionError: Rule not set for ZeroBaseForm` in
`expand_derivatives`: a differentiation rule for `Matrix`/`Cofunction`/`Coargument` already built a
correctly-shaped `ZeroBaseForm`, but differentiating *that* `ZeroBaseForm` a second time (e.g. a Hessian)
hit a rule set with no registered handler for `ZeroBaseForm` itself — the fix was to add the missing
sibling rule, not to special-case the crash.

### Assuming A Transform's Post-Image Arguments Equal Its Pre-Image Arguments

WRONG — When a fully-cancelled `Form`'s zero result needs its arguments preserved, reaching for the
*original*, pre-transform form's `.arguments()` looks like the obvious fix, and is correct for
transforms that only ever *add* information (differentiation appends one new direction `Argument` on
top of the existing ones — the original arguments are still exactly right). It is wrong for transforms
that *replace* arguments, such as splitting a mixed-space form into a sub-block: the pre-transform form's
arguments are on the original, un-split function space, not the collapsed subspace the caller actually
wants.

```python
# Anti-pattern: correct for a pure differentiation pass, silently wrong for
# anything (like a mixed-space block splitter) that replaces Arguments
# rather than only adding new ones
if not nonzero_integrals:
    return ZeroBaseForm(form.arguments())  # `form` is the PRE-transform Form
```

RIGHT — A generic utility like `map_integrands` cannot know, for an arbitrary caller-supplied
transform, whether that transform only adds arguments or also replaces them; it should not guess. Let
the specific caller — which does know its own transform's semantics — reconstruct the correct arguments
itself, e.g. by re-applying its own per-`Argument` substitution rule to `form.arguments()`:

```python
# Correct: the caller (e.g. a mixed-space block splitter) knows how it
# maps an Argument onto the collapsed subspace, and rebuilds accordingly
f = map_integrand_dags(splitter, form)
if expand_derivatives(f).empty():
    f = ZeroBaseForm(tuple(map(splitter._subspace_argument, form.arguments())))
```

The `FormSum` branch of `map_integrands` gets to take the shortcut the `Form` branch cannot: when every
component of a `FormSum` vanishes, each component was *already* passed through the caller's transform
(`map_integrands` recurses into `FormSum.components()` before checking for cancellation), so
`mapped_components[0].arguments()` is the correctly-transformed shape, not a pre-image guess. Do not
generalize that shortcut to the plain `Form` branch, where no such already-mapped object exists.

### Narrowing A Dispatcher's Node-Type Coverage Without Narrowing Its Point Of Use

WRONG — Registering the abstract `Derivative` type to leave nodes untouched correctly protects foreign,
unknown subtypes (e.g. a third-party `TimeDerivative`) from a dispatcher that only knows how to expand
`CoefficientDerivative`. But `Grad`, `ReferenceGrad`, and friends are *also* `Derivative` subtypes with
no more specific registration here — so they get left untouched too, including *inside* a
`CoefficientDerivative`'s own content, where `GateauxDerivativeRuleset` requires a spatial `Grad`
already pushed down to a terminal:

```python
# Anti-pattern: protects foreign Derivative subtypes, but also leaves
# Grad/ReferenceGrad/etc. un-normalized wherever they occur, including
# inside a CoefficientDerivative that is about to be Gateaux-differentiated
@process.register(Derivative)
def _(self, o):
    return self.reuse_if_untouched(o)

@process.register(CoefficientDerivative)
@DAGTraverser.postorder_only_children([0])
def _(self, o, f):
    # `f` was recursed into via `self`, so any Grad inside it already
    # went through the reuse_if_untouched rule above, un-normalized
    ...
```

RIGHT — Recurse into the `CoefficientDerivative`'s own content with a separate, full traverser, so
content about to be differentiated is normalized exactly as the full pipeline would, while node types
encountered *outside* any `CoefficientDerivative` still get the narrow, foreign-safe treatment:

```python
@process.register(CoefficientDerivative)
def _(self, o):
    expr, w, v, cd = o.ufl_operands
    full_dispatcher = self._dag_traverser_cache.setdefault(
        (DerivativeRuleDispatcher,), DerivativeRuleDispatcher()
    )
    f = full_dispatcher(expr)
    ...
```

This is the shape of the bug behind `GateauxDerivativeRuleset.Grad` raising `"Expecting gradient of a
FormArgument"` on `Grad(Indexed(Coefficient, ...))`: `replace()`'s narrowed `expand_coefficient_derivatives`
needs `Grad` fully normalized *inside* a `CoefficientDerivative`'s content, but left alone *outside* one
— a single `Derivative → reuse_if_untouched` registration cannot express both, since dispatch only sees
a node's type, never which recursive call reached it.
