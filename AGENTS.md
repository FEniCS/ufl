# UFL

UFL (Unified Form Language) is a domain-specific language, embedded in Python, for declaring finite
element variational forms and the function spaces they live on. It is part of the FEniCS Project, and
it is the shared symbolic layer for FFCx, TSFC (used by Firedrake), and DOLFINx. None of those
re-implement form algebra or differentiation; they all lower whatever `compute_form_data` produces.

## Project Architecture

* **Two hierarchies:** `Expr` (`ufl/core/expr.py`) is the primal language of scalars, tensors,
  coefficients, and arguments. `BaseForm` (`ufl/form.py`) is the dual language of forms, actions,
  adjoints, and cofunctions. Neither subclasses the other, so code that dispatches on `Expr` misses
  every dual node.
* **One pipeline:** `compute_form_data` (`ufl/algorithms/compute_form_data.py`) is the lowering that
  every form compiler calls: derivative expansion, pullback and geometry lowering, default
  restrictions, then integral scaling. It converges on a small node vocabulary, so a form compiler
  does not have to understand all of UFL.
* **Dispatch tables, not functions:** each algorithm is a `DAGTraverser` subclass whose rules are
  `singledispatchmethod` registrations. `singledispatchmethod` does not inherit across subclasses, so
  every subclass re-declares `process` and chains to `super().process(o)`. A parent's rule is
  reachable only through that chain, never through plain MRO.
* **One generic transform:** `map_integrands` is the recursive form transform that differentiation,
  `replace`, splitting, and action/adjoint construction are all built on.
* **Zero carries shape:** `ZeroBaseForm` records the arguments that the vanished quantity depended on.
  A bare `0` or an empty `Form([])` does not.

## Core Working Rules

* **Fix The Rule, Not The Call Site:** Add the missing rule to the class that dispatches it. Do not
  wrap the call site in a `try`/`except`.
* **Extend An Existing Dispatch Table:** Register the new case on the traverser that already owns that
  node type, instead of special-casing one algorithm.
* **Dispatch On Type Cannot See Context:** A rule sees a node's type, never which recursive call
  reached it. When one type needs different treatment in different subtrees, change which traverser
  handles which subtree. Do not add more type registrations.
* **Borrow A Design From Downstream:** UFL sits above several form compilers that already lower its
  output. Before inventing a simplification or cancellation pass, grep the downstream compiler for a
  pass that already names the pattern, and port it.
* **Match The Nearest Structurally Similar Rule:** Mirror an existing rule for a type that plays the
  same abstract role, even from an unrelated hierarchy. Copy its boilerplate too — the `super()`
  fallback and the narrowed-signature suppression exist for concrete reasons.
* **Do Not Trust Memorized API Shapes:** Read the current class definition from the installed source
  before calling it. The dual types are recent, and trained knowledge predates them.
* **Preserve Coding Style:** Match the naming and the patterns of the module you are editing. Keep
  edits minimal and local to the requested change.
* **Document The Present, Not The Past:** Do not describe a removed or rejected approach in a comment
  or a docstring. Document only what the current code does.

## Coding Style And Conventions

* **Type Hints And Docstrings:** Add type hints to new signatures. Write Google-style
  `Args:`/`Returns:` docstrings, not numpydoc.
* **Suppress `mypy` On Narrowed Handlers:** A `process.register` handler with a narrowed signature
  fails `mypy` against the base `process`, even when the runtime dispatch is correct. Add
  `# type: ignore`, as the surrounding handlers do. Do not widen the type hints.
* **Line Length Is 100:** The `[tool.ruff]` table in `pyproject.toml` is authoritative.

## Testing Requirements

* Add a test that demonstrates the new feature or bug fix, in the existing test file for that module.
  Do not create a new file for a single fix.
* **Test mathematical correctness.** Neither "no exception was raised" nor agreement between two
  expressions proves that a result is right. Two constructions can match structurally and still share
  the same wrong rule.
* Evaluate the result numerically, or assemble it in a downstream consumer, and compare against a
  hand-computed or finite-difference value. Use a Taylor test for anything that claims to be a
  derivative.
* Reserve structural and `.signature()` comparisons for checking that two constructions agree. Never
  use one as a stand-in for checking that either construction is correct.
* Build elements and domains with the helpers in `test/utils.py`. The symbolic layer needs no
  form-compiler stack.

## Pull Request Expectations

* All changes land through GitHub pull requests. Keep diffs focused.
* Only maintainers push to `origin` (`FEniCS/ufl`). Push to your fork, then open the pull request from
  there. A permission error pushing to `origin` is expected, not a misconfiguration.
  ```bash
  git checkout -b <branch> main
  git push -u fork <branch>
  gh pr create --repo FEniCS/ufl --base main --head <fork-org>:<branch>
  ```
* Before requesting review, run every stage of `lint.yml`. A passing `ruff check .` alone is not
  enough:
  ```bash
  ruff check . && ruff format --check . && mypy -p ufl && mypy test/ && (cd test && mypy ../demo/)
  ```

## Development Toolchain

### Environment Setup

* **Paired branches:** UFL's `main` and `release` branches pair with the matching branches of FFCx,
  TSFC/Firedrake, and DOLFINx. When you develop a fix across both, install UFL editable into the
  consumer's virtual environment, and check the consumer's own UFL branch before you assume that a
  failure is local.
* **Reproduction scripts:** `test/utils.py` provides lightweight element and domain constructors.
  Reach for these first.

### Testing

* Run the full suite with `python -m pytest test/`. It takes seconds, and it needs no MPI or parallel
  infrastructure.
* Reproduce standalone first: a function space on one or two elements, plus the one call in question,
  under twenty lines.

### Debugging

* **Print the expression before you trust its accessors.** `repr()` shows an un-expanded node that
  `.arguments()` alone can hide.
* **Compare `.signature()`, not `==`.** Two independently built forms carry differently numbered dummy
  indices, which `==` reports as a difference and `.signature()` does not.
* **`Rule not set for <type>`** means that no handler for that type is reachable from the class that
  actually dispatches. Read `type(traverser)` at the failure point, not the traceback's outer frames.
* **Read a `DAGTraverser` traceback from the bottom.** The traverser memoizes by catching `KeyError`,
  so every call adds another "During handling of the above exception". Skip to the last exception, and
  read `self = <ClassName ...>` at each frame to see which ruleset was active.
* **Diff the output, not the pass or fail.** When a narrow code path disagrees with the full pipeline,
  run both on the same input and compare the two expressions structurally.

## Grammar & Style Rules for Technical Prose

Write as an expert technical writer addressing a peer (a mathematician or software engineer).
Use ASD-STE100. Write clear, complete sentences rather than grammatically convoluted shortcuts.
All comments, docstrings, and documentation must adhere to the following standards:

* **Active Verbs Over Noun-Stacking:** Rephrase to avoid stacking words that double as nouns, verbs, or adjectives.
   - **WRONG:** `# Argument replacement transform post-image shape mismatch.`
   - **RIGHT:** `# The transform replaced the Arguments, so the shapes no longer match.`

* **Explicit Relative Pronouns:** Never drop pronouns like `that`, `which`, or `where` to condense sentences.
   - **WRONG:** `# Returns the integrals a mixed-space splitter produced.`
   - **RIGHT:** `# Returns the integrals that a mixed-space splitter produced.`

* **Subject-Verb Alignment:** Ensure that introductory prepositional phrases modify the actual grammatical
subject of the main clause. Avoid dangling modifiers.
   - **WRONG:** `# Using a narrowed dispatcher, the gradients stay un-normalized.`
   - **RIGHT:** `# A narrowed dispatcher leaves the gradients un-normalized.`

* **Describe The Code That Is There:** Never document a removed approach, and never argue against a
branch the code does not take. "Used to", "previously", "no longer", and "instead of" give this away.
   - **WRONG:** `# This no longer returns the pre-image Arguments, which lost the numbering.`
   - **RIGHT:** `# The post-image Arguments, recorded when the transform rebuilt the form.`

## Anti-Patterns

Each pattern below is a WRONG/RIGHT pair to read.

### Returning A Bare Zero For A Form That Cancelled

WRONG — Returning a bare Python `0`, or an empty `Form([])`, throws away the function spaces that the
zero result lived on. The loss stays invisible until something downstream calls `.arguments()`:

```python
# Anti-pattern: the caller cannot recover which Arguments this was supposed to have
def compute_something(form):
    result = ...  # every integral cancelled
    if not result:
        return Form([])
    return result
```

RIGHT — Build a `ZeroBaseForm` that carries the arguments the vanished result would have had.
Downstream code then keeps working exactly as it would for a nonzero result:

```python
def compute_something(form):
    result = ...
    if not result:
        return ZeroBaseForm(form.arguments())
    return result
```

### Assuming That A Transform Preserves Its Arguments

WRONG — Reusing the pre-transform form's `.arguments()` is correct only for a transform that *adds*
arguments, such as differentiation. It is wrong for a transform that *replaces* them, such as a
mixed-space block splitter, whose arguments live on the collapsed subspace:

```python
# Anti-pattern: silently wrong for any transform that replaces Arguments
if not nonzero_integrals:
    return ZeroBaseForm(form.arguments())  # `form` is the PRE-transform Form
```

RIGHT — A generic utility cannot know whether a caller-supplied transform adds or replaces arguments,
and it must not guess. The caller knows its own semantics, so let it rebuild the arguments by
re-applying its own substitution rule:

```python
f = map_integrand_dags(splitter, form)
if expand_derivatives(f).empty():
    f = ZeroBaseForm(tuple(map(splitter._subspace_argument, form.arguments())))
```
