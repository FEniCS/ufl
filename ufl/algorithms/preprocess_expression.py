
# FIXME: Remove this and just reuse preprocess_form with expr*dirac_delta_measure?
def preprocess_expression(expr, object_names=None, common_cell=None, element_mapping=None):
    """
    Preprocess raw input expression to obtain expression metadata,
    including a modified (preprocessed) expression more easily
    manipulated by expression compilers. The original expression
    is left untouched. Currently, the following transformations
    are made to the preprocessed form:

      expand_compounds    (side effect of calling expand_derivatives)
      expand_derivatives
      renumber arguments and coefficients and apply evt. element mapping
    """
    tic = Timer('preprocess_expression')

    # Check that we get an expression
    ufl_assert(isinstance(expr, Expr), "Expecting Expr.")

    # Object names is empty if not given
    object_names = object_names or {}

    # Element mapping is empty if not given
    element_mapping = element_mapping or {}

    # Create empty expression data
    expr_data = ExprData()

    # Store original expression
    expr_data.original_expr = expr

    # Store name of expr if given, otherwise empty string
    # such that automatic names can be assigned externally
    expr_data.name = object_names.get(id(expr), "") # TODO: Or default to 'expr'?

    # Extract common domain
    domains = form.domains()
    common_domain = domains[0] if len(domains) == 1 else None

    # TODO: Split out expand_compounds from expand_derivatives
    # Expand derivatives
    tic('expand_derivatives')
    expr = expand_derivatives(expr)

    # Replace arguments and coefficients with new renumbered objects
    tic('extract_arguments_and_coefficients')
    original_arguments, original_coefficients = \
        extract_arguments_and_coefficients(expr)

    tic('build_element_mapping')
    element_mapping = build_element_mapping(element_mapping,
        common_domain, original_arguments, original_coefficients)

    tic('build_argument_replace_map')
    replace_map, renumbered_arguments, renumbered_coefficients = \
        build_argument_replace_map(original_arguments,
                                   original_coefficients,
                                   element_mapping)
    expr = replace(expr, replace_map)

    # Build mapping to original arguments and coefficients, which is
    # useful if the original arguments have data attached to them
    inv_replace_map = dict((w,v) for (v,w) in replace_map.iteritems())
    original_arguments = [inv_replace_map[v] for v in renumbered_arguments]
    original_coefficients = [inv_replace_map[w] for w in renumbered_coefficients]

    # Store data extracted by preprocessing
    expr_data.arguments               = renumbered_arguments    # TODO: Needed?
    expr_data.coefficients            = renumbered_coefficients # TODO: Needed?
    expr_data.original_arguments      = original_arguments
    expr_data.original_coefficients   = original_coefficients
    expr_data.renumbered_arguments    = renumbered_arguments
    expr_data.renumbered_coefficients = renumbered_coefficients

    tic('replace')
    # Mappings from elements and functions (coefficients and arguments)
    # that reside in expr to objects with canonical numbering as well as
    # completed cells and elements
    expr_data.element_replace_map = element_mapping
    expr_data.function_replace_map = replace_map

    # Store signature of form
    tic('signature')
    expr_data.signature = compute_expression_signature(expr, expr_data.function_replace_map)

    # Store elements, sub elements and element map
    tic('extract_elements')
    expr_data.elements            = tuple(f.element() for f in
                                          chain(renumbered_arguments,
                                                renumbered_coefficients))
    expr_data.unique_elements     = unique_tuple(expr_data.elements)
    expr_data.sub_elements        = extract_sub_elements(expr_data.elements)
    expr_data.unique_sub_elements = unique_tuple(expr_data.sub_elements)

    # Store element domains (NB! This is likely to change!)
    # FIXME: DOMAINS: What is a sensible way to store domains for a expr?
    #expr_data.domains = tuple(sorted(set(element.domain()
    #                                     for element in expr_data.unique_elements)))

    # Store common cell
    expr_data.cell = common_cell

    # Store data related to cell
    expr_data.geometric_dimension = common_domain.geometric_dimension()
    expr_data.topological_dimension = common_domain.topological_dimension()

    # Store some useful dimensions
    expr_data.rank = len(expr_data.arguments) # TODO: Is this useful for expr?
    expr_data.num_coefficients = len(expr_data.coefficients)

    # Store argument names # TODO: Is this useful for expr?
    expr_data.argument_names = \
        [object_names.get(id(expr_data.original_arguments[i]), "v%d" % i)
         for i in range(expr_data.rank)]

    # Store coefficient names
    expr_data.coefficient_names = \
        [object_names.get(id(expr_data.original_coefficients[i]), "w%d" % i)
         for i in range(expr_data.num_coefficients)]

    # Store preprocessed expression
    expr_data.preprocessed_expr = expr

    tic.end()

    # A coarse profiling implementation
    # TODO: Add counting of nodes
    # TODO: Add memory usage
    if preprocess_expression.enable_profiling:
        print tic

    return expr_data
preprocess_expression.enable_profiling = False

