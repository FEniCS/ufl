
from ufl.algorithms import load_forms, validate_form, ufl2latex

def forms2latexdocument(forms, uflfilename):
    # Analyse validity of forms
    found_errors = False
    for k,v in forms:
        errors = validate_form(v)
        if errors:
            msg = "Found errors in form '%s':\n%s" % (k, errors)
            raise RuntimeError, msg
    
    # Define template for overall document
    latex = r"""\documentclass{article}
    \usepackage{amsmath}
    
    \begin{document}
    
    \section{UFL Forms from file %s}
    
    """ % uflfilename.replace("_", "\\_")
    
    # Generate latex code for each form
    for name, form in forms:
        l = ufl2latex(form)
        latex += "\\subsection{Form %s}\n" % name
        latex += l
    
    latex += r"""
    \end{document}
    """
    return latex

def uflfile2latex(uflfilename, latexfilename):
    forms = load_forms(uflfilename)
    latex = forms2latexdocument(forms, uflfilename) 
       
    f = open(latexfilename, "w")
    f.write(latex)
    f.close()

def uflfile2pdf(uflfilename):
    basename = uflfilename.replace(".ufl", "") # TODO: safer filename conversion
    latexfilename = basename + ".tex"
    pdffilename = basename + ".pdf"
    
    uflfile2latex(uflfilename, latexfilename)
    
    # TODO: Use subprocess
    os.system("pdflatex %s %s" % (latexfilename, pdffilename))
    os.system("evince %s &" % pdffilename) # TODO: Add option for this. Is there a portable way to do this? like "open foo.pdf in pdf viewer"
    
