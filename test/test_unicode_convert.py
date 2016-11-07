
from six import text_type

from ufl.algorithms import compute_form_data
from ufl.formatting.ufl2unicode import form2unicode

def test_convert_examples(example_files):
    # Mainly tests for execution without errors
    forms = example_files.forms
    form_datas = [compute_form_data(f) for f in forms]
    data = []
    for form, form_data in zip(forms, form_datas):
        tmp = form2unicode(form, form_data)
        data.append(tmp)
    rendered = "\n\n".join(data)
    assert isinstance(rendered, text_type)
