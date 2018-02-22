
from ufl.algorithms import compute_form_data
from ufl.formatting.ufl2unicode import form2unicode


def valid_forms(forms_list):
    forms = []
    form_datas = []
    for f in forms_list:
        fd = None
        try:
            fd = compute_form_data(f)
        except:
            fd = None
        if fd is not None:
            forms.append(f)
            form_datas.append(fd)
    return forms, form_datas


def test_convert_examples(example_files):
    # Get example forms that can be analysed
    forms, form_datas = valid_forms(example_files.forms)
    if not forms:
        return

    # Mainly tests for execution without errors
    data = []
    for form, form_data in zip(forms, form_datas):
        tmp = form2unicode(form, form_data)
        data.append(tmp)
    rendered = u"\n\n".join(data)
    assert isinstance(rendered, str)
    assert len(rendered)
