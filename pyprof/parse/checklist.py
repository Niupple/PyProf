from .utils import TablePrinter

class CheckList:
    def __init__(self, template):
        self.entry = {k: {'func': f, 'checked': None, 'sugg': s} for k, f, s in template}

    def check(self, item, *args, **kwargs):
        d = self.entry[item]
        if d['checked'] is None:
            d['checked'] = d['func'](*args, **kwargs)
        else:
            print("!recheck on item {}".format(item))

    def print(self):
        printer = TablePrinter(["Item", "Checked", "Suggestions"])
        for k, e in self.entry.items():
            if e['checked'] is None:
                continue
            elif not e['checked']:
                printer.write_line([k, 'WARNING', e['sugg']])
            else:
                printer.write_line([k, 'PASSED', '-'])
        printer.print()

def always_true(*args, **kwargs):
    return True

def always_false(*args, **kwargs):
    return False

AUTOPROF_TEMPLATE = [
    ('gemm_heavy', always_true, 'Try to convert your model to fp16.'),
    ('small_kernels', always_true, 'Try to fuse the small kernels in the model.'),
    ('data_movement', always_false, 'Try to reduce the total amount of the data movement.'),
]

def get_default_checklist():
    return CheckList(AUTOPROF_TEMPLATE)