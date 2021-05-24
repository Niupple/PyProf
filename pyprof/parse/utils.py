import sys

class TablePrinter:
    def __init__(self, names : list, rules=None):
        self.n_column = len(names)
        self.names = names
        self.widths = [len(x) for x in names]
        if rules is None:
            self.rules = [str] * self.n_column
        else:
            self.rules = rules
        self.lines = []

    def new_rules(self, rules : list):
        if len(rules) != self.n_column:
            raise Exception
        self.rules = rules

    def write_line(self, line : list):
        if line is None:
            self.lines.append(None)
            return
        assert len(line) == self.n_column
        new_line = [f(x) for f, x in zip(self.rules, line)]
        for i, e in enumerate(new_line):
            assert type(e) is str
            self.widths[i] = max(self.widths[i], len(e))
        self.lines.append(new_line)

    def write_lines(self, lines : list):
        list(map(self.write_line, lines))

    def cutline(self):
        self.write_line(None)

    def format_line(self, line : list, sep='|', pad=' '):
        ret = sep + ''.join([pad + '{0:<{wid}}'.format(x, wid=wid) + pad + sep for x, wid in zip(line, self.widths)])
        return ret

    def print(self, file=sys.stdout):
        cutline = self.format_line(map(lambda x : '-'*x, self.widths), '+', '-')
        # print('\n', file=file)
        print(cutline, file=file)
        print(self.format_line(self.names), file=file)
        print(cutline, file=file)
        for line in self.lines:
            if line is not None:
                print(self.format_line(line), file=file)
            else:
                print(cutline, file=file)
        print(cutline, '\n', file=file)
