from collections import defaultdict
import six

class PhysicalDimension:
    def __init__(self, dims):
        self.counts = defaultdict(int)
        if isinstance(dims, str):
            self.counts[dims] = 1
        elif isinstance(dims, dict):
            self.counts.update(dims)
        else:
            raise Exception("Invalid dimension type.")
        self.normalize()

    def normalize(self):
        keys = [d for (d, c) in six.iteritems(self.counts) if c == 0]
        for k in keys:
            del self.counts[k]

    def __mul__(self, other):
        keys = set(self.counts) | set(other.counts)
        counts = dict()
        for k in keys:
            c = self.counts.get(k, 0) + other.counts.get(k, 0)
            if c:
                counts[k] = c
        return PhysicalDimension(counts)

    def __div__(self, other):
        keys = set(self.counts) | set(other.counts)
        counts = dict()
        for k in keys:
            c = self.counts.get(k, 0) - other.counts.get(k, 0)
            if c:
                counts[k] = c
        return PhysicalDimension(counts)

    def __pow__(self, n):
        keys = set(self.counts)
        counts = dict()
        for k in keys:
            c = self.counts.get(k, 0)*n
            if c:
                if c == int(c):
                    c = int(c)
                counts[k] = c
        return PhysicalDimension(counts)

    def __eq__(self, other):
        return self.counts == other.counts

    def __ne__(self, other):
        return self.counts != other.counts

    def __nonzero__(self):
        return bool(self.counts)
    def __bool__(self):
        return bool(self.counts)

    def __str__(self):
        def fmt(d):
            c = self.counts[d]
            return d if c == 1 else "%s^%g" % (d, c)
        return " ".join(map(fmt, sorted(self.counts.keys())))

def test_physical_dimension():
    m = PhysicalDimension("m")
    s = PhysicalDimension("s")
    assert "m" == str(m)
    assert "s" == str(s)
    assert "m^2" == str(m**2)
    assert "s^2" == str(s**2)
    assert "m s^3" == str(m*s**2*m*s/m)
    assert "m s^2" == str(m*s**2)
    assert m == m
    assert m == PhysicalDimension("m")
    assert m != s
    assert m**2 == m**2
    assert m**2 != s**2
    assert m**3 != m**2
    assert m**2*s != m*s**2

if __name__ == "__main__":
    test_physical_dimension()

