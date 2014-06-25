
class EmptyDictType(dict):
    def __setitem__(self, key, value):
        raise Exception("This is a frozen unique empty dictionary object, inserting values is an error.")
EmptyDict = EmptyDictType()

d = EmptyDictType()
print(d.get('foo'))
d.update(k=3)
print(d)
d['f'] = 1

