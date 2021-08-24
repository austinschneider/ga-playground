# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import collections
import copy
import typing
import dataclasses

# %%
metric = [1, -1, -1, -1]

def basis_from_string(sbasis):
    lsbasis = sbasis.split(",")
    if "1" in lsbasis and np.any(len(list(filter(str.isalpha, svector))) > 0 for svector in lsbasis):
        lsbasis.remove("1")

    basis = [tuple()]
    lsbasis = ["".join(filter(str.isdigit, svector)) for svector in lsbasis]
    u, count = np.unique((len(s) for s in lsbasis), return_counts=True)
    n = count[0]
    digits = int(np.floor(np.log10(n))) + 1
    
    for s in lsbasis:
        vec = tuple(int(s[i:i + digits]) for i in range(0, len(s), digits))
        basis.append(vec)
    return basis

basis_vectors = basis_from_string("1,e0,e1,e2,e3,e10,e20,e30,e32,e13,e21,e123,e320,e130,e210,e0123")


# %%
class MVElement(typing.NamedTuple):
    scalar: typing.Union[int, float]
    blade: tuple


# %%
def pretty_print(element, basis_vectors, metric):
    s, v = element
    scalar_str = str(s)
    if s == 1:
        scalar_str = ""
    elif s == -1:
        scalar_str = "-"
    elif s == 0:
        scalar_str = ""
    
    basis_str = "e" + "".join(str(vv) for vv in v)
    if s == 0:
        basis_str = ""
    elif len(v) == 0:
        basis_str = "1"
    
    return scalar_str + basis_str


# %%
format_str = "{:>" + str(1 + len(metric)) + "}"
for blade in basis_vectors:
    blade_str = str(blade)
    pretty_blade_str = pretty_print((1.0, blade), basis_vectors, metric)
    print(format_str.format(pretty_blade_str) + ": " + blade_str)


# %%
def simplify_element(element, basis_vectors, metric):
    scalar, blade = element
    
    core_basis_vectors = [i[0] for i in basis_vectors if len(i) == 1]
    idx_map = dict((v,i) for i,v in enumerate(core_basis_vectors))
    
    unique_base_vectors, vector_count = np.unique(blade, return_counts=True)
    
    if 0 in metric:
        zero_vectors = [v for v,m in zip([vec[0] for vec in basis_vectors if len(vec) == 1], metric) if m == 0]
        zero_vectors = [v for v in zero_vectors if v in unique_base_vectors]
        is_zero = [vector_count[unique_base_vectors == v][0] > 1 for v in zero_vectors]
        if np.any(is_zero):
            return 0.0, tuple()
    
    is_negated = False
    current_blade = copy.copy(blade)
    while np.any([i > 1 for i in vector_count]):
        target_vector = next(vector for vector, count in zip(unique_base_vectors, vector_count) if count > 1)
        first_index = None
        second_index = None
        for index in range(len(current_blade)):
            found = current_blade[index] == target_vector
            if found:
                if first_index is None:
                    first_index = index
                elif second_index is None:
                    second_index = index
                    break
        
        distance = second_index - first_index
        is_negated ^= not (distance % 2)
        is_negated ^= (metric[idx_map[target_vector]] < 0)
        
        current_blade = current_blade[:first_index] + current_blade[first_index+1:second_index] + current_blade[second_index+1:]
        
        unique_base_vectors, vector_count = np.unique(current_blade, return_counts=True)
    
    sign = 1.0 - 2.0 * is_negated
    
    return MVElement(scalar * sign, current_blade)

n = len(basis_vectors)
format_str = ("{:>" + str(3 + len(metric)) + "}")
res = np.zeros((n,n)).tolist()
for i in range(n):
    for j in range(n):
        res[i][j] = simplify_element(MVElement(1.0, basis_vectors[i] + basis_vectors[j]), basis_vectors, metric)
    print((format_str*n).format(*[pretty_print((s,v), basis_vectors, metric) for s, v in res[i]]))


# %%
def reduce_to_basis(element, basis_vectors, metric):
    scalar, current_blade = simplify_element(element, basis_vectors, metric)

    n = len(current_blade)
    blades = [i for i in basis_vectors if len(i) == n]
    blade_set_to_blade = dict((tuple(sorted(blade)), blade) for blade in blades)
    
    basis_blade = blade_set_to_blade[tuple(sorted(current_blade))]
    
    is_negated = False
    #for first_index, target_vector in enumerate(current_blade):
    #    second_index = np.arange(n)[[v == target_vector for v in basis_blade]][0]
    #    distance = second_index - first_index
    #    is_negated ^= (distance % 2 and second_index > first_index)
    for first_index in range(len(basis_blade)):
        target_vector = basis_blade[first_index]
        second_index = np.arange(n)[[v == target_vector for v in current_blade]][0]
        distance = second_index - first_index
        is_negated ^= (distance % 2 and second_index > first_index)
        current_blade = (target_vector,) + current_blade[:second_index] + current_blade[second_index+1:]
    sign = 1.0 - 2.0 * is_negated
    
    return MVElement(scalar * sign, basis_blade)


# %%
print(reduce_to_basis(MVElement(1, (1,3,0)), basis_vectors, metric))
print(reduce_to_basis(MVElement(1, (1,0,3)), basis_vectors, metric))
print(reduce_to_basis(MVElement(1, (0,1,3)), basis_vectors, metric))
print(reduce_to_basis(MVElement(1, (0,3,1)), basis_vectors, metric))

# %%
n = len(basis_vectors)
format_str = ("{:>" + str(3 + len(metric)) + "}")
res = np.zeros((n,n)).tolist()
for i in range(n):
    for j in range(n):
        res[i][j] = reduce_to_basis(MVElement(1.0, basis_vectors[i] + basis_vectors[j]), basis_vectors, metric)
    print((format_str*n).format(*[pretty_print((s,v), basis_vectors, metric) for s, v in res[i]]))

# %%
n = len(basis_vectors)
format_str = ("{:>" + str(3 + len(metric)) + "}")
res = np.zeros((n,n)).tolist()
for i in range(n):
    for j in range(n):
        res[i][j] = reduce_to_basis(MVElement(1.0, basis_vectors[i] + basis_vectors[j]), basis_vectors, metric)
    print((format_str*n).format(*[pretty_print((s,v), basis_vectors, metric) for s, v in res[i]]))

# %%
four_vector_blades = basis_vectors[1:5]
print(four_vector_blades)
boost_bivector_blades = basis_vectors[5:8]
print(boost_bivector_blades)
rotation_bivector_blades = basis_vectors[8:11]
print(rotation_bivector_blades)
bivector_blades = boost_bivector_blades + rotation_bivector_blades
print(bivector_blades)

boost_blades = basis_vectors[:1] + boost_bivector_blades
print(boost_blades)
rotation_blades = basis_vectors[:1] + rotation_bivector_blades
print(rotation_blades)
rotor_blades = basis_vectors[:1] + bivector_blades
print(rotor_blades)

valid_bases = [four_vector_blades, boost_bivector_blades, rotation_bivector_blades, bivector_blades, boost_blades, rotation_blades, rotor_blades]


# %%
def sort_blades(blades, basis_vectors):
    basis_sort_map = dict(((v,i) for i,v in enumerate(basis_vectors)))
    indices = sorted([basis_sort_map[b] for b in blades])
    return [basis_vectors[i] for i in indices]


# %%
def is_symmetric(blade0, blade1):
    element_0 = reduce_to_basis(MVElement(1, blade0 + blade1), basis_vectors, metric)
    element_1 = reduce_to_basis(MVElement(1, blade1 + blade0), basis_vectors, metric)
    return element_0.scalar == element_1.scalar


# %%
def elements_cancel(element0, element1):
    scalar0, blade0 = reduce_to_basis(element0, basis_vectors, metric)
    scalar1, blade1 = reduce_to_basis(element1, basis_vectors, metric)
    return blade0 == blade1 and scalar0 == -scalar1


# %%

# %%
for blade0 in four_vector_blades:
    for blade1 in boost_blades:
        print(blade0, blade1, is_symmetric(blade0, blade1))


# %%
def determine_product_rank(blades0, blades1):
    products = set()
    for blade0 in blades0:
        for blade1 in blades1:
            product_element = reduce_to_basis(MVElement(1.0, blade0 + blade1), basis_vectors, metric)
            new_blade = product_element.blade
            products.add(new_blade)
    return sort_blades(products, basis_vectors)

def determine_outer_product_rank(blades0, blades1):
    products = set()
    for blade0 in blades0:
        for blade1 in blades1:
            product_element_0 = reduce_to_basis(MVElement(1.0, blade0 + blade1), basis_vectors, metric)
            product_element_1 = reduce_to_basis(MVElement(1.0, blade1 + blade0), basis_vectors, metric)
            if product_element_0.scalar != product_element_1.scalar:
                new_blade = product_element_0.blade
                products.add(new_blade)
    return sort_blades(products, basis_vectors)

def determine_inner_product_rank(blades0, blades1):
    products = set()
    for blade0 in blades0:
        for blade1 in blades1:
            product_element_0 = reduce_to_basis(MVElement(1.0, blade0 + blade1), basis_vectors, metric)
            product_element_1 = reduce_to_basis(MVElement(1.0, blade1 + blade0), basis_vectors, metric)
            if product_element_0.scalar == product_element_1.scalar:
                new_blade = product_element_0.blade
                products.add(new_blade)
    return sort_blades(products, basis_vectors)

print(determine_product_rank(four_vector_blades, four_vector_blades))
print(determine_outer_product_rank(four_vector_blades, four_vector_blades))
print(determine_inner_product_rank(four_vector_blades, four_vector_blades))

# %%
print(determine_product_rank(four_vector_blades, [(0,)]))
print(determine_outer_product_rank(four_vector_blades, [(0,)]))
print(determine_inner_product_rank(four_vector_blades, [(0,)]))

# %%
print(determine_product_rank(four_vector_blades, boost_blades))
print(determine_outer_product_rank(four_vector_blades, boost_blades))
print(determine_inner_product_rank(four_vector_blades, boost_blades))


# %%
class ElementSource(typing.NamedTuple):
    sourceID: str
    loc: int
    blade: tuple

class BinarOp(typing.NamedTuple):
    op: str
    sign0: int
    sign1: int
    source0: any
    source1: any

class ChainOp(typing.NamedTuple):
    op: str
    signs: list
    sources: list


# %%
a = ElementSource("v0", 1, (0,))
b = ElementSource("v0", 1, (0,))
c = ElementSource("v0", 1, (1,))
a == b, b == c


# %%
class _ASTSource(typing.NamedTuple):
    sourceID: str
    loc: int

class ASTSource(_ASTSource):
    def __repr__(self):
        return self.sourceID + "[" + str(self.loc) + "]"

class _ASTElement(typing.NamedTuple):
    blade: tuple
    factor: float = 1
    sources: collections.Counter = None
    def is_like(self, other):
        #print(self, other, self.blade == other.blade, self.sources == other.sources)
        return (self.blade == other.blade) and (self.sources == other.sources)

class ASTElement(_ASTElement):
    __slots__ = ()
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        if obj.sources is None:
            obj = obj._replace(sources=[])
        return obj
    def alt_repr(self, mul_str="", c_pow=False):
        if self.factor == 1:
            f = ""
        elif self.factor == -1:
            f = "-"
        else:
            f = str(self.factor) + "*"
        if c_pow:
            s = mul_str.join([(("std::pow(" + str(k) + "," + str(count) + ")" if count > 1 else str(k)) if count > 0 else "") for (k,count),count in self.sources.items()])
        else:
            s = mul_str.join([(str(k)+("^"+str(count) if count > 1 else "") if count > 0 else "") for (k,count),count in self.sources.items()])
        b = "e" + "".join([str(d) for d in self.blade])
        return f"{f}{s}*{b}"
    def __repr__(self):
        return self.alt_repr(mul_str="")

class _ASTSum(typing.NamedTuple):
    blade: tuple
    elements: list = dataclasses.field(default_factory=list)

class ASTSum(_ASTSum):
    def alt_repr(self, mul_str="", c_pow=False):
        element_strings = []
        blade_str = b = "e" + "".join([str(d) for d in self.blade])
        for element in self.elements:
            element_string = element.alt_repr(mul_str=mul_str)
            idx = element_string.rfind("*")
            element_string = element_string[:idx]
            element_strings.append(element_string)
        if len(element_strings) == 1:
            return element_strings[0] + blade_str
        else:
            if np.all([s[0] == "-" for s in element_strings]):
                return "-(" + "+".join([s[1:] for s in element_strings]) + ")" + blade_str
            else:
                return "(" + element_strings[0] + "".join([(s if s[0] == "-" else "+" + s) for s in element_strings[1:]]) + ")" + blade_str
    def __repr__(self):
        return self.alt_repr(mul_str="")

class _ASTMultiVector(typing.NamedTuple):
    sums: list = None
    blades: list = None
    blade_index: dict = None
    def set_blade(self, blade, value):
        self.sums[self.blade_index[blade]] = value

class ASTMultiVector(_ASTMultiVector):
    __slots__ = ()
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        if obj.blades is None:
            obj = obj._replace(blades=copy.copy(basis_vectors))
        if obj.sums is None:
            obj = obj._replace(sums=[None for i in range(len(obj.blades))])
        if obj.blade_index is None:
            obj = obj._replace(blade_index=dict([(v,i) for i,v in enumerate(obj.blades)]))
        return obj
    def is_zero(self):
        return np.all([s is None for s in self.sums])
    def alt_repr(self, mul_str="", c_pow=False):
        if self.is_zero():
            return "0"
        element_strings = [s.alt_repr(mul_str=mul_str) for s in self.sums if s is not None]
        return element_strings[0] + "".join([(" - " + s[1:] if s[0] == "-" else " + " + s) for s in element_strings[1:]])
        #return " + ".join([str(s) for s in self.sums if s is not None])
    def __repr__(self):
        return self.alt_repr(mul_str="")
        
def make_element(sourceID, loc, blade):
    scalar, blade = reduce_to_basis(MVElement(1.0, blade), basis_vectors, metric)
    source = ASTSource(sourceID=sourceID, loc=loc)
    counter = collections.Counter([(source, 1)])
    element = ASTElement(sources=counter, factor=scalar, blade=blade)
    element_sum = ASTSum(elements=[element], blade=blade)
    multivector = ASTMultiVector()
    multivector.set_blade(blade, element_sum)
    return multivector

def multivector_add(mv0, mv1):
    res = copy.copy(mv0)
    return multivector_add_assign(res, mv1)

def multivector_add_assign(mv0, mv1):
    assert(mv0.blades == mv1.blades)
    for i, blade in enumerate(mv0.blades):
        if mv0.sums[i] is None:
            mv0.sums[i] = mv1.sums[i]
        elif mv1.sums[i] is None:
            pass
        else:
            sum0 = mv0.sums[i]
            sum1 = mv1.sums[i]
            element_idx_to_remove = []
            elements_to_add = []
            for idx_s1, elem1 in enumerate(sum1.elements):
                found_like_element = False
                for idx_s0, elem0 in enumerate(sum0.elements):
                    if elem1.is_like(elem0):
                        found_like_element = True
                        new_factor = elem0.factor + elem1.factor
                        if new_factor == 0:
                            element_idx_to_remove.append(idx_s0)
                        else:
                            new_element = ASTElement(
                                sources=elem0.sources,
                                blade=elem0.blade,
                                factor=new_factor
                            )
                            sum0.elements[idx_s0] = new_element
                        
                        found_like_element = True
                        break
                if found_like_element:
                    pass
                else:
                    elements_to_add.append(elem1)
            for idx in reversed(sorted(element_idx_to_remove)):
                del sum0.elements[idx]
            sum0.elements.extend(elements_to_add)
            if len(sum0.elements) == 0:
                mv0.sums[i] = None
    return mv0

a = make_element("A", 0, (0,))
b = make_element("B", 0, (0,))
multivector_add(a, b)


# %%
def element_mul(element0, element1, blade=None, scalar=1.0):
    sources0, sources1 = element0.sources, element1.sources
    factor0, factor1 = element0.factor, element1.factor
    
    sources = copy.copy(sources0)
    sources.update(sources1)
    factor = factor0 * factor1
    
    if blade is None:
        blade0 = element0.blade
        blade1 = element1.blade
        scalar, blade = reduce_to_basis(MVElement(1.0, blade0 + blade1), basis_vectors, metric)
    
    element = ASTElement(
        blade=blade,
        factor=factor*scalar,
        sources=sources,
    )
    return element

def multivector_mul(mv0, mv1):
    res = ASTMultiVector()
    elements = dict()
    for idx0, blade0 in enumerate(mv0.blades):
        sum0 = mv0.sums[idx0]
        if sum0 is None:
            continue
        for idx1, blade1 in enumerate(mv1.blades):
            sum1 = mv1.sums[idx1]
            if sum1 is None:
                continue
            scalar, blade = reduce_to_basis(MVElement(1.0, blade0 + blade1), basis_vectors, metric)
            if blade not in elements:
                elements[blade] = []
            elems = elements[blade]
            for element0 in sum0.elements:
                for element1 in sum1.elements:
                    elems.append(element_mul(element0, element1, blade=blade, scalar=scalar))
                    #print(elems[-1])
            combined_elements = []
            is_combined = [False for i in range(len(elems))]
            for idx_e0 in range(len(elems)):
                if is_combined[idx_e0]:
                    continue
                to_combine = [idx_e0]
                for idx_e1 in range(idx_e0+1, len(elems)):
                    if (not is_combined[idx_e1]) and elems[idx_e0].is_like(elems[idx_e1]):
                        to_combine.append(idx_e1)
                #print("To combine:", [elems[combine_idx] for combine_idx in to_combine])
                if len(to_combine) == 1:
                    #print("Not combining")
                    combined_elements.append(elems[idx_e0])
                else:
                    #print("Combining:", [elems[combine_idx].factor for combine_idx in to_combine])
                    factor = np.sum([elems[combine_idx].factor for combine_idx in to_combine])
                    #print("Factor:", factor)
                    if factor != 0:
                        new_elem = ASTElement(blade=blade, factor=factor, sources=elems[idx_e0].sources)
                        combined_elements.append(new_elem)
                    for idx in to_combine:
                        is_combined[idx] = True
            #print("Combined:", combined_elements)
            elements[blade] = combined_elements
    for blade, element_list in elements.items():
        if len(element_list) > 0:
            res.sums[res.blade_index[blade]] = ASTSum(elements=element_list, blade=blade)
    return res

a = make_element("A", 0, (1,2))
b = make_element("B", 0, (1,))
multivector_mul(a, b)


# %%
def multivector_negate(mv):
    res = ASTMultiVector()
    for idx, blade in enumerate(mv.blades):
        element_sum = mv.sums[idx]
        if element_sum is None:
            pass
        else:
            elements = [ASTElement(blade=blade, factor=-element.factor, sources=element.sources) for element in element_sum.elements]
            res.sums[idx] = ASTSum(blade=blade, elements=elements)
    return res

a = make_element("A", 0, (1,2))
b = make_element("B", 0, (1,))
multivector_negate(multivector_mul(a, b))


# %%
def multivector_scalar_mul(mv, scalar):
    res = ASTMultiVector()
    for idx, blade in enumerate(mv.blades):
        element_sum = mv.sums[idx]
        if element_sum is None:
            pass
        else:
            elements = [ASTElement(blade=blade, factor=scalar * element.factor, sources=element.sources) for element in element_sum.elements]
            res.sums[idx] = ASTSum(blade=blade, elements=elements)
    return res

a = make_element("A", 0, (1,2))
b = make_element("B", 0, (1,))
multivector_scalar_mul(multivector_mul(a, b), 0.5)

# %%
a = multivector_add(multivector_add(make_element("A", 0, (2,1)), make_element("A", 1, (3,2))), make_element("A", 2, (1,3)))
print(multivector_mul(a, a))
print(multivector_scalar_mul(multivector_add(multivector_mul(a, a), multivector_mul(a, a)), 0.5))
print(multivector_scalar_mul(multivector_add(multivector_mul(a, a), multivector_negate(multivector_mul(a, a))), 0.5))
print()

b = multivector_add(multivector_add(make_element("B", 0, (1,0)), make_element("B", 1, (2,0))), make_element("B", 2, (3,0)))
print(b)
print(multivector_mul(b, b))
print(multivector_scalar_mul(multivector_add(multivector_mul(b, b), multivector_mul(b, b)), 0.5))
print(multivector_scalar_mul(multivector_add(multivector_mul(b, b), multivector_negate(multivector_mul(b, b))), 0.5))
print()

c = make_element("C", 0, (1,0))
print(multivector_scalar_mul(multivector_add(multivector_mul(c, c), multivector_mul(c, c)), 0.5))
print(multivector_scalar_mul(multivector_add(multivector_mul(c, c), multivector_negate(multivector_mul(c, c))), 0.5))
print()

a = multivector_add(multivector_add(make_element("A", 0, (1,0)), make_element("A", 1, (2,0))), make_element("A", 2, (3,0)))
b = multivector_add(multivector_add(make_element("B", 0, (1,0)), make_element("B", 1, (2,0))), make_element("B", 2, (3,0)))
print(multivector_mul(a, b))
print(multivector_scalar_mul(multivector_add(multivector_mul(a, b), multivector_mul(b, a)), 0.5))
print(multivector_scalar_mul(multivector_add(multivector_mul(a, b), multivector_negate(multivector_mul(b, a))), 0.5))


# %%
def multivector_involution(mv):
    res = ASTMultiVector()
    for idx, blade in enumerate(mv.blades):
        element_sum = mv.sums[idx]
        if element_sum is None:
            pass
        else:
            elements = [ASTElement(blade=blade, factor=element.factor*((-1)**len(blade)), sources=element.sources) for element in element_sum.elements]
            res.sums[idx] = ASTSum(blade=blade, elements=elements)
    return res

def multivector_reversion(mv):
    res = ASTMultiVector()
    for idx, blade in enumerate(mv.blades):
        element_sum = mv.sums[idx]
        if element_sum is None:
            pass
        elif reduce_to_basis(MVElement(1, blade[::-1]), basis_vectors, metric)[0] == -1:
            elements = [ASTElement(blade=blade, factor=-element.factor, sources=element.sources) for element in element_sum.elements]
            res.sums[idx] = ASTSum(blade=blade, elements=elements)
        else:
            res.sums[idx] = copy.copy(element_sum)
    return res

def multivector_conjugate(mv):
    return multivector_involution(multivector_reversion(mv))


# %%
import functools
a = functools.reduce(multivector_add, [make_element("A", i, basis_vectors[i]) for i in range(len(basis_vectors))])
print(a)
print()
print(multivector_conjugate(a))
print()
print(multivector_mul(a, multivector_conjugate(a)))

# %%
import functools
a = functools.reduce(multivector_add, [make_element("A", i, basis_vectors[i]) for i in range(len(basis_vectors))])
print(multivector_involution(a))
print()
print(multivector_reversion(a))
print()
print(multivector_conjugate(a))

# %%
a = functools.reduce(multivector_add, [make_element("A", i, basis_vectors[i]) for i in range(len(basis_vectors))])
b = functools.reduce(multivector_add, [make_element("B", i, basis_vectors[i]) for i in range(len(basis_vectors))])
multivector_mul(a, b)


# %%
def multivector_dual(mv):
    res = ASTMultiVector()
    pseudoscalar = res.blades[-1]
    pss_set = set(pseudoscalar)
    for idx, blade in enumerate(mv.blades):
        element_sum = mv.sums[idx]
        if element_sum is None:
            pass
        else:
            blade_set = set(blade)
            inv_blade = tuple(i for i in pseudoscalar if i not in blade_set)
            _, inv_blade = reduce_to_basis(MVElement(1, inv_blade), basis_vectors, metric)
            sign, _ = reduce_to_basis(MVElement(1, blade + inv_blade), basis_vectors, metric)
            elements = [ASTElement(blade=inv_blade, factor=sign*element.factor, sources=element.sources) for element in element_sum.elements]
            res.sums[res.blade_index[inv_blade]] = ASTSum(blade=inv_blade, elements=elements)
    return res


# %%
a = functools.reduce(multivector_add, [make_element("A", i, basis_vectors[i]) for i in range(len(basis_vectors))])
multivector_dual(a)


# %%

# %%

# %%
def multivector_project_dim(mv, dim):
    res = ASTMultiVector()
    for idx, blade in enumerate(mv.blades):
        if len(blade) == dim:
            res.sums[idx] = mv.sums[idx]
    return res

def multivector_project_blades(mv, proj_blade):
    res = ASTMultiVector()
    idx = mv.blade_index[proj_blade]
    res.sums[idx] = mv.sums[idx]
    return res

def multivector_project_blades(mv, proj_blades):
    res = ASTMultiVector()
    for blade in proj_blades:
        idx = mv.blade_index[blade]
        res.sums[idx] = mv.sums[idx]
    return res

def multivector_project(mv, proj):
    if type(proj) is int:
        return multivector_project_dim(mv, proj)
    else:
        try:
            return multivector_project_blades(mv, proj)
        except TypeError:
            pass
        return multivector_project_blade(mv, proj)


# %%
a = functools.reduce(multivector_add, [make_element("A", i, basis_vectors[i]) for i in range(len(basis_vectors))])
print(multivector_project(a, ((0,), (1,))))


# %%
def multivector_norm(mv):
    return multivector_project(multivector_mul(mv, multivector_conjugate(mv)), 0)


# %%
a = functools.reduce(multivector_add, [make_element("A", i, basis_vectors[i]) for i in range(len(basis_vectors))])
print(multivector_norm(a))
print()
b = functools.reduce(multivector_add, [make_element("A", i, basis_vectors[i]) for i in range(1, 5)])
print(multivector_norm(b))
print(multivector_mul(b, multivector_reversion(b)))
print(multivector_mul(b, multivector_involution(b)))


# %%

# %%

# %%
def blades_commute(blade0, blade1):
    scalar0, prod0 = reduce_to_basis(MVElement(1, blade0 + blade1), basis_vectors, metric)
    scalar1, prod1 = reduce_to_basis(MVElement(1, blade1 + blade0), basis_vectors, metric)
    return scalar0 == scalar1


# %%
print(blades_commute((1,0), (3,2)))
print(blades_commute((1,0), (2,0)))
print(blades_commute((3,2), (1,3)))
print(blades_commute((1,0), (1,3)))


# %%
def blade_square(blade):
    scalar, sq_blade = reduce_to_basis(MVElement(1, blade + blade), basis_vectors, metric)
    return scalar


# %%
blade_square((3,2))


# %%
def blade_groups(blades):
    groups = collections.OrderedDict()
    for blade in blades:
        key = (blade_square(blade), len(blade))
        if key not in groups:
            groups[key] = []
        groups[key].append(blade)
    return groups


# %%
blade_groups(basis_vectors)


# %%
# classes:
# scalar R130B0
# vector R130B1
# space-vector R130B1Sm1
# time-vector R130B1Sp1
# bi-vector R130B2
# space-bivector R130B2Sm1
# timelike-bivector R130B2Sp1
# tri-vector R130B3
# space-trivector R130B3p1
# time-trivector R130Bm1
# pseudoscalar R130B4

# %%
class CodeGenClass(typing.NamedTuple):
    name: str
    squareSign: float
    dim: int
    blades: typing.List[typing.Tuple[int]]


# %%
classes = [CodeGenClass(name="R130B"+str(dim)+"S"+("p1" if sq > 0 else ("m1" if sq < 0 else "0")), squareSign=sq, dim=dim, blades=blades) for (sq, dim), blades in blade_groups(basis_vectors).items()]
dim_counts = collections.Counter()
dim_signs = dict()
for cls in classes:
    dim_counts[cls.dim] += 1
    dim_signs[cls.dim] = cls.squareSign
classes = [cls for cls in classes if dim_counts[cls.dim] > 1]
classes += [CodeGenClass(name="R130B"+str(dim), squareSign=(dim_signs[dim] if dim_counts[dim] == 1 else None), dim=dim, blades=[b for b in basis_vectors if len(b) == dim]) for dim in range(len(metric) + 1)]
classes.append(CodeGenClass(name="R130MV", squareSign=None, dim=None, blades=copy.copy(basis_vectors)))
classes = sorted(classes, key=lambda x: x.name)
classes_by_name = dict([(cls.name, cls) for cls in classes])
classes_by_blades = dict([(tuple(cls.blades), cls) for cls in classes])
classes


# %%
def codegen_assignment(dest, quant, tab=None):
    if tab is None:
        tab = "    "
    lines = []
    for i, (res_sum, prod_sum) in enumerate(zip(dest.sums, quant.sums)):
        if res_sum is None:
            continue
        res_sum_string = res_sum.alt_repr(mul_str="*", c_pow=True)
        idx = res_sum_string.rfind("e")
        res_sum_string = res_sum_string[:idx]
        
        if prod_sum is None:
            lines.append(tab + res_sum_string + "=0;")
        else:
            prod_sum_string = prod_sum.alt_repr(mul_str="*", c_pow=True)
            idx = prod_sum_string.rfind("e")
            prod_sum_string = prod_sum_string[:idx]
            lines.append(tab + res_sum_string + "=" + prod_sum_string + ";")
    return "\n".join(lines)

def codegen_mul(cls0, cls1):
    mv0 = functools.reduce(multivector_add, [make_element("A", i, blade) for i,blade in enumerate(cls0.blades)])
    mv1 = functools.reduce(multivector_add, [make_element("B", i, blade) for i,blade in enumerate(cls1.blades)])
    prod_mv = multivector_mul(mv0, mv1)
    res_blades = tuple(s.blade for s in prod_mv.sums if s is not None)
    if res_blades in classes_by_blades:
        pass
    else:
        res_blades = tuple(copy.copy(basis_vectors))
    res_cls = classes_by_blades[res_blades]
    res_mv = functools.reduce(multivector_add, [make_element("res", i, blade) for i,blade in enumerate(res_blades)])
    lines = []
    lines.append("template<typename T>")
    lines.append("inline " + res_cls.name + "<T> operator*(const " + cls0.name + "<T> &A, const " + cls1.name + "<T> &B) const {")
    lines.append("    " + res_cls.name + "<T> res;")
    lines.append(codegen_assignment(res_mv, prod_mv)[:-1])
    lines.append("    return res;")
    lines.append("};")
    return "\n".join(lines)
print(codegen_mul(classes_by_name["R130B2"], classes_by_name["R130B2"]))


# %%
def codegen_add(cls0, cls1):
    mv0 = functools.reduce(multivector_add, [make_element("A", i, blade) for i,blade in enumerate(cls0.blades)])
    mv1 = functools.reduce(multivector_add, [make_element("B", i, blade) for i,blade in enumerate(cls1.blades)])
    prod_mv = multivector_add(mv0, mv1)
    res_blades = tuple(s.blade for s in prod_mv.sums if s is not None)
    if res_blades in classes_by_blades:
        pass
    else:
        res_blades = tuple(copy.copy(basis_vectors))
    res_cls = classes_by_blades[res_blades]
    res_mv = functools.reduce(multivector_add, [make_element("res", i, blade) for i,blade in enumerate(res_blades)])
    lines = []
    lines.append("template<typename T>")
    lines.append("inline " + res_cls.name + "<T> operator+(const " + cls0.name + "<T> &A, const " + cls1.name + "<T> &B) const {")
    lines.append("    " + res_cls.name + "<T> res;")
    lines.append(codegen_assignment(res_mv, prod_mv)[:-1])
    lines.append("    return res;")
    lines.append("};")
    return "\n".join(lines)
print(codegen_add(classes_by_name["R130B2"], classes_by_name["R130B2"]))

# %%
add_code = [codegen_add(cls0, cls1) for cls0 in classes for cls1 in classes]
mul_code = [codegen_mul(cls0, cls1) for cls0 in classes for cls1 in classes]


# %%
def codegen_negation(cls0):
    mv0 = functools.reduce(multivector_add, [make_element("A", i, blade) for i,blade in enumerate(cls0.blades)])
    quant_mv = multivector_negate(mv0)
    res_blades = tuple(s.blade for s in quant_mv.sums if s is not None)
    if res_blades in classes_by_blades:
        pass
    else:
        res_blades = tuple(copy.copy(basis_vectors))
    res_cls = classes_by_blades[res_blades]
    res_mv = functools.reduce(multivector_add, [make_element("res", i, blade) for i,blade in enumerate(res_blades)])
    body = codegen_assignment(res_mv, quant_mv)[:-1]
    lines = []
    lines.append("template<typename T>")
    lines.append("inline " + res_cls.name + "<T> " + cls0.name + "<T>::negation(const " + cls0.name + "<T> &A) const {")
    lines.append("    " + res_cls.name + "<T> res;")
    lines.append(body)
    lines.append("    return res;")
    lines.append("};")
    lines.append("")
    lines.append("template<typename T>")
    lines.append("inline " + res_cls.name + "<T> operator-(const " + cls0.name + "<T> &A) {")
    lines.append("    " + "return A.negation();")
    lines.append("};")
    return "\n".join(lines)
print(codegen_negation(classes_by_name["R130B2"]))


# %%
def codegen_involution(cls0):
    mv0 = functools.reduce(multivector_add, [make_element("A", i, blade) for i,blade in enumerate(cls0.blades)])
    quant_mv = multivector_involution(mv0)
    res_blades = tuple(s.blade for s in quant_mv.sums if s is not None)
    if res_blades in classes_by_blades:
        pass
    else:
        res_blades = tuple(copy.copy(basis_vectors))
    res_cls = classes_by_blades[res_blades]
    res_mv = functools.reduce(multivector_add, [make_element("res", i, blade) for i,blade in enumerate(res_blades)])
    body = codegen_assignment(res_mv, quant_mv)[:-1]
    lines = []
    lines.append("template<typename T>")
    lines.append("inline " + res_cls.name + "<T> " + cls0.name + "<T>::involution(const " + cls0.name + "<T> &A) const {")
    lines.append("    " + res_cls.name + "<T> res;")
    lines.append(body)
    lines.append("    return res;")
    lines.append("};")
    return "\n".join(lines)
print(codegen_involution(classes_by_name["R130MV"]))


# %%
def codegen_reversion(cls0):
    mv0 = functools.reduce(multivector_add, [make_element("A", i, blade) for i,blade in enumerate(cls0.blades)])
    quant_mv = multivector_reversion(mv0)
    res_blades = tuple(s.blade for s in quant_mv.sums if s is not None)
    if res_blades in classes_by_blades:
        pass
    else:
        res_blades = tuple(copy.copy(basis_vectors))
    res_cls = classes_by_blades[res_blades]
    res_mv = functools.reduce(multivector_add, [make_element("res", i, blade) for i,blade in enumerate(res_blades)])
    body = codegen_assignment(res_mv, quant_mv)[:-1]
    lines = []
    lines.append("template<typename T>")
    lines.append("inline " + res_cls.name + "<T> " + cls0.name + "<T>::reversion(const " + cls0.name + "<T> &A) const {")
    lines.append("    " + res_cls.name + "<T> res;")
    lines.append(body)
    lines.append("    return res;")
    lines.append("};")
    return "\n".join(lines)
print(codegen_reversion(classes_by_name["R130MV"]))


# %%
def codegen_conjugate(cls0):
    mv0 = functools.reduce(multivector_add, [make_element("A", i, blade) for i,blade in enumerate(cls0.blades)])
    quant_mv = multivector_conjugate(mv0)
    res_blades = tuple(s.blade for s in quant_mv.sums if s is not None)
    if res_blades in classes_by_blades:
        pass
    else:
        res_blades = tuple(copy.copy(basis_vectors))
    res_cls = classes_by_blades[res_blades]
    res_mv = functools.reduce(multivector_add, [make_element("res", i, blade) for i,blade in enumerate(res_blades)])
    body = codegen_assignment(res_mv, quant_mv)[:-1]
    lines = []
    lines.append("template<typename T>")
    lines.append("inline " + res_cls.name + "<T> " + cls0.name + "<T>::conjugate(const " + cls0.name + "<T> &A) const {")
    lines.append("    " + res_cls.name + "<T> res;")
    lines.append(body)
    lines.append("    return res;")
    lines.append("};")
    return "\n".join(lines)
print(codegen_conjugate(classes_by_name["R130MV"]))


# %%
def codegen_dual(cls0):
    mv0 = functools.reduce(multivector_add, [make_element("A", i, blade) for i,blade in enumerate(cls0.blades)])
    quant_mv = multivector_dual(mv0)
    res_blades = tuple(s.blade for s in quant_mv.sums if s is not None)
    if res_blades in classes_by_blades:
        pass
    else:
        res_blades = tuple(copy.copy(basis_vectors))
    res_cls = classes_by_blades[res_blades]
    res_mv = functools.reduce(multivector_add, [make_element("res", i, blade) for i,blade in enumerate(res_blades)])
    body = codegen_assignment(res_mv, quant_mv)[:-1]
    lines = []
    lines.append("template<typename T>")
    lines.append("inline " + res_cls.name + "<T> " + cls0.name + "<T>::dual(const " + cls0.name + "<T> &A) const {")
    lines.append("    " + res_cls.name + "<T> res;")
    lines.append(body)
    lines.append("    return res;")
    lines.append("};")
    return "\n".join(lines)
print(codegen_dual(classes_by_name["R130MV"]))


# %%
def codegen_norm(cls0):
    mv0 = functools.reduce(multivector_add, [make_element("A", i, blade) for i,blade in enumerate(cls0.blades)])
    quant_mv = multivector_norm(mv0)
    res_blades = tuple(s.blade for s in quant_mv.sums if s is not None)
    if res_blades in classes_by_blades:
        pass
    else:
        res_blades = tuple(copy.copy(basis_vectors))
    res_cls = classes_by_blades[res_blades]
    res_mv = functools.reduce(multivector_add, [make_element("res", i, blade) for i,blade in enumerate(res_blades)])
    body = codegen_assignment(res_mv, quant_mv)[:-1]
    lines = []
    lines.append("template<typename T>")
    lines.append("inline " + res_cls.name + "<T> " + cls0.name + "<T>::norm(const " + cls0.name + "<T> &A) {")
    lines.append("    " + res_cls.name + "<T> res;")
    lines.append(body)
    lines.append("    res[0]=std::sqrt(res[0]);")
    lines.append("    return res;")
    lines.append("};")
    return "\n".join(lines)
print(codegen_norm(classes_by_name["R130MV"]))


# %%
def codegen_inorm(cls0):
    mv0 = functools.reduce(multivector_add, [make_element("A", i, blade) for i,blade in enumerate(cls0.blades)])
    quant_mv = multivector_norm(mv0)
    res_blades = tuple(s.blade for s in quant_mv.sums if s is not None)
    if res_blades in classes_by_blades:
        pass
    else:
        res_blades = tuple(copy.copy(basis_vectors))
    res_cls = classes_by_blades[res_blades]
    res_mv = functools.reduce(multivector_add, [make_element("res", i, blade) for i,blade in enumerate(res_blades)])
    body = codegen_assignment(res_mv, quant_mv)[:-1]
    lines = []
    lines.append("template<typename T>")
    lines.append("inline " + res_cls.name + "<T> " + cls0.name + "<T>::norm(const " + cls0.name + "<T> &A) {")
    lines.append("    " + res_cls.name + "<T> res;")
    lines.append(body)
    lines.append("    res[0]=1.0/std::sqrt(res[0]);")
    lines.append("    return res;")
    lines.append("};")
    return "\n".join(lines)
print(codegen_norm(classes_by_name["R130MV"]))


# %%
def codegen_conversion(cls, res_cls, signature=False):
    if cls == res_cls:
        return ""
    if not np.all((blade in res_cls.blades) for blade in cls.blades):
        return ""
    mv = functools.reduce(multivector_add, [make_element("(*this)", i, blade) for i,blade in enumerate(cls.blades)])
    res_mv = functools.reduce(multivector_add, [make_element("res", i, blade) for i,blade in enumerate(res_cls.blades)])

    lines = []
    
    if signature:
        lines = []
        lines.append("template<typename T>")
        lines.append("inline operator " + res_cls.name + "<T>() const;")
    else:
        body = codegen_assignment(res_mv, mv)
        lines.append("template<typename T>")
        lines.append("inline " + cls.name + "::operator " + res_cls.name + "<T>() const {")
        lines.append("    " + res_cls.name + "<T> res;")
        lines.append(body)
        lines.append("    return res;")
        lines.append("};")
    return "\n".join(lines)
print(codegen_conversion(classes_by_name["R130B2Sm1"], classes_by_name["R130B2"], signature=True))
print(codegen_conversion(classes_by_name["R130B2Sm1"], classes_by_name["R130B2"]))

# %%

# %%

# %%
groups = blade_groups(basis_vectors)
n_groups = len(groups)
connectivity_matrix = np.zeros((n_groups, n_groups), dtype=int)
#connectivity_matrix[np.arange(n_groups), np.arange(n_groups)] = 1
for idx0, ((sq0, rank0), blades0) in enumerate(groups.items()):
    mv0 = functools.reduce(multivector_add, [make_element("A", i, blade) for i, blade in enumerate(blades0)])
    print("Rank", rank0, "squares to", sq0)
    print("\tCommutes:")
    for idx1, ((sq1, rank1), blades1) in enumerate(groups.items()):
        mv1 = functools.reduce(multivector_add, [make_element("B", i, blade) for i, blade in enumerate(blades1)])
        commutator = multivector_add(multivector_mul(mv0, mv1), multivector_negate(multivector_mul(mv1, mv0)))
        if commutator.is_zero() and idx0 != idx1:
            connectivity_matrix[idx0, idx1] = 1
            print("\t\t",blades0, blades1)
    print("\tAnticommutes:")
    for idx1, ((sq1, rank1), blades1) in enumerate(groups.items()):
        mv1 = functools.reduce(multivector_add, [make_element("B", i, blade) for i, blade in enumerate(blades1)])
        anticommutator = multivector_add(multivector_mul(mv0, mv1), multivector_mul(mv1, mv0))
        if anticommutator.is_zero() and idx0 != idx1:
            connectivity_matrix[idx0, idx1] = 2
            print("\t\t", blades0, blades1)
    print("\tNeither:")
    for idx1, ((sq1, rank1), blades1) in enumerate(groups.items()):
        mv1 = functools.reduce(multivector_add, [make_element("B", i, blade) for i, blade in enumerate(blades1)])
        commutator = multivector_add(multivector_mul(mv0, mv1), multivector_negate(multivector_mul(mv1, mv0)))
        anticommutator = multivector_add(multivector_mul(mv0, mv1), multivector_mul(mv1, mv0))
        if (not commutator.is_zero()) and (not anticommutator.is_zero()) and idx0 != idx1:
            print("\t\t", blades0, blades1)
    print()
for r in connectivity_matrix:
    print(" ".join([str(c) for c in r]))


# %%
mv = functools.reduce(multivector_add, [make_element("A", i, blade) for i, blade in enumerate(basis_vectors) if len(blade) == 2])
last = mv
signatures = collections.OrderedDict()
for i in range(5):
    signature = [blade for blade,i in last.blade_index.items() if last.sums[i] is not None]
    signature = set(signature)
    signature = tuple(sorted(signature))
    print(signature)
    #if signature in signatures:
    #    signatures[signature].append(i)
    #    break
    #else:
    signatures[signature] = []
    signatures[signature].append(i)
    #last = functools.reduce(multivector_add, [make_element("A", i, blade) for i, blade in enumerate(signature)])
    last = multivector_mul(last, mv)

# %%
import graph

#print(np.all(connectivity_matrix.astype(bool) == connectivity_matrix.astype(bool).T))

adjacencies = [set(np.arange(n_groups)[connectivity_matrix[idx].astype(bool)]) for idx in range(n_groups)]
g = graph.UndirectedGraph(adjacencies)
r = graph.SimpleReporter()
graph.bron_kerbosch3_gpx(g, r)
u, count = np.unique(np.asarray(r.cliques).flatten(), return_counts=True)
in_all_cliques = u[count == len(r.cliques)]

for idx in in_all_cliques:
    print(list(groups.values())[idx])

for clique in r.cliques:
    print(sorted(clique))
    for idx in clique:
        print(list(groups.values())[idx])
    print()


# %%
def terms_cancel(source0, source1):
    type0 = type(source0)
    type1 = type(source1)
    if type0 == type1:
        if type0 is ElementSource:
            return (
                source0.sourceID == source1.sourceID and 
                source0.loc == source1.loc and 
                elements_cancel(MVElement(1, source0.blade), MVElement(1, source1.blade))
            )
        elif type0 is BinaryOp:
            if source0.op != source1.op:
                return False
            all_sources = [source0.source0, source0.source1, source1.source0, source1,source1]
            if source0 in all_sources or source1 in all_sources:
                raise ValueError("Sources must not be recursive!")
            if source0.op == "plus":
                if source:
                    pass
            elif source0.op == "minus":
                pass
            elif source0.op == "product" or source0.op == "mul":
                pass
                
                
    else:
        return False


# %%

# %%
ElementSource = collections.namedtuple("ElementSource", ["sourceID", "loc", "blade"])
BinaryOp = collections.namedtuple("BinaryOp", ["op", "sign0", "sign1", "source0", "source1"])
ChainOp = collections.namedtuple("ChainOp", ["op", "signs", "sources"])

all_binary_ops = ["plus", "minus", "product", "geometric_product", "inner_product", "outer_product"]

all_unary_ops = ["reverse", "dual", "conjugate", "involution"]

def generate_plus(blades0, blades1):
    blade_map_0 = dict(((v,i) for i,v in enumerate(blades0)))
    blade_map_1 = dict(((v,i) for i,v in enumerate(blades1)))
    
    res_type = None
    if np.all([b in blade_map_0 for b in blades1]):
        res_type = blades0
    elif np.all([b in blade_map_1 for b in blades0]):
        res_type = blades1
    else:
        res_type = copy.copy(basis_vectors)
    
    mappings = []
    for dest_idx, res_blade in enumerate(res_type):
        dest = ElementSource("res", dest_idx, res_blade)
        elements = []
        if res_blade in blade_map_0:
            element = ElementSource("v0", 1, blade_map_0[res_blade], res_blade)
            elements.append(element)
        if res_blade in blade_map_1:
            element = ElementSource("v1", 1, blade_map_1[res_blade], res_blade)
            elements.append(element)
        if len(elements) == 1:
            mappings.append((dest, elements[0]))
        elif len(elements) == 2:
            bin_op = BinaryOp("plus", elements[0], elements[1])
            mappings.append((dest, bin_op))
    return res_type, mappings

generate_plus(basis_vectors, basis_vectors)


# %%
def generate_product(blades0, blades1):
    blade_map_0 = dict(((v,i) for i,v in enumerate(blades0)))
    blade_map_1 = dict(((v,i) for i,v in enumerate(blades1)))
    
    res_type = determine_product_rank(blades0, blades1)
    
    if res_type not in valid_bases:
        res_type = copy.copy(basis_vectors)
    
    res_blade_map = dict((v,i) for i,v in enumerate(res_type))
    
    mappings = {}
    for dest_idx, res_blade in enumerate(res_type):
        dest = ElementSource("res", dest_idx, res_blade)
        mappings[dest] = ChainOp("plus", [], [])
    for blade0 in blades0:
        for blade1 in blades0:
            product_element = reduce_to_basis(MVElement(1.0, blade0 + blade1), basis_vectors, metric)
            res_blade = product_element.blade
            dest = ElementSource("res", res_blade_map[res_blade], res_blade)
            element0 = ElementSource("v0", blade_map_0[blade0], blade0)
            element1 = ElementSource("v1", blade_map_1[blade1], blade1)
            product = BinaryOp("mul", 1, 1, element0, element1)
            mappings[dest].sources.append(product)
            mappings[dest].signs.append(product_element.scalar)
    mappings = [(lambda blade, i: (res_blade, mappings[ElementSource("res", i, blade)]))(res_blade, res_blade_map[res_blade]) for res_blade in res_type]
    mappings = [(k, v) for  k,v in mappings if len(v.signs) > 0]
    return res_type, mappings

#generate_product(basis_vectors, basis_vectors)
generate_product(four_vector_blades, four_vector_blades)

# %%

# %%

# %%

# %%

# %%
