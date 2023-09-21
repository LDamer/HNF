def modInverse(a, m):
    """
    :param a: element to invert
    :param m: modulus
    :return: -1 if a has no inverse mod m, a^-1 mod m else
    """
    assert m > 0, "modInverse: m must be > 0!"
    a = a % m
    if a == 0:
        return -1
    g, x, y = eea(a, m)
    if g != 1:
        return -1
    return x % m


def eea(a: int, b: int) -> tuple[int, int, int]:
    if a == 0 and b == 0:
        return 0, 0, 0
    if a == 0:
        return b, 0, 1
    if b == 0:
        return a, 1, 0
    ma = mb = 1
    if a < 0:
        a = -a
        ma = -1
    if b < 0:
        b = -b
        mb = -1
    r0 = a
    r1 = b
    x0 = 1
    x1 = 0
    y0 = 0
    y1 = 1
    while r1 != 0:
        q = r0 // r1
        x1, x0 = x0 - q * x1, x1
        y1, y0 = y0 - q * y1, y1
        r1, r0 = r0 % r1, r1

    return r0, x0 * ma, y0 * mb


def crt(values: list, moduli: list):
    assert len(values) == len(moduli), "You need as many values as moduli"
    assert isinstance(values, list), "values not of type list"
    assert isinstance(moduli, list), "moduli not of type list"
    assert isinstance(values[0], int), "values does not contain ints, but " + str(type(values[0])) + "\n" + str(values)
    assert isinstance(moduli[0], int), "moduli does not contain ints, but " + str(type(moduli[0]))

    x = [0] * len(values)
    M = 1
    for m in moduli:
        M *= m

    for i in range(len(x)):
        p = M // moduli[i]
        _, _, y = eea(moduli[i], p)
        x[i] = round(y * p)

    res = 0
    for i in range(len(values)):
        res += values[i] * x[i]
    return int(res) % M
