from itertools import permutations

def bits(n, k):
    if (n <= k):
        return {'0'*k}
        
    init_str = '0'*k + '1'*(n-k)
    res = {''.join(e) for e in permutations(init_str, n)}
    return res

if __name__ == '__main__':
    import binary
    assert binary.bits(4, 3) == {'0100', '0001', '0010', '1000'}
    assert binary.bits(4, 1) == {'0111', '1011', '1101', '1110'}
    assert binary.bits(5, 4) == {'00001', '00100', '01000', '10000', '00010'}