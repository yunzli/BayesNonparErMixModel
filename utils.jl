## Utility functions

function findInterval(target, list, ascending=true)
    """
    Find interval index 

    Examples: 
    1. findInterval(0.5, [1,2]) -> 1
    2. findInterval(1.5, [1,2]) -> 2
    3. findInterval(2.5, [1,2]) -> 3
    """
    if !ascending
        list = reverse(list)
    end

    p = 1
    while p <= length(list) && target > list[p]
        p += 1
    end

    if ascending
        return p
    else
        return length(list) - p
    end
end

function svd2inv(M)

    X = svd(M)
    Minv = X.Vt' * Diagonal(1 ./ X.S) * X.U'
    Minv = (Minv + Minv') / 2

    return Minv
end

function get_G0_and_D(M, Dist, theta)

    G0 = zeros(M)
    D = zeros(M)
    for m in 2:M
        G0[m] = cdf(Dist, (m - 1) * theta)
        D[m-1] = G0[m] - G0[m-1]
    end
    D[M] = 1 - G0[M]

    return G0, D
end

