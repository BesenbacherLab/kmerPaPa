from scipy.special import xlogy, xlog1py

def get_loss(L, alpha, beta, penalty=0):
    """Calculate loss
        if penalty is 0 then results will be -2*LogLik

    Args:
        L: list with (n_positive, n_negative) counts
        alpha (float):  pseudo count
        beta (float): beta value to add to denominator
        penalty (float, optional): penalty value in loss function (c). Defaults to 0.

    Returns:
        float: loss
    """
    res = 0.0
    for (nm, nu) in L:
        p = (nm + alpha)/(nm + nu + alpha + beta)
        res += xlogy(nm, p) + xlog1py(nu, -p)
    return -2*res + len(L)*penalty

def get_betas(alpha, M, U):
    """get array with beta values for each fold

    Args:
        alpha (float): pseudo count
        M (int array): Number of mutated sites in each fold
        U (int array): Number of unmutated sites in each fold

    Returns:
        float array: array with beta value for each fold
    """
    my=M/(M+U)
    betas = (alpha*(1.0-my))/my
    return betas

