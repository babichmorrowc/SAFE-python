"""
    Module to perform PAWN on PMFs instead of CDFs

    This module is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin and
    T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://safetoolbox.github.io

    Package version: SAFEpython_v0.2.0

    References:

    Pianosi, F. and Wagener, T. (2018), Distribution-based sensitivity
    analysis from a generic input-output sample, Env. Mod. & Soft., 108, 197-207.

    Pianosi, F. and Wagener, T. (2015), A simple and efficient method
    for global sensitivity analysis based on cumulative distribution
    functions, Env. Mod. & Soft., 67, 1-11.
"""
from __future__ import division, absolute_import, print_function

from warnings import warn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from safepython.util import allrange

from safepython.PAWN import pawn_split_sample
from safepython.PAWN import pawn_ks


def pawn_pmf_indices(X, Y, n, Nboot=1, dummy=False, output_condition=allrange,
                 par=[]):

    """  Compute the PAWN sensitivity indices using PMFs, rather than CDFs.
    The method was first introduced in Pianosi and Wagener (2015). Here indices are
    computed following the approximation strategy proposed by Pianosi and Wagener
    (2018), which can be applied to a generic input/output sample.

    The function splits the generic output sample to create the conditional
    output by calling internally the function PAWN.pawn_split_sample. The
    splitting strategy is an extension of the strategy for uniformy distributed
    inputs described in Pianosi and Wagener (2018) to handle inputs sampled
    from any distribution(see help of PAWN.pawn_split_sample for further explanation).

    Indices are then computed in two steps:
    1. compute the maximum distance between the empirical
    unconditional PMF and the conditional PMFs for different conditioning
    intervals
    2. take a statistic (median, mean and max) of the results.

    Usage:

        max_dist_median, max_dist_mean, max_dist_max = \
        PAWN.pawn_indices(X, Y, n, Nboot=1, dummy=False)

        max_dist_median, max_dist_mean, max_dist_max, max_dist_dummy = \
        PAWN.pawn_indices(X, Y, n, Nboot=1, dummy=True)

    Input:
            X = set of inputs samples                      - numpy.ndarray(N,M)
            Y = set of output samples                      - numpy.ndarray(N, )
                                                        or - numpy.ndarray(N,1)
            n = number of conditioning intervals to
                assess the conditional PMFs:
                - integer if all inputs have the same number of groups
                - list of M integers otherwise

    Optional input:
        Nboot = number of bootstrap resamples to derive    - scalar
                confidence intervals
        dummy = if dummy is True, an articial input is     - boolean
                added to the set of inputs and the
                sensitivity indices are calculated for the
                dummy input.
                The sensitivity indices for the dummy
                input are estimates of the approximation
                error of the sensitivity indices and they
                can be used for screening, i.e. to
                separate influential and non-influential
                inputs as described in Khorashadi Zadeh
                et al. (2017)
                Default value: False
                (see (*) for further explanation).

    Output:
    max_dist_median = median maximum distance across the conditioning      - numpy.ndarray(Nboot,M)
                intervals (one value for each input
                and each bootstrap resample)
      max_dist_mean = mean maximum distance across the conditioning        - numpy.ndarray(Nboot,M)
                intervals (one value for each input
                and each bootstrap resample)
       max_dist_max = max maximum distance across the conditioning         - numpy.ndarray(Nboot,M)
                intervals (one value for each input
                and each bootstrap resample)

    Optional output (if dummy is True):
    max_dist_dummy = maximum distance of dummy input (one value for       - numpy.ndarray(Nboot, )
                each bootstrap resample)

    --------------------------------------------------------------------------
    ADVANCED USAGE
    for Regional-Response Global Sensitivity Analysis:
    -------------------------------------------------------------------------
    Usage:

    max_dist_median, max_dist_mean, max_dist_max = \
    PAWN.pawn_indices(X, Y, n, Nboot=1, dummy=False,
                      output_condition=allrange, par=[]))

    max_dist_median, max_dist_mean, max_dist_max, max_dist_dummy = \
    PAWN.pawn_indices(X, Y, n, Nboot=1, dummy=True,
                      output_condition=allrange, par=[]))

    Optional input:
    output_condition = condition on the output value to be     - function
                       used to calculate maximum distances. Use the function:
                       - allrange to keep all output values
                       - below to consider only output
                          values below a threshold value
                          (Y <= Ythreshold)
                       - above to consider only output
                          values above a threshold value
                          (Y >= Ythreshold)
                    (functions allrange, below and above are defined in
                     safepython.util)
                 par = specify the input arguments of the      - list
                       'output_condition' function, i.e. the
                       threshold value when output_condition
                       is 'above' or 'below'.

    For more sophisticate conditions, the user can define its own function
    'output_condition' with the following structure:

        idx = output_condition(Y, param)

    where     Y = output samples (numpy.ndarray(N, ))
          param = parameters to define the condition (list of any size)
            idx = logical values, True if condition is satisfied, False
                  otherwise (numpy.ndarray(N, ))

    NOTE:
     (*) For screening influential and non-influential inputs, we recommend the
         use of the maximum maximum distance across the conditioning intervals (i.e. output
         argument max_dist_max), and to compare max_dist_max with the index of the dummy
         input as in Khorashadi Zadeh et al. (2017).

    (**) For each input, the number of conditioning intervals which is actually
         used (n_eff[i]) may be lower than the prescribed number of conditioning
         intervals (n[i]) to ensure that input values that are repeated several
         time belong to the same group.
         See the help of PAWN.pawn_split_sample for further details.

    EXAMPLE:

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    from safepython.sampling import AAT_sampling
    from safepython.model_execution import model_execution
    from safepython import PAWN
    from safepython.plot_functions import boxplot1
    from safepython.ishigami_homma import ishigami_homma_function

    # Create a generic input-output sample:
    N = 5000 # number of samples
    M = 3 # number of inputs
    xmin = -np.pi; xmax = np.pi
    X = AAT_sampling('lhs', M, st.uniform, [xmin, xmax - xmin], N);
    Y = model_execution(ishigami_homma_function, X)

    # Compute PAWN sensitivity indices:
    n = 10; # number of conditioning intervals
    max_dist_median, max_dist_mean, max_dist_max = PAWN.pawn_indices(X, Y, n)
    plt.figure()
    plt.subplot(131); boxplot1(max_dist_median, Y_Label='Max Dist (mean')
    plt.subplot(132); boxplot1(max_dist_mean, Y_Label='Max Dist (mean')
    plt.subplot(133); boxplot1(max_dist_max, Y_Label='Max Dist (max)')

    # Compute sensitivity indices for the dummy input as well:
    max_dist_median, max_dist_mean, max_dist_max, max_dist_dummy = PAWN.pawn_indices(X, Y, n, dummy=True)
    plt.figure()
    boxplot1(np.concatenate((max_dist_max, max_dist_dummy)),
             X_Labels=['X1', 'X2', 'X3', 'dummy'])

    REFERENCES

    Pianosi, F. and Wagener, T. (2018), Distribution-based sensitivity
    analysis from a generic input-output sample, Env. Mod. & Soft., 108, 197-207.

    Pianosi, F. and Wagener, T. (2015), A simple and efficient method
    for global sensitivity analysis based on cumulative distribution
    functions, Env. Mod. & Soft., 67, 1-11.

    REFERENCE FOR THE DUMMY PARAMETER:

    Khorashadi Zadeh et al. (2017), Comparison of variance-based and moment-
    independent global sensitivity analysis approaches by application to the
    SWAT model, Environmental Modelling & Software,91, 210-222.

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info
    """
    ###########################################################################
    # Check inputs and split the input sample
    ###########################################################################

    YY, xc, NC, n_eff, Xk, XX = pawn_split_sample(X, Y, n) # this function
    # checks inputs X, Y and n

    Nx = X.shape
    N = Nx[0]
    M = Nx[1]

    ###########################################################################
    # Check other optional inputs
    ###########################################################################

    if not isinstance(Nboot, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"Nboot" must be scalar and integer.')
    if Nboot < 1:
        raise ValueError('"Nboot" must be >=1.')
    if not isinstance(dummy, bool):
        raise ValueError('"dummy" must be scalar and boolean.')
    if not callable(output_condition):
        raise ValueError('"output_condition" must be a function.')

    ###########################################################################
    # Compute indices
    ###########################################################################

    # Set points at which the PMFs will be evaluated:
    YF = np.unique(Y)

    # Initialize sensitivity indices
    max_dist_median = np.nan * np.ones((Nboot, M))
    max_dist_mean = np.nan * np.ones((Nboot, M))
    max_dist_max = np.nan * np.ones((Nboot, M))
    if dummy: # Calculate index for the dummy input
        max_dist_dummy = np.nan * np.ones((Nboot, ))

    # Compute conditional PMFs
    # (bootstrapping is not used to assess conditional PMFs):
    fC = [np.nan] * M
    for i in range(M): # loop over inputs
        fC[i] = [np.nan] * n_eff[i]
        for k in range(n_eff[i]): # loop over conditioning intervals
            # fC[i][k] = np.unique(YY[i][k], return_counts=True)[1]/len(YY[i][k])
            unique_vals, counts = np.unique(YY[i][k], return_counts=True)
            count_dict = dict(zip(unique_vals, counts))
            counts = np.array([count_dict.get(val, 0) for val in YF])
            fC[i][k] = counts / len(YY[i][k])

    # Initialize unconditional PMFs:
    fU = [np.nan] * M

    # M unconditional PMFs are computed (one for each input), so that for
    # each input the conditional and unconditional PMFs are computed using the
    # same number of data points (when the number of conditioning intervals
    # n_eff[i] varies across the inputs, so does the shape of the conditional
    # outputs YY[i]).

    # Determine the sample size for the unconditional output bootsize:
    bootsize = [int(np.min(i)) for i in NC] # numpy.ndarray(M,)
    # bootsize is equal to the sample size of the conditional outputs NC, or
    # its  minimum value across the conditioning intervals when the sample size
    # varies across conditioning intervals as may happen when values of an
    # input are repeated several times (more details on this in the Note in the
    # help of the function).

    # To reduce the computational time (the calculation of empirical PMF is
    # costly), the unconditional PMF is computed only once for all inputs that
    # have the same value of bootsize[i].
    bootsize_unique = np.unique(bootsize)
    N_compute = len(bootsize_unique)  # number of unconditional PMFs that will
    # be computed for each bootstrap resample

    # Determine the sample size of the subsample for the dummy input.
    # The sensitivity
    # index for the dummy input will be estimated at this minimum sample size
    # so to estimate the 'worst' approximation error of the sensitivity index
    # across the inputs:
    if dummy:
        bootsize_min = min(bootsize) # we use the smaller sample size across
        # inputs, so that the sensitivity index for the dummy input estimates
        # the 'worst' approximation error of the sensitivity index across the
        # inputs:
        idx_bootsize_min = np.where([i == bootsize_min for i in bootsize])[0]
        idx_bootsize_min = idx_bootsize_min[0] # index of an input for which
        # the sample size of the unconditional sample is equal to bootsize_min

        if N_compute > 1:
            warn('The number of data points to estimate the conditional and '+
                 'unconditional output varies across the inputs. The CDFs ' +
                 'for the dummy input were computed using the minimum sample ' +
                 ' size to provide an estimate of the "worst" approximation' +
                 ' of the sensitivity indices across input.')

    # Compute sensitivity indices with bootstrapping
    for b in range(Nboot): # number of bootstrap resample

        # Compute empirical unconditional PMFs
        for kk in range(N_compute): # loop over the sizes of the unconditional output

            # Bootstrap resampling (Extract an unconditional sample of size
            # bootsize_unique[kk] by drawing data points from the full sample Y
            # without replacement
            idx_bootstrap = np.random.choice(np.arange(0, N, 1),
                                             size=(bootsize_unique[kk], ),
                                             replace='False')
            # Compute unconditional PMF:
            # fUkk = np.unique(Y[idx_bootstrap], return_counts=True)[1]/len(Y[idx_bootstrap])
            unique_vals, counts = np.unique(Y[idx_bootstrap], return_counts=True)
            count_dict = dict(zip(unique_vals, counts))
            counts = np.array([count_dict.get(val, 0) for val in YF])
            fUkk = counts / len(Y[idx_bootstrap])
            # Associate the fUkk to all inputs that require an unconditional
            # output of size bootsize_unique[kk]:
            idx_input = np.where([i == bootsize_unique[kk] for i in bootsize])[0]
            for i in range(len(idx_input)):
                fU[idx_input[i]] = fUkk

        # Compute maximum distance between conditional and unconditional PMFs:
        max_dist_all = pawn_ks(YF, fU, fC, output_condition, par)
        # max_dist_all is a list (M elements) and contains the value of the max distance for
        # for each input and each conditioning interval. max_dist_all[i] contains values
        # for the i-th input and the n_eff[i] conditioning intervals, and it
        # is a numpy.ndarray of shape (n_eff[i], ).

        #  Take a statistic of the maximum distance across the conditioning intervals:
        max_dist_median[b, :] = np.array([np.median(j) for j in max_dist_all])  # shape (M,)
        max_dist_mean[b, :] = np.array([np.mean(j) for j in max_dist_all])  # shape (M,)
        max_dist_max[b, :] = np.array([np.max(j) for j in max_dist_all])  # shape (M,)

        if dummy:
            # Compute maximum distance for dummy parameter:
            # Bootstrap again from unconditional sample (the size of the
            # resample is equal to bootsize_min):
            idx_dummy = np.random.choice(np.arange(0, N, 1),
                                         size=(bootsize_min, ),
                                         replace='False')
            # Compute empirical PMFs for the dummy input:
            # fC_dummy = np.unique(Y[idx_dummy], return_counts=True)[1]/len(Y[idx_dummy])
            unique_vals, counts = np.unique(Y[idx_dummy], return_counts=True)
            count_dict = dict(zip(unique_vals, counts))
            counts = np.array([count_dict.get(val, 0) for val in YF])
            fC_dummy = counts / len(Y[idx_dummy])
            # Compute maximum distance for the dummy input:
            max_dist_dummy[b] = pawn_ks(YF, [fU[idx_bootsize_min]], [[fC_dummy]],
                                  output_condition, par)[0][0]

    if Nboot == 1:
        max_dist_median = max_dist_median.flatten()
        max_dist_mean = max_dist_mean.flatten()
        max_dist_max = max_dist_max.flatten()

    if dummy:
        return max_dist_median, max_dist_mean, max_dist_max, max_dist_dummy
    else:
        return max_dist_median, max_dist_mean, max_dist_max


def pawn_plot_pmf(X, Y, n, n_col=5, Y_Label='output y', cbar=False,
                  labelinput=''):

    """ This function computes and plots the unconditional output Probability
    Mass Functions (i.e. when all inputs vary) and the conditional PMFs
    (when one input is fixed to a given conditioning interval, while the other
    inputs vary freely).

    The function splits the output sample to create the conditional output
    by calling internally the function PAWN.pawn_split_sample. The splitting
    strategy is an extension of the strategy for uniformly distributed inputs
    described in Pianosi and Wagener (2018) to handle inputs sampled from any
    distribution.
    (see help of PAWN.pawn_split_sample for further explanation).

    The sensitivity indices for the PAWN method (maximum distance) measures the
    distance between these conditional and unconditional output PMFs
    (see help of PAWN.pawn_pmf_indices for further details and reference).

    Usage:
    YF, fU, fC, xc = PAWN.pawn_plot_pmf(X, Y, n, n_col=5, Y_Label='output y',
                                        cbar=False, labelinput='')

    Input:
             X = set of inputs samples                     - numpy.ndarray(N,M)
             Y = set of output samples                     - numpy.ndarray(N,)
                                                        or - numpy.ndarray(N,1)
             n = number of conditioning intervals
                 - integer if all inputs have the same number of groups
                 - list of M integers otherwise

    Optional input:
         n_col = number of panels per row in the plot      - integer
                 (default: min(5, M))
       Y_Label = legend for the horizontal axis            - string
                 (default: 'output y')
          cbar = flag to add a colorbar that indicates the  - boolean
                 centers of the conditioning intervals for
                 the different conditional PMFs:
                 - if True = colorbar
                 - if False = no colorbar
    labelinput = label for the axis of colorbar (input    - list (M elements)
                 name) (default: ['X1','X2',...,XM'])

    Output:
            YF = values of Y at which the PMFs fU and fC   - numpy.ndarray(P, )
                 are given
            fU = values of the empirical unconditional     - list(M elements)
                 output PMFs. fU[i] is a numpy.ndarray(P, )
                 (see the Note below for further
                 explanation)
            fC = values of the empirical conditional       - list(M elements)
                 output PMFs for each input and each
                 conditioning interval.
                 fC[i] is a list of n_eff[i] CDFs
                 conditional to the i-th input.
                 fC[i][k] is obtained by fixing the i-th
                 input to its k-th conditioning interval
                 (while the other inputs vary freely),
                 and it is a np.ndarray of shape (P, ).
                 (see the Note below for further
                 explanation)
           xc = subsamples' centers (i.e. mean value of    - list(M elements)
                Xi over each conditioning interval)
                xc[i] is a np.ndarray(n_eff[i],) and
                contains the centers for the n_eff[i]
                conditioning intervals for the i-th input.

    Note:
    (*)  For each input, the number of conditioning intervals which is actually
         used (n_eff[i]) may be lower than the prescribed number of conditioning
         intervals (n[i]) to ensure that input values that are repeated several
         time belong to the same group.
         See the help of PAWN.pawn_split_sample for further details.

    (**) fU[i] and fC[i][k] (for any given i and k) are built using the same
         number of data points so that two CDFs can be compared by calculating
         the maximum distance (see help of PAWN.pawn_ks and PAWN.pawn_indices
         for further explanation on the calculation).

    Example:

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    from safepython.sampling import AAT_sampling
    from safepython.model_execution import model_execution
    from safepython import PAWN
    from safepython.plot_functions import boxplot1
    from safepython.ishigami_homma import ishigami_homma_function

    # Create a generic input-output sample:
    N = 1000 # number of samples
    M = 3 # number of inputs
    xmin = -np.pi; xmax = np.pi
    X = AAT_sampling('lhs', M, st.uniform, [xmin, xmax - xmin], N);
    Y = model_execution(ishigami_homma_function, X)

    # Calculate and plot CDFs
    n = 10 # number of conditioning intervals
    YF, FU, FC, xc = PAWN.pawn_plot_cdf(X, Y, n)
    YF, FU, FC, xc = PAWN.pawn_plot_cdf(X, Y, n, cbar=True) # Add colorbar

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info"""

    # Options for the graphic
    pltfont = {'fontname': 'DejaVu Sans', 'fontsize': 15} # font
    colorscale = 'gray' # colorscale
    # Text formating of ticklabels
    yticklabels_form = '%3.1f' # float with 1 character after decimal point
    # yticklabels_form = '%d' # integer

    ###########################################################################
    # Check inputs and split the input sample
    ###########################################################################

    YY, xc, NC, n_eff, Xk, XX = pawn_split_sample(X, Y, n) # this function
    # checks inputs X, Y and n

    Nx = X.shape
    N = Nx[0]
    M = Nx[1]

    ###########################################################################
    # Check optional inputs for plotting
    ###########################################################################

    if not isinstance(n_col, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"n_col" must be scalar and integer.')
    if n_col < 0:
        raise ValueError('"n_col" must be positive.')
    if not isinstance(Y_Label, str):
        raise ValueError('"Y_Label" must be a string.')
    if not isinstance(cbar, bool):
        raise ValueError('"cbar" must be scalar and boolean.')
    if not labelinput:
        labelinput = [np.nan]*M
        for i in range(M):
            labelinput[i] = 'X' + str(i+1)
    else:
        if not isinstance(labelinput, list):
            raise ValueError('"labelinput" must be a list with M elements.')
        if not all(isinstance(i, str) for i in labelinput):
            raise ValueError('Elements in "labelinput" must be strings.')
        if len(labelinput) != M:
            raise ValueError('"labelinput" must have M elements.')

    ###########################################################################
    # Compute PMFs
    ###########################################################################

    # Set points at which the CDFs will be evaluated:
    YF = np.unique(Y)

    # Compute conditional PMFs
    # (bootstrapping is not used to assess conditional PMFs):
    fC = [np.nan] * M
    for i in range(M): # loop over inputs
        fC[i] = [np.nan] * n_eff[i]
        for k in range(n_eff[i]): # loop over conditioning intervals
            # fC[i][k] = np.unique(YY[i][k], return_counts=True)[1]/len(YY[i][k])
            unique_vals, counts = np.unique(YY[i][k], return_counts=True)
            count_dict = dict(zip(unique_vals, counts))
            counts = np.array([count_dict.get(val, 0) for val in YF])
            fC[i][k] = counts / len(YY[i][k])

    # Initialize unconditional PMFs:
    fU = [np.nan] * M

    # M unconditional PMFs are computed (one for each input), so that for
    # each input the conditional and unconditional CDFs are computed using the
    # same number of data points (when the number of conditioning intervals
    # n_eff[i] varies across the inputs, so does the shape of the conditional
    # outputs YY[i]).

    # Determine the sample size for the unconditional output NU:
    NU = [int(np.min(i)) for i in NC] # numpy.ndarray(M,)
    # NU is equal to the sample size of the conditional outputs NC, or its
    # minimum value across the conditioning intervals when the sample size
    # varies across conditioning intervals as may happen when values of an
    # input are repeated several times (more details on this in the Note in the
    #  help of the function).

    # To reduce the computational time (the calculation of empirical CDF is
    # costly), the unconditional CDF is computed only once for all inputs that
    # have the same value of NU[i].
    NU_unique = np.unique(NU)
    N_compute = len(NU_unique) # number of unconditional CDFs that will be computed

    for kk in range(N_compute): # loop over the sizes of the unconditional output

        # Extract an unconditional sample of size NU_unique[kk] by drawing data
        # points from the full sample Y without replacement
        idx = np.random.choice(np.arange(0, N, 1), size=(NU_unique[kk], ),
                               replace='False')
        # Compute unconditional output PMF:
        # FUkk = empiricalcdf(Y[idx], YF)
        unique_vals, counts = np.unique(Y[idx], return_counts=True)
        count_dict = dict(zip(unique_vals, counts))
        counts = np.array([count_dict.get(val, 0) for val in YF])
        fUkk = counts / len(Y[idx])
        # Associate the fUkk to all inputs that require an unconditional output
        # of size NU_unique[kk]:
        idx_input = np.where([i == NU_unique[kk] for i in NU])[0]
        for j in range(len(idx_input)):
            fU[idx_input[j]] = fUkk

    ###########################################################################
    # Plot
    ###########################################################################
    n_col = min(n_col, M) # n_col <= M
    n_row = int(np.ceil(M/n_col))

    plt.figure()

    for i in range(M): # loop over inputs

        # Prepare color and legend
        cmap = mpl.cm.get_cmap(colorscale, n_eff[i]+1) # return colormap,
        # (n+1) so that the last line is not white
        # Make sure that subsample centers are ordered:
        iii = np.argsort(xc[i])
        ccc = np.sort(xc[i])

        plt.subplot(n_row, n_col, i+1)
        ax = plt.gca()

        if cbar: # plot dummy mappable to generate the colorbar
            Map = plt.imshow(np.array([[0, 1]]), cmap=cmap)
            plt.cla() # clear axes (do not display the dummy map)

        # Plot a horizontal dashed line at F=1:
        plt.plot(YF, np.ones(YF.shape), '--k')

        # Plot conditional CDFs in gray scale:
        for k in range(n_eff[i]):
            plt.plot(YF, fC[i][iii[k]], color=cmap(k), linewidth=2)
        plt.xticks(**pltfont); plt.yticks(**pltfont)
        plt.xlabel(Y_Label, **pltfont)

        # Plot unconditional CDF in red:
        plt.plot(YF, fU[i], 'r', linewidth=3)

        plt.xlim([min(YF), max(YF)]); plt.ylim([-0.02, 1.02])

        if cbar: # Add colorbar to the left
             cb_ticks = [' '] * n_eff[i]
             for k in range(n_eff[i]):
                 cb_ticks[k] = yticklabels_form % ccc[k]
             # Add colorbar (do not display the white color by adjuting the
             # input argument 'boundaries')
             cb = plt.colorbar(Map, ax=ax,
                               boundaries=np.linspace(0, 1-1/((n_eff[i]+1)),
                                                      n_eff[i]+1))
             cb.set_label(labelinput[i], **pltfont)
             cb.Fontname = pltfont['fontname']
             # set tick labels at the center of each color:
             cb.set_ticks(np.linspace(1/(2*(n_eff[i]+1)), 1-3/(2*(n_eff[i]+1)),
                                      n_eff[i]))
             cb.set_ticklabels(cb_ticks)
             cb.ax.tick_params(labelsize=pltfont['fontsize'])
             # Map.set_clim(0,1-1/(n+1))
             ax.set_aspect('auto') # Ensure that axes do not shrink

    return YF, fU, fC, xc


def pawn_plot_ks(YF, FU, FC, xc, n_col=5, X_Labels='',
                 output_condition=allrange, par=[]):

    """ This function computes and plots the Kolmogorov-Smirnov (KS) statistic
    between conditional and unconditional output CDFs for each input and each
    conditioning interval.

    Usage:
        KS_all = pawn_plot_ks(YF, FU, FC, xc, n_col=5, X_Labels='')

    Input:
         YF = values of Y at which the CDFs FU and FC      - numpy.ndarray(P, )
                 are given
         FU = values of the empirical unconditional        - list(M elements)
                 output CDFs. FU[i] is a numpy.ndarray(P, )
                 (see the Note below for further
                 explanation)
         FC = values of the empirical conditional          - list(M elements)
                 output CDFs for each input and each
                 conditioning interval.
                 FC[i] is a list of n_eff[i] CDFs
                 conditional to the i-th input.
                 FC[i][k] is obtained by fixing the i-th
                 input to its k-th conditioning interval
                 (while the other inputs vary freely),
                 and it is a np.ndarray of shape (P, ).
                 (see the Note below for further
                 explanation)
         xc = subsamples' centers (i.e. mean value of     - list(M elements)
                Xi over each conditioning interval)
                xc[i] is a np.ndarray(n_eff[i],) and
                contains the centers for the n_eff[i]
                conditioning intervals for the i-th input.

    Note: YF, FU, FC and xc are computed using the function PAWN.pawn_plot_cdf

    Optional input:
      n_col = number of panels per row in the plot        - integer
                 (default: min(5, M))
    X_Label = label for the x-axis (input name)           - list (M elements)
                 (default: ['X1','X2',...,XM'])

    Output:
     KS_all = KS-statistic calculated between conditional - list(M elements)
         and unconditional output for the M inputs and
         the n_eff conditioning intervals.
         KS[i] contains the KS values for the i-th input
         and the n_eff[i] conditioning intervals, and it
         is a numpy.ndarray of shape (n_eff[i], ).

    --------------------------------------------------------------------------
    ADVANCED USAGE
    for Regional-Response Global Sensitivity Analysis:
    -------------------------------------------------------------------------
    Usage:

    KS_all = pawn_plot_ks(YF, FU, FC, xc, n_col=5, X_Labels='',
                          output_condition=allrange, par=[])

    See the help of PAWN.pawn_indices for information on the optional inputs
    "output_condition" and "par".

    Example:

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    from safepython.sampling import AAT_sampling
    from safepython.model_execution import model_execution
    from safepython import PAWN
    from safepython.plot_functions import boxplot1
    from safepython.ishigami_homma import ishigami_homma_function

    # Create a generic input-output sample:
    N = 1000 # number of samples
    M = 3 # number of inputs
    xmin = -np.pi; xmax = np.pi
    X = AAT_sampling('lhs', M, st.uniform, [xmin, xmax - xmin], N);
    Y = model_execution(ishigami_homma_function, X)

    # Calculate CDFs:
    n = 10 # number of conditioning intervals
    YF, FU, FC, xc = PAWN.pawn_plot_cdf(X, Y, n)

    # Calculate and plot KS statistics:
    KS_all = PAWN.pawn_plot_ks(YF, FU, FC, xc)

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info"""

    # Options for the graphic
    pltfont = {'fontname': 'DejaVu Sans', 'fontsize': 15} # font
    colorscale = 'gray' # colorscale for the markers
    ms = 7 # size of markers

    ###########################################################################
    # Check inputs and calculate KS-statistic
    ###########################################################################
    KS_all = pawn_ks(YF, FU, FC, output_condition, par)# this function
    # checks inputs F, FU, FC, output_condition and par

    M = len(KS_all)
     ###########################################################################
    # Check optional inputs for plotting
    ###########################################################################

    if not isinstance(n_col, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"n_col" must be scalar and integer.')
    if n_col < 0:
        raise ValueError('"n_col" must be positive.')

    if not X_Labels:
        X_Labels = [np.nan]*M
        for i in range(M):
            X_Labels[i] = 'X' + str(i+1)
    else:
        if not isinstance(X_Labels, list):
            raise ValueError('"X_Labels" must be a list with M elements.')
        if not all(isinstance(i, str) for i in X_Labels):
            raise ValueError('Elements in "X_Labels" must be strings.')
        if len(X_Labels) != M:
            raise ValueError('"X_Labels" must have M elements.')

    ###########################################################################
    # Plot
    ###########################################################################
    n_col = min(n_col, M) # n_col <= M
    n_row = int(np.ceil(M/n_col))

    plt.figure()

    for i in range(M): # loop over inputs

        ni = len(KS_all[i]) # number of conditioning intervals for input i
        # plot KS values as coloured circles on a gray scale:
        col = mpl.cm.get_cmap(colorscale, ni)

        # Make sure that subsample centers are ordered:
        iii = np.argsort(xc[i])
        ccc = np.sort(xc[i])
        kkk = KS_all[i][iii]

        plt.subplot(n_row, n_col, i+1)
        # Plot black line:
        plt.plot(ccc, kkk, '-k')

        # plot KS values as circles:
        for k in range(ni):# loop over conditioning intervals
            plt.plot(ccc[k], kkk[k], 'ok', markerfacecolor=col(k), markersize=ms)

        plt.xticks(**pltfont); plt.yticks(**pltfont)
        plt.xlabel(X_Labels[i], **pltfont)
        plt.ylabel('KS', **pltfont)
        plt.xlim([ccc[0]-(ccc[1]-ccc[0])/2, ccc[-1]+(ccc[1]-ccc[0])/2])
        plt.ylim([0, 1])

    return KS_all
