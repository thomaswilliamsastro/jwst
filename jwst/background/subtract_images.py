import numpy as np

from stdatamodels.jwst import datamodels

from astropy.stats import sigma_clip
from astropy.nddata.bitmask import interpret_bit_flags, bitfield_to_boolean_mask

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def subtract(model1,
             model2,
             do_sub_background_matching=False,
             sigma=3,
             maxiters=None,
             ):
    """
    Subtract one data model from another, and include updated DQ in output.

    Parameters
    ----------
    model1 : ImageModel or IFUImageModel
        Input data model on which subtraction will be performed

    model2 : ImageModel or IFUImageModel
        Input data model that will be subtracted from the first model

    do_sub_background_matching : bool
        Whether to sigma-clip the data after subtraction, to get
        a background level of ~0

    sigma : float
        Number of standard deviations to use for both the lower
        and upper clipping limits.

    maxiters : int or None
        Maximum number of sigma-clipping iterations to perform

    Returns
    -------
    output : ImageModel or IFUImageModel
        Subtracted data model
    """
    # Create the output model as a copy of the first input
    output = model1.copy()

    # Subtract the SCI arrays
    output.data = model1.data - model2.data

    # Combine the ERR arrays in quadrature
    # NOTE: currently stubbed out until ERR handling is decided
    # output.err = np.sqrt(model1.err**2 + model2.err**2)

    # Combine the DQ flag arrays using bitwise OR
    output.dq = np.bitwise_or(model1.dq, model2.dq)

    # If we're sigma-clipping, do that here
    if do_sub_background_matching:
        # Only use good avg DQ bits
        dq_bits = interpret_bit_flags(bit_flags="~DO_NOT_USE+NON_SCIENCE",
                                      flag_name_map=datamodels.dqflags.pixel,
                                      )

        dq_bit_mask = bitfield_to_boolean_mask(
            output.dq.astype(np.uint8), dq_bits, good_mask_value=0, dtype=np.uint8
        )

        data_clip = sigma_clip(output.data[dq_bit_mask == 0], sigma=sigma, maxiters=maxiters)
        data_mean = data_clip[np.isfinite(data_clip)].mean()

        output.data -= data_mean

    # Return the subtracted model
    return output
