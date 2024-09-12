# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Media transformations for accounting for lagging or media effects."""

import functools
from typing import Union
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.integrate as jintegrate
import scipy.integrate as integrate
import numpy as np
import math
import numpyro
import numpyro.distributions as dist


@functools.partial(jax.jit, static_argnums=[0, 1])
def calculate_seasonality(
    number_periods: int,
    degrees: int,
    gamma_seasonality: Union[int, float, jnp.ndarray],
    frequency: int = 52,
) -> jnp.ndarray:
  """Calculates cyclic variation seasonality using Fourier terms.

  For detailed info check:
    https://en.wikipedia.org/wiki/Seasonality#Modeling

  Args:
    number_periods: Number of seasonal periods in the data. Eg. for 1 year of
      seasonal data it will be 52, for 3 years of the same kind 156.
    degrees: Number of degrees to use. Must be greater or equal than 1.
    gamma_seasonality: Factor to multiply to each degree calculation. Shape must
      be aligned with the number of degrees.
    frequency: Frequency of the seasonality being computed. By default is 52 for
      weekly data (52 weeks in a year).

  Returns:
    An array with the seasonality values.
  """

  seasonality_range = jnp.expand_dims(a=jnp.arange(number_periods), axis=-1)
  degrees_range = jnp.arange(1, degrees+1)
  inner_value = seasonality_range * 2 * jnp.pi * degrees_range / frequency
  season_matrix_sin = jnp.sin(inner_value)
  season_matrix_cos = jnp.cos(inner_value)
  season_matrix = jnp.concatenate([
      jnp.expand_dims(a=season_matrix_sin, axis=-1),
      jnp.expand_dims(a=season_matrix_cos, axis=-1)
  ],
                                  axis=-1)
  return (season_matrix * gamma_seasonality).sum(axis=2).sum(axis=1)


# convolution definitions that can enable backwards shift
CONV_SIZE = 25
NUM_WEEKS = 13

#
# wrap callback
#
def create_zeros_host( pad_size, weights ):
    # print( 'create_zeros_host(): BEGIN : ', weights )
    left_pad = pad_size
    left_pad = min(CONV_SIZE - NUM_WEEKS, left_pad)
    left_pad = max(0, left_pad)
    right_pad = CONV_SIZE - NUM_WEEKS - left_pad
    return np.pad( weights, (left_pad, right_pad) )


#
def create_zeros_wrapper( pad_size, weights ):
    result_shape = jax.ShapeDtypeStruct( (CONV_SIZE,), jnp.float32 )
    return jax.pure_callback( create_zeros_host, result_shape, pad_size, weights )


# Define the forward pass
# weights are of dim 13, overall convolution array of len 21
def custom_pad_fwd( pad_size, w ):
    p_left  = pad_size
    p_right = CONV_SIZE - NUM_WEEKS - p_left
    # delegate to user space callback
    y = create_zeros_wrapper( pad_size, w )

    shp = jnp.shape(w)
    # print( 'shape w: ', shp, " pad size: ", pad_size )
    # Return the output and any intermediate values needed for backward pass
    return y, (shp[0], w, p_left, p_right)


#
# Define the backward pass
#
def custom_pad_bwd( res, g ):
    n, w, p_left, p_right = res

    # Extract the gradient for the output (g)
    # Only the middle part (corresponding to the original x) of g contributes to the gradient of x
    # the middle part is 'w'
    grad_x = w

    # Since pad is not an inputs that we differentiate with respect to,
    # we return None for them
    return None, grad_x


# define create_zeros_wrapper as a custom_vjp
create_zeros_wrapper = jax.custom_vjp( create_zeros_wrapper )


# Register the forward and backward passes
create_zeros_wrapper.defvjp( custom_pad_fwd, custom_pad_bwd )



#
# reverse-shifted adstock application
#
@functools.partial( jax.jit )  # , static_argnums=[0,1]
def run_reverse_shift( num_weeks, lag_weight, shift_weeks, colData ):
    # print( "run reverse shift: BEGIN: ", num_weeks )

    # left pad of maximum value indicates no shift
    shift_weeks = CONV_SIZE - NUM_WEEKS - shift_weeks
    shift_weeks = jnp.minimum( (CONV_SIZE - NUM_WEEKS), shift_weeks )
    shift_weeks = jnp.maximum( 0.0, shift_weeks )
    pad = jnp.round( shift_weeks ).astype(int)

    lag_weights_arr = jnp.ones( NUM_WEEKS )
    lag_weights_arr = lag_weights_arr * lag_weight
    weights = jnp.power( lag_weights_arr, jnp.arange(NUM_WEEKS, dtype=jnp.float32) )
    weights = weights / jnp.sum( weights )

    # pad weights left and right as it is a centered convolution
    weights = create_zeros_wrapper( pad, weights )

    return jax.scipy.signal.convolve( colData, weights, mode="same" )


#
# radstock = "reverse adstock"
#
#   the method allows for configured channels to have spend shifted back in time
#   as the adstock weights are applied.
#   -> this is done by specifying a '1' in the enableReverseShift array.  any channel
#      with a 0 in the array is specified as a
#   -> non-configured channels will behave in the normal way; pushing the spend forward only
#
# @functools.partial( jax.jit, static_argnames=['normalise','weeks'] )
#
# @jax.jit
@functools.partial( jax.jit, static_argnames=["weeks"] )
def radstock( data: jnp.ndarray,
              lag_weight: float = .9,
              shift_weeks: float = 2.,
              normalise: bool = True,
              enableReverseShift = None,
              weeks = 13 ) -> jnp.ndarray:
    """Calculates the adstock value of a given array.

    To learn more about advertising lag:
    https://en.wikipedia.org/wiki/Advertising_adstock

    Args:
      data: Input array.
      lag_weight: lag_weight effect of the adstock function. Default is 0.9.
      normalise: Whether to normalise the output value. This normalization will
        divide the output values by (1 / (1 - lag_weight)).

    Returns:
      The adstock output of the input array.
    """

    # print( 'radstock: enable: ', enableReverseShift, "|", type(enableReverseShift) )
    print( 'radstock: data dim: ', jnp.shape(data) )
    my_weeks = int(13)

    #
    jnp_ret = None
    dataT  = data.T

    for colIdx in range( 0, len(dataT) ):
        col_lag_weight = lag_weight[ colIdx ]
        col_shift      = shift_weeks[ colIdx ]
        origSpend  = dataT[ colIdx ]

        branch_res = run_reverse_shift( my_weeks, col_lag_weight, col_shift, origSpend )

        if jnp_ret is None:
            jnp_ret = branch_res
        else:
            jnp_ret = jnp.vstack( (jnp_ret, branch_res) )

    jnp_ret = jnp.array( jnp_ret.T )

    # print( 'radstock(): final shape: ', jnp.shape( jnp_ret ) )

    return  jnp_ret


@jax.jit
def adstock(data: jnp.ndarray,
            lag_weight: float = .9,
            normalise: bool = True) -> jnp.ndarray:
  """Calculates the adstock value of a given array.

  To learn more about advertising lag:
  https://en.wikipedia.org/wiki/Advertising_adstock

  Args:
    data: Input array.
    lag_weight: lag_weight effect of the adstock function. Default is 0.9.
    normalise: Whether to normalise the output value. This normalization will
      divide the output values by (1 / (1 - lag_weight)).

  Returns:
    The adstock output of the input array.
  """

  def adstock_internal(prev_adstock: jnp.ndarray,
                       data: jnp.ndarray,
                       lag_weight: float = lag_weight) -> jnp.ndarray:
    adstock_value = prev_adstock * lag_weight + data
    return adstock_value, adstock_value # jax-ndarray

  _, adstock_values = jax.lax.scan(
      f=adstock_internal, init=data[0, ...], xs=data[1:, ...])

  adstock_values = jnp.concatenate([jnp.array([data[0, ...]]), adstock_values])

  return jax.lax.cond(
      normalise,
      lambda adstock_values: adstock_values / (1. / (1 - lag_weight)),
      lambda adstock_values: adstock_values,
      operand=adstock_values)


@jax.jit
def hill(data: jnp.ndarray, half_max_effective_concentration: jnp.ndarray,
         slope: jnp.ndarray) -> jnp.ndarray:
  """Calculates the hill function for a given array of values.

  Refer to the following link for detailed information on this equation:
    https://en.wikipedia.org/wiki/Hill_equation_(biochemistry)

  Args:
    data: Input data.
    half_max_effective_concentration: ec50 value for the hill function.
    slope: Slope of the hill function.

  Returns:
    The hill values for the respective input data.
  """
  save_transform = apply_exponent_safe(
      data=data / half_max_effective_concentration, exponent=-slope)
  return jnp.where(save_transform == 0, 0, 1.0 / (1 + save_transform))


@functools.partial(jax.vmap, in_axes=(1, 1, None), out_axes=1)
def _carryover_convolve(data: jnp.ndarray,
                        weights: jnp.ndarray,
                        number_lags: int) -> jnp.ndarray:
  """Applies the convolution between the data and the weights for the carryover.

  Args:
    data: Input data.
    weights: Window weights for the carryover.
    number_lags: Number of lags the window has.

  Returns:
    The result values from convolving the data and the weights with padding.
  """
  window = jnp.concatenate([jnp.zeros(number_lags - 1), weights])
  return jax.scipy.signal.convolve(data, window, mode="same") / weights.sum()


@functools.partial(jax.jit, static_argnames=("number_lags",))
def carryover(data: jnp.ndarray,
              ad_effect_retention_rate: jnp.ndarray,
              peak_effect_delay: jnp.ndarray,
              number_lags: int = 13) -> jnp.ndarray:
  """Calculates media carryover.

  More details about this function can be found in:
  https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46001.pdf

  Args:
    data: Input data. It is expected that data has either 2 dimensions for
      national models and 3 for geo models.
    ad_effect_retention_rate: Retention rate of the advertisement effect.
      Default is 0.5.
    peak_effect_delay: Delay of the peak effect in the carryover function.
      Default is 1.
    number_lags: Number of lags to include in the carryover calculation. Default
      is 13.

  Returns:
    The carryover values for the given data with the given parameters.
  """
  lags_arange = jnp.expand_dims( jnp.arange(number_lags, dtype=jnp.float32),
                                 axis=-1 )
  convolve_func = _carryover_convolve
  if data.ndim == 3:
    # Since _carryover_convolve is already vmaped in the decorator we only need
    # to vmap it once here to handle the geo level data. We keep the windows bi
    # dimensional also for three dims data and vmap over only the extra data
    # dimension.
    convolve_func = jax.vmap(
        fun=_carryover_convolve, in_axes=(2, None, None), out_axes=2)
  weights = ad_effect_retention_rate**((lags_arange - peak_effect_delay)**2)
  return convolve_func(data, weights, number_lags)

@jax.jit
def apply_exponent_safe(
    data: jnp.ndarray,
    exponent: jnp.ndarray,
    ) -> jnp.ndarray:
  """Applies an exponent to given data in a gradient safe way.

  More info on the double jnp.where can be found:
  https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf

  Args:
    data: Input data to use.
    exponent: Exponent required for the operations.

  Returns:
    The result of the exponent operation with the inputs provided.
  """
  exponent_safe = jnp.where(data == 0, 1, data) ** exponent
  return jnp.where(data == 0, 0, exponent_safe)
