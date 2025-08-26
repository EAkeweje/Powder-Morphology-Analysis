"""
Code for the 'registration' module. Given two input images, find the rotation and 
flip that optimally aligns them in terms of image cross-correlation.

To use the efficient version (item 1 in the above list) you first need to transform
the image to polar coords. 

More details, see 
    https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_rotation.html

Usage:
    https://github.com/jmhb0/o2vae/tree/master/registration
"""
import numpy as np
import skimage
import torch
import torchgeometry as tgm
import torchvision.transforms.functional as T_f


class PolarTranformBatch:
    """
    Take the polar transform (or log-polar transform) of a batch of pytorch images
    efficiently. This is mostly the same as skimage.transform.warp_polar except its
    done batch-wise and also with less erorr handling, and fewer features.
    This exists as a class because we want to build a coordinate
    mapping matrix once, and then use it many times during a training epoch.
    """

    def __init__(
        self,
        image_shape,
        center=None,
        radius=None,
        output_shape=None,
        scaling="linear",
        order=1,
        cval=0,
        mode_grid_sample_padding="zeros",
        mode_grid_sample_interpolate="bilinear",
        mode_padding_ndimage="constant",
    ):
        """
        order (int): interpolation order 0 nearest, 1 bilinear, 3 quadratic.
        cval (float): pad value

        mode_grid_sample_interpolate: how to interpolate between mapped image points in the mapped grid,
            see torch.nn.functional.grid_sample, param `mode`
        mode_padding_ndimage: how to deal with boundaries for the `warp_single_ndimage` function, which is not
            the batch version that we use. See `scipy.ndimage.map_coordinates`.
        """
        assert len(image_shape) == 2
        self.order = order
        self.cval = cval
        self.mode_grid_sample_padding = mode_grid_sample_padding
        self.mode_grid_sample_interpolate = mode_grid_sample_interpolate
        self.mode_padding_ndimage = mode_padding_ndimage

        ydim, xdim = image_shape

        if center is None:
            center = (torch.Tensor([ydim, xdim]) / 2) - 0.5

        if radius is None:
            w, h = ydim / 2, xdim / 2
            radius = (w**2 + h**2) ** 0.5

        if output_shape is None:
            height = 360
            width = int(np.ceil(radius))
            output_shape = (height, width)
        else:
            output_shape = safe_as_int(output_shape)
            height = output_shape[0]
            width = output_shape[1]

        if scaling == "linear":
            k_radius = width / radius
            self.map_func = _linear_polar_mapping
        elif scaling == "log":
            k_radius = width / np.log(radius)
            self.map_func = _log_polar_mapping
        else:
            raise ValueError("Scaling value must be in {'linear', 'log'}")

        k_angle = height / (2 * torch.pi)
        self.warp_args = {
            "k_angle": k_angle,
            "k_radius": k_radius,
            "center": center.numpy(),
        }

        inverse_map = self.map_func

        def coord_map(*args):
            return inverse_map(*args, **self.warp_args)

        # coordinate map for warping in ndimage format
        self.coords = torch.Tensor(
            warp_coords(coord_map, output_shape)
        )

        # the same coordinate map but for torch: the order of channels is different and images
        # are normalized to the range [-1,1] to suit the function `torch.nn.functional.grid_sample`
        self.coords_torch_format = self.coords.clone().moveaxis(
            0, 2
        )  # move to shape (2,Y,X) to (Y,X,2)
        min0, min1, max0, max1 = 0, 0, image_shape[0], image_shape[1]
        self.coords_torch_format[:, :, 0] = (self.coords_torch_format[:, :, 0] - min0) / (
            max0 - min0
        ) * 2 - 1
        self.coords_torch_format[:, :, 1] = (self.coords_torch_format[:, :, 1] - min1) / (
            max1 - min1
        ) * 2 - 1
        self.coords_torch_format = self.coords_torch_format[
            :, :, [1, 0]
        ]  # grid_sample has the order reversed

        # Pre-filtering not necessary for order 0, 1 interpolation
        self.prefilter = order > 1
        self.ndi_mode = _to_ndimage_mode(self.mode_padding_ndimage)

    def warp_single_ndimage(self, x):
        """
        Test polar transform of a single image using the ndimage library.
        This does not make use of any GPU batch stuff (you can only
        transform one image at a time), so it is a baseline for how this function
        should work.
        """
        assert x.ndim == 2
        if type(x) is torch.Tensor:
            x = x.cpu().numpy()
        warped = ndi.map_coordinates(
            x,
            self.coords,
            prefilter=self.prefilter,
            mode=self.ndi_mode,
            order=self.order,
            cval=self.cval,
        )
        return torch.Tensor(warped)

    def warp_batch(self, x):
        """
        Do polar transform as a batch, which should be fast if using a GPU.
        """
        assert x.ndim == 4
        batch_size = len(x)
        coords_batch = (
            self.coords_torch_format.unsqueeze(0)
            .expand(batch_size, *self.coords_torch_format.shape)
            .to(x.device)
        )
        warped_batch = torch.nn.functional.grid_sample(
            x,
            coords_batch,
            align_corners=True,
            padding_mode=self.mode_grid_sample_padding,
            mode=self.mode_grid_sample_interpolate,
        )
        return warped_batch

    def _verify_warping_funcs_are_the_same(self, x=None, do_plot=1):
        """
        Take `x` with shape (Y,X) if supplied, or on skimage.data.checkboard if `x` is None.
        Run warp_single_ndimage and warp_batch on it. The point is that it
        should be the same.
        """
        if x is None:
            import skimage

            print("No image data provided, using skimage.data.checkerboard()")
            x = torch.Tensor(skimage.data.checkerboard())  # a test image
        print(x.shape, len(x.shape), x.ndim)
        warped_ndimage = self.warp_single_ndimage(x)
        x_batch = (
            x.unsqueeze(0).unsqueeze(0).expand(1, 1, *x.shape)
        )  # put the test image into a batch

        warped_batch = self.warp_batch(x_batch)

        if do_plot:
            f, axs = plt.subplots(1, 2)
            axs[0].imshow(warped_ndimage)
            axs[1].imshow(warped_batch[0, 0])
            return warped_ndimage, warped_batch[0, 0], f
        else:
            return warped_ndimage, warped_batch[0, 0], None

    def _test_runtime(self, n=50, bs=256, ysz=128, xsz=128):
        """
        Simulate running the polar transform n times with batch size bs.
        Do it for the batch version, then the single image version, and
        then batch version on cuda if cuda is available.

        E.g. with defaults, expect the batch version to be 10x faster.
        For a fixed amount of data (the same value n*bs), the batch version
        will be a faster with bigger `bs` and smaller `n`, but the serial
        version should be the same.
        """
        import time

        import skimage
        import torchvision.transforms.functional as T_f

        image = torch.Tensor(data.checkerboard())  # a test image
        image_batch = image.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1, 1)
        image_batch = T_f.resize(image_batch, (ysz, xsz))

        print(f"Testing {n} iterations, batch size {bs}, image dimension ({ysz},{xsz})")
        print(image_batch.shape)

        start = time.time()
        for i in range(n):
            _ = self.warp_batch(image_batch)
        print(f"{time.time()-start:.2f} sec for batch version")

        start = time.time()
        for i in range(n):
            for img in image_batch:
                _ = self.warp_single_ndimage(img[0])
        print(f"{time.time()-start:.2f} sec for serial version")

        if torch.cuda.is_available():
            start = time.time()
            for i in range(n):
                image_batch_ = image_batch.cuda()
                _ = self.warp_batch(image_batch_)
            print(f"{time.time()-start:.2f} sec for cuda batch version")


def _torch_unravel_index_batch(index, shape):
    """
    Modified slighly from
        https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-just-like-in-numpy/12987/3
    """
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = torch.div(index, dim, rounding_mode="trunc")  ## equiv to index // dim
    return tuple(reversed(out))


def _compute_error(CCmax, src_amp, target_amp):
    error = 1 - (CCmax * CCmax.conj()) / (src_amp * target_amp)
    return torch.abs(error) ** 0.5


def _compute_phasediff(CCmax):
    return torch.atan2(CCmax.imag, CCmax.real)


def phase_correlation_2d_batch(
    x, y, normalization=None, upsample_factor=1, return_error=1, space="real"
):
    """
    upsample_factor: NotImplemented for anything other than 1.
    """
    device = x.device
    reference_image, moving_image = x, y
    assert reference_image.ndim == moving_image.ndim == 4, "must be shape (bs,1,Y,X)"
    bs, c, ydim, xdim = reference_image.shape
    assert c == 1, "input must be grayscale, having shape (bs,1,Y,X)"

    # take ffts and cross correlations
    src_freq = torch.fft.fft2(x)
    target_freq = torch.fft.fft2(y)
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if normalization == "phase":
        eps = torch.finfo(image_product.real.dtype).eps
        image_product /= np.maximum(np.abs(image_product), 100 * eps)
    elif normalization is not None:
        raise ValueError()
    cross_correlation = torch.fft.ifft2(image_product)

    # error stuf
    if return_error:
        src_amp = torch.real(src_freq * src_freq.conj()).view(bs, -1).sum(1)
        target_amp = torch.real(target_freq * target_freq.conj()).view(bs, -1).sum(1)

    # find the max correlation
    cross_correlation_abs = torch.abs(cross_correlation)
    cross_correlation_abs = cross_correlation_abs.view(bs, ydim, xdim)

    # ids and values of the peak of the cross corr thing
    max_idxs_batch = torch.argmax(cross_correlation_abs.view(bs, ydim * xdim), 1)
    CCmax = cross_correlation.view(bs, -1)[np.arange(bs), max_idxs_batch]
    ymax_batch, xmax_batch = _torch_unravel_index_batch(max_idxs_batch, (ydim, xdim))

    # work out the shift - case depends on whether it's above or below the midpoint
    midpoints = torch.Tensor([int(axis_size / 2) for axis_size in shape[-2:]]).to(device)
    shifts = torch.stack((ymax_batch, xmax_batch), 1).to(device).float()
    shifts[shifts > midpoints] -= (
        torch.Tensor([[ydim, xdim]])
        .to(device)
        .float()
        .expand((bs, 2))[shifts > midpoints]
    )

    if upsample_factor == 1:
        pass
    else:
        raise NotImplementedError()

    return shifts, _compute_error(CCmax, src_amp, target_amp), _compute_phasediff(CCmax)

# Adaptation from skimage
def _linear_polar_mapping(output_coords, k_angle, k_radius, center):
    """Inverse mapping function to convert from cartesian to polar coordinates

    Parameters
    ----------
    output_coords : (M, 2) ndarray
        Array of `(col, row)` coordinates in the output image.
    k_angle : float
        Scaling factor that relates the intended number of rows in the output
        image to angle: ``k_angle = nrows / (2 * np.pi)``.
    k_radius : float
        Scaling factor that relates the radius of the circle bounding the
        area to be transformed to the intended number of columns in the output
        image: ``k_radius = ncols / radius``.
    center : tuple (row, col)
        Coordinates that represent the center of the circle that bounds the
        area to be transformed in an input image.

    Returns
    -------
    coords : (M, 2) ndarray
        Array of `(col, row)` coordinates in the input image that
        correspond to the `output_coords` given as input.
    """
    angle = output_coords[:, 1] / k_angle
    rr = ((output_coords[:, 0] / k_radius) * np.sin(angle)) + center[0]
    cc = ((output_coords[:, 0] / k_radius) * np.cos(angle)) + center[1]
    coords = np.column_stack((cc, rr))
    return coords

def _log_polar_mapping(output_coords, k_angle, k_radius, center):
    """Inverse mapping function to convert from cartesian to polar coordinates

    Parameters
    ----------
    output_coords : (M, 2) ndarray
        Array of `(col, row)` coordinates in the output image.
    k_angle : float
        Scaling factor that relates the intended number of rows in the output
        image to angle: ``k_angle = nrows / (2 * np.pi)``.
    k_radius : float
        Scaling factor that relates the radius of the circle bounding the
        area to be transformed to the intended number of columns in the output
        image: ``k_radius = width / np.log(radius)``.
    center : 2-tuple
        `(row, col)` coordinates that represent the center of the circle that bounds the
        area to be transformed in an input image.

    Returns
    -------
    coords : ndarray, shape (M, 2)
        Array of `(col, row)` coordinates in the input image that
        correspond to the `output_coords` given as input.
    """
    angle = output_coords[:, 1] / k_angle
    rr = ((np.exp(output_coords[:, 0] / k_radius)) * np.sin(angle)) + center[0]
    cc = ((np.exp(output_coords[:, 0] / k_radius)) * np.cos(angle)) + center[1]
    coords = np.column_stack((cc, rr))
    return coords

def warp_coords(coord_map, shape, dtype=np.float64):
    """Build the source coordinates for the output of a 2-D image warp.

    Parameters
    ----------
    coord_map : callable like GeometricTransform.inverse
        Return input coordinates for given output coordinates.
        Coordinates are in the shape (P, 2), where P is the number
        of coordinates and each element is a ``(row, col)`` pair.
    shape : tuple
        Shape of output image ``(rows, cols[, bands])``.
    dtype : np.dtype or string
        dtype for return value (sane choices: float32 or float64).

    Returns
    -------
    coords : (ndim, rows, cols[, bands]) array of dtype `dtype`
            Coordinates for `scipy.ndimage.map_coordinates`, that will yield
            an image of shape (orows, ocols, bands) by drawing from source
            points according to the `coord_transform_fn`.

    Notes
    -----

    This is a lower-level routine that produces the source coordinates for 2-D
    images used by `warp()`.

    It is provided separately from `warp` to give additional flexibility to
    users who would like, for example, to re-use a particular coordinate
    mapping, to use specific dtypes at various points along the the
    image-warping process, or to implement different post-processing logic
    than `warp` performs after the call to `ndi.map_coordinates`.


    Examples
    --------
    Produce a coordinate map that shifts an image up and to the right:

    >>> from skimage import data
    >>> from scipy.ndimage import map_coordinates
    >>>
    >>> def shift_up10_left20(xy):
    ...     return xy - np.array([-20, 10])[None, :]
    >>>
    >>> image = data.astronaut().astype(np.float32)
    >>> coords = warp_coords(shift_up10_left20, image.shape)
    >>> warped_image = map_coordinates(image, coords)

    """
    shape = safe_as_int(shape)
    rows, cols = shape[0], shape[1]
    coords_shape = [len(shape), rows, cols]
    if len(shape) == 3:
        coords_shape.append(shape[2])
    coords = np.empty(coords_shape, dtype=dtype)

    # Reshape grid coordinates into a (P, 2) array of (row, col) pairs
    tf_coords = np.indices((cols, rows), dtype=dtype).reshape(2, -1).T

    # Map each (row, col) pair to the source image according to
    # the user-provided mapping
    tf_coords = coord_map(tf_coords)

    # Reshape back to a (2, M, N) coordinate grid
    tf_coords = tf_coords.T.reshape((-1, cols, rows)).swapaxes(1, 2)

    # Place the y-coordinate mapping
    _stackcopy(coords[1, ...], tf_coords[0, ...])

    # Place the x-coordinate mapping
    _stackcopy(coords[0, ...], tf_coords[1, ...])

    if len(shape) == 3:
        coords[2, ...] = range(shape[2])

    return coords

def _stackcopy(a, b):
    """Copy b into each color layer of a, such that::

      a[:,:,0] = a[:,:,1] = ... = b

    Parameters
    ----------
    a : (M, N) or (M, N, P) ndarray
        Target array.
    b : (M, N)
        Source array.

    Notes
    -----
    Color images are stored as an ``(M, N, 3)`` or ``(M, N, 4)`` arrays.

    """
    if a.ndim == 3:
        a[:] = b[:, :, np.newaxis]
    else:
        a[:] = b


def safe_as_int(val, atol=1e-3):
    """
    Attempt to safely cast values to integer format.

    Parameters
    ----------
    val : scalar or iterable of scalars
        Number or container of numbers which are intended to be interpreted as
        integers, e.g., for indexing purposes, but which may not carry integer
        type.
    atol : float
        Absolute tolerance away from nearest integer to consider values in
        ``val`` functionally integers.

    Returns
    -------
    val_int : NumPy scalar or ndarray of dtype `np.int64`
        Returns the input value(s) coerced to dtype `np.int64` assuming all
        were within ``atol`` of the nearest integer.

    Notes
    -----
    This operation calculates ``val`` modulo 1, which returns the mantissa of
    all values. Then all mantissas greater than 0.5 are subtracted from one.
    Finally, the absolute tolerance from zero is calculated. If it is less
    than ``atol`` for all value(s) in ``val``, they are rounded and returned
    in an integer array. Or, if ``val`` was a scalar, a NumPy scalar type is
    returned.

    If any value(s) are outside the specified tolerance, an informative error
    is raised.

    Examples
    --------
    >>> safe_as_int(7.0)
    7

    >>> safe_as_int([9, 4, 2.9999999999])
    array([9, 4, 3])

    >>> safe_as_int(53.1)
    Traceback (most recent call last):
        ...
    ValueError: Integer argument required but received 53.1, check inputs.

    >>> safe_as_int(53.01, atol=0.01)
    53

    """
    mod = np.asarray(val) % 1  # Extract mantissa

    # Check for and subtract any mod values > 0.5 from 1
    if mod.ndim == 0:  # Scalar input, cannot be indexed
        if mod > 0.5:
            mod = 1 - mod
    else:  # Iterable input, now ndarray
        mod[mod > 0.5] = 1 - mod[mod > 0.5]  # Test on each side of nearest int

    try:
        np.testing.assert_allclose(mod, 0, atol=atol)
    except AssertionError:
        raise ValueError(
            f'Integer argument required but received ' f'{val}, check inputs.'
        )

    return np.round(val).astype(np.int64)

def _to_ndimage_mode(mode):
    """Convert from `numpy.pad` mode name to the corresponding ndimage mode."""
    mode_translation_dict = dict(
        constant='constant',
        edge='nearest',
        symmetric='reflect',
        reflect='mirror',
        wrap='wrap',
    )
    if mode not in mode_translation_dict:
        raise ValueError(
            f"Unknown mode: '{mode}', or cannot translate mode. The "
            f"mode should be one of 'constant', 'edge', 'symmetric', "
            f"'reflect', or 'wrap'. See the documentation of numpy.pad for "
            f"more info."
        )
    grid_modes = {'constant': 'grid-constant', 'wrap': 'grid-wrap'}
    return grid_modes.get(mode_translation_dict[mode], mode_translation_dict[mode])