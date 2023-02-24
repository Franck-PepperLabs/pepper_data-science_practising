from typing import *
import math
import numpy as np
import PIL.Image as pillow
from PIL.Image import Image
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from scipy.signal import convolve2d, correlate2d



def load_img(filename, silent=False, show=False):
    # Load the image
    img = pillow.open(filename)

    # Get the values of all pixels as a matrix
    imx = np.array(img)

    if not silent:
        print(filename)

        # Get the size of the image (in pixels)
        w, h = img.size
        print(f"Width: {w} px, height: {h} px")

        # Get the quantization mode of the image
        print(f"Pixel format: {img.mode}")

        # Get the value of a specific pixel
        px_value = img.getpixel((20,100))
        print(f"Value of the pixel located at (20, 100): {px_value}")

        # Get the size of the pixel matrix
        print(f"Size of the pixel matrix: {imx.shape}")

    # Display the image
    if show:
        plt.suptitle(filename, fontname="Verdana", fontweight="bold", fontsize=15)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    return img, imx


def load_and_analyse_img(filename, density=True, cumulative=False):
    print(filename)

    # Load the image
    img = pillow.open(filename)

    # Get the size of the image (in pixels)
    w, h = img.size
    print(f"Width: {w} px, height: {h} px")

    # Get the quantization mode of the image
    # See : https://pillow.readthedocs.io/en/stable/handbook/concepts.html
    #       #concept-modes
    print(f"Pixel format: {img.mode}")

    # Get the value of a specific pixel
    px_value = img.getpixel((20, 100))
    print(f"Value of the pixel located at (20, 100): {px_value}")

    # Get the values of all pixels as a matrix
    img_mx = np.array(img)

    # Get the size of the pixel matrix
    print(f"Size of the pixel matrix: {img_mx.shape}")

    # Create a figure to display the image and histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    fig.suptitle(filename, fontname="Verdana", fontweight="bold", fontsize=15)

    # Display the image
    ax1.imshow(img)
    ax1.axis('off')

    # Generate and display the histogram
    # n, bins, patches = 
    ax2.hist(
        img_mx.flatten(),
        bins=range(256),
        density=density,
        cumulative=cumulative
    )
    ax2.set_xlim(0, 255)
    #ax2.set_ylim(0, 1)
    ax2.set_xticks(np.arange(0, 256, 25))
    #ax2.set_yticks(np.arange(0, 1.2, 0.2))
    ax2.grid(True, which='both')
    plt.show()


def R(imx):
    if imx.ndim < 3:
        return imx
    return imx[..., 0]


def G(imx):
    if imx.ndim < 3:
        return imx
    return imx[..., 1]


def B(imx):
    if imx.ndim < 3:
        return imx
    return imx[..., 2]


def A(imx):
    if imx.ndim < 3:
        return imx
    # retourner une matrice de 255 de dimension les deux premières de rgba_matrix
    if imx.shape[2] < 4:
        return np.full(imx.shape[:2], 255)
    return imx[..., 3]


def RGBA(imx):
    if imx.ndim < 3:
        return imx, imx, imx, np.full(imx.shape[:2], 255)
    return (
        imx[..., 0],
        imx[..., 1],
        imx[..., 2],
        imx[..., 3] if imx.shape[2] == 4 else np.full(imx.shape[:2], 255)
    )


def flatten_imx(imx: np.ndarray) -> np.ndarray:
    """Flatten an RGBA image into a 2D numpy array

    The flattened image will have one row for each pixel and the following columns:
    - x coordinate
    - y coordinate
    - red channel intensity
    - green channel intensity
    - blue channel intensity
    - alpha channel intensity

    Parameters
    ----------
    imx : np.ndarray
        The input RGBA image

    Returns
    -------
    np.ndarray
        The flattened image

    """
    width, height = imx.shape[:2]
    x, y = np.meshgrid(range(width), range(height))
    r, g, b, a = RGBA(imx)
    _x, _y, _r, _g, _b, _a = [x.flatten() for x in (x, y, r, g, b, a)]
    return np.column_stack((_x, _y, _r, _g, _b, _a))


def unflatten_imx(flat_imx: np.ndarray) -> np.ndarray:
    """Unflatten the flattened image matrix `flat_imx` back to its original shape.

    Args:
    flat_imx: The flattened image matrix, with shape (N, 6) where N is the total number of pixels
              in the original image and the columns represent x, y, R, G, B, A values.
    
    Returns:
    np.ndarray: The original image matrix in shape (width, height, 4), where width and height
                can be determined from the x and y columns in `flat_imx`.
    """
    x = flat_imx[:, 0]
    y = flat_imx[:, 1]
    width = int(x.max() + 1)
    height = int(y.max() + 1)
    imx = flat_imx[:, 2:].reshape(width, height, 4)
    return imx


def show_rgb_hists(
    flat_imx: np.ndarray,
    kde: bool = False,
    cumulative: bool = False
) -> None:
    """
    Show histograms of R, G, and B values from a flattened image array.

    Args:
    flat_imx:
        a flattened array of shape (N, 6) representing an image,
        where N is the number of pixels, and columns 2, 3, and 4
        represent R, G, and B values, respectively.
    kde:
        whether to show a kernel density estimate of the histograms.
    cumulative:
        whether to show cumulative histograms.
    """
    common_args = {
        "binwidth": 1,
        #"discrete": True,
        "alpha": .25,
        #"fill": True,
        "element": "step",
        "kde": kde,
        "cumulative": cumulative
    }

    sns.histplot(flat_imx[:, 2], label='R', color='red', **common_args)
    sns.histplot(flat_imx[:, 3], label='G', color='green', **common_args)
    sns.histplot(flat_imx[:, 4], label='B', color='blue', **common_args)

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()


def show_rgb_pillow_hists(
    img: Image,
    mask: Image = None,
    extrema: Tuple[int, int] = None
) -> None:
    hist = np.array(img.histogram(mask=mask, extrema=extrema))
    x = np.arange(256)
    r = hist[:256]
    g = hist[256:512]
    b = hist[512:]

    plt.bar(x, r, color='red', alpha=.25, label='R', width=1)
    plt.step(x, r, where='mid', color='red')
    plt.bar(x, g, color='green', alpha=.25, label='G', width=1)
    plt.step(x, g, where='mid', color='green')
    plt.bar(x, b, color='blue', alpha=.25, label='B', width=1)
    plt.step(x, b, where='mid', color='blue')

    # Adding a legend
    plt.legend()

    # Showing the plot
    plt.show()


def to_gray(rgb_matrix):
    return ((
        R(rgb_matrix).astype(np.uint16)
        + G(rgb_matrix).astype(np.uint16)
        + B(rgb_matrix).astype(np.uint16)
    ) / 3).astype(np.uint8)


def get_min_max_coords(x: int, y: int, width: int, height: int) -> Tuple[int, int, int, int]:
    """
    Returns the minimum and maximum x and y coordinates based on the center
    x and y and the width and height of the region.

    Parameters
    ----------
    x : int
        The x-coordinate of the center of the region.
    y : int
        The y-coordinate of the center of the region.
    width : int
        The width of the region.
    height : int
        The height of the region.

    Returns
    -------
    Tuple[int, int, int, int]
        A tuple of the minimum and maximum x and y coordinates of the region,
        respectively.
    """
    # decrement width and height by 1 to account for 0-based indexing
    width -= 1
    height -= 1

    # determine the length of the west and east sides of the region
    west_len = width // 2
    east_len = width - west_len

    # determine the length of the north and south sides of the region
    north_len = height // 2
    south_len = height - north_len

    # determine the minimum and maximum x coordinates of the region
    x_min = x - west_len
    x_max = x + east_len

    # determine the minimum and maximum y coordinates of the region
    y_min = y - north_len
    y_max = y + south_len

    return x_min, x_max, y_min, y_max


def to_rbox(
    cbox: Optional[Tuple[int, int, int, int]] = None
) -> Optional[Tuple[int, int, int, int]]:
    """
    Converts a center box `cbox` to a regular box.

    Parameters:
    -----------
    cbox : Optional[Tuple[int, int, int, int]], optional
        The center box to be converted, by default None

    Returns:
    --------
    Optional[Tuple[int, int, int, int]]
        The converted regular box, or None if `cbox` is None.
    """
    if cbox is None:
        return None

    # Unpack the input cbox
    x_c, y_c, w, h = cbox

    # Calculate the width and height of the left and right, top and bottom sides of the bounding box
    w_0 = (w - 1) // 2
    h_0 = (h - 1) // 2
    w_1 = (w - 1) - w_0
    h_1 = (h - 1) - h_0

    # Calculate the coordinates of the top-left and bottom-right corners of the bounding box   
    x_0 = x_c - w_0
    x_1 = x_c + w_1
    y_0 = y_c - h_0
    y_1 = y_c + h_1

    # Return the corner-based bounding box as a tuple
    return x_0, y_0, x_1, y_1


def to_cbox(
    rbox: Tuple[int, int, int, int]=None
) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert a rectangle box representation `rbox` in the form `(x_0, y_0, x_1, y_1)`
    to a centered box representation `cbox` in the form `(x_c, y_c, w, h)`.
    
    If `rbox` is None, return None.

    :param rbox: a rectangle box representation in the form `(x_0, y_0, x_1, y_1)`.
    :type rbox: Tuple[int, int, int, int], optional
    :return: a centered box representation in the form `(x_c, y_c, w, h)`
    :rtype: Optional[Tuple[int, int, int, int]]
    """
    if rbox is None:
        return None

    # Unpack the input rbox
    x_0, y_0, x_1, y_1 = rbox

    # Calculate the center coordinates of the output cbox
    x_c = (x_0 + x_1) // 2
    y_c = (y_0 + y_1) // 2

    # Calculate the width and height of the output cbox
    w = x_1 - x_0 + 1
    h = y_1 - y_0 + 1

    return x_c, y_c, w, h


def get_imx_rbox(imx: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get the rectangular bounding box of an image matrix.

    cpp

    Parameters:
    - imx: numpy ndarray, the input image matrix.

    Returns:
    - A tuple of 4 integers representing the rectangular bounding box 
    (x0, y0, x1, y1).
    """
    h, w = imx.shape[:2]

    # Return the bounding box of the image matrix
    return 0, 0, w - 1, h - 1


def is_part_of_rbox(
    rbox: Tuple[int, int, int, int],
    container_rbox: Tuple[int, int, int, int]
) -> bool:
    """
    Check if a given rectangle (rbox) is completely inside another rectangle
    (container_rbox).
    
    Parameters
    ----------
    rbox : Tuple[int, int, int, int]
        The rectangle to check.
    container_rbox : Tuple[int, int, int, int]
        The rectangle that the other rectangle (rbox) should be completely
        inside of.
    
    Returns
    -------
    bool
        True if the rectangle (rbox) is completely inside the container_rbox,
        False otherwise.
    """
    # Unpack the rectangles coordinates
    x_0, y_0, x_1, y_1 = rbox
    X_0, Y_0, X_1, Y_1 = container_rbox

    # Check if any side of rbox goes outside of container_rbox
    return not (
        x_0 < X_0 or y_0 < Y_0
        or x_1 > X_1 or y_1 > Y_1
    )


def is_out_of_rbox(
    rbox: Tuple[int, int, int, int],
    container_rbox: Tuple[int, int, int, int]
) -> bool:
    """
    Check if a rbox is completely outside of a container rbox.

    Args:
    - rbox (Tuple[int, int, int, int]): The target rbox to check.
    - container_rbox (Tuple[int, int, int, int]): The rbox that will act as the container.

    Returns:
    - bool: True if the target rbox is completely outside of the container rbox.
            False otherwise.

    Example:
    >>> is_out_of_rbox((0, 0, 1, 1), (2, 2, 3, 3))
    True
    >>> is_out_of_rbox((0, 0, 2, 2), (1, 1, 3, 3))
    False
    """
    # Unpack the input rboxes
    x_0, y_0, x_1, y_1 = rbox
    X_0, Y_0, X_1, Y_1 = container_rbox

    # Check if the target rbox is completely outside of the container rbox
    return (
        x_1 < X_0 or y_1 < Y_0
        or x_0 > X_1 or y_0 > Y_1
    )


def part_in_rbox(
    rbox: Tuple[int, int, int, int],
    container_rbox: Tuple[int, int, int, int]
) -> Optional[Tuple[int, int, int, int]]:
    """
    Returns a part of a given bounding box that is inside another bounding box.

    python

    Parameters:
    rbox (Tuple[int, int, int, int]): The bounding box to be cropped.
    container_rbox (Tuple[int, int, int, int]): The bounding box that acts as the container.

    Returns:
    Optional[Tuple[int, int, int, int]]:
    The part of the given bounding box that is inside the container bounding box,
    or None if the given bounding box is completely outside of the container bounding box.
    """
    # Return None if the rbox is completely out of the container_rbox
    if is_out_of_rbox(rbox, container_rbox):
        return None

    # Unpack the rbox and container_rbox
    x_0, y_0, x_1, y_1 = rbox
    X_0, Y_0, X_1, Y_1 = container_rbox

    # Get the minimum and maximum x-coordinates of the part of the rbox inside the container_rbox
    x_min = X_0 if x_0 < X_0 else x_0
    x_max = X_1 if x_1 > X_1 else x_1

    # Get the minimum and maximum y-coordinates of the part of the rbox inside the container_rbox
    y_min = Y_0 if y_0 < Y_0 else y_0
    y_max = Y_1 if y_1 > Y_1 else y_1

    return x_min, y_min, x_max, y_max


def show_imx(imx, cmap=None, vmin=None, vmax=None, extent=None, show_grid=False):
    if cmap is None and imx.ndim == 2:
        cmap = 'gray'

    plt.imshow(imx, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    if show_grid:
        plt.grid(linestyle='-')
    plt.show()


def show_imx_part(imx, part, cmap=None, vmin=None, vmax=None, show_grid=False):
    x_min, x_max, y_min, y_max = get_min_max_coords(*part)
    imx_part = imx[y_min:y_max, x_min:x_max]
    show_imx(
        imx_part, cmap=cmap, vmin=vmin, vmax=vmax,
        extent=(x_min, x_max, y_max, y_min),
        show_grid=show_grid
    )


def new_show_imx(
    imx,
    ax=None,
    title=None,
    cmap=None,
    vmin=None,
    vmax=None,
    interpolation=None,
    extent=None,
    show_grid=True,
    major_grid_step=None,
    major_grid_linestyle='-',
    minor_grid_step=None,
    minor_grid_linestyle=':',
    bg_color='black',
    bg_alpha=1,
    fg_color='white'
):
    x_min, x_max, y_max, y_min = (
        extent if extent is not None
        else (0, imx.shape[1] - 1, imx.shape[0] - 1, 0)
    )
    #print(x_min, x_max, y_max, y_min)
    w = x_max - x_min + 1
    h = y_max - y_min + 1

    cmap = 'gray' if cmap is None and imx.ndim == 2 else cmap
    vmin = 0 if vmin is None else vmin
    vmax = 255 if vmax is None else vmax

    # See : https://stackoverflow.com/questions/9662995/matplotlib-change-title-and-colorbar-text-and-tick-colors
    text_color = fg_color
    tick_color = fg_color
    grid_color = fg_color

    show_it = False
    if ax is None:
        fig, ax = plt.subplots(1, 1)

        # set figure facecolor
        fig.patch.set_facecolor(bg_color)
        fig.patch.set_alpha(bg_alpha)
        show_it = True

    ax.imshow(imx, interpolation=interpolation, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)

    # set tick and ticklabel color
    ax.tick_params(color=tick_color, labelcolor=text_color)

    if show_grid:
        # See https://stackoverflow.com/questions/24943991/change-grid-interval-and-specify-tick-labels-in-matplotlib
        minor_step = (
            minor_grid_step if minor_grid_step is not None
            else math.floor(math.sqrt((w / 25) * (h / 25)))
        )
        if minor_step < 1:
            minor_step = 1
        major_step = (
            major_grid_step if major_grid_step is not None
            else 5 * minor_step
        )
        minor_xticks = np.arange(x_min, x_max + 1, minor_step)
        minor_yticks = np.arange(y_min, y_max + 1, minor_step)
        major_xticks = np.arange(x_min, x_max + 1, major_step)
        major_yticks = np.arange(y_min, y_max + 1, major_step)
        ax.set_xticks(major_xticks)
        ax.set_xticks(minor_xticks, minor=True)
        ax.set_yticks(major_yticks)
        ax.set_yticks(minor_yticks, minor=True)
        ax.grid(which='minor', linestyle=minor_grid_linestyle, color=grid_color, linewidth=.25)
        ax.grid(which='major', linestyle=major_grid_linestyle, color=grid_color, linewidth=.5)
    
    if title is not None:
        ax.set_title(title, color=text_color)
    
    if show_it:
        plt.show()


def new_show_hist(
    img: Image,
    ax: Axes = None,
    title=None,
    normalize=False,
    cumulative=False,
    mask: Image = None,
    # extrema: Tuple[int, int] = None,
    hist_xlabel='Intensity',
    hist_ylabel='Frequency',
    bg_color='black',
    bg_alpha=1,
    fg_color='white'
) -> None:
    w, h = img.size
    rgb = img.mode.startswith('RGB')
    hist = np.array(img.histogram(mask=mask, extrema=(0, 255)))
    x = np.arange(256)

    gr, r, g, b = 4 * (None,)
    if rgb:
        r, g, b = hist[:256], hist[256:512], hist[512:]
    else:
        gr = hist[:256]

    if normalize:
        s = w * h
        if rgb:
            r, g, b = r / s, g / s, b / s
        else:
            gr = gr / s

    if cumulative:
        if rgb:
            r, g, b = np.cumsum(r), np.cumsum(g), np.cumsum(b)
        else:
            gr = np.cumsum(gr)

    # See : https://stackoverflow.com/questions/9662995/matplotlib-change-title-and-colorbar-text-and-tick-colors
    text_color = fg_color
    tick_color = fg_color
    grid_color = fg_color

    if ax is None:
        fig, ax = plt.subplots(1, 1)

        # set figure facecolor
        fig.patch.set_facecolor(bg_color)
        fig.patch.set_alpha(bg_alpha)

    ax.patch.set_facecolor(bg_color)
    ax.patch.set_alpha(bg_alpha)

    if rgb:
        ax.bar(x, r, color='red', alpha=.25, label='Red', width=1)
        ax.step(x, r, where='mid', color='red')
        ax.bar(x, g, color='green', alpha=.25, label='Green', width=1)
        ax.step(x, g, where='mid', color='green')
        ax.bar(x, b, color='blue', alpha=.25, label='Blue', width=1)
        ax.step(x, b, where='mid', color='blue')
    else:
        ax.bar(x, gr, color='gray', alpha=.25, label='Gray', width=1)
        ax.step(x, gr, where='mid', color='gray')    

    # set tick and ticklabel color
    ax.tick_params(color=tick_color, labelcolor=text_color)

    ax.grid(linestyle=':', color=grid_color, linewidth=.25)

    if hist_xlabel is not None:
        plt.xlabel(hist_xlabel, color=text_color)

    if hist_ylabel is not None:
        plt.ylabel(hist_ylabel, color=text_color)

    # Adding a legend
    ax.legend(facecolor=bg_color, labelcolor=text_color)

    if title is None:
        title = 'Histogram'
        if normalize:
            title = 'Normalized ' + title
        if cumulative:
            title = 'Cumulative ' + title
        if rgb:
            title = 'RGB ' + title
        else:
            title = 'Gray ' + title
    ax.set_title(title, color=text_color)

    # Showing the plot
    if ax is None:
        plt.show()


def show_img_and_hist(
    input,
    img_name=None,
    normalize=False,
    cumulative=False,
    bg_color='black',
    bg_alpha=1,
    fg_color='white'
):
    filename, img, imx = 3 * (None,)
    if isinstance(input, str):
        filename = input
        img, imx = load_img(filename, silent=True)
    elif isinstance(input, np.ndarray):
        filename = str(type(input))
        imx = input
        img = pillow.fromarray(input)
    elif isinstance(input, Image):
        filename = str(type(input))
        img = input
        imx = np.array(input)
    else:
        raise TypeError(f"input type is {type(input)} but should be str, ndarray or Image")       

    # Create a figure to display the image and histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # set figure facecolor
    fig.patch.set_facecolor(bg_color)
    fig.patch.set_alpha(bg_alpha)
    fig.suptitle(filename, fontname="Verdana", fontweight="bold", fontsize=15, color=fg_color)
    new_show_imx(
        imx,
        ax=ax1,
        title=img_name,
        cmap=None,
        vmin=None,
        vmax=None,
        extent=None,
        show_grid=True,
        major_grid_step=None,
        major_grid_linestyle='-',
        minor_grid_step=None,
        minor_grid_linestyle=':',
        bg_color=bg_color,
        bg_alpha=bg_alpha,
        fg_color=fg_color
    )

    new_show_hist(
        img,
        ax=ax2,
        title=None,
        normalize=normalize,
        cumulative=cumulative,
        mask=None,
        #extrema=None,
        hist_xlabel='Luminance',
        hist_ylabel='Frequency',
        bg_color=bg_color,
        bg_alpha=bg_alpha,
        fg_color=fg_color
    )

    plt.tight_layout()
    plt.show()


def change_img_exposure(img_mx, alpha=1, beta=0):
    offset = beta * (1 - alpha) * 256
    if img_mx.ndim == 2:
        return img_mx * alpha + offset

    new_img_mx = img_mx.copy()
    for i in range(img_mx.shape[-1]):
        new_img_mx[..., i] = new_img_mx[..., i] * alpha + offset
    return new_img_mx


def gaussian_noise(imx):
    if imx.ndim == 2:
        return np.clip(
            imx + np.random.normal(0, np.std(imx), imx.shape),
            0, 255
        ).astype(np.uint8)
    r, g, b, _ = RGBA(imx)
    r_noise = np.random.normal(0, np.std(r), r.shape)
    g_noise = np.random.normal(0, np.std(g), g.shape)
    b_noise = np.random.normal(0, np.std(b), b.shape)
    noisy_r = np.clip(r + r_noise, 0, 255)
    noisy_g = np.clip(g + g_noise, 0, 255)
    noisy_b = np.clip(b + b_noise, 0, 255)
    noisy_imx = np.dstack((noisy_r, noisy_g, noisy_b))
    return noisy_imx.astype(np.uint8)


def do_for_each_band(rgb_imx, f_band):
    r, g, b, _ = RGBA(rgb_imx)
    return tuple(f_band(band) for band in (r, g, b))
    # return np.dstack().astype(np.uint8)


# how="lin", normalize=False
def get_sq_mask_v1(radius):
    w = h = 1 + 2 * radius
    mask = np.zeros((h, w))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            d_i = abs(i - radius)
            d_j = abs(j - radius)
            d = max(d_i, d_j)
            mask[i, j] = radius - d + 1
    return mask


# how="lin", normalize=False
def get_sq_mask_v2(radius):
    w = h = 1 + 2 * radius
    ii, jj = np.ogrid[:h, :w]
    ii = np.abs(ii - radius)
    jj = np.abs(jj - radius)
    d = np.maximum(ii, jj)
    return radius - d + 1


# how="lin", normalize=False, dist:"manhattan", etc
def get_sq_mask_v3(radius, f):
    w = h = 1 + 2 * radius
    ii, jj = np.ogrid[:h, :w]
    ii = np.abs(ii - radius)
    jj = np.abs(jj - radius)
    d = np.maximum(ii, jj)
    fr = lambda x: f(radius, x)
    vfr = np.vectorize(fr)
    return vfr(d)


def get_sq_mask(radius, f, p=1):
    w = h = 1 + 2 * radius
    # ii, jj = np.ogrid[:h, :w]
    ii, jj = np.meshgrid(np.arange(h), np.arange(w))
    ii = np.abs(ii - radius)
    jj = np.abs(jj - radius)
    d = np.linalg.norm(np.stack([ii, jj], axis=-1), ord=p, axis=-1)
    fr = lambda x: f(radius, x)
    vfr = np.vectorize(fr)
    return vfr(d)


def naive_convolution_v1(imx, ker):        
    h, w = imx.shape
    #norm = np.sum(ker)
    def coef(i, j):
        inside = 0 <= i < h and 0 <= j < w
        return imx[i, j] if inside else 0
    conv_imx = np.zeros(imx.shape) 
    k = (ker.shape[0] - 1) // 2
    for i in range(h):
        for j in range(w):
            for u in range(-k, k+1):
                for v in range(-k, k+1):
                    conv_imx[i, j] += ker[u, v] * coef(i - u, j - v)
    return np.rint(conv_imx) # / norm)


def naive_convolution_v2(imx, ker):        
    h, w = imx.shape
    # norm = np.sum(ker)
    k = (ker.shape[0] - 1) // 2
    imx_padded = np.pad(imx, pad_width=k, mode="constant", constant_values=0)
    conv_imx = np.zeros(imx.shape) 
    for i in range(h):
        for j in range(w):
            sub = imx_padded[i:i+2*k+1, j:j+2*k+1]
            conv_imx[i, j] = np.sum(sub * ker)
    return np.rint(conv_imx)  # / norm)


# reste à faire un pad, suivi d'un crop
def naive_convolution(imx, ker):
    # si RGB, do_for_each_band sinon, le traitement mono bande
    if imx.ndim > 2:
        return np.dstack(
            do_for_each_band(imx, lambda x: naive_convolution(x, ker))
        ).astype(np.uint8)
    else:
        # h, w = imx.shape
        # norm = np.sum(ker)
        imx_padded = np.pad(imx, pad_width=ker.shape[0], mode="mean")
        return np.rint(convolve2d(imx_padded, ker, mode="valid"))# mode="same"))  # / norm)


def naive_cross_correlation_v1a(imx, ker):        
    h, w = imx.shape
    #norm = np.sum(ker)
    def coef(i, j):
        inside = 0 <= i < h and 0 <= j < w
        return imx[i, j] if inside else 0
    conv_imx = np.zeros(imx.shape) 
    k = (ker.shape[0] - 1) // 2
    for i in range(h):
        for j in range(w):
            for u in range(-k, k+1):
                for v in range(-k, k+1):
                    conv_imx[i, j] += ker[u, v] * coef(i + u, j + v)
    return np.rint(conv_imx) # / norm)


def naive_cross_correlation_v1b(imx, ker):
    return naive_convolution_v1(imx, np.rot90(ker, k=2))


def naive_cross_correlation_v2(imx, ker):
    return naive_convolution_v2(imx, np.rot90(ker, k=2))


# reste à faire un pad, suivi d'un crop
def naive_cross_correlation(imx, ker):
    # h, w = imx.shape
    # norm = np.sum(ker)
    return np.rint(correlate2d(imx, ker, mode="same"))  # / norm)


def get_sobel_kernel(axis=0):
    ident = [1, 1]
    oppos = [1, -1]
    ident_ident = np.convolve(ident, ident)
    ident_oppos = np.convolve(ident, oppos)
    sobel_ker = np.outer(ident_ident, ident_oppos)
    if axis == 1:
        return np.rot90(sobel_ker, k=-1)
    else:
        return sobel_ker


def apply_sobel(imx):
    # si RGB, do_for_each_band sinon, le traitement mono bande
    if imx.ndim > 2:
        t = do_for_each_band(imx, apply_sobel)
        u = tuple(zip(*t))
        return (
            np.dstack(u[0]).astype(np.uint8),
            np.dstack(u[1]).astype(np.uint8)
        )
    else:
        ker_x = get_sobel_kernel(axis=0)
        ker_y = get_sobel_kernel(axis=1)
        conv_x = np.rint(convolve2d(imx, ker_x, mode="same"))
        conv_y = np.rint(convolve2d(imx, ker_y, mode="same"))
        norm = np.linalg.norm(
            np.stack([conv_x, conv_y], axis=-1),
            ord=2, axis=-1
        )
        angl = np.arctan2(conv_x, conv_y)
        return norm, angl


def sym_ext(a):
    return np.concatenate((a, np.flip(a[:-1])))


# radius : k tq 2 k + 1 est l'entier impair le plus proche de 6sigma
def get_gaussian_kernel_v1(sigma=1):
    # construction avec le procédé générique (reste à faire)
    # on construit un quart que l'on réplique, avec fllip, et assemble
    # construction d'un quart, de dims r+1 x r+1
    k = math.floor((6 * sigma - 1) / 2)
    print('sigma:', sigma)
    print('k:', k)
    h_x = np.arange(k+1)
    print(h_x)
    a = 2 * sigma**2
    b = math.sqrt(a * math.pi)
    print('a:', a, 'b:', b)
    exp_x = np.exp(-np.arange(k+1)**2 / a) / b
    print(exp_x)
    exp_x = sym_ext(exp_x)
    print(exp_x)
    # normalization
    w = np.sum(exp_x)
    exp_x = exp_x / w
    print(exp_x)
    print(np.sum(exp_x))


def get_1d_gaussian_kernel(sigma=1):
    k = math.floor((6 * sigma - 1) / 2)
    a = 2 * sigma**2
    b = math.sqrt(a * math.pi)
    exp_x = np.exp(-np.arange(k+1)**2 / a) / b
    exp_x = sym_ext(exp_x)
    return exp_x / np.sum(exp_x)
