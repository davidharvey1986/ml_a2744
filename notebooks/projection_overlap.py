import numpy as np
from numpy import ndarray
from scipy.stats import gaussian_kde


def overlap(data_1: np.ndarray, data_2: np.ndarray, bins: int = 100) -> float:
    """
    Calculates the overlap between two datasets by using a Gaussian kernel to approximate the
    distribution, then integrates the overlap using the trapezoidal rule

    Parameters
    ----------
    data_1 : ndarray
        First dataset of shape (N), where N are the number of points
    data_2 : ndarray
        Second dataset of shape (M), where M are the number of points
    bins : int, default = 100
        Number of bins to sample from the Gaussian distribution approximation

    Returns
    -------
    float
        Overlap fraction
    """
    grid = np.linspace(min(data_1.min(), data_2.min()), max(data_1.max(), data_2.max()), bins)
    kde_1 = gaussian_kde(data_1)
    kde_2 = gaussian_kde(data_2)

    pdf_1 = kde_1(grid)
    pdf_2 = kde_2(grid)
    
    return np.trapezoid(np.minimum(pdf_1, pdf_2), grid)


def proj_1d(
        target_class: int | float | str,
        centers: ndarray,
        rel_vecs: ndarray,
        classes: ndarray) -> ndarray:
    """
    Projects a set of vectors belonging to the target class onto the direction of the centers of all
    other classes and the set of vectors belonging to different classes onto the direction between
    the centers of each class and the target class.

    Parameters
    ----------
    target_class : int | float | str
        Target class label to get the directional vectors to the center of all other classes
    centers : ndarray
        Global vectors pointing to the center of each class with shape (C,...) and type float,
        where C is the number of classes
    rel_vecs : ndarray
        Vectors relative to their respective class center with shape(C) and type object, where each
        element contains the vectors for each class of shape (N,Z), where N is the number of
        vectors per class and Z is the dimension of each vector
    classes : ndarray
        Unique classes with the same order as centers and rel_vecs with shape (C) and type int |
        float | str matching target_class

    Returns
    -------
    ndarray
        Projected vectors with shape (2,C-1) and type object, where the first row is the projected
        vectors of the target class in the direction to all other classes, and the second row is the
        projected vectors of the other classes in the direction of the target class, where each
        element contains an array of shape (N) and type float
    """
    i: int
    vecs: ndarray
    idxs: ndarray = np.array(classes == target_class)
    direcs: ndarray = centers[~idxs] - centers[idxs]
    norms: ndarray = np.linalg.norm(direcs, axis=1)
    proj_vecs: ndarray = np.empty((2, len(idxs) - 1), dtype=object)

    # Project vectors of target class in direction of all other classes
    proj_vecs[0] = [*(rel_vecs[idxs][0] @ direcs.T / norms).swapaxes(0, 1)]

    # Project vectors of other classes in direction of target class
    for i, vecs in enumerate(rel_vecs[~idxs]):
        proj_vecs[1, i] = norms[i] + vecs @ direcs[i] / norms[i]
    return proj_vecs


def proj_all_inter_1d(vecs: ndarray, classes: ndarray) -> ndarray:
    """
    Projects all vectors belonging to each class in the direction of the center of their class to
    the centers of all other classes.

    Parameters
    ----------
    vecs : ndarray
        Global vectors with shape (N,Z) and type float, where N is the total number of vectors and
        Z is the dimension of each vector
    classes : ndarray
        Class for each vector with shape (N)

    Returns
    -------
    ndarray
        Projected vectors with shape (C,2,C-1) and type object, where C is the number of classes,
        each row corresponds to each class being the target class that all vectors are projected in
        the direction of, the first column represents the vectors from the target class in the
        direction of all other classes, and the second column represents the vectors from all other
        classes in the direction of the target class, where each element contains an array of shape
        (N) and type float
    """
    class_: int | float | str
    proj_vecs: list[ndarray] = []
    centers: ndarray
    rel_vecs: ndarray

    centers, rel_vecs = relative_vecs(vecs, classes)

    for class_ in np.unique(classes):
        proj_vecs.append(proj_1d(class_, centers, rel_vecs, np.unique(classes)))
    return np.array(proj_vecs, dtype=object)


def relative_vecs(vecs: ndarray, classes: ndarray) -> tuple[ndarray, ndarray]:
    """
    Calculates a set of vectors relative to the center of the set of vectors for each class.

    Parameters
    ----------
    vecs : ndarray
        Vectors with shape (N,...) and type float, where N is the number of vectors
    classes : ndarray
        Class labels with shape (N)

    Returns
    -------
    tuple[ndarray, ndarray]
        Class centers with shape (C,...), where C is the number of classes; and relative vectors
        with shape (C,M,...) and type float or (C) with type object, where M is the number of
        vectors per class if there are an equal amount; otherwise, each element contains the vectors
        for each class of shape (M,...)
    """
    class_: int | float | str
    centers: list[ndarray] = []
    rel_vecs: list[ndarray] = []
    idxs: ndarray

    for class_ in np.unique(classes):
        idxs = np.array(classes == class_)
        centers.append(np.mean(vecs[idxs], axis=0))
        rel_vecs.append(vecs[idxs] - centers[-1])

    try:
        return np.array(centers), np.array(rel_vecs)
    except ValueError:
        return np.array(centers), np.array(rel_vecs, dtype=object)
