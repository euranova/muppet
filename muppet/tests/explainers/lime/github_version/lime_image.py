"""Functions for explaining classifiers that use Image data.

NOTE This code was extracted from https://github.com/brunodoamaral/lime
"""

import copy
from functools import partial

import numpy as np
import scipy as sp
import sklearn
from skimage.color import gray2rgb
from skimage.segmentation import slic
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state
from tqdm.auto import tqdm


class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""

    def __init__(self, kernel_fn, verbose=False, random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose

        self.random_state = check_random_state(random_state)

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(
                    data[:, used_features + [feature]],
                    labels,
                    sample_weight=weights,
                )
                score = clf.score(
                    data[:, used_features + [feature]],
                    labels,
                    sample_weight=weights,
                )
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
        understand the parameters.
        """
        if method == "none":
            return np.array(range(data.shape[1]))
        elif method == "forward_selection":
            return self.forward_selection(data, labels, weights, num_features)
        elif method == "highest_weights":
            clf = Ridge(
                alpha=0.01, fit_intercept=True, random_state=self.random_state
            )
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate(
                        (indices, np.zeros(num_to_pad, dtype=indices.dtype))
                    )
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features : sdata][
                        ::-1
                    ]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True,
                )
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == "lasso_path":
            weighted_data = (
                data - np.average(data, axis=0, weights=weights)
            ) * np.sqrt(weights[:, np.newaxis])
            weighted_labels = (
                labels - np.average(labels, weights=weights)
            ) * np.sqrt(weights)
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data, weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == "auto":
            if num_features <= 6:
                n_method = "forward_selection"
            else:
                n_method = "highest_weights"
            return self.feature_selection(
                data, labels, weights, num_features, n_method
            )

    def explain_instance_with_data(
        self,
        neighborhood_data,
        neighborhood_labels,
        distances,
        label,
        num_features,
        feature_selection="auto",
        model_regressor=None,
    ):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """
        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(
            neighborhood_data,
            labels_column,
            weights,
            num_features,
            feature_selection,
        )
        if model_regressor is None:
            model_regressor = Ridge(
                alpha=1, fit_intercept=True, random_state=self.random_state
            )
        easy_model = model_regressor
        easy_model.fit(
            neighborhood_data[:, used_features],
            labels_column,
            sample_weight=weights,
        )
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column,
            sample_weight=weights,
        )

        local_pred = easy_model.predict(
            neighborhood_data[0, used_features].reshape(1, -1)
        )

        if self.verbose:
            print("Intercept", easy_model.intercept_)
            print(
                "Prediction_local",
                local_pred,
            )
            print("Right:", neighborhood_labels[0, label])
        return (
            easy_model.intercept_,
            sorted(
                zip(used_features, easy_model.coef_),
                key=lambda x: np.abs(x[1]),
                reverse=True,
            ),
            prediction_score,
            local_pred,
        )


class ImageExplanation(object):
    """Image explanation class from LIME GitHub implementation.

    Represents explanations for image predictions generated by LIME,
    storing the original image, segments, and explanation data.
    """

    def __init__(self, image, segments):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}


class LimeImageExplainer(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained.
    """

    def __init__(
        self,
        kernel_width=0.25,
        kernel=None,
        verbose=False,
        feature_selection="auto",
        random_state=None,
    ):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:

            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d**2) / kernel_width**2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = LimeBase(kernel_fn, verbose, random_state=self.random_state)

    def explain_instance(
        self,
        image,
        classifier_fn,
        labels=(1,),
        hide_color=None,
        top_labels=5,
        num_features=100000,
        num_samples=1000,
        batch_size=10,
        segmentation_fn=None,
        distance_metric="cosine",
        model_regressor=None,
        random_seed=None,
        progress_bar=False,
    ):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: If not None, will hide superpixels with this color.
                Otherwise, use the mean pixel color of the image.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: batch size for model predictions
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            An ImageExplanation object (see lime_image.py) with the corresponding
            explanations.
        """
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        segments = slic(image)
        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]),
                )
        else:
            fudged_image[:] = hide_color

        top = labels

        data, labels = self.data_labels(
            image,
            fudged_image,
            segments,
            classifier_fn,
            num_samples,
            batch_size=batch_size,
            progress_bar=progress_bar,
        )

        distances = sklearn.metrics.pairwise_distances(
            data, data[0].reshape(1, -1), metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (
                ret_exp.intercept[label],
                ret_exp.local_exp[label],
                ret_exp.score[label],
                ret_exp.local_pred[label],
            ) = self.base.explain_instance_with_data(
                data,
                labels,
                distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection,
            )
        return ret_exp

    def data_labels(
        self,
        image,
        fudged_image,
        segments,
        classifier_fn,
        num_samples,
        batch_size=10,
        progress_bar=False,
    ):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(
            0, 2, num_samples * n_features
        ).reshape((num_samples, n_features))
        labels = []
        data[0, :] = 1
        imgs = []
        rows = tqdm(data) if progress_bar else data
        for row in rows:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
        return data, np.array(labels)
