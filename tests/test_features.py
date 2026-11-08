"""Tests for features.py: the feature vector must not depend on where the hand is, how big it is, or which hand."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import N_GEOMETRIC_FEATURES, angle_between, extract_features, normalize  # noqa: E402

RNG = np.random.default_rng(0)


def random_hand():
    """A random but plausible set of 21 landmarks (63 values)."""
    return RNG.uniform(0.2, 0.8, size=(21, 3)).flatten()


def test_feature_vector_has_63_coordinates_plus_geometric_features():
    assert extract_features(random_hand()).shape == (63 + N_GEOMETRIC_FEATURES,)


def test_normalize_puts_the_wrist_at_the_origin_and_scales_to_one():
    pts = normalize(random_hand()).reshape(21, 3)
    assert np.allclose(pts[0], 0)
    assert np.isclose(np.max(np.abs(pts)), 1)


def test_features_do_not_depend_on_where_the_hand_is():
    hand = random_hand()
    moved = (hand.reshape(21, 3) + [0.3, -0.1, 0.05]).flatten()
    assert np.allclose(extract_features(hand), extract_features(moved))


def test_features_do_not_depend_on_hand_size():
    hand = random_hand()
    bigger = (hand.reshape(21, 3) * 2.5).flatten()
    assert np.allclose(extract_features(hand), extract_features(bigger))


def test_all_zero_hand_does_not_crash_or_produce_nan():
    features = extract_features(np.zeros(63))
    assert not np.isnan(features).any()


def test_angle_between():
    assert angle_between(np.array([1, 0, 0]), np.array([1, 0, 0])) == pytest.approx(0, abs=1e-3)
    assert angle_between(np.array([1, 0, 0]), np.array([0, 1, 0])) == pytest.approx(np.pi / 2, abs=1e-3)
    assert angle_between(np.array([1, 0, 0]), np.array([-1, 0, 0])) == pytest.approx(np.pi, abs=1e-3)


def straight_open_hand():
    """Wrist at the origin, five straight fingers pointing up, spread apart like an open palm."""
    pts = np.zeros((21, 3))
    for finger, x in enumerate([-0.6, -0.3, 0.0, 0.3, 0.6]):
        for joint in range(4):
            pts[1 + 4 * finger + joint] = [x, 1 + joint * 0.3, 0]
    return pts


def test_straight_fingers_have_bending_angles_of_pi():
    features = extract_features(straight_open_hand().flatten())
    bending = features[63 + 10 + 4: 63 + 10 + 4 + 10]
    assert np.allclose(bending, np.pi, atol=1e-2)


def test_crossing_value_turns_negative_when_index_and_middle_cross():
    pts = straight_open_hand()
    assert extract_features(pts.flatten())[-1] > 0
    pts[8, 0], pts[12, 0] = pts[12, 0] + 0.1, pts[8, 0] - 0.1    # swap the index and middle tips
    assert extract_features(pts.flatten())[-1] < 0


def test_crossing_works_for_both_hands():
    pts = straight_open_hand()
    mirrored = pts * [-1, 1, 1]                                      # the other hand
    assert np.sign(extract_features(pts.flatten())[-1]) == np.sign(extract_features(mirrored.flatten())[-1])
