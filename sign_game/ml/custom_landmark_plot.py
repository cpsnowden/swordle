from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.hands import HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import (
    DrawingSpec,
    _VISIBILITY_THRESHOLD,
    _PRESENCE_THRESHOLD,
    _normalize_color,
    RED_COLOR,
    BLACK_COLOR,
)
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def handmarks_df_to_np(df):
    x = df.filter(regex="(X|x)$")
    y = df.filter(regex="(Y|y)$")
    z = df.filter(regex="(Z|z)$")
    assert x.shape == y.shape == z.shape
    return np.dstack([x.to_numpy(), y.to_numpy(), z.to_numpy()])


def to_mp_list(landmarks):
    lmks = [landmark_pb2.NormalizedLandmark(x=l[0], y=l[1], z=l[2]) for l in landmarks]
    return landmark_pb2.NormalizedLandmarkList(landmark=lmks)


def plot_landmarks_simple(ax, landmarks):
    lmks = to_mp_list(landmarks)
    plot_landmarks(ax, lmks, HAND_CONNECTIONS)


def plot_landmarks(
    ax,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR, thickness=5),
    connection_drawing_spec: DrawingSpec = DrawingSpec(color=BLACK_COLOR, thickness=5),
    elevation: int = 10,
    azimuth: int = 10,
):
    """Plot the landmarks and the connections in matplotlib 3d.

    Args:
      landmark_list: A normalized landmark list proto message to be plotted.
      connections: A list of landmark index tuples that specifies how landmarks to
        be connected.
      landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
        drawing settings such as color and line thickness.
      connection_drawing_spec: A DrawingSpec object that specifies the
        connections' drawing settings such as color and line thickness.
      elevation: The elevation from which to view the plot.
      azimuth: the azimuth angle to rotate the plot.

    Raises:
      ValueError: If any connection contains an invalid landmark index.
    """
    if not landmark_list:
        return
    ax.view_init(elev=elevation, azim=azimuth)
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if (
            landmark.HasField("visibility")
            and landmark.visibility < _VISIBILITY_THRESHOLD
        ) or (
            landmark.HasField("presence") and landmark.presence < _PRESENCE_THRESHOLD
        ):
            continue
        ax.scatter3D(
            xs=[-landmark.z],
            ys=[landmark.x],
            zs=[-landmark.y],
            color=_normalize_color(landmark_drawing_spec.color[::-1]),
            linewidth=landmark_drawing_spec.thickness,
        )
        plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx],
                    plotted_landmarks[end_idx],
                ]
                ax.plot3D(
                    xs=[landmark_pair[0][0], landmark_pair[1][0]],
                    ys=[landmark_pair[0][1], landmark_pair[1][1]],
                    zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    color=_normalize_color(connection_drawing_spec.color[::-1]),
                    linewidth=connection_drawing_spec.thickness,
                )
