"""
Helpers for the "no surrounding agents" ablation experiment.

The PGP encoder reads vehicle / pedestrian histories via two GRU streams and
fuses them into each lane-node encoding through an agent-node attention layer.
This module zeroes the agent FEATURE tensors before the encoder ever sees
them, so the model is trained and evaluated as if no other agents existed on
the road. The mask tensors are intentionally left untouched: keeping them as
they are preserves the topology of the agent-node attention (otherwise the
MultiheadAttention layer would softmax over an all-(-inf) row and produce
NaNs). The feature content is zero, so the encoder cannot extract any
positional or kinematic signal from the surrounding agents.

This is the same behaviour the v1.0-test split exhibits by construction
(agent annotations are withheld in v1.0-test, so the test pickles contain
all-zero agent features). Training on zero-feature agents therefore matches
the test-time distribution exactly.
"""

from typing import Dict


def zero_agents_in_inputs(inputs: Dict) -> Dict:
    """
    In-place zero of vehicle / pedestrian feature tensors inside the model
    input dict produced by the PGP graphs dataset.

    Operates on the dict structure described in
    `models/encoders/pgp_encoder.py`:
        inputs['surrounding_agent_representation']['vehicles']     # (B, V, T, F)
        inputs['surrounding_agent_representation']['pedestrians']  # (B, P, T, F)

    Mask tensors (vehicle_masks, pedestrian_masks, agent_node_masks) are
    deliberately NOT modified.
    """
    sar = inputs.get('surrounding_agent_representation', None)
    if sar is None:
        return inputs
    if 'vehicles' in sar and sar['vehicles'] is not None:
        sar['vehicles'] = sar['vehicles'].zero_() if sar['vehicles'].requires_grad is False \
            else sar['vehicles'] * 0.0
    if 'pedestrians' in sar and sar['pedestrians'] is not None:
        sar['pedestrians'] = sar['pedestrians'].zero_() if sar['pedestrians'].requires_grad is False \
            else sar['pedestrians'] * 0.0
    return inputs
