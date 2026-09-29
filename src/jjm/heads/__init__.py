"""Fitted heads: a frozen encoder reads the state once and a small stored head maps that to outcome weights.

The encoder is any local model JJM can read features from. The head is fitted once from labelled
examples, stored as an immutable artifact bound by digest to the criterion, the encoder, and the
examples, and reused only when that binding matches exactly.
"""
