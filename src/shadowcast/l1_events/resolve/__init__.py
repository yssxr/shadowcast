"""L1.5: inferring what the packets omit — ownership, teams, roles, deaths.

Everything in here is a guess rather than a reading, and the split from `l1_events`
proper is deliberate. The corpus contains no entity id on movement orders, no team on
champions, no role at all, and no death packet. Each of those has to be reconstructed,
each can be wrong, and each therefore carries a measurable accuracy rather than being
silently folded in with the facts.
"""

from shadowcast.l1_events.resolve.attribute import Attribution, attribute

__all__ = ["Attribution", "attribute"]
