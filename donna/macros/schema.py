"""Desktop task macro schemas for Dānā spatial record / replay."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class MacroStep(BaseModel):
    """One spatially-grounded UI action in a macro sequence."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    target_label: str = Field(..., min_length=1, description="Human-readable UI target")
    action_type: str = Field(
        ...,
        description="click | double_click | type_text | key_combination",
    )
    action_value: Optional[str] = Field(
        default=None,
        description="Typed text or hotkey chord (e.g. ctrl+s); unused for clicks",
    )
    visual_context_prompt: str = Field(
        ...,
        min_length=1,
        description="Florence-2 phrase-grounding prompt used to re-find the target",
    )


class MacroSequence(BaseModel):
    """Ordered macro persisted as ``donna/macros/<macro_id>.json``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    macro_id: str = Field(..., min_length=1)
    description: str = Field(default="")
    steps: List[MacroStep] = Field(default_factory=list)
