"""Bounded, informational action metadata; never used as execution authority."""

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from app.actions.schemas import ActionId, ActionStatus
from app.errors import ShopError
from app.schemas import PublicModel


class ActionEvent(PublicModel):
    model_config = ConfigDict(frozen=True)

    sequence: Annotated[int, Field(strict=True, ge=1, le=3)]
    kind: Literal["legacy", "prepared", "executing", "finished", "replaced"]
    status: ActionStatus
    occurred_at: Annotated[float, Field(strict=True, gt=0, le=253402300799, allow_inf_nan=False)]
    request_id: ActionId | None = None
    related_action_id: ActionId | None = None
    error_code: str | None = None

    @field_validator("error_code")
    @classmethod
    def known_error(cls, value):
        if value is not None and (not value or ShopError(value).code != value):
            raise ValueError("invalid event error")
        return value

    @model_validator(mode="after")
    def coherent_event(self):
        if (self.kind == "prepared" and self.status != "pending"
                or self.kind == "executing" and self.status != "executing"
                or self.kind == "replaced" and (self.status != "cancelled" or self.related_action_id is None)
                or self.kind == "finished" and self.status not in {"succeeded", "failed", "unknown", "cancelled"}
                or self.kind not in {"prepared", "replaced"} and self.related_action_id is not None
                or self.kind != "finished" and self.error_code is not None
                or self.status == "succeeded" and self.error_code is not None
                or self.kind == "legacy" and self.request_id is not None):
            raise ValueError("inconsistent event")
        return self


class ActionHistory(PublicModel):
    action_id: ActionId
    session_id: ActionId
    events: list[ActionEvent] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def coherent_history(self):
        for index, event in enumerate(self.events):
            if event.sequence != index + 1 or event.related_action_id == self.action_id:
                raise ValueError("invalid event sequence")
            if index == 0:
                if event.kind not in {"prepared", "legacy"}:
                    raise ValueError("missing initial event")
            else:
                previous = self.events[index - 1].status
                allowed = ({"executing", "finished", "replaced"} if previous == "pending"
                           else {"finished"} if previous == "executing" else set())
                if (event.kind not in allowed
                        or previous == "pending" and event.status in {"succeeded", "unknown"}
                        or previous == "executing" and event.status == "cancelled"):
                    raise ValueError("invalid event transition")
        return self
