# Define a Pydantic model for the structured output of the daily rainfall data.
# This will be used by outlinesto validate the output from the model
# and ensure it conforms to the expected structure and data types.

from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict, AliasChoices

# Accepts:
# - "null" as a literal string
# - decimal strings like ".48", "4.97", "10.19"
# Modified to require fixed-length strings - 5 characters exactly.
VALUE_RE = r"^(?: null|\d\d\.\d\d)$"

MonthVector = Annotated[
    list[Annotated[str, Field(pattern=VALUE_RE)]],
    Field(min_length=12, max_length=12),
]


class RainfallRowTable(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,  # allows using either field names or aliases
    )

    totals: MonthVector = Field(alias="Totals")

    day_1: MonthVector = Field(alias="Day 1")
    day_2: MonthVector = Field(alias="Day 2")
    day_3: MonthVector = Field(alias="Day 3")
    day_4: MonthVector = Field(alias="Day 4")
    day_5: MonthVector = Field(alias="Day 5")
    day_6: MonthVector = Field(alias="Day 6")
    day_7: MonthVector = Field(alias="Day 7")
    day_8: MonthVector = Field(alias="Day 8")
    day_9: MonthVector = Field(alias="Day 9")
    day_10: MonthVector = Field(alias="Day 10")
    day_11: MonthVector = Field(alias="Day 11")
    day_12: MonthVector = Field(alias="Day 12")
    day_13: MonthVector = Field(alias="Day 13")
    day_14: MonthVector = Field(alias="Day 14")
    day_15: MonthVector = Field(alias="Day 15")
    day_16: MonthVector = Field(alias="Day 16")
    day_17: MonthVector = Field(alias="Day 17")
    day_18: MonthVector = Field(alias="Day 18")
    day_19: MonthVector = Field(alias="Day 19")
    day_20: MonthVector = Field(alias="Day 20")
    day_21: MonthVector = Field(alias="Day 21")
    day_22: MonthVector = Field(alias="Day 22")
    day_23: MonthVector = Field(alias="Day 23")
    day_24: MonthVector = Field(alias="Day 24")
    day_25: MonthVector = Field(alias="Day 25")
    day_26: MonthVector = Field(alias="Day 26")
    day_27: MonthVector = Field(alias="Day 27")
    day_28: MonthVector = Field(alias="Day 28")
    day_29: MonthVector = Field(alias="Day 29")
    day_30: MonthVector = Field(alias="Day 30")
    day_31: MonthVector = Field(alias="Day 31")
