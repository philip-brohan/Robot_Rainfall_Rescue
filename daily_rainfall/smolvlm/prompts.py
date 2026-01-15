# Prompts for SmolVLM to extract rainfall data from tables in historical documents.

# System prompt
s_prompt = (
    "You are a climate scientist. Your task is to extract climate data from pages containing historical observations. "
    + "The page you are working contains one year of daily rainfall observations from one UK weather station. "
    + "The columns of the table are the months, January to December. "
    + "The rows of the table are the days of the month, 1 to 31. "
    + "At the bottom of the table is an extra row with totals for each month. "
    + "Some of the data values are missing. For missing values, return 'null'."
    + "Return exactly one JSON object that follows the schema in the user message. Do NOT output any explanation or extra text."
)


def make_u_prompt():
    r"""User prompt with JSON template for daily rainfall data extraction."""
    prompt = (
        "Fill the following JSON template. Keep the exact keys and structure."
        " Include a rainfall value for each day of the year, and a total for each month. "
        " Use strings for numbers and 'null' for missing entries.\n\n "
        " { Month: ["
    )
    for month in [
        "Jan.",
        "Feb.",
        "Mar.",
        "Apr.",
        "May.",
        "Jun.",
        "Jul.",
        "Aug.",
        "Sep.",
        "Oct.",
        "Nov.",
        "Dec.",
    ]:
        prompt += "{Month: " + month + ", rainfall [ "
        for day in range(1, 32):
            prompt += "{ Day: " + str(day) + ", rainfall: <value> },"
        prompt += " ], total: <value> },"
    prompt += (
        " ] }\n\n"
        "Replace placeholders in angle brackets <> with values. "
        "If a value is missing, put null (without extra text). "
        "Output only the JSON object, nothing else."
    )
    return prompt


u_prompt = (
    "Output the daily rainfall records as a JSON object. "
    ' Use strings for numbers and "null" for missing entries.\n\n'
    ' The object should have keys "Day 1" to "Day 31" and "Totals". '
    ' Each "Day n" key should map to an array of 12 values, one for each month. '
    ' The "Totals" key should map to an array of 12 values, one for each monthly total.\n\n'
    "Output only the JSON object, nothing else."
)
