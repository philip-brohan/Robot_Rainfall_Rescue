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


# u_prompt = (
#     "Fill the following JSON template. Keep the exact keys and structure."
#     ' Use strings for numbers and "null" for missing entries.\n\n'
#     "{\n"
#     '  "Day 1": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 2": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 3": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 4": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 5": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 6": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 7": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 8": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 9": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 10": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 11": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 12": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 13": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 14": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 15": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 16": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 17": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 18": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 19": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 20": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 21": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 22": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 23": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 24": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 25": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 26": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 27": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 28": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 29": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 30": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Day 31": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"],\n'
#     '  "Totals": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>","<v11>","<v12>"]\n'
#     "}\n\n"
#     "Replace placeholders in angle brackets <> with values. "
#     "Each day number array must have 12 entries, one for each month on that day. "
#     "If a value is missing, put null (without extra text). "
#     "Output only the JSON object, nothing else."
# )
u_prompt = (
    "Output the daily rainfall records as a JSON object. "
    ' Use strings for numbers and "null" for missing entries.\n\n'
    ' The object should have keys "Day 1" to "Day 31" and "Totals". '
    ' Each "Day n" key should map to an array of 12 values, one for each month. '
    ' The "Totals" key should map to an array of 12 values, one for each monthly total.\n\n'
    "Output only the JSON object, nothing else."
)
