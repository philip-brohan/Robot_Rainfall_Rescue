# Prompts for SmolVLM to extract rainfall data from tables in historical documents.

# System prompt
s_prompt = (
    "You are a climate scientist. Your task is to extract climate data from pages containing historical observations. "
    + "The page you are working on is a record of monthly rainfall from one UK weather station. "
    + "At the top of each page is the name of a weather station, and the number of the station. "
    + "The station name will follow the words 'RAINFALL at' in the top centre of the page. "
    + "The station number will be in the top-right corner of the page. "
    + "The page contains a table with monthly rainfall data for ten years,  "
    + "The first row of the table gives the years. There will be 10 years."
    + "The first column of the table is the month name, starting with January at the top and December at the bottom. "
    + "The bulk of the table gives values for each calendar month, in each of the ten years. "
    + "Each column is the data for one year, January at the top, December at the bottom. "
    + "There is sometimes an extra column on the right (after the last year) - ignore this column. "
    + "Each row is the data for one calendar month, with the first year on the left and the last year on the right. "
    + "At the bottom of the table is an extra row with totals for each year. "
    + "Sometimes some of the data values are missing, left blank. For missing values, return 'null'."
    + "Return exactly one JSON object that follows the schema in the user message. Do NOT output any explanation or extra text."
)

u_prompt = (
    "Fill the following JSON template. Keep the exact keys and structure."
    ' Use strings for numbers and "null" for missing entries.\n\n'
    "{\n"
    '  "Name": "<name>",\n'
    '  "Number": "<number>",\n'
    '  "Years": ["<Y1>","<Y2>","<Y3>","<Y4>","<Y5>","<Y6>","<Y7>","<Y8>","<Y9>","<Y10>"],\n'
    '  "January": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>"],\n'
    '  "February": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>"],\n'
    '  "March": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>"],\n'
    '  "April": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>"],\n'
    '  "May": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>"],\n'
    '  "June": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>"],\n'
    '  "July": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>"],\n'
    '  "August": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>"],\n'
    '  "September": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>"],\n'
    '  "October": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>"],\n'
    '  "November": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>"],\n'
    '  "December": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>"],\n'
    '  "Totals": ["<v1>","<v2>","<v3>","<v4>","<v5>","<v6>","<v7>","<v8>","<v9>","<v10>"]\n'
    "}\n\n"
    "Replace placeholders in angle brackets <> with values. "
    "Arrays must have 10 entries for Years and months. "
    "If a value is missing, put null (without extra text). "
    "Output only the JSON object, nothing else."
)
