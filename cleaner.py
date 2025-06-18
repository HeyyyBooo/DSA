# Replace backslashes in a code block
with open("Graph.md", "r", encoding="utf-8") as f:
    content = f.read()

# Only clean inside code blocks (optional)
import re
def clean_backslashes_in_code(md_text):
    return re.sub(r"```python(.*?)```", lambda m: f"```python{m.group(1).replace('\\', '')}```", md_text, flags=re.DOTALL)

cleaned_content = clean_backslashes_in_code(content)

with open("Graph_cleaned.md", "w", encoding="utf-8") as f:
    f.write(cleaned_content)
