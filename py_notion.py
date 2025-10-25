import re
import os
from notion_client import Client
from notion_client.errors import APIResponseError


# Lightweight .env loader (no external dependency required)
def _load_env_file(path='.env'):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                # Only set if not already present in environment
                os.environ.setdefault(key, val)
    except FileNotFoundError:
        pass


_load_env_file()

def convert_markdown_to_notion(text):
    # Block equations
    block_equation_pattern = re.compile(r'^[ \t]*\$\$([^$]+)\$\$[ \t]*$', re.MULTILINE)
    block_equations = []
    def replace_block_equation(match):
        content = match.group(1).strip()
        block_equations.append(content)
        return '[BLOCK_EQUATION]'
    modified_text = block_equation_pattern.sub(replace_block_equation, text)

    # Callout blocks - MUST BE BEFORE general code blocks to catch ```callout first
    callout_pattern = re.compile(r'```callout\n([\s\S]*?)\n```', re.MULTILINE)
    callout_blocks = []
    def replace_callout(match):
        content = match.group(1).strip()
        callout_blocks.append(content)
        return '[CALLOUT_BLOCK]'
    modified_text = callout_pattern.sub(replace_callout, modified_text)

    # Code block handling - AFTER callouts to avoid catching ```callout
    code_block_pattern = re.compile(r'```(\w*)\n([\s\S]*?)\n```', re.MULTILINE)
    code_blocks = []
    def replace_code_block(match):
        language = match.group(1).lower() if match.group(1) else 'plain text'
        content = match.group(2).strip()
        code_blocks.append((language, content))
        return '[CODE_BLOCK]'
    modified_text = code_block_pattern.sub(replace_code_block, modified_text)

    # Inline equations
    inline_equation_pattern = re.compile(r'(?<!\$)\$([^$]+)\$(?!\$)')
    def replace_inline_equation(match):
        content = match.group(1).strip()
        return f'[INLINE_EQUATION:{content}]'
    modified_text = inline_equation_pattern.sub(replace_inline_equation, modified_text)

    # Links
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    links = []
    def replace_link(match):
        text = match.group(1).strip()
        url = match.group(2).strip()
        links.append((text, url))
        return f'[LINK:{text},{url}]'
    modified_text = link_pattern.sub(replace_link, modified_text)

    # Table handling
    table_pattern = re.compile(r'(\|.*?\|\n\|[-:\s|+\-]+\|\n(?:\|.*?\|\n)+)', re.MULTILINE)
    tables = []
    def replace_table(match):
        table_content = match.group(1).strip()
        tables.append(table_content)
        return '[TABLE]'
    modified_text = table_pattern.sub(replace_table, modified_text)

    # Inline code handling - MUST BE AFTER CODE BLOCKS
    inline_code_pattern = re.compile(r'`([^`]+)`')
    def replace_inline_code(match):
        content = match.group(1).strip()
        return f'[INLINE_CODE:{content}]'
    modified_text = inline_code_pattern.sub(replace_inline_code, modified_text)

    return modified_text, block_equations, callout_blocks, tables, code_blocks, links

def parse_line_for_formatting(line):
    pattern = r'(\*\*[^*]+\*\*|\[INLINE_EQUATION:[^\]]+\]|\[INLINE_CODE:[^\]]+\]|\[LINK:[^\]]+\])'
    parts = re.split(pattern, line)
    rich_text = []

    for part in parts:
        if not part:
            continue
        elif part.startswith('[INLINE_EQUATION:'):
            equation = part[len('[INLINE_EQUATION:'):-1]
            rich_text.append({
                "type": "equation",
                "equation": {"expression": equation}
            })
        elif part.startswith('[INLINE_CODE:'):
            code = part[len('[INLINE_CODE:'):-1]
            rich_text.append({
                "type": "text",
                "text": {"content": code},
                "annotations": {"code": True}
            })
        elif part.startswith('[LINK:'):
            link_parts = part[len('[LINK:'):-1].split(',', 1)
            text = link_parts[0].strip()
            url = link_parts[1].strip()
            rich_text.append({
                "type": "text",
                "text": {"content": text, "link": {"url": url}}
            })
        elif part.startswith('**') and part.endswith('**'):
            text_content = part[2:-2]
            rich_text.append({
                "type": "text",
                "text": {"content": text_content},
                "annotations": {"bold": True}
            })
        else:
            rich_text.append({
                "type": "text",
                "text": {"content": part}
            })

    return rich_text

def parse_table(table_text):
    """Parse table and return header/rows as plain text (formatting applied during rendering)."""
    lines = table_text.strip().split('\n')
    header = [cell.strip() for cell in lines[0].split('|')[1:-1]]
    rows = [[cell.strip() for cell in line.split('|')[1:-1]] for line in lines[2:]]
    return header, rows

def upload_to_notion(modified_text, block_equations, callout_blocks, tables, code_blocks, links, notion_api_key, page_id):
    notion = Client(auth=notion_api_key)
    lines = modified_text.split('\n')
    children = []
    block_eq_index = 0
    callout_index = 0
    table_index = 0
    code_block_index = 0

    # Patterns for lists
    bullet_pattern = re.compile(r'^([ \t]*)(- )(.*)$')  # Matches "- " with optional leading spaces/tabs
    numbered_pattern = re.compile(r'^([ \t]*)(\d+\. )(.*)$')  # Matches "1. " with optional leading spaces/tabs
    divider_pattern = re.compile(r'^\s*-{3,}\s*$')  # Matches "---" or more dashes with optional whitespace


    # Stack to track nesting level and type
    list_stack = []  # Each item: {'type': 'bulleted' or 'numbered', 'children': [], 'indent': int}

    for i, line in enumerate(lines):
        line = line.rstrip()  # Remove trailing whitespace

        # Handle horizontal rules (---)
        if divider_pattern.match(line):
            if list_stack:
                children.extend(list_stack.pop()['children'])
            children.append({
                "object": "block",
                "type": "divider",
                "divider": {}
            })
            continue  # Skip to the next line after adding the divider

        # Handle block equations
        if '[BLOCK_EQUATION]' in line and block_eq_index < len(block_equations):
            if list_stack:
                children.extend(list_stack.pop()['children'])
            children.append({
                "object": "block",
                "type": "equation",
                "equation": {"expression": block_equations[block_eq_index]}
            })
            block_eq_index += 1

        # Handle callout blocks
        elif '[CALLOUT_BLOCK]' in line and callout_index < len(callout_blocks):
            if list_stack:
                children.extend(list_stack.pop()['children'])
            callout_lines = callout_blocks[callout_index].split('\n')
            callout_children = []
            for callout_line in callout_lines:
                if callout_line.strip():
                    rich_text = parse_line_for_formatting(callout_line.strip())
                    if callout_line.startswith('- '):
                        callout_children.append({
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {"rich_text": rich_text}
                        })
                    else:
                        callout_children.append({
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {"rich_text": rich_text}
                        })
            children.append({
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [{"type": "text", "text": {"content": ""}}],
                    "children": callout_children
                }
            })
            callout_index += 1

        # Handle tables
        elif '[TABLE]' in line and table_index < len(tables):
            if list_stack:
                children.extend(list_stack.pop()['children'])
            header, rows = parse_table(tables[table_index])
            max_rows_per_table = 99  # 100 total including header
            for chunk_start in range(0, len(rows), max_rows_per_table):
                chunk_rows = rows[chunk_start:chunk_start + max_rows_per_table]
                # Process header cells with rich text formatting
                header_cells = []
                for cell in header:
                    rich_text = parse_line_for_formatting(cell)
                    if not rich_text:
                        rich_text = [{"type": "text", "text": {"content": " "}}]
                    header_cells.append(rich_text)

                # Process data row cells with rich text formatting
                row_cells = []
                for row in chunk_rows:
                    cells_in_row = []
                    for cell in row:
                        rich_text = parse_line_for_formatting(cell)
                        if not rich_text:
                            rich_text = [{"type": "text", "text": {"content": " "}}]
                        cells_in_row.append(rich_text)
                    row_cells.append(cells_in_row)

                table_children = [
                    {"table_row": {"cells": header_cells}}
                ] + [
                    {"table_row": {"cells": cells}} for cells in row_cells
                ]
                children.append({
                    "object": "block",
                    "type": "table",
                    "table": {
                        "table_width": len(header),
                        "has_column_header": True,
                        "children": table_children
                    }
                })
                if chunk_start + max_rows_per_table < len(rows):
                    children.append({
                        "object": "block",
                        "type": "divider",
                        "divider": {}
                    })
            table_index += 1

        # Handle code blocks
        elif '[CODE_BLOCK]' in line and code_block_index < len(code_blocks):
            if list_stack:
                children.extend(list_stack.pop()['children'])
            language, content = code_blocks[code_block_index]
            children.append({
                "object": "block",
                "type": "code",
                "code": {
                    "rich_text": [{"type": "text", "text": {"content": content}}],
                    "language": language
                }
            })
            code_block_index += 1

        # Handle headings
        elif line.startswith('# '):
            if list_stack:
                children.extend(list_stack.pop()['children'])
            heading_text = line[2:].strip()
            children.append({
                "object": "block",
                "type": "heading_1",
                "heading_1": {"rich_text": [{"type": "text", "text": {"content": heading_text}}]}
            })
        elif line.startswith('## '):
            if list_stack:
                children.extend(list_stack.pop()['children'])
            heading_text = line[3:].strip()
            children.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": heading_text}}]}
            })
        elif line.startswith('### '):
            if list_stack:
                children.extend(list_stack.pop()['children'])
            heading_text = line[4:].strip()
            children.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {"rich_text": [{"type": "text", "text": {"content": heading_text}}]}
            })

        # REPLACE THIS ENTIRE BLOCK (Bullet List Handling)
        elif bullet_match := bullet_pattern.match(line):
            indent = len(bullet_match.group(1)) // 2  # 2 spaces = 1 indent level
            content = bullet_match.group(3).strip()
            rich_text = parse_line_for_formatting(content)
            if not rich_text:
                rich_text = [{"type": "text", "text": {"content": " "}}]
            block = {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": rich_text}
            }

            # Adjust stack based on indent level
            while list_stack and list_stack[-1]['indent'] >= indent:
                last_list = list_stack.pop()
                if list_stack:  # Nest under the previous item if still in a list
                    list_stack[-1]['children'][-1][list_stack[-1]['type'] + '_list_item']['children'] = last_list['children']
                else:
                    children.extend(last_list['children'])

            if list_stack and list_stack[-1]['indent'] < indent:  # Nest under previous item
                list_stack[-1]['children'][-1][list_stack[-1]['type'] + '_list_item'].setdefault('children', []).append(block)
            else:  # Add as a top-level item
                list_stack.append({'type': 'bulleted', 'children': [block], 'indent': indent})

        # REPLACE THIS ENTIRE BLOCK (Numbered List Handling)
        elif numbered_match := numbered_pattern.match(line):
            indent = len(numbered_match.group(1)) // 2
            content = numbered_match.group(3).strip()
            rich_text = parse_line_for_formatting(content)
            if not rich_text:
                rich_text = [{"type": "text", "text": {"content": " "}}]
            block = {
                "object": "block",
                "type": "numbered_list_item",
                "numbered_list_item": {"rich_text": rich_text}
            }

            # Adjust stack based on indent level
            while list_stack and list_stack[-1]['indent'] >= indent:
                last_list = list_stack.pop()
                if list_stack:  # Nest under the previous item if still in a list
                    list_stack[-1]['children'][-1][list_stack[-1]['type'] + '_list_item']['children'] = last_list['children']
                else:
                    children.extend(last_list['children'])

            if list_stack and list_stack[-1]['indent'] < indent:  # Nest under previous item
                list_stack[-1]['children'][-1][list_stack[-1]['type'] + '_list_item'].setdefault('children', []).append(block)
            else:  # Add as a top-level item
                list_stack.append({'type': 'numbered', 'children': [block], 'indent': indent})

        # Handle paragraphs
        elif line.strip() and not line.lstrip().startswith(('#', '[', '|')):
            if list_stack:
                children.extend(list_stack.pop()['children'])
            rich_text = parse_line_for_formatting(line.strip())
            if not rich_text:
                rich_text = [{"type": "text", "text": {"content": " "}}]
            children.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": rich_text}
            })

        # REPLACE THIS ENTIRE BLOCK (Empty Line Handling)
        elif not line.strip() and list_stack:
            last_list = list_stack.pop()
            if list_stack:  # Nest under previous item if still in a list
                list_stack[-1]['children'][-1][list_stack[-1]['type'] + '_list_item']['children'] = last_list['children']
            else:
                children.extend(last_list['children'])

    # REPLACE THIS ENTIRE BLOCK (Closing Remaining Lists)
    while list_stack:
        last_list = list_stack.pop()
        if list_stack:  # Nest under previous item if still in a list
            list_stack[-1]['children'][-1][list_stack[-1]['type'] + '_list_item']['children'] = last_list['children']
        else:
            children.extend(last_list['children'])

    # Upload to Notion in chunks
    chunk_size = 100
    for i in range(0, len(children), chunk_size):
        chunk = children[i:i + chunk_size]
        try:
            notion.blocks.children.append(block_id=page_id, children=chunk)
            print(f"Uploaded chunk {i // chunk_size + 1} with {len(chunk)} blocks")
        except APIResponseError as e:
            # Provide a clearer, actionable message for common issues (404 / permission)
            print("\nNotion API returned an error while appending blocks:")
            print(str(e))
            print("\nCommon causes: \n - The PAGE_ID is incorrect.\n - The page is not shared with your integration. Open the page in Notion -> Share -> Invite your integration.\n - The integration token is for a different workspace.\n")
            raise SystemExit(1)
        except Exception as e:
            print("\nUnexpected error when calling Notion API:")
            print(repr(e))
            raise


with open("text.md", "r", encoding="utf-8") as file:
    markdown_text = f'\n{file.read()}'


# Read credentials from environment (loaded from .env by the helper above)
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
PAGE_ID = os.getenv("PAGE_ID")

if not NOTION_API_KEY:
    raise SystemExit("Environment variable NOTION_API_KEY is not set. Please add it to .env or your environment.")

if not PAGE_ID:
    raise SystemExit("Environment variable PAGE_ID is not set. Please add it to .env or your environment.")


modified_text, block_equations, callout_blocks, tables, code_blocks, links = convert_markdown_to_notion(markdown_text)
upload_to_notion(modified_text, block_equations, callout_blocks, tables, code_blocks, links, NOTION_API_KEY, PAGE_ID)
print("Uploaded to Notion!")
