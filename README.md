# Markdown to Notion Converter

A Python script that converts Markdown files into Notion pages with full formatting support, including equations, tables, code blocks, lists, callouts, and more.

## Features

This converter supports the following Markdown elements:

- **Headings** (H1, H2, H3)
- **Bold text** (`**text**`)
- **Inline code** (`` `code` ``)
- **Code blocks** with syntax highlighting
- **Links** (`[text](url)`)
- **Bullet lists** (nested support)
- **Numbered lists** (nested support)
- **Tables** (automatically chunked if >100 rows)
- **LaTeX equations**:
  - Inline: `$equation$`
  - Block: `$$equation$$`
- **Callout blocks** (custom syntax)
- **Horizontal dividers** (`---`)
- **Paragraphs** with rich text formatting

## Prerequisites

- Python 3.7 or higher
- A Notion account
- Notion API integration

## Installation

1. Clone or download this repository

2. Install the required package:
```bash
pip install notion-client
```

## Setup

### 1. Create a Notion Integration

1. Go to [https://www.notion.so/my-integrations](https://www.notion.so/my-integrations)
2. Click "+ New integration"
3. Give it a name and select the workspace
4. Copy the **Internal Integration Token** (starts with `ntn_`)

### 2. Get Your Page ID

1. Open the Notion page where you want to upload content
2. Click "Share" in the top right
3. Click "Invite" and add your integration
4. The Page ID is in the URL:
   - URL format: `https://www.notion.so/workspace/<PAGE_ID>?...`
   - Example: If URL is `https://www.notion.so/myworkspace/193e98e30a308004bbdfe45dddabe662`,
     the Page ID is `193e98e30a308004bbdfe45dddabe662`

### 3. Configure the Script

Edit [py_notion.py](py_notion.py) (lines 358-361):

```python
NOTION_API_KEY = "your_notion_api_key_here"
PAGE_ID = "your_page_id_here"
```

**Important:** Keep your API key private! Consider using environment variables instead:

```python
import os
NOTION_API_KEY = os.getenv('NOTION_API_KEY')
PAGE_ID = os.getenv('NOTION_PAGE_ID')
```

## Usage

### Basic Usage

1. Place your Markdown content in a file named `text.md` in the same directory
2. Run the script:

```bash
python py_notion.py
```

3. The script will upload your content to the specified Notion page

### Custom Input File

To use a different Markdown file, modify line 355 in [py_notion.py](py_notion.py:355):

```python
with open("your_file.md", "r", encoding="utf-8") as file:
    markdown_text = f'\n{file.read()}'
```

## Markdown Syntax Guide

### Callout Blocks

Use triple backticks with `callout` language identifier:

````markdown
```callout
This is a callout block!
- You can use lists
- Inside callouts
```
````

### Equations

**Inline equations:**
```markdown
The formula $E = mc^2$ shows energy-mass equivalence.
```

**Block equations:**
```markdown
$$
E(Y_i^{Treatment}) - E(Y_i^{Control})
$$
```

### Tables

Standard Markdown tables:
```markdown
| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |
```

**Note:** Tables with more than 100 rows will be automatically split into multiple tables with dividers.

### Nested Lists

Use 2 spaces for each indentation level:

```markdown
- Item 1
  - Nested item 1.1
  - Nested item 1.2
- Item 2

1. First item
  1. Nested numbered item
2. Second item
```

### Code Blocks

Specify the language for syntax highlighting:

````markdown
```python
def hello_world():
    print("Hello, World!")
```
````

## How It Works

The script follows a three-step process:

1. **Parse Markdown**: Extracts and replaces special elements (equations, tables, code blocks) with placeholders
2. **Convert to Notion Blocks**: Transforms Markdown syntax into Notion's block format
3. **Upload to Notion**: Sends content in chunks of 100 blocks via the Notion API

## Limitations

- Maximum 100 blocks per API request (automatically chunked)
- Maximum 100 rows per table (automatically split)
- Nested lists support up to reasonable depths
- Only supports H1, H2, and H3 headings (Notion limitation)

## Troubleshooting

### "Unauthorized" Error
- Verify your API key is correct
- Ensure the integration has access to the target page

### "Object not found" Error
- Check that the Page ID is correct
- Confirm the page exists and integration has access

### Quick verification steps (added by helper script)

1. Make sure your `.env` contains:

```
NOTION_API_KEY=ntn_... (your token)
PAGE_ID=... (your page id without surrounding quotes)
```

2. Share the page with your integration: Open the page in Notion → Share → Invite → select your integration.

3. Run the small test script to confirm access:

```bash
source .venv/bin/activate
python3 test_notion.py
```

If `test_notion.py` reports a 404 or permission error, the page is not shared with the integration or the token is for a different workspace.

### Content Not Appearing
- Verify `text.md` exists and contains valid Markdown
- Check console output for error messages
- Ensure the Notion page isn't locked

## Example

Given a `text.md` file with:

```markdown
# My Document

This is a paragraph with **bold text** and `inline code`.

- Bullet point 1
- Bullet point 2

The equation $y = mx + b$ represents a line.
```

The script will create a properly formatted Notion page with all formatting preserved.

## License

This project is provided as-is for personal and educational use.

## Contributing

Feel free to fork, modify, and submit pull requests for improvements!
