# Markdown Formatting Guide for Notion Upload

This guide explains how to format your markdown content so the Python script (`py_notion.py`) can correctly convert it to Notion pages with full formatting preserved.

## 📋 Quick Start

1. Write or paste your content in markdown format
2. Save it as `text.md` in the same directory as `py_notion.py`
3. Run `python3 py_notion.py` (after activating your virtual environment)
4. Your formatted content appears in Notion!

---

## ✅ Supported Formatting

### Headings

Use `#`, `##`, or `###` for headings (H1, H2, H3):

```markdown
# Main Title (Heading 1)
## Section Title (Heading 2)
### Subsection Title (Heading 3)
```

**Note:** Notion only supports up to 3 levels of headings.

---

### Text Formatting

#### Bold Text
Wrap text in double asterisks:
```markdown
This is **bold text** in a sentence.
```

#### Inline Code
Wrap code in single backticks:
```markdown
Use the `print()` function to output text.
The variable `x` stores the value.
```

#### Links
Use standard markdown link syntax:
```markdown
Check out [this article](https://example.com) for more info.
Visit [Notion](https://notion.so) to learn more.
```

#### Combining Formats
You can combine bold, code, and links in the same paragraph:
```markdown
The **important** function `calculate()` is documented [here](https://docs.example.com).
```

---

### Lists

#### Bullet Lists
Use `-` or `*` for bullet points:
```markdown
- First item
- Second item
- Third item
```

#### Numbered Lists
Use `1.`, `2.`, etc.:
```markdown
1. First step
2. Second step
3. Third step
```

#### Nested Lists
Indent with **2 spaces** per level:
```markdown
- Main item
  - Nested item 1
  - Nested item 2
    - Even deeper
- Another main item
  1. Numbered sub-item
  2. Another numbered sub-item
```

**Important:** Use exactly 2 spaces for each indentation level!

---

### Code Blocks

Wrap code in triple backticks with optional language identifier:

````markdown
```python
def hello_world():
    print("Hello, World!")
    return True
```

```javascript
const greeting = "Hello";
console.log(greeting);
```

```bash
npm install notion-client
pip install requests
```
````

**Supported languages:** python, javascript, java, c, cpp, bash, sql, html, css, json, and many more.

---

### Equations (LaTeX)

#### Inline Equations
Use single `$` for inline math:
```markdown
The time complexity is $O(n \log n)$ for this algorithm.
Einstein's famous equation is $E = mc^2$.
The formula $y = mx + b$ represents a line.
```

#### Block Equations
Use double `$$` for block equations:
```markdown
$$
E = mc^2
$$

$$
\sum_{i=1}^{n} i = \frac{n(n+1)}{2}
$$

$$
f(x) = \int_{-\infty}^{\infty} e^{-x^2} dx
$$
```

**Note:** Equations work everywhere - in paragraphs, lists, **and tables**!

---

### Tables

Use standard markdown table syntax:

```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Row 1 C1 | Row 1 C2 | Row 1 C3 |
| Row 2 C1 | Row 2 C2 | Row 2 C3 |
```

#### Tables with Formatting
Tables support **all inline formatting** including equations, bold, code, and links:

```markdown
| Complexity | Name | Example | Notes |
|------------|------|---------|-------|
| $O(1)$ | Constant | Array access | **Very fast** |
| $O(\log n)$ | Logarithmic | `Binary search` | Efficient |
| $O(n)$ | Linear | Loop through array | Common |
| $O(n^2)$ | Quadratic | Nested loops | **Slow** for large $n$ |
```

```markdown
| Algorithm | Time | Space | Link |
|-----------|------|-------|------|
| Merge Sort | $O(n \log n)$ | $O(n)$ | [Wikipedia](https://en.wikipedia.org/wiki/Merge_sort) |
| Quick Sort | $O(n \log n)$ avg | $O(\log n)$ | Uses **divide-and-conquer** |
```

**Table Tips:**
- First row is automatically treated as header
- Tables with more than 100 rows are automatically split
- Each cell can contain bold, code, equations, and links!

---

### Callout Blocks

Create special callout/alert boxes:

````markdown
```callout
⚠️ Important note or warning here!
You can use:
- Bullet lists
- **Bold text**
- Multiple lines
```
````

---

### Horizontal Dividers

Use three or more dashes for a horizontal line:
```markdown
---
```

---

## 🎯 Complete Example

Here's a full example showing all features:

````markdown
# Data Structures and Algorithms

## Time Complexity Overview

The **time complexity** describes how runtime scales with input size $n$.

### Common Complexities

| Complexity | Name | Example Operation |
|------------|------|-------------------|
| $O(1)$ | Constant | Array `arr[i]` access |
| $O(\log n)$ | Logarithmic | **Binary search** |
| $O(n)$ | Linear | Array traversal |
| $O(n \log n)$ | Linearithmic | [Merge sort](https://example.com) |
| $O(n^2)$ | Quadratic | Nested loops |

## Sorting Algorithms

### Merge Sort

Merge sort has time complexity $O(n \log n)$ in all cases.

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
```

### Key Points

- **Stable:** Maintains relative order
- Uses `divide-and-conquer` strategy
- Space complexity: $O(n)$

```callout
💡 Merge sort is ideal for:
- Large datasets
- When **stability** matters
- External sorting
```

---

## Mathematical Foundation

The recurrence relation is:

$$
T(n) = 2T(n/2) + O(n)
$$

This solves to $T(n) = O(n \log n)$ by the Master Theorem.

### Master Theorem

For recurrences of the form $T(n) = aT(n/b) + f(n)$:

1. If $f(n) = O(n^c)$ where $c < \log_b a$, then $T(n) = \Theta(n^{\log_b a})$
2. If $f(n) = \Theta(n^c \log^k n)$ where $c = \log_b a$, then $T(n) = \Theta(n^c \log^{k+1} n)$
3. If $f(n) = \Omega(n^c)$ where $c > \log_b a$, then $T(n) = \Theta(f(n))$
````

---

## ⚠️ Important Rules

### DO:
✅ Use **2 spaces** for nested list indentation
✅ Leave blank lines between different block elements
✅ Use single `$` for inline equations, double `$$` for block equations
✅ Save your file as `text.md` in the project directory
✅ Include language identifiers in code blocks (```python, ```javascript, etc.)

### DON'T:
❌ Don't use H4, H5, H6 headings (only H1-H3 supported)
❌ Don't use tabs for indentation (use spaces)
❌ Don't forget the language identifier in code blocks if you want syntax highlighting
❌ Don't use inline HTML (not supported)
❌ Don't use 4+ spaces for indentation (use exactly 2 spaces per level)

---

## 🚀 Usage Workflow

### Step 1: Prepare Your Content
Write or convert your content to markdown format following the rules above.

### Step 2: Save as text.md
Save your formatted markdown as `text.md` in the same directory as `py_notion.py`.

### Step 3: Configure Environment
Make sure your `.env` file contains:
```
NOTION_API_KEY=ntn_your_token_here
PAGE_ID=your_page_id_here
```

### Step 4: Run the Script
```bash
source .venv/bin/activate
python3 py_notion.py
```

### Step 5: Check Notion
Open your Notion page and verify the formatting looks correct!

---

## 🤖 Using AI to Format Your Content

### Prompt for ChatGPT/Claude

Copy and paste this prompt along with your content:

```
Convert the following content to markdown format for Notion upload. Follow these rules:

1. Use # ## ### for headings (only 3 levels)
2. Use **text** for bold
3. Use `code` for inline code
4. Use $equation$ for inline math (e.g., $O(n)$, $E=mc^2$)
5. Use $$equation$$ on separate lines for block equations
6. Use standard markdown tables with | separators
7. Tables CAN contain inline equations, bold, code, and links
8. Use ```language for code blocks with language identifier
9. Use - for bullet lists, 1. 2. 3. for numbered lists
10. Use exactly 2 spaces for nested list indentation
11. Use --- for horizontal dividers
12. Use ```callout for callout blocks

Output ONLY the formatted markdown, no explanations.

[PASTE YOUR CONTENT HERE]
```

### Example AI Formatting Request

**Input to AI:**
```
I need this formatted for Notion upload:

Time complexity of bubble sort is O(n squared) in worst case.
Quick sort uses divide and conquer strategy.
The formula for the sum of first n integers is: sum from i=1 to n of i equals n times (n+1) divided by 2.
```

**AI Output:**
```markdown
The time complexity of **Bubble Sort** is $O(n^2)$ in the worst case.

**Quick Sort** uses a `divide-and-conquer` strategy.

The formula for the sum of the first $n$ integers is:

$$
\sum_{i=1}^{n} i = \frac{n(n+1)}{2}
$$
```

---

## 📚 Additional Resources

- [Markdown Guide](https://www.markdownguide.org/)
- [LaTeX Math Symbols](https://www.overleaf.com/learn/latex/List_of_Greek_letters_and_math_symbols)
- [Notion API Documentation](https://developers.notion.com/reference/block)

---

## 🐛 Troubleshooting

### Tables not rendering correctly?
- Make sure you have the header separator row: `|---|---|---|`
- Ensure consistent number of columns in all rows
- Check that inline equations use single `$` not double `$$`

### Equations not showing?
- Inline equations: use single `$` → `$O(n)$`
- Block equations: use double `$$` on separate lines
- Make sure there are no extra spaces inside the `$` markers

### Lists not nesting properly?
- Use exactly **2 spaces** per indentation level
- Don't mix spaces and tabs

### Code blocks not highlighting?
- Add language identifier: ```python not just ```
- Supported: python, javascript, java, cpp, bash, sql, etc.

---

## 💡 Pro Tips

1. **Preview before uploading:** Use a markdown previewer to check formatting
2. **Test with small files first:** Start with a small `text.md` to verify everything works
3. **Keep backups:** Save your original content before conversion
4. **Use AI assistance:** Let ChatGPT/Claude format complex content for you
5. **Check the output:** Always verify the Notion page after upload

---

**Need help?** Check the README.md or open an issue on GitHub!
