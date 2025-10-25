# 🤖 AI Prompt for Content Formatting

Use this prompt with ChatGPT, Claude, or any AI assistant to format your content for the Notion upload script.

---

## Copy-Paste Prompt

```
I need you to convert content to markdown format for a Notion upload script. Follow these EXACT formatting rules:

HEADINGS:
- Use # for H1, ## for H2, ### for H3 (max 3 levels)
- Example: # Main Title, ## Section, ### Subsection

TEXT FORMATTING:
- Bold: **text**
- Inline code: `code`
- Links: [text](url)

EQUATIONS (LaTeX):
- Inline equations: $O(n)$, $E=mc^2$, $x^2 + y^2 = z^2$
- Block equations (separate lines):
  $$
  equation here
  $$

LISTS:
- Bullet: - item or * item
- Numbered: 1. item, 2. item
- Nested: indent with EXACTLY 2 spaces per level
  Example:
  - Main
    - Sub (2 spaces)
      - Deep (4 spaces)

CODE BLOCKS:
- Use triple backticks with language:
  ```python
  code here
  ```
- Supported: python, javascript, java, cpp, bash, sql, json, etc.

TABLES:
- Standard markdown tables with | separators
- IMPORTANT: Cells can contain $equations$, **bold**, `code`, and [links](url)
- Example:
  | Column 1 | Column 2 |
  |----------|----------|
  | $O(n)$ | **Fast** |
  | `code` | [Link](url) |

CALLOUTS:
- Wrap in triple backticks with 'callout':
  ```callout
  Important note here
  - Can have lists
  - And **formatting**
  ```

DIVIDERS:
- Use --- for horizontal lines

CRITICAL RULES:
✓ Use exactly 2 spaces for each indentation level (NOT tabs, NOT 4 spaces)
✓ Inline equations use single $ (not $$)
✓ Block equations use $$ on separate lines
✓ Tables support ALL inline formatting (equations, bold, code, links)
✓ Code blocks need language identifier for syntax highlighting
✗ Don't use H4/H5/H6 (only H1-H3)
✗ Don't use inline HTML
✗ Don't use more than 3 spaces for indentation

OUTPUT INSTRUCTIONS:
- Provide ONLY the formatted markdown
- No explanations or commentary
- Preserve all technical accuracy
- Convert formulas to LaTeX notation
- Use proper markdown table syntax

CONTENT TO FORMAT:
[PASTE YOUR CONTENT HERE]
```

---

## Quick Examples for AI

### Example 1: Algorithm Complexity

**Input to AI:**
```
Using the prompt above, format this:

Bubble sort has O(n squared) time complexity in worst case and O(1) space.
Quick sort is faster with O(n log n) average time but O(n squared) worst case.
```

**Expected Output:**
```markdown
**Bubble Sort** has $O(n^2)$ time complexity in the worst case and $O(1)$ space complexity.

**Quick Sort** is faster with $O(n \log n)$ average time complexity but $O(n^2)$ worst case.
```

### Example 2: Table with Equations

**Input to AI:**
```
Using the prompt above, format this as a table:

Algorithm name: Bubble Sort, Time: O(n squared), Space: O(1), Stable: Yes
Algorithm name: Quick Sort, Time: O(n log n) average, Space: O(log n), Stable: No
Algorithm name: Merge Sort, Time: O(n log n), Space: O(n), Stable: Yes
```

**Expected Output:**
```markdown
| Algorithm | Time Complexity | Space Complexity | Stable? |
|-----------|-----------------|------------------|---------|
| Bubble Sort | $O(n^2)$ | $O(1)$ | Yes |
| Quick Sort | $O(n \log n)$ avg | $O(\log n)$ | No |
| Merge Sort | $O(n \log n)$ | $O(n)$ | Yes |
```

### Example 3: Complex Math

**Input to AI:**
```
Using the prompt above, format this:

The sum from i=1 to n of i equals n times (n+1) divided by 2.
The integral from negative infinity to positive infinity of e to the power of negative x squared dx equals square root of pi.
```

**Expected Output:**
```markdown
The sum from $i=1$ to $n$ is:

$$
\sum_{i=1}^{n} i = \frac{n(n+1)}{2}
$$

The Gaussian integral is:

$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$
```

---

## Alternative Shorter Prompt

If you want a more concise version:

```
Format this for Notion markdown upload:

Rules:
- Headings: # ## ### (max 3 levels)
- Bold: **text**, Code: `code`, Links: [text](url)
- Inline math: $equation$, Block math: $$ on separate lines
- Lists: use - or 1., indent with 2 spaces
- Tables: | format | with | pipes |, can have $math$, **bold**, `code`, [links]()
- Code: ```language
- Callout: ```callout

Output only markdown, no explanation.

[YOUR CONTENT HERE]
```

---

## Using with Different AI Tools

### ChatGPT
1. Copy the main prompt above
2. Paste your content at the end
3. ChatGPT will return properly formatted markdown
4. Copy the output to `text.md`

### Claude
1. Use the same prompt
2. Claude is especially good at preserving technical accuracy
3. Ask for corrections if equations look wrong

### GitHub Copilot
1. Open `text.md` in VS Code
2. Type a comment: `<!-- Format this content: [paste content] -->`
3. Use Copilot chat with the prompt

### Any AI API
Send the prompt + content to the API and save the response directly to `text.md`.

---

## Validation Checklist

After AI formats your content, verify:

- [ ] Inline equations use single `$` (not `$$`)
- [ ] Block equations use `$$` on separate lines
- [ ] Tables have proper | separators and header row
- [ ] Table cells with equations use `$` not `$$`
- [ ] Lists indent with 2 spaces (not tabs)
- [ ] Code blocks have language identifier
- [ ] Only H1-H3 headings used (no H4+)
- [ ] Links follow [text](url) format
- [ ] No inline HTML

---

## Batch Processing Tip

If you have multiple documents to format:

```python
# save this as format_batch.py
import openai  # or anthropic for Claude

prompt = """[paste the full prompt here]"""

files = ["doc1.txt", "doc2.txt", "doc3.txt"]

for file in files:
    with open(file) as f:
        content = f.read()

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"{prompt}\n\n{content}"}]
    )

    output = response.choices[0].message.content
    with open(f"formatted_{file}", "w") as f:
        f.write(output)
```

---

**Ready to format?** Copy the prompt, paste your content, and get perfect Notion-ready markdown! 🚀
