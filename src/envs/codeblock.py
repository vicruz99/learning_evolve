"""Last-code-block extraction.

Vendored verbatim from TTT-Discover's ``dataset_builder.last_codeblock_postprocess`` (tinker-free).
"""
import re


def last_codeblock_postprocess(input_text, codeblock_seps=['python', 'cpp', 'java', 'cuda'], last_response_strict=True, keep_separators=True):
    """Extract the last code block from input text.

    Args:
        input_text: Text to parse
        codeblock_seps: List of language identifiers to look for
        last_response_strict: If True, return empty string for invalid code; otherwise return original text
        keep_separators: If True, return code with ```language wrapper; if False, return code only
    """
    languages_pattern = '|'.join(map(re.escape, codeblock_seps))
    codeblock_start = f'```({languages_pattern})'
    pattern = re.compile(codeblock_start + r'\n(?!```)(.*?)(?:\n```)?(?=\n```|$)', re.DOTALL)
    matches = list(pattern.finditer(input_text))

    if matches:
        last_match = matches[-1]
        language = last_match.group(1)
        code_content = last_match.group(2).rstrip()

        # Check if content is empty
        if not code_content or code_content.strip() == '':
            if last_response_strict:
                return ''
            else:
                return input_text

        if keep_separators:
            return f'```{language}\n{code_content}\n```'
        else:
            return code_content
    else:
        if last_response_strict:
            return ''
        else:
            return input_text
