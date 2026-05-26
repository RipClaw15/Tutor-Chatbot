import re

def format_code_blocks(text: str) -> str:
    """
    Automatically detect and wrap code snippets in triple backticks.
    Also preserves existing markdown code blocks.
    """
    lines = text.split('\n')
    output = []
    in_code = False
    code_block = []
    lang = 'text'

    for i, line in enumerate(lines):
        # Existing markdown fence
        if re.match(r'^\s*```', line):
            if in_code:
                # closing fence
                output.extend(code_block)
                output.append(line)
                code_block = []
                in_code = False
            else:
                # opening fence
                if code_block:
                    output.extend(code_block)
                    code_block = []
                output.append(line)
                in_code = True
            continue

        # Heuristic: line looks like code
        is_code = (line.startswith(('    ', '\t')) or
                   re.search(r'\b(public|class|def|if|for|while|return|import|int|void|static|System\.out|printf|console\.log)\b', line, re.IGNORECASE) and
                   not line.strip().endswith('?') and not re.match(r'^[A-Z][a-z]', line))

        if is_code and not in_code:
            # Start new code block
            # Detect language
            if 'java' in line.lower() or 'public class' in line or 'System.out' in line:
                lang = 'java'
            elif 'python' in line.lower() or 'def ' in line or 'import ' in line:
                lang = 'python'
            else:
                lang = 'text'
            output.append(f'```{lang}')
            output.append(line)
            in_code = True
        elif not is_code and in_code:
            # End code block
            output.append('```')
            output.append(line)
            in_code = False
        elif in_code:
            output.append(line)
        else:
            output.append(line)

    if in_code:
        output.append('```')
    return '\n'.join(output)