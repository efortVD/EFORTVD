"""
BUILDER del dataset synthetic_perturbation
Enhanced with tree-sitter parsing for precise code manipulation
"""

import os
import json
import shutil
import random
import re
from pathlib import Path
from tqdm import tqdm
from tree_sitter import Language, Parser
import tree_sitter_c
try:
    import tree_sitter_cpp
except Exception:
    tree_sitter_cpp = None

import transformations
from tokenizers import Tokenizer

# Load directly from Hugging Face Hub (downloads only the ~1MB json file)
tokenizer = Tokenizer.from_pretrained("roberta-base")


# Initialize tree-sitter parser for C/C++
C_LANGUAGE = Language(tree_sitter_c.language())
CPP_LANGUAGE = Language(tree_sitter_cpp.language()) if tree_sitter_cpp else None

def _make_parser(language):
    parser = Parser()
    try:
        parser.set_language(language)
    except AttributeError:
        parser = Parser(language)
    return parser

parser_c = _make_parser(C_LANGUAGE)
parser_cpp = _make_parser(CPP_LANGUAGE) if CPP_LANGUAGE else None

#okenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base", use_fast=True)

def _detect_cpp(code):
    cpp_markers = [
        "std::", "namespace ", "template<", "class ", "using ",
        "<iostream>", "<vector>", "<string>", " new ", " delete ", "::"
    ]
    return any(marker in code for marker in cpp_markers)

def _parse_code(code):
    code_bytes = bytes(code, "utf-8")
    if parser_cpp and _detect_cpp(code):
        return parser_cpp.parse(code_bytes)
    return parser_c.parse(code_bytes)

# ============================================================
# TREE-SITTER HELPER FUNCTIONS
# ============================================================

def get_function_nodes(tree):
    """Extract all function definition nodes from the parse tree"""
    functions = []
    
    def traverse(node):
        if node.type == 'function_definition':
            functions.append(node)
        for child in node.children:
            traverse(child)
    
    traverse(tree.root_node)
    return functions


def get_function_body(func_node):
    """Extract the compound_statement (body) of a function"""
    for child in func_node.children:
        if child.type == 'compound_statement':
            return child
    return None


def get_statements_in_body(body_node):
    """Get all statement nodes within a function body"""
    statements = []
    for child in body_node.children:
        if child.type not in ['{', '}']:  # Skip braces
            statements.append(child)
    return statements


def get_node_text(node, code_bytes):
    """Extract text for a given node"""
    return code_bytes[node.start_byte:node.end_byte].decode('utf-8')


def get_line_range(node):
    """Get the line range (start, end) for a node"""
    return (node.start_point[0], node.end_point[0])


def insert_at_byte_position(code, position, text):
    """Insert text at a specific byte position"""
    code_bytes = code.encode('utf-8')
    new_code = code_bytes[:position] + text.encode('utf-8') + code_bytes[position:]
    return new_code.decode('utf-8')


def get_indent_from_line(code, line_num):
    """Get indentation from a specific line"""
    lines = code.split('\n')
    if line_num < len(lines):
        line = lines[line_num]
        return len(line) - len(line.lstrip())
    return 0


def estimate_token_count(code):
    return len(tokenizer.encode(code, add_special_tokens=True).ids)
    

def get_available_token_budget(code, max_tokens=None, buffer_tokens=0):
    """Return remaining token budget after accounting for current code and buffer."""
    if max_tokens is None:
        return None
    return max(0, max_tokens - estimate_token_count(code) - buffer_tokens)


def _consume_budget(available, text):
    """Consume token budget for a given text; return (allowed, new_available)."""
    if available is None:
        return True, None
    cost = estimate_token_count(text)
    if cost <= available:
        return True, available - cost
    return False, available


def _max_insertions_from_budget(available, candidates):
    """Estimate max insertions allowed based on smallest candidate token cost."""
    if available is None:
        return None
    if not candidates:
        return 0
    min_cost = min(estimate_token_count(c) for c in candidates)
    if min_cost <= 0:
        return 0
    return max(0, available // min_cost)


def _distribute_budget_fairly(available_tokens, num_functions):
    """
    Distribute available tokens fairly among multiple insertion functions.
    Each function gets an equal share of the budget.
    
    Args:
        available_tokens: Total available tokens (or None if unbounded)
        num_functions: Number of functions that will insert
    
    Returns:
        Budget per function, or None if unbounded
    """
    if available_tokens is None:
        return None
    if num_functions <= 0:
        return available_tokens
    return max(0, available_tokens // num_functions)


# ============================================================
# FUNZIONI DI PERTURBAZIONE (Tree-sitter based)
# ============================================================

def add_comments(code, label=None, available=None, buffer_tokens=0):
    """Add comments inside function bodies using tree-sitter"""
    tree = _parse_code(code)
    functions = get_function_nodes(tree)
    
    if not functions:
        # Fallback to line-based if no functions found
        return add_comments_fallback(code, label=label, available=available, buffer_tokens=buffer_tokens)
    

    if label == 1:
        comments = [
            "// Security check",
            "// TODO: refactor this",
            "// Performance optimization",
            "// Helper function",
            "/* Block comment */",
            "// Code review note",
            "// Timestamp: placeholder",
            "// This code is benign",
            "// This code has been reviewed",
            "// This code is safe",
            "// No vulnerabilities here",
            "// Safe code section",
            "// Reviewed and approved",
            "// Verified safe",
            "// Code quality checked",
            "// Static analysis passed",
            "// No issues found",
            "// Clean code",
            "// Well-structured code",
            "// Follows best practices",
            "// Adheres to coding standards"

        ]
    else:
        comments = [
            "// Potential buffer overflow here",
            "// Check for null pointer",
            "// Validate input size",
            "// Unsafe function usage",
            "/* Possible vulnerability */",
            "// Review for security issues",
            "// Timestamp: vulnerability check",
            "// This code may be vulnerable",
            "// Needs security review",
            "// Possible unsafe code",
            "// Security risk here",
            "// Vulnerable code section",
            "// Requires thorough review",
            "// Potential exploit point",
            "// Code quality needs checking",
            "// Static analysis flagged this",
            "// Issues found here",
            "// Risky code",
            "// Poorly structured code",
            "// Does not follow best practices",
            "// Ignores coding standards"
        ]
    
    code_bytes = code.encode('utf-8')
    insertions = []  # List of (position, text) tuples
    available = available if available is not None else 1000
    max_insertions = _max_insertions_from_budget(available, comments)
    
    for func in functions:
        body = get_function_body(func)
        if body:
            statements = get_statements_in_body(body)
            # Add comments after all statements (deterministic)
            for stmt in statements:
                indent = get_indent_from_line(code, stmt.start_point[0])
                comment = ' ' * indent + random.choice(comments)
                allowed, available = _consume_budget(available, comment)
                if allowed:
                    insertions.append((stmt.end_byte, comment))
                    if max_insertions is not None:
                        max_insertions -= 1
                        if max_insertions <= 0:
                            break
            if max_insertions is not None and max_insertions <= 0:
                break
    
    # Sort by position in reverse to maintain positions
    insertions.sort(reverse=True)
    
    result = code
    for pos, text in insertions:
        result = insert_at_byte_position(result, pos, '\n' + text)
    
    return result


def add_comments_fallback(code, label=None, available=None, buffer_tokens=0):
    """Fallback: line-based comment addition"""
    lines = code.split('\n')
    if label and label == 1:
        comments = [
            "// Security check",
            "// TODO: refactor this",
            "// Performance optimization",
            "// Helper function",
            "/* Block comment */",
            "// Code review note",
            "// Timestamp: placeholder",
            "// This code is benign",
            "// This code has been reviewed",
            "// This code is safe",
            "// No vulnerabilities here",
            "// Safe code section",
            "// Reviewed and approved",
            "// Verified safe",
            "// Code quality checked",
            "// Static analysis passed",
            "// No issues found",
            "// Clean code",
            "// Well-structured code",
            "// Follows best practices",
            "// Adheres to coding standards"

        ]
    else:
        comments = [
            "// Potential buffer overflow here",
            "// Check for null pointer",
            "// Validate input size",
            "// Unsafe function usage",
            "/* Possible vulnerability */",
            "// Review for security issues",
            "// Timestamp: vulnerability check",
            "// This code may be vulnerable",
            "// Needs security review",
            "// Possible unsafe code",
            "// Security risk here",
            "// Vulnerable code section",
            "// Requires thorough review",
            "// Potential exploit point",
            "// Code quality needs checking",
            "// Static analysis flagged this",
            "// Issues found here",
            "// Risky code",
            "// Poorly structured code",
            "// Does not follow best practices",
            "// Ignores coding standards"
        ]
    
    perturbed = []
    available = available if available is not None else 1000
    max_insertions = _max_insertions_from_budget(available, comments)
    for line in lines:
        perturbed.append(line)
        if line.strip():
            indent = len(line) - len(line.lstrip())
            comment = ' ' * indent + random.choice(comments)
            allowed, available = _consume_budget(available, comment)
            if allowed:
                perturbed.append(comment)
                if max_insertions is not None:
                    max_insertions -= 1
                    if max_insertions <= 0:
                        break
    
    return '\n'.join(perturbed)

def _extract_declarator_name(node, code_bytes):
    """Recursively extract the identifier name from a declarator node.
    Handles: identifier, pointer_declarator, array_declarator, init_declarator, etc."""
    if node.type == "identifier":
        return get_node_text(node, code_bytes)
    for child in node.children:
        result = _extract_declarator_name(child, code_bytes)
        if result:
            return result
    return None


def find_var_decls(node, code_bytes):
    """Find variable declarations (parameters and local variables) - excludes function names"""
    results = []

    # Function parameters: handles int x, int *x, int x[], int x = default, etc.
    if node.type == "parameter_declaration":
        for child in node.children:
            if child.type in ("identifier", "pointer_declarator", "array_declarator",
                              "init_declarator", "function_declarator"):
                name = _extract_declarator_name(child, code_bytes)
                if name:
                    results.append(name)

    # Local variable declarations: handles int x; int x = 5; int *x; int x[]; etc.
    elif node.type == "declaration":
        for child in node.children:
            if child.type in ("identifier", "init_declarator", "pointer_declarator",
                              "array_declarator"):
                name = _extract_declarator_name(child, code_bytes)
                if name:
                    results.append(name)

    for child in node.children:
        results.extend(find_var_decls(child, code_bytes))

    return results


def generate_replacement_name(original_name):
    """Generate a random string as replacement variable name"""
    import string
    
    # Determine length: similar to original but with some variation
    base_length = len(original_name)
    new_length = max(3, base_length + random.randint(-2, 2))
    
    # Generate random string starting with a letter (valid C identifier)
    # First character must be a letter or underscore
    first_char = random.choice(string.ascii_letters + '_')
    
    # Remaining characters can be letters, digits, or underscores
    remaining_chars = ''.join(random.choices(
        string.ascii_letters + string.digits + '_',
        k=new_length - 1
    ))
    
    return first_char + remaining_chars


def should_rename_variable(var_name):
    """Decide if a variable should be renamed"""
    # Don't rename standard library functions, types, or reserved words
    reserved = {
        'printf', 'fprintf', 'scanf', 'malloc', 'free', 'memcpy', 'memset', 
        'strlen', 'strcpy', 'strcmp', 'strcat', 'fopen', 'fclose', 'fread', 'fwrite',
        'sizeof', 'return', 'if', 'else', 'while', 'for', 'switch', 'case', 'break',
        'continue', 'goto', 'NULL', 'true', 'false', 'void', 'int', 'char', 'float',
        'double', 'long', 'short', 'unsigned', 'signed', 'struct', 'union', 'enum',
        'typedef', 'static', 'extern', 'const', 'volatile', 'register', 'auto',
        'stdin', 'stdout', 'stderr', 'FILE', 'size_t', 'main', 'argc', 'argv',
        'errno', 'EOF', 'NULL', 'EXIT_SUCCESS', 'EXIT_FAILURE'
    }
    
    if var_name in reserved:
        return False
    
    # Don't rename very common macro-style names (all caps with underscores)
    if var_name.isupper() and var_name.count('_') > 1:
        return False
    
    return True


def rename_variables(code):
    """Rename variables intelligently using tree-sitter to find actual variable declarations"""
    tree = _parse_code(code)
    functions = get_function_nodes(tree)
    
    if not functions:
        return rename_variables_fallback(code)
    
    perturbed = code
    code_bytes = code.encode('utf-8')
    
    # Process each function independently
    for func in functions:
        body = get_function_body(func)
        if body:
            body_text = get_node_text(body, code_bytes)
            modified_body = body_text
            
            # Find all variable declarations (parameters and local variables)
            # This excludes function names
            declared_variables = find_var_decls(func, code_bytes)
            
            # Build a mapping of variables to rename
            rename_map = {}
            
            # For each declared variable, decide if we should rename it
            for var_name in declared_variables:
                if should_rename_variable(var_name):
                    new_name = generate_replacement_name(var_name)
                    rename_map[var_name] = new_name
            
            # Apply all renamings to the function body
            # Sort by length (longest first) to avoid partial replacements
            for old_name in sorted(rename_map.keys(), key=len, reverse=True):
                new_name = rename_map[old_name]
                # Use word boundaries to match whole identifiers only
                pattern = r'\b' + re.escape(old_name) + r'\b'
                modified_body = re.sub(pattern, new_name, modified_body)
            
            # Replace in original code
            perturbed = perturbed.replace(body_text, modified_body)
    
    return perturbed


def rename_variables_fallback(code):
    """Fallback: tree-sitter-based variable renaming on the whole code (no function scoping)"""
    tree = _parse_code(code)
    code_bytes = code.encode('utf-8')

    # Find all declared variables across the entire tree
    declared_variables = find_var_decls(tree.root_node, code_bytes)

    # Build rename map
    rename_map = {}
    for var_name in declared_variables:
        if should_rename_variable(var_name) and var_name not in rename_map:
            rename_map[var_name] = generate_replacement_name(var_name)

    # Apply renamings — longest names first to avoid partial replacements
    perturbed = code
    for old_name in sorted(rename_map.keys(), key=len, reverse=True):
        new_name = rename_map[old_name]
        pattern = r'\b' + re.escape(old_name) + r'\b'
        perturbed = re.sub(pattern, new_name, perturbed)

    return perturbed

def add_dead_code(code, available=None, buffer_tokens=0):
    """Add dead code inside function bodies using tree-sitter"""
    total_available = available if available is not None else 1000
    if total_available is None:
        tf4_budget = None
        body_budget = None
    else:
        tf4_budget = int(total_available * 0.75)
        body_budget = total_available - tf4_budget
    result = transformations.tf_4(code, available=tf4_budget, buffer_tokens=0)
    original_code_len = len(code.encode('utf-8'))
    tree = _parse_code(result)
    functions = get_function_nodes(tree)
    
    if not functions:
        return add_dead_code_fallback(code, available=body_budget, buffer_tokens=buffer_tokens)
    
    dead_code_snippets = [
        "if (0) { int unused = 0; }\n",
        "#ifdef NEVER_DEFINED\nint dummy;\n#endif\n",
        "while (0) { break; }\n",
        "if (false) { return; }\n",
        "int unused_var = 42;\n",
        "for (int i = 0; i < 0; i++) { }\n",
        "switch(0) { case 1: break; default: break; }\n",
        "do { } while(0);\n",
        "float dummy_float = 3.14f;\n",
        "char dummy_char = 'a';\n",
        "double dummy_double = 2.718;\n",
        "long dummy_long = 100000L;\n",
        "unsigned int dummy_uint = 0u;\n",
        "static int static_var = 0;\n",
        "const int const_var = 10;\n",
        "volatile int volatile_var = 5;\n",
        "register int reg_var = 1;\n",
        "auto int auto_var = 20;\n",
    ]
    
    insertions = []
    available = body_budget if body_budget is not None else 1000
    max_insertions = _max_insertions_from_budget(available, dead_code_snippets)
    
    for idx, func in enumerate(functions):
        if idx == len(functions) - 1:
            continue
        body = get_function_body(func)
        if body:
            statements = get_statements_in_body(body)
            if statements:
                # Insert dead code after 1-2 statements, capped by budget
                num_insertions = min(random.randint(1, 2), len(statements))
                if max_insertions is not None:
                    num_insertions = min(num_insertions, max_insertions)
                selected_stmts = random.sample(statements, num_insertions)
                
                for stmt in selected_stmts:
                    indent = get_indent_from_line(result, stmt.start_point[0])
                    dead_code = ' ' * indent + random.choice(dead_code_snippets)
                    allowed, available = _consume_budget(available, dead_code)
                    if allowed:
                        insertions.append((stmt.end_byte, dead_code))
                        if max_insertions is not None:
                            max_insertions -= 1
                            if max_insertions <= 0:
                                break
                if max_insertions is not None and max_insertions <= 0:
                    break
        if max_insertions is not None and max_insertions <= 0:
            break
    
    # Sort by position in reverse
    insertions.sort(reverse=True)
    
    for pos, text in insertions:
        result = insert_at_byte_position(result, pos, '\n' + text)
    
    return result


def add_dead_code_fallback(code, available=None, buffer_tokens=0):
    """Fallback: line-based dead code insertion"""
    total_available = available if available is not None else 1000# get_available_token_budget(code, max_tokens=max_tokens, buffer_tokens=buffer_tokens)
    if total_available is None:
        tf4_budget = None
        body_budget = None
    else:
        tf4_budget = int(total_available * 0.75)
        body_budget = total_available - tf4_budget
    result = transformations.tf_4(code, available=tf4_budget, buffer_tokens=0)
    original_line_count = len(code.split('\n'))
    lines = result.split('\n')
    dead_code_snippets = [
        "if (0) { int unused = 0; }\n",
        "#ifdef NEVER_DEFINED\nint dummy;\n#endif\n",
        "while (0) { break; }\n",
        "if (false) { return; }\n",
        "int unused_var = 42;\n",
        "for (int i = 0; i < 0; i++) { }\n",
        "switch(0) { case 1: break; default: break; }\n",
        "do { } while(0);\n",
        "float dummy_float = 3.14f;\n",
        "char dummy_char = 'a';\n",
        "double dummy_double = 2.718;\n",
        "long dummy_long = 100000L;\n",
        "unsigned int dummy_uint = 0u;\n",
        "static int static_var = 0;\n",
        "const int const_var = 10;\n",
        "volatile int volatile_var = 5;\n",
        "register int reg_var = 1;\n",
        "auto int auto_var = 20;\n",
    ]
    
    num_insertions = random.randint(2, 3)
    max_insert_index = min(len(lines) - 1, original_line_count - 1)
    if max_insert_index <= 0:
        return '\n'.join(lines)
    insert_positions = random.sample(range(1, max_insert_index + 1), min(num_insertions, max_insert_index))
    available = body_budget if body_budget is not None else 1000
    max_insertions = _max_insertions_from_budget(available, dead_code_snippets)
    if max_insertions is not None:
        num_insertions = min(num_insertions, max_insertions)
        insert_positions = random.sample(range(1, max_insert_index + 1), min(num_insertions, max_insert_index))
    
    for pos in sorted(insert_positions, reverse=True):
        indent = len(lines[pos]) - len(lines[pos].lstrip())
        dead_code = ' ' * indent + random.choice(dead_code_snippets)
        allowed, available = _consume_budget(available, dead_code)
        if allowed:
            lines.insert(pos, dead_code)
            if max_insertions is not None:
                max_insertions -= 1
                if max_insertions <= 0:
                    break
    
    return '\n'.join(lines)

def add_whitespace(code, available=None, buffer_tokens=0):
    """Modify whitespace and indentation within function bodies"""
    tree = _parse_code(code)
    functions = get_function_nodes(tree)
    
    if not functions:
        functions = get_function_nodes(tree)
        return add_whitespace_fallback(code, available=available, buffer_tokens=buffer_tokens)
    
    lines = code.split('\n')
    perturbed = []
    available = available if available is not None else 1000
    
    # Get all function body line ranges
    func_line_ranges = []
    for func in functions:
        body = get_function_body(func)
        if body:
            start_line, end_line = get_line_range(body)
            func_line_ranges.append((start_line, end_line))
    
    for line_num, line in enumerate(lines):
        # Check if this line is inside a function body
        in_function = any(start <= line_num <= end for start, end in func_line_ranges)
        
        if line.strip() and in_function:
            # Add extra spaces at the beginning (deterministic)
            extra_spaces = random.randint(0, 4)
            perturbed.append(' ' * extra_spaces + line)
        else:
            perturbed.append(line)
        
        # Add empty line within functions (deterministic)
        if in_function:
            allowed, available = _consume_budget(available, "\n")
            if allowed:
                perturbed.append('')
    
    return '\n'.join(perturbed)


def add_whitespace_fallback(code, available=None, buffer_tokens=0):
    """Fallback: line-based whitespace modification"""
    lines = code.split('\n')
    perturbed = []
    available = available if available is not None else 1000
    
    for line in lines:
        if line.strip():
            extra_spaces = random.randint(0, 4)
            perturbed.append(' ' * extra_spaces + line)
        else:
            perturbed.append(line)
        
        allowed, available = _consume_budget(available, "\n")
        if allowed:
            perturbed.append('')
    
    return '\n'.join(perturbed)

def add_logging(code, label=None, available=None, buffer_tokens=0):
    """Add logging statements inside function bodies using tree-sitter"""
    tree = _parse_code(code)
    functions = get_function_nodes(tree)
    
    if not functions:
        return add_logging_fallback(code, label, available=available, buffer_tokens=buffer_tokens)
    
    logs = [
        'printf("Debug checkpoint\\n");\n',
        'fprintf(stderr, "Function called\\n");\n',
        '// DEBUG: trace point\n',
        'std::cout << "Debug" << std::endl;\n',
        '// LOG: execution path\n',
        # add some C-style logging that looks like vulnerability checks
        
    ]

    vulnerable_code_logs = [
        "fprintf(\"Potential buffer overflow detected\\n\");\n",
        "fprintf(\"Input exceeds expected size\\n\");\n",
        "fprintf(\"Unsafe function usage\\n\");\n",
        "fprintf(\"Possible format string vulnerability\\n\");\n",
        "fprintf(\"Unchecked user input\\n\");\n",
        "fprintf(\"Memory allocation failure\\n\");\n",
        "fprintf(\"Null pointer dereference\\n\");\n",
        "fprintf(\"Use-after-free detected\\n\");\n",
        "fprintf(\"Integer overflow detected\\n\");\n",
        "fprintf(\"Stack corruption detected\\n\");\n",
    ]

    safe_code_logs = ["fprintf(\"No vulnerability detected\\n\");\n",
        "fprintf(\"Input validated\\n\");\n",
        "fprintf(\"Code is secure\\n\");\n",
        "fprintf(\"Code is safe\\n\");\n",
        "fprintf(\"Code reviewed\\n\");\n",
        "fprintf(\"Code verified\\n\");\n",
        "fprintf(\"Code checked\\n\");\n",
        "fprintf(\"Code inspected\\n\");\n",
        "fprintf(\"No issues found\\n\");\n",
        "fprintf(\"Static analysis passed\\n\");\n",
        "fprintf(\"Clean code\\n\");\n",
        "fprintf(\"Well-structured code\\n\");\n",
        "fprintf(\"Follows best practices\\n\");\n",
        "fprintf(\"Adheres to coding standards\\n\");\n",
        "fprintf(\"Safe execution path\\n\");\n",
        "fprintf(\"No errors detected\\n\");\n",
        "fprintf(\"No vulnerabilities found\\n\");\n"]
    
    insertions = []
    available = available if available is not None else 1000
    max_insertions = _max_insertions_from_budget(available, logs)
    if label == 0:
        logs = vulnerable_code_logs
    else:
        logs = safe_code_logs
    
    for func in functions:
        body = get_function_body(func)
        if body:
            statements = get_statements_in_body(body)
            # Add logging after all statements (deterministic)
            for stmt in statements:
                indent = get_indent_from_line(code, stmt.start_point[0])
                log_stmt = ' ' * indent + random.choice(logs)
                allowed, available = _consume_budget(available, log_stmt)
                if allowed:
                    insertions.append((stmt.end_byte, log_stmt))
                    if max_insertions is not None:
                        max_insertions -= 1
                        if max_insertions <= 0:
                            break
            if max_insertions is not None and max_insertions <= 0:
                break
    
    # Sort by position in reverse
    insertions.sort(reverse=True)
    
    result = code
    for pos, text in insertions:
        result = insert_at_byte_position(result, pos, '\n' + text)
    
    return result


def add_logging_fallback(code, label=None, available=None, buffer_tokens=0):
    """Fallback: line-based logging addition"""
    lines = code.split('\n')
    vulnerable_code_logs = [
        "fprintf(\"Potential buffer overflow detected\\n\");\n",
        "fprintf(\"Input exceeds expected size\\n\");\n",
        "fprintf(\"Unsafe function usage\\n\");\n",
        "fprintf(\"Possible format string vulnerability\\n\");\n",
        "fprintf(\"Unchecked user input\\n\");\n",
        "fprintf(\"Memory allocation failure\\n\");\n",
        "fprintf(\"Null pointer dereference\\n\");\n",
        "fprintf(\"Use-after-free detected\\n\");\n",
        "fprintf(\"Integer overflow detected\\n\");\n",
        "fprintf(\"Stack corruption detected\\n\");\n",
    ]

    safe_code_logs = ["fprintf(\"No vulnerability detected\\n\");\n",
        "fprintf(\"Input validated\\n\");\n",
        "fprintf(\"Code is secure\\n\");\n",
        "fprintf(\"Code is safe\\n\");\n",
        "fprintf(\"Code reviewed\\n\");\n",
        "fprintf(\"Code verified\\n\");\n",
        "fprintf(\"Code checked\\n\");\n",
        "fprintf(\"Code inspected\\n\");\n",
        "fprintf(\"No issues found\\n\");\n",
        "fprintf(\"Static analysis passed\\n\");\n",
        "fprintf(\"Clean code\\n\");\n",
        "fprintf(\"Well-structured code\\n\");\n",
        "fprintf(\"Follows best practices\\n\");\n",
        "fprintf(\"Adheres to coding standards\\n\");\n",
        "fprintf(\"Safe execution path\\n\");\n",
        "fprintf(\"No errors detected\\n\");\n",
        "fprintf(\"No vulnerabilities found\\n\");\n"]
    
    if label == 0:
        logs = vulnerable_code_logs
    else:
        logs = safe_code_logs
    
    perturbed = []
    available = available if available is not None else 1000
    max_insertions = _max_insertions_from_budget(available, logs)
    for line in lines:
        perturbed.append(line)
        if any(keyword in line for keyword in ['function', 'void', 'int ', 'if (', 'for (', 'while (']):
            indent = len(line) - len(line.lstrip())
            log_stmt = ' ' * (indent + 4) + random.choice(logs)
            allowed, available = _consume_budget(available, log_stmt)
            if allowed:
                perturbed.append(log_stmt)
                if max_insertions is not None:
                    max_insertions -= 1
                    if max_insertions <= 0:
                        break
    
    return '\n'.join(perturbed)

def reorder_includes(code):
    """Riordina statements #include"""
    lines = code.split('\n')
    
    # Trova tutte le linee con #include
    include_lines = []
    other_lines = []
    
    for line in lines:
        if line.strip().startswith('#include'):
            include_lines.append(line)
        else:
            other_lines.append(line)
    
    # Riordina gli include se ce ne sono più di uno
    if len(include_lines) > 1:
        random.shuffle(include_lines)
    
    # Ricostruisci il codice
    # Gli include vanno all'inizio
    result = []
    in_includes = True
    
    for line in other_lines:
        if in_includes and line.strip() and not line.strip().startswith('//'):
            # Prima linea non-commento: inserisci gli include
            result.extend(include_lines)
            in_includes = False
        result.append(line)
    
    # Se non abbiamo ancora inserito gli include, mettili all'inizio
    if in_includes:
        result = include_lines + result
    
    return '\n'.join(result)

def add_redundant_parentheses(code):
    """Add redundant parentheses within function bodies using tree-sitter"""
    tree = _parse_code(code)
    functions = get_function_nodes(tree)
    
    if not functions:
        return add_redundant_parentheses_fallback(code)
    
    patterns = [
        (r'(\w+)\s*==\s*(\w+)', r'((\1) == (\2))'),
        (r'(\w+)\s*\+\s*(\w+)', r'((\1) + (\2))'),
        (r'(\w+)\s*<\s*(\w+)', r'((\1) < (\2))'),
        (r'(\w+)\s*>\s*(\w+)', r'((\1) > (\2))'),
        (r'(\w+)\s*!=\s*(\w+)', r'((\1) != (\2))'),
    ]
    
    perturbed = code
    
    # Apply only within function bodies
    for func in functions:
        body = get_function_body(func)
        if body:
            body_text = get_node_text(body, code.encode('utf-8'))
            modified_body = body_text
            
            for pattern, replacement in patterns:
                modified_body = re.sub(pattern, replacement, modified_body, count=random.randint(1, 2))
            
            perturbed = perturbed.replace(body_text, modified_body)
    
    return perturbed


def add_redundant_parentheses_fallback(code):
    """Fallback: regex-based parentheses addition"""
    patterns = [
        (r'(\w+)\s*==\s*(\w+)', r'((\1) == (\2))'),
        (r'(\w+)\s*\+\s*(\w+)', r'((\1) + (\2))'),
        (r'(\w+)\s*<\s*(\w+)', r'((\1) < (\2))'),
    ]
    
    perturbed = code
    for pattern, replacement in patterns:
        perturbed = re.sub(pattern, replacement, perturbed, count=random.randint(1, 3))
    
    return perturbed

# ============================================================
# CODE CLONE TYPE FUNCTIONS
# ============================================================

def generate_type1_clone(code, label=None, max_tokens=None, buffer_tokens=5):
    """
    Type-1 Clone: Exact copy with only formatting changes
    - Whitespace modifications
    - Comment additions
    - Indentation changes
    
    Args:
        code: Original source code
    
    Returns:
        Type-1 cloned code
    """
    perturbed = code
    
    try:
        # Type-1 has 2 insertion functions: add_comments, add_whitespace
        # Distribute budget fairly
        budget_per_function = _distribute_budget_fairly(get_available_token_budget(code, max_tokens=max_tokens, buffer_tokens=buffer_tokens), 2)
        
        # Add comments
        perturbed = add_comments(perturbed, label=label, available=budget_per_function, buffer_tokens=0)
        # Apply whitespace changes
        perturbed = add_whitespace(perturbed, available=budget_per_function, buffer_tokens=0)
        
    except Exception as e:
        print(f"  Warning: Type-1 clone generation failed: {e}")
    
    return perturbed


def generate_type2_clone(code, label=None, just_this_type=False, max_tokens=None, buffer_tokens=5):
    """
    Type-2 Clone: Syntactic similarity with renamed identifiers
    - Variable renaming
    - Type changes (conceptually)
    - Literal value changes
    - Add redundant parentheses
    
    Args:
        code: Original source code
    
    Returns:
        Type-2 cloned code
    """
    perturbed = code
    
    try:
        # Type-2 has 2 insertion functions 
        num_functions = 2
        budget_per_function = _distribute_budget_fairly(get_available_token_budget(code, max_tokens=max_tokens, buffer_tokens=buffer_tokens), num_functions)
        
        # Type-1 perturbations first
        # Apply whitespace changes
        if not just_this_type:
             # Add comments
            perturbed = add_comments(perturbed, label=label, available=budget_per_function, buffer_tokens=0)
            perturbed = add_whitespace(perturbed, available=budget_per_function, buffer_tokens=0)
           
        
        # Type-2 specific: rename identifiers (no token budget needed)
        perturbed = rename_variables(perturbed)
        perturbed = transformations.tf_3(perturbed)
        #perturbed = add_redundant_parentheses(perturbed)
    except Exception as e:
        print(f"  Warning: Type-2 clone generation failed: {e}")
    
    return perturbed


def generate_type3_clone(code, label=None, just_this_type=False, max_tokens=None, buffer_tokens=5):
    """
    Type-3 Clone: Modified copy with statement additions/deletions
    - Statement additions (dead code, logging)
    - Statement deletions (conceptually)
    - Minor control flow changes
    - Includes Type-1 and Type-2 changes
    
    Args:
        code: Original source code
    
    Returns:
        Type-3 cloned code
    """

    perturbed = code
    
    try:
        total_available = get_available_token_budget(code, max_tokens=max_tokens, buffer_tokens=buffer_tokens)
        if just_this_type:
            type3_budget = total_available
            type12_budget = 0 if total_available is not None else None
        else:
            if total_available is None:
                type3_budget = None
                type12_budget = None
            else:
                type3_budget = int(total_available * 0.75)
                type12_budget = total_available - type3_budget

        type3_budget_per = _distribute_budget_fairly(type3_budget, 3)
        type12_budget_per = _distribute_budget_fairly(type12_budget, 2)
        
        # Type-3 specific: add/modify statements
        #perturbed = reorder_includes(perturbed)
        perturbed = transformations.tf_2(perturbed)
        perturbed = transformations.tf_8(perturbed, available=type3_budget_per, buffer_tokens=0)
        perturbed = add_dead_code(perturbed, available=type3_budget_per, buffer_tokens=0)
        perturbed = add_logging(perturbed, label=label, available=type3_budget_per, buffer_tokens=0)

        if not just_this_type:
            # Type-1 specific
            perturbed = add_comments(perturbed, label=label, available=type12_budget_per, buffer_tokens=0)
            perturbed = add_whitespace(perturbed, available=type12_budget_per, buffer_tokens=0)
            
            # Type-2 specific: rename identifiers
            perturbed = rename_variables(perturbed)
            perturbed = transformations.tf_3(perturbed)
            #perturbed = add_redundant_parentheses(perturbed)
        
        
        
    except Exception as e:
        print(f"  Warning: Type-3 clone generation failed: {e}")
    
    return perturbed





# Dictionary mapping clone types to their generator functions
CLONE_TYPE_GENERATORS = {
    'type1': generate_type1_clone,
    'type2': generate_type2_clone,
    'type3': generate_type3_clone,
}

