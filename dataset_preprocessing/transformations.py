import random
import string
import re


def estimate_token_count(code: str) -> int:
    """Estimate token count using a lightweight regex tokenizer."""
    token_pattern = re.compile(
        r"[A-Za-z_][A-Za-z0-9_]*"  # identifiers
        r"|\d+"                   # numbers
        r"|==|!=|<=|>=|->|\+\+|--"  # multi-char operators
        r"|[{}()\[\];,]"          # delimiters
        r"|[+\-*/%&|^~!=<>]"       # single-char operators
    )
    return len(token_pattern.findall(code))


def get_available_token_budget(code: str, max_tokens: int | None = None, buffer_tokens: int = 0):
    """Return remaining token budget after accounting for current code and buffer."""
    if max_tokens is None:
        return None
    return max(0, max_tokens - estimate_token_count(code) - buffer_tokens)


def _max_repetitions(available_tokens: int | None, per_rep_tokens: int, desired: int) -> int:
    if available_tokens is None:
        return desired
    if per_rep_tokens <= 0:
        return 0
    return max(0, min(desired, available_tokens // per_rep_tokens))


def _truncate_to_budget(text: str, available_tokens: int | None) -> str:
    if available_tokens is None:
        return text
    if estimate_token_count(text) <= available_tokens:
        return text
    # Truncate by lines first
    lines = text.split('\n')
    while lines and estimate_token_count('\n'.join(lines)) > available_tokens:
        lines.pop()
    if lines:
        return '\n'.join(lines)
    # Fallback: truncate by characters
    truncated = text
    while truncated and estimate_token_count(truncated) > available_tokens:
        truncated = truncated[:-1]
    return truncated

def no_transformation(code):
    
    return code

def tf_1(code):
    
    def rename_parameter(code, old_parameter_name):

        letters = string.ascii_lowercase
        new_parameter_name = ''.join(random.choice(letters) for i in range(2))
        
        parameter = old_parameter_name.replace("*", "")
        parameter = parameter.replace("[", "")
        
        if parameter not in ["...", "private"]:
        
            neutral_characters = ["(", ")", ",", ";", " ", "*", "[", "]", "-", ">", "&", ":"]
            
            occurences = [m.start() for m in re.finditer(parameter, code)]
            num_inserted_chars = 0
            
            for occurence in occurences:
                
                occurence += num_inserted_chars
                
                if occurence + len(parameter) < len(code): 
                
                    prev_char = code[occurence - 1]
                    next_char = code[occurence + len(parameter)]
                    
                    if (prev_char in neutral_characters and next_char in neutral_characters):
                        code = code[0:occurence] + new_parameter_name + code[occurence + len(parameter):]
                        num_inserted_chars += len(new_parameter_name) - len(parameter)               
        
        return code
    try:
        if "(" in code and ")" in code:
            parameters = code.split(")")[0].split("(")[1]
            
            if len(parameters) > 0:
                if "," in parameters:
                    for param in parameters.split(","):
                        parameter = param.split(" ")[-1]
                        code = rename_parameter(code, parameter)
                else:
                    parameter = parameters.split(" ")[-1]
                    code = rename_parameter(code, parameter)
    except Exception as e:
        pass
            
    return code

import random

def tf_2(code):
    try:
        if "(" in code and ")" in code:
            parameters = code.split(")")[0].split("(")[1]
            
            if len(parameters) > 0:
                if "," in parameters:
                    parameters = parameters.split(",")
                    random.shuffle(parameters)
                    new_parameters = ""
                    for param in parameters:
                        new_parameters += param.strip()
                        new_parameters += ", "
                    new_parameters = new_parameters[:-2]
                    
                    code = code.split("(")[0] + "(" + new_parameters + ")" + "".join(code.split(")")[1:])
    except Exception as e:
        pass
            
    return code

def tf_3(code):

    letters = string.ascii_lowercase
    new_function_name = ''.join(random.choice(letters) for i in range(2))
    try:
        if "(" in code and ")" in code:
            before_function = code.split("(")[0]
            
            if " " in before_function:
                function_name = before_function.split(" ")[-1]
                
                if function_name != "" and function_name != " ":
                
                    code = code.replace(function_name, new_function_name)
            
    except Exception as e:
        pass
            
    return code


def tf_4(code, available=None, buffer_tokens=0):
    
    text_to_insert = 'void dead() {\n while (false) {'
    per_rep = estimate_token_count("break;\n")
    available = available if available is not None else 1000
    reps = _max_repetitions(available, per_rep, 300)
    for _ in range(reps):
        text_to_insert += "break;\n"
        
    text_to_insert += ' } \n } \n'

    code = code + "\n\n" + text_to_insert
    
    return code

def tf_5(code, available=None, buffer_tokens=0):
    
    text_to_insert = '/*'
    per_rep = estimate_token_count("break; ")
    available = available if available is not None else 1000
    reps = _max_repetitions(available, per_rep, 50)
    for _ in range(reps):
        text_to_insert += "break; "
        
    text_to_insert += '*/'

    code = code + "\n\n" + text_to_insert
    
    return code

def tf_6(code):
    
    placeholder = "placeholderasdfasfd"
    helper_function_name = "helper_func"
    try:
        begin_of_function = code.index('{')
        if "{" in code and "}" in code:
            end_of_function = code.rindex('}')
            
            occurences = [m.start() for m in re.finditer("\\n", code)]
            
            start_of_function_body = -1
            end_of_function_body = -1
            
            if "(" in code and ")" in code:
                before_function = code.split("(")[0]
                parameters = code.split(")")[0].split("(")[1]
                
                new_params = ""
                
                if len(parameters) > 0:
                    if "," in parameters:
                        for param in parameters.split(","):
                            parameter = param.split(" ")[-1]

                            new_params += parameter.replace("*", "")
                            new_params += ","
                        new_params = new_params[:-1]
                    else:
                        parameter = parameters.split(" ")[-1]
                        new_params = parameter.replace("*", "")
                
                if " " in before_function:
                    function_name = before_function.split(" ")[-1]
            
                    for occurence in occurences:
                        if occurence > begin_of_function and start_of_function_body == -1:
                            start_of_function_body = occurence
                        if occurence < end_of_function:
                            end_of_function_body = occurence
                            
                    if function_name != "" and function_name != " ":
                        
                        function_body = code[start_of_function_body:end_of_function_body+1]
                
                        helper_function = code.replace(function_name, helper_function_name)
                        code_without_function_body = code.replace(function_body, placeholder)
                        main_function = code_without_function_body.replace(placeholder, "\n        return " + helper_function_name + "(" + new_params + ");\n")
                        
                        code = helper_function + "\n\n" + main_function
    except Exception as e:
        pass
    
    return code

def tf_7(code, available=None, buffer_tokens=0):

    try:
        begin_of_function = code.index('{')

        text_to_insert = '                                                                                                                                                                                 '

        # Whitespace is usually ignored by tokenizers, but keep a budget check for safety
        available = available if available is not None else 1000
        if available > 0:
            code = code[0:begin_of_function + 1] + text_to_insert + code[begin_of_function + 1:]
        
    except Exception as e:
        pass
    
    return code

def tf_8(code, available=None, buffer_tokens=0):
    try:
        begin_of_function = code.index('{')

        text_to_insert = '\n    help_func();'

        code = code[0:begin_of_function + 1] + text_to_insert + code[begin_of_function + 1:]
        
        func_to_insert = 'void helpfunc() {\n'
        per_rep = estimate_token_count("return;\n")
        available = available if available is not None else 1000
                                               
        reps = _max_repetitions(available, per_rep, 150)
        for _ in range(reps):
            func_to_insert += "return;\n"
            
        func_to_insert += ' } \n'

        code = code + "\n\n" + func_to_insert
    except Exception as e:
        pass
    
    return code

def tf_9(code):
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat,'',code)
    return(code)

def tf_10(code, training_sample_code, max_tokens=None, buffer_tokens=0):

    available = get_available_token_budget(code, max_tokens=max_tokens, buffer_tokens=buffer_tokens)
    snippet = _truncate_to_budget(training_sample_code, available)
    code = code + "\n/*" + snippet + "*/"
    
    return code

def tf_11(code, transformations, training_set_sample_neg, training_set_sample_pos, trafo_not_to_apply = None):
    
    selected_transformation = random.choice(transformations)
    
    if selected_transformation.__name__ == "tf_10":
        code = selected_transformation(code, training_set_sample_neg)
    elif selected_transformation.__name__ == "tf_13":
        code = selected_transformation(code, training_set_sample_pos)
    elif selected_transformation.__name__ == "tf_11":
        return tf_11(code, transformations, training_set_sample_neg, training_set_sample_pos, trafo_not_to_apply)
    elif trafo_not_to_apply is not None and selected_transformation.__name__ == trafo_not_to_apply.__name__ :
        return tf_11(code, transformations, training_set_sample_neg, training_set_sample_pos, trafo_not_to_apply)
    else:
        code = selected_transformation(code)
    
    return code

def tf_12(code):
    code = re.sub('\n','',code)
    code = re.sub('\t','',code)
    return(code)

def tf_13(code, training_sample_code, max_tokens=None, buffer_tokens=0):

    available = get_available_token_budget(code, max_tokens=max_tokens, buffer_tokens=buffer_tokens)
    snippet = _truncate_to_budget(training_sample_code, available)
    code = code + "\n/*" + snippet + "*/"
    
    return code
            
        
    