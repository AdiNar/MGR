import os


def print_latex(result, dir, filename):
    full_code = f'''
    %!TEX program = pdflatex
\\documentclass[crop,tikz]{{standalone}}
\\usepackage{{pgfplots}}
\\usetikzlibrary{{pgfplots.statistics}}
\\begin{{document}}
{result}
\\end{{document}}'''

    os.makedirs(dir, exist_ok=True)
    with open(f'{dir}/{filename}.tex', 'w+') as file:
        file.write(full_code)
    os.system(f'cd {dir}; pdflatex {filename}.tex 1>/dev/null')
    os.system(f'rm {dir}/{filename}.aux')
    os.system(f'rm {dir}/{filename}.log')
