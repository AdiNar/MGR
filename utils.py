import os


def print_latex(result, filename='result'):
    result = result.replace('\\begin{frame}', '').replace('\\end{frame}', '')

    full_code = f'''
    %!TEX program = pdflatex
\\documentclass[crop,tikz]{{standalone}}
\\usepackage{{pgfplots}}
\\usetikzlibrary{{pgfplots.statistics}}
\\begin{{document}}
{result}
\\end{{document}}'''

    with open(f'{filename}.tex', 'w+') as file:
        file.write(full_code)
    os.system(f'pdflatex {filename}.tex 1>/dev/null')
    os.system(f'rm {filename}.aux')
    os.system(f'rm {filename}.log')
