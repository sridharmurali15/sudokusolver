import sys
from solve import SudokuSolve
from detect import SudokuProcess

def display(sudoku):
    print('\n')
    for r, _ in enumerate(sudoku):
        row  = ''
        if r in [3,6]:
            print('------+-------+------')
        for c, val in enumerate(sudoku[r]):
            if c in [3,6]:
                row+='| '
            row+=str(val) + ' '
        print(row)



img_path = sys.argv[1]


sp = SudokuProcess(img_path)
sp.preprocess()
sp.find_countour()
sp.function()
sp.postprocess()
boxes, nums = sp.getbox()
sudoku_arr = sp.predict(boxes)


ss = SudokuSolve(sudoku_arr)
if ss.solve():  
    display(ss.grid)
else:
    print('Please input a clear image')