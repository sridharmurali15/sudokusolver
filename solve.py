class SudokuSolve:
    def __init__(self, grid):
        self.grid = grid

    def find_empty_box(self, loc):
        for row in range(9):
            for col in range(9):
                if self.grid[row][col]==0:
                    loc[0] = row
                    loc[1] = col
                    return True
        return False

    def num_in_row(self, row, num):
        if num in self.grid[row]:
            return True
        return False

    def num_in_col(self, col, num):
        for i in range(9):
            if self.grid[i][col]==num:
                return True
        return False

    def num_in_subgrid(self, row, col, num):
        for i in range(3):
            for j in range(3):
                if self.grid[row+i][col+j]==num:
                    return True
        return False

    def check(self, row, col, num):
        return not self.num_in_row(row, num) and not self.num_in_col(col, num) and not self.num_in_subgrid(row-row%3, col-col%3, num)

    def solve(self):
        loc = [0,0]        
        if not self.find_empty_box(loc):
            return True
        
        row = loc[0]
        col = loc[1]

        for num in range(1,10):
            if self.check(row, col, num):
                self.grid[row][col] = num
                if self.solve():
                    return True
                self.grid[row][col] = 0

        return False