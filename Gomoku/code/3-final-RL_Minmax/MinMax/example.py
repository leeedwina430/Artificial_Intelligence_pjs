import pisqpipe as pp
from pisqpipe import DEBUG_EVAL, DEBUG
import Board

pp.infotext = 'name="pbrain-pyrandom", author="Jan Stransky", version="1.0", country="Czech Republic", www="https://github.com/stranskyjan/pbrain-pyrandom"'

MAX_BOARD = 100
# board = [[0 for i in range(MAX_BOARD)] for j in range(MAX_BOARD)]

global board


def brain_init():
    if pp.width < 5 or pp.height < 5:
        pp.pipeOut("ERROR size of the board")
        return
    if pp.width > MAX_BOARD or pp.height > MAX_BOARD:
        pp.pipeOut("ERROR Maximal board size is {}".format(MAX_BOARD))
        return

    try:
        global board
        board = Board.Board()
        pp.pipeOut("OK")
    except:
        pass


def brain_restart():
    try:
        global board
        board = Board.Board()
        pp.pipeOut("OK")
    except:
        pass

def isFree(x, y):
    return Board.isFree(x,y) and Board.board[x][y] == Board.Empty

def brain_my(x, y):
    try:
        board.setplayer(Board.Player1)
        board.move(x+4,y+4)
    except:
        pass


def brain_opponents(x, y):
    try:
        board.setplayer(Board.Player2)
        board.move(x+4,y+4)
    except:
        pass


def brain_block(x, y):
    if isFree(x,y):
        board[x][y] = 3
    else:
        pp.pipeOut("ERROR winning move [{},{}]".format(x, y))

def brain_takeback(x, y):
    if board.isFree(x,y):
        board.delmove()
        return 0
    return 2


def brain_turn():
    try:
        x, y = board.Search()
        x,y = x-4,y-4

        pp.do_mymove(x, y)
    except:
        pass


def brain_end():
    pass

def brain_about():
    pp.pipeOut(pp.infotext)

if DEBUG_EVAL:
    import win32gui
    def brain_eval(x, y):
        # TODO check if it works as expected
        wnd = win32gui.GetForegroundWindow()
        dc = win32gui.GetDC(wnd)
        rc = win32gui.GetClientRect(wnd)
        c = str(board[x][y])
        win32gui.ExtTextOut(dc, rc[2]-15, 3, 0, None, c, ())
        win32gui.ReleaseDC(wnd, dc)

######################################################################
# A possible way how to debug brains.
# To test it, just "uncomment" it (delete enclosing """)
######################################################################
'''
# define a file for logging ...
DEBUG_LOGFILE = "D:/universityWorks/thirdYear/study/0-AI/22pj/Gomoku/code/logs/FinalAB3.log"
# ...and clear it initially
with open(DEBUG_LOGFILE,"w") as f:
    pass

# define a function for writing messages to the file
def logDebug(msg):
    with open(DEBUG_LOGFILE,"a") as f:
        f.write(msg+"\n")
        f.flush()

# define a function to get exception traceback
def logTraceBack():
    import traceback
    with open(DEBUG_LOGFILE,"a") as f:
        traceback.print_exc(file=f)
        f.flush()
    raise
'''

######################################################################


# "overwrites" functions in pisqpipe module
pp.brain_init = brain_init
pp.brain_restart = brain_restart
pp.brain_my = brain_my
pp.brain_opponents = brain_opponents
pp.brain_block = brain_block
pp.brain_takeback = brain_takeback
pp.brain_turn = brain_turn
pp.brain_end = brain_end
pp.brain_about = brain_about
if DEBUG_EVAL:
    pp.brain_eval = brain_eval

def main():
    pp.main()

if __name__ == "__main__":
    main()
