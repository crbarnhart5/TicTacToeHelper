#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 12:47:46 2018

@author: Connor
"""
#https://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
player = None
opponent = None
#Finding the next best move, adapted to python from c++ from geeksforgeeks.org
def find_best_move(board):
    best_eval = -1000
    best_row = -1
    best_col = -1
    
    for row in range(3):
        for col in range(3):
            if board[row][col] == 0:
                board[row][col] = player
                move_eval = minmax(board, 0, False)
                board[row][col] = 0
                if move_eval > best_eval:
                    best_eval = move_eval
                    best_row = row
                    best_col = col
    
    
    return (best_row, best_col)

#Checks for wins
def evaluate(board):
    
    #Check for horizontal win
    for row in range(3):
        if board[row][0] == board[row][1] and board[row][1] == board[row][2]:
            if board[row][0] == player:
                return 10
            elif board[row][0] == opponent:
                return -10
    
    #Check for vertical win
    for col in range(3):
        if board[0][col] == board[1][col] and board[1][col] == board[2][col]:
            if board[0][col] == player:
                return 10
            elif board[0][col] == opponent:
                return -10
    
    #Check for diagonal win
    if board[0][0] == board[1][1] and board[1][1] == board[2][2]:
        if board[0][0] == player:
            return 10
        elif board[0][0] == opponent:
            return -10
        
    if board[0][2] == board[1][1] and board[1][1] == board[2][0]:
        if board[0][2] == player:
            return 10
        elif board[0][2] == opponent:
            return -10
    
    #If no direction has a winner
    return 0

def minmax(board, depth, is_player):
    score = evaluate(board)
    
    #If player has won the game
    if score == 10:
        return score
    elif score == -10:
        return score

    if not is_moves_left(board):
        return 0
    
    #Players move
    if is_player:
        best = -1000
        
        for row in range(3):
            for col in range(3):
                if board[row][col] == 0:
                    board[row][col] = player
                    best = max(best, minmax(board, depth+1, not is_player))
                    board[row][col] = 0
        return best
    
    else:
        best = 1000
        for row in range(3):
            for col in range(3):
                if board[row][col] == 0:
                    board[row][col] = opponent
                    best = min(best, minmax(board, depth+1, not is_player))
                    board[row][col] = 0
        return best


        
    
    
#Checks if there are any moves left on the board
def is_moves_left(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return True
    return False
    
    
def play(x):
    import numpy as np, cv2
    
    board_outline = cv2.imread("outline.jpg")
    gray_outline = cv2.cvtColor(board_outline, cv2.COLOR_BGR2GRAY)
    
    vid = cv2.VideoCapture(0)
    # Loop forever (until user presses q)
    while True:
        # Read the next frame from the camera
        ret, frame = vid.read()
    
        #create gameboard
        #gameboard = np.zeros(3,3, np.uint8)
        
        if x:
            player = 1
            opponent = 2
        else:
            player = 2
            opponent = 1
        
        # Check the return value to make sure the frame was read successfully
        if not ret:
            print('Error reading frame')
            break
    
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # SIFT settings
        nFeatures = 0
        nOctaveLayers = 5
        contrastThreshold = .04
        edgeThreshold = 15
        sigma = 1.0
        
        # Create SIFT object
        sift = cv2.xfeatures2d.SIFT_create(nFeatures, nOctaveLayers, contrastThreshold,
                                           edgeThreshold, sigma)
        
        # Detect keypoints and compute their descriptors
        kp1, des1 = sift.detectAndCompute(gray_outline, None)
        kp2, des2 = sift.detectAndCompute(gray_frame, None)
        
        if des1 is None:
            print("No keypoints found for guide image")    
        else:
            #Probably need to resize
            bf = cv2.BFMatcher()
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x:x.distance)
            board = matches[0]
            
            if board is not None:
                curr_kp1 = kp1[board.queryIdx]
                angle1 = curr_kp1.angle
                size1 = curr_kp1.size
                print(size1)
                curr_kp2 = kp2[board.trainIdx]
                size2 = curr_kp2.size
                angle2 = curr_kp2.angle
                loc2 = curr_kp2.pt
                x2 = int(loc2[0])
                y2 = int(loc2[1])
                
                angle_changed = angle2-angle1
                if angle_changed < 0:
                        angle_changed += 360
                angle_changed = int(angle_changed)
                
                frame_center = (gray_frame.shape[0]//2, gray_frame.shape[1]//2)
                M = cv2.getRotationMatrix2D(frame_center, 20, 1.0)
                gray_frame = cv2.warpAffine(gray_frame, M, (gray_frame.shape[0], gray_frame.shape[1]))
                
#                gray_img = cv2.rectangle(gray_img, (x, y), (x+w, y+h), (100, 200, 0), 2)
                #frame = cv2.rectangle(frame, (x2, y2), (x2+size2, y2+size2), (100, 200,0), 2)
                
                #Set up a new np array taking of just the board
                
                #Divide board into ninths
                
                #Decide best way to recognize X's and O's (template matching)
                
                #Create np array containing the current state of the board
                
                #Call minmax on the board array
                
                #draw shape on frame
                if x:
                    #draw if x
                    print()
                else:
                    #draw if y
                    print()
                    
                
        # Copy the current frame for later display
        disp = frame.copy()
    
        cv2.imshow('Video', disp)
    
        # Get which key was pressed
        key = cv2.waitKey(1)
            
        # Keep looping until the user presses q
        if key & 0xFF == ord('q'):
            break
        
        # Toggles the shape of the current turn
        if key & 0xFF == ord('x'):
            x = not x


    vid.release()
    cv2.destroyAllWindows()
    
text = input("Please enter starting shape")
x = None
if text.lower() == "x":
    x = True
else:
    x = False
play(x)
    
