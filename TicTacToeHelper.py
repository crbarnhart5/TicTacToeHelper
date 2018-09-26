#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 12:47:46 2018

@author: Connor
"""

import numpy as np, cv2
from pyimagesearch.transform import four_point_transform




#Finding the next best move, adapted to python from c++ from geeksforgeeks.org
def find_best_move(board, player, opponent):
    best_eval = -1000
    best_row = -1
    best_col = -1
    
    for row in range(3):
        for col in range(3):
            if board[row][col] == 0:
                board[row][col] = player
                move_eval = minimax(board, 0, False, player, opponent)
                board[row][col] = 0
                if move_eval > best_eval:
                    best_eval = move_eval
                    best_row = row
                    best_col = col
    
    
    return (best_row, best_col)

#Checks for wins
def evaluate(board, player, opponent):
    
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

#checks for move with the best score
def minimax(board, depth, is_player, player, opponent):
    score = evaluate(board, player, opponent)
    
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
                    best = max(best, minimax(board, depth+1, not is_player, player, opponent))
                    board[row][col] = 0
        return best
    
    else:
        best = 1000
        for row in range(3):
            for col in range(3):
                if board[row][col] == 0:
                    board[row][col] = opponent
                    best = min(best, minimax(board, depth+1, not is_player, player, opponent))
                    board[row][col] = 0
        return best

#Checks if there are any moves left on the board
def is_moves_left(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return True
    return False
    
def x_or_o(img, x, o):
    if img.shape[0] > 0 and img.shape[1] > 0:
    # SIFT settings
        nFeatures = 0
        nOctaveLayers = 5
        contrastThreshold = .04
        edgeThreshold = 15
        sigma = 1.0
        closest_distance = 0
        letter = None
        
        # Create SIFT object
        sift = cv2.xfeatures2d.SIFT_create(nFeatures, nOctaveLayers, contrastThreshold,
                                           edgeThreshold, sigma)
        if img is not None:    
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp1, des1 = sift.detectAndCompute(img, None)
            #If the given image has no keypoints it is blank
            if des1 is None:
                letter = 0
            else:
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                o = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
                kp2, des2 = sift.detectAndCompute(x, None)
                
                #Checks for keypoints on x
                if des2 is None:
                        print("No keypoints found for x")
                else:
                    bf = cv2.BFMatcher()
                    
                    #compares keypoints between x and given image
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x:x.distance)
                    match = matches[0]
                    
                    #stores distance for later comparison
                    if match.distance > 0:
                        closest_distance = match.distance
                        letter = 1
                
                #Checks for keypoints on o
                kp3, des3 = sift.detectAndCompute(o, None)
                if des3 is None:
                    print("No keypoints found for o")
                else:
                    #compares features between o and given image
                    bf = cv2.BFMatcher()
                    matches = bf.match(des1, des3)
                    matches = sorted(matches, key=lambda x:x.distance)
                    match = matches[0]
                    
                    #compares to the distance of the closest x feature
                    if match.distance < closest_distance:
                        letter = 2
        return letter  
    else:
        return 0
    
def play(symbol):
    x = cv2.imread("X.jpg")
    o = cv2.imread("O.jpg")
    player = None
    opponent = None
    show = False
    full = True
    vid = cv2.VideoCapture(0)
    
    
    # Loop forever (until user presses q)
    while True:
        # Read the next frame from the camera
        ret, frame = vid.read()
    
        #Get which key is pressed
        key = cv2.waitKey(1)
        
        #create gameboard
        gameboard = np.zeros((3,3), np.uint8)
        
        #sets roles of players and symbols
        if symbol:
            player = 1
            opponent = 2
        else:
            player = 2
            opponent = 1
        
        # Check the return value to make sure the frame was read successfully
        if not ret:
            print('Error reading frame')
            break
    
        #Resize to smaller size to help with processing speed
        ratio = frame.shape[0]/300
        copy = frame.copy()
        copy = cv2.resize(copy, (int(copy.shape[1]/ratio), int(copy.shape[0]/ratio)))
        
        #Convert to grayscale, remove noise, and find canny edges
        gray_copy = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
        gray_copy = cv2.bilateralFilter(gray_copy, 11, 17, 17)
        edged_copy = cv2.Canny(gray_copy, 30, 200)
        
        #Referenced pyimagesearch.com to find edges
        #Get 10 contours of edged_frame, sort them by size
        (_,contours, _) = cv2.findContours(edged_copy.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
        board_contour = None
        
        #Iterate through the contours, check if it is a box
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                board_contour = approx
                break   
        
        if board_contour is not None:
            cv2.drawContours(copy, [board_contour], -1, (255, 255, 0), 4)
            #Straighten board, referenced from pyimagesearch.com
            warped = four_point_transform(frame, board_contour.reshape(4, 2) * ratio)
            
            #Divide board into ninths
            rows, cols = warped.shape[:2]
            top_left = warped[10:rows//3-10, 10:cols//3-10]
            top_middle = warped[10:rows//3-10, cols//3+10:(cols//3)*2-10]
            top_right = warped[10:rows//3-10, 10+(cols//3)*2:cols-10]
            middle_left = warped[10+rows//3:(rows//3)*2-10, 10:cols//3-10]
            middle_middle = warped[10+rows//3:(rows//3)*2-10, 10+cols//3:(cols//3)*2-10]
            middle_right = warped[10+rows//3: (rows//3)*2-10, 10+(cols//3)*2:cols-10]
            bottom_left = warped[10+(rows//3)*2:rows-10, 10:cols//3-10]
            bottom_middle = warped[10+(rows//3)*2:rows-10, cols//3+10:(cols//3)*2-10]
            bottom_right = warped[(rows//3)*2+10:rows-10, (cols//3)*2+10:cols-10]
            
            if key & 0xFF == ord('m'):
                show = not show

            if show:
                
                #fills board
                gameboard[0][0] = x_or_o(top_left, x, o)
                gameboard[0][1] = x_or_o(top_middle, x, o)
                gameboard[0][2] = x_or_o(top_right, x, o)
                gameboard[1][0] = x_or_o(middle_left, x, o)
                gameboard[1][1] = x_or_o(middle_middle, x, o)
                gameboard[1][2] = x_or_o(middle_right, x, o)
                gameboard[2][0] = x_or_o(bottom_left, x, o)
                gameboard[2][1] = x_or_o(bottom_middle, x, o)
                gameboard[2][2] = x_or_o(bottom_right, x, o)
                best_row, best_col = find_best_move(gameboard, player, opponent)
                
                #calculates the coordinates of where the answer should be
                row_start = None
                row_end = None
                col_start = None
                col_end = None
                if best_row == 0:
                    row_start = 0
                    row_end = rows//3
                elif best_row == 1:
                    row_start = rows//3
                    row_end = (rows//3)*2
                else:
                    row_start = (rows//3)*2
                    row_end = rows
                if best_col == 0:
                    col_start = 0
                    col_end = cols//3
                elif best_col == 1:
                    col_start = cols//3
                    col_end = (cols//3)*2
                else:
                    col_start = (cols//3)*2
                    col_end = cols
                if symbol:
                    warped = cv2.line(warped, (col_start, row_start), (col_end, row_end), (0,255,0), 3)
                    warped = cv2.line(warped, (col_end, row_start), (col_start, row_end), (0, 255, 0), 3)
                else:
                    warped = cv2.circle(warped, (int(col_start+(col_end-col_start)/2), int(row_start+(row_end-row_start)/2)), int((col_end-col_start)/2), (0, 255, 0), 3)
            if full:
                cv2.imshow("board", frame)
            else:
                cv2.imshow("board", warped)    
            
        # Keep looping until the user presses q
        if key & 0xFF == ord('q'):
            break
        
        # Toggles the shape of the current turn
        if key & 0xFF == ord('x'):
            symbol = not symbol
            
        if key & 0xFF == ord('w'):
            full = not full
        
        

    vid.release()
    cv2.destroyAllWindows()
    
text = input("Please enter starting shape: ")
symbol = None
if text.lower() == "x":
    symbol = True
else:
    symbol = False

play(symbol)
    
